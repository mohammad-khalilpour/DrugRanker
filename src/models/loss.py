import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class PairPushLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1):
        super(PairPushLoss, self).__init__()
        self.loss = nn.Softplus(beta=1, threshold=50)
        self.relu = nn.ReLU()
        self.alpha = alpha
        self.beta = beta
        #self.gamma = gamma

    def forward(self, diff, labels1, labels2, sign):
        y = np.array(labels1) == np.array(labels2)
        pn_pairs = (y == 0)   # for (sens, insens) or (insens, sens) pairs
        pp_pairs = (y == 1) & (np.array(labels1) == 1)   # for (sens,sens) pairs
        nn_pairs = (y == 1) & (np.array(labels1) == 0)   # for (insens,insens) pairs
        # sign should be 1 if (+,-) pair and -1 if (-,+) pair

        bloss = 0
        if sum(pp_pairs):
            bloss = self.alpha*torch.mean(self.loss(-sign[pp_pairs]*diff[pp_pairs]), dim=0)
        if sum(pn_pairs):
            bloss += (1-self.alpha-self.beta)*torch.mean(self.loss(-sign[pn_pairs]*diff[pn_pairs]), dim=0)
        if sum(nn_pairs):
            bloss += self.beta*torch.mean(self.loss(-sign[nn_pairs]*diff[nn_pairs]), dim=0)
        return bloss

class ListOneLoss(nn.Module):
    def __init__(self, M=1):
        super(ListOneLoss, self).__init__()
        self.M = M

    def forward(self, y_pred, y_true):
        pred_max = f.softmax(y_pred/self.M, dim=0) + 1e-9
        true_max = f.softmax(-y_true/self.M, dim=0)  # need to reverse the sign
        pred_log = torch.log(pred_max)
        return torch.mean(-torch.sum(true_max*pred_log))

class ListAllLoss(nn.Module):
    def __init__(self, M=0.5):
        super(ListAllLoss, self).__init__()
        self.M = M

    def forward(self, y_pred, y_label):
        pred_max = f.softmax(y_pred/self.M, dim=1) + 1e-9
        pred_log = torch.log(pred_max)
        return torch.mean(-torch.sum(y_label*pred_log))

class LambdaLoss(nn.Module):
    def __init__(self, eps=1e-10, padded_value_indicator=-1, weighing_scheme=None, k=None, sigma=1.0, mu=10.0,
                 reduction="mean", reduction_log="binary"):
        super(LambdaLoss, self).__init__()
        self.eps = eps
        self.padded_value_indicator = padded_value_indicator
        self.weighing_scheme = weighing_scheme
        self.k = k
        self.sigma = sigma
        self.mu = mu
        self.reduction = reduction
        self.reduction_log = reduction_log

    def forward(self, y_pred, y_true):
        device = y_pred.device
        y_pred = y_pred.clone() 
        y_true = y_true.clone()

        k = self.k or y_true.shape[1]

        y_true = y_true.float()
        y_true = (y_true.max()-y_true) + y_true.min()

        padded_mask = y_true == self.padded_value_indicator
        y_pred[padded_mask] = float("-inf")
        y_true[padded_mask] = float("-inf")

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs >= 0)

        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        if self.k is not None:
            ndcg_at_k_mask[:self.k, :self.k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :self.k], dim=-1).clamp(min=self.eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        if self.weighing_scheme is None:
            weights = 1.0
        else:
            weights = self.weighing_scheme(G, D, self.mu, true_sorted_by_preds)

        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.0)
        weighted_probas = (torch.sigmoid(self.sigma * scores_diffs).clamp(min=self.eps) ** weights).clamp(min=self.eps)

        if self.reduction_log == "natural":
            losses = torch.log(weighted_probas)
        elif self.reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            raise ValueError("Reduction logarithm base can be either natural or binary")

        if self.reduction == "sum":
            loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
        elif self.reduction == "mean":
            loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss

def lambdaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])

class ApproxNDCGLoss(nn.Module):                 
    def __init__(self, eps=1e-10, padded_value_indicator=-1, alpha=1.):
        """
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :param alpha: score difference weight used in the sigmoid function
        """
        super(ApproxNDCGLoss, self).__init__()
        self.eps = eps
        self.padded_value_indicator = padded_value_indicator
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        """
        Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
        Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :return: loss value, a torch.Tensor
        """
        device = y_pred.device
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        y_true = y_true.float()
        y_true = (y_true.max()-y_true) + y_true.min()

        padded_mask = y_true == self.padded_value_indicator
        y_pred[padded_mask] = float("-inf")
        y_true[padded_mask] = float("-inf")

        # Here we sort the true and predicted relevancy scores.
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)
        padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=self.eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
        scores_diffs[~padded_pairs_mask] = 0.
        approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-self.alpha * scores_diffs).clamp(min=self.eps)), dim=-1)
        approx_D = torch.log2(1. + approx_pos)
        approx_NDCG = torch.sum((G / approx_D), dim=-1)

        return -torch.mean(approx_NDCG)
        
class NeuralNDCG(nn.Module):
    def __init__(self, padded_value_indicator=-1, temperature=1.0, powered_relevancies=True, k=None,
                 stochastic=False, n_samples=32, beta=0.1, log_scores=True):
        super(NeuralNDCG, self).__init__()
        self.padded_value_indicator = padded_value_indicator
        self.temperature = temperature
        self.powered_relevancies = powered_relevancies
        self.k = k
        self.stochastic = stochastic
        self.n_samples = n_samples
        self.beta = beta
        self.log_scores = log_scores

    def forward(self, y_pred, y_true):
        dev = y_pred.device

        y_true = y_true.float()
        y_true = (y_true.max()-y_true) + y_true.min()

        k = self.k or y_true.shape[1]
        mask = (y_true == self.padded_value_indicator)

        if self.stochastic:
            P_hat = stochastic_neural_sort(
                y_pred.unsqueeze(-1),
                n_samples=self.n_samples,
                tau=self.temperature,
                mask=mask,
                beta=self.beta,
                log_scores=self.log_scores,
                device=dev,
            )
        else:
            P_hat = deterministic_neural_sort(
                y_pred.unsqueeze(-1),
                tau=self.temperature,
                mask=mask,
                device=dev,
            ).unsqueeze(0)

        # Perform Sinkhorn scaling to obtain doubly stochastic permutation matrices
        P_hat = sinkhorn_scaling(
            P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
            mask.repeat_interleave(P_hat.shape[0], dim=0),
            tol=1e-6, max_iter=50
        )
        P_hat = P_hat.view(
            int(P_hat.shape[0] / y_pred.shape[0]),
            y_pred.shape[0],
            P_hat.shape[1],
            P_hat.shape[2]
        )

        P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.0)
        y_true_masked = y_true.masked_fill(mask, 0.0).unsqueeze(-1).unsqueeze(0)

        if self.powered_relevancies:
            y_true_masked = torch.pow(2.0, y_true_masked) - 1.0

        ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
        discounts = (torch.tensor(1.0) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.0)).to(dev)
        discounted_gains = ground_truth * discounts

        if self.powered_relevancies:
            idcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)
        else:
            idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)

        discounted_gains = discounted_gains[:, :, :k]
        ndcg = discounted_gains.sum(dim=-1) / (idcg + 1e-10)
        idcg_mask = idcg == 0.0
        ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.0)

        assert (ndcg < 0.0).sum() == 0, "Every NDCG should be non-negative"
        if idcg_mask.all():
            return torch.tensor(0.0, device=dev)

        mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])
        return -1.0 * mean_ndcg


def sinkhorn_scaling(mat, mask=None, tol=1e-6, max_iter=50):
    """
    Sinkhorn scaling procedure.
    :param mat: a tensor of square matrices of shape N x M x M, where N is batch size
    :param mask: a tensor of masks of shape N x M
    :param tol: Sinkhorn scaling tolerance
    :param max_iter: maximum number of iterations of the Sinkhorn scaling
    :return: a tensor of (approximately) doubly stochastic matrices
    """
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)

    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=1e-10)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=1e-10)

        if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
            break

    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)

    return mat


def deterministic_neural_sort(s, tau, mask, device="cuda"):
    """
    Deterministic neural sort.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :return: approximate permutation matrices of shape [batch_size, slate_length, slate_length]
    """

    n = s.size()[1]
    one = torch.ones((n, 1), dtype=torch.float32, device=device)
    s = s.masked_fill(mask[:, :, None], -1e8)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    A_s = A_s.masked_fill(mask[:, :, None] | mask[:, None, :], 0.0)

    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))

    temp = [n - m + 1 - 2 * (torch.arange(n - m, device=device) + 1) for m in mask.squeeze(-1).sum(dim=1)]
    temp = [t.type(torch.float32) for t in temp]
    temp = [torch.cat((t, torch.zeros(n - len(t), device=device))) for t in temp]
    scaling = torch.stack(temp).type(torch.float32).to(device)  # type: ignore

    s = s.masked_fill(mask[:, :, None], 0.0)
    C = torch.matmul(s, scaling.unsqueeze(-2))

    P_max = (C - B).permute(0, 2, 1)
    P_max = P_max.masked_fill(mask[:, :, None] | mask[:, None, :], -np.inf)
    P_max = P_max.masked_fill(mask[:, :, None] & mask[:, None, :], 1.0)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


def stochastic_neural_sort(s, n_samples, tau, mask, beta=1.0, log_scores=True, eps=1e-10, device="cuda"):
    """
    Stochastic neural sort. Please note that memory complexity grows by factor n_samples.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param n_samples: number of samples (approximations) for each permutation matrix
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :param beta: scale parameter for the Gumbel distribution
    :param log_scores: whether to apply the logarithm function to scores prior to Gumbel perturbation
    :param eps: epsilon for the logarithm function
    :return: approximate permutation matrices of shape [n_samples, batch_size, slate_length, slate_length]
    """
    batch_size = s.size()[0]
    n = s.size()[1]
    s_positive = s + torch.abs(s.min())
    samples = beta * sample_gumbel([n_samples, batch_size, n, 1], device=device)
    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * batch_size, n, 1)
    mask_repeated = mask.repeat_interleave(n_samples, dim=0)

    P_hat = deterministic_neural_sort(s_perturb, tau, mask_repeated)
    P_hat = P_hat.view(n_samples, batch_size, n, n)
    return P_hat


def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=-1):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=-1):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)