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
    def __init__(self, eps=1e-10, padded_value_indicator=-1, weighing_scheme=None, k=None, sigma=1., mu=10.,
                 reduction="sum", reduction_log="binary"):
        super(LambdaLoss, self).__init__()
        self.eps = eps
        self.padded_value_indicator = padded_value_indicator
        self.weighting_scheme = weighing_scheme
        self.k = k
        self.sigma = sigma
        self.mu = mu
        self.reduction = reduction
        self.reduction_log = reduction_log

    def forward(self, y_pred, y_true):
        device = y_pred.device
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        padded_mask = y_true == self.padded_value_indicator
        y_pred[padded_mask] = float("-inf")
        y_true[padded_mask] = float("-inf")

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if self.weighing_scheme != "ndcgLoss1_scheme":
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:self.k, :self.k] = 1

        true_sorted_by_preds.clamp_(min=0.0)
        y_true_sorted.clamp_(min=0.0)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1.0 + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :self.k], dim=-1).clamp(min=self.eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        if self.weighing_scheme is None:
            weights = 1.0
        else:
            weights = globals()[self.weighing_scheme](G, D, self.mu, true_sorted_by_preds)  # type: ignore

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


class NeuralNDCGLoss(nn.Module):
    def __init__(self, padded_value_indicator=-1, temperature=1.0, powered_relevancies=True, k=None,
                 stochastic=False, n_samples=32, beta=0.1, log_scores=True):
        super(NeuralNDCGLoss, self).__init__()
        self.padded_value_indicator = padded_value_indicator
        self.temperature = temperature
        self.powered_relevancies = powered_relevancies
        self.k = k
        self.stochastic = stochastic
        self.n_samples = n_samples
        self.beta = beta
        self.log_scores = log_scores

    def forward(self, y_pred, y_true):
        device = y_pred.device

        if self.k is None:
            self.k = y_true.shape[1]

        mask = (y_true == self.padded_value_indicator)

        if self.stochastic:
            P_hat = stochastic_neural_sort(y_pred.unsqueeze(-1), n_samples=self.n_samples, tau=self.temperature,
                                           mask=mask, beta=self.beta, log_scores=self.log_scores)
        else:
            P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=self.temperature, mask=mask).unsqueeze(0)

        P_hat = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
                                 mask.repeat_interleave(P_hat.shape[0], dim=0), tol=1e-6, max_iter=50)
        P_hat = P_hat.view(int(P_hat.shape[0] / y_pred.shape[0]), y_pred.shape[0], P_hat.shape[1], P_hat.shape[2])

        P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.0)
        y_true_masked = y_true.masked_fill(mask, 0.0).unsqueeze(-1).unsqueeze(0)

        if self.powered_relevancies:
            y_true_masked = torch.pow(2.0, y_true_masked) - 1.0

        ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
        discounts = (1.0 / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.0)).to(device)
        discounted_gains = ground_truth * discounts

        if self.powered_relevancies:
            idcg = dcg(y_true, y_true, ats=[self.k]).permute(1, 0)
        else:
            idcg = dcg(y_true, y_true, ats=[self.k], gain_function=lambda x: x).permute(1, 0)

        discounted_gains = discounted_gains[:, :, :self.k]
        ndcg = discounted_gains.sum(dim=-1) / (idcg + DEFAULT_EPS)
        idcg_mask = idcg == 0.0
        ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.0)

        if idcg_mask.all():
            return torch.tensor(0.0, device=device)

        mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])
        return -1.0 * mean_ndcg


import torch
import torch.nn as nn


class LambdaRankLoss(nn.Module):
    def __init__(self, eps=1e-6, sigma=1.0, k=None, reduction='mean'):
        """
        LambdaRank loss implementation.
        :param eps: Numerical stability factor for log and division operations.
        :param sigma: A parameter that controls the steepness of the sigmoid function in LambdaRank.
        :param k: Rank at which the loss is truncated (optional).
        :param reduction: Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
        """
        super(LambdaRankLoss, self).__init__()
        self.eps = eps
        self.sigma = sigma
        self.k = k
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Compute the LambdaRank loss.
        :param y_pred: Model's predicted scores, shape [batch_size, slate_length]
        :param y_true: Ground truth relevance scores, shape [batch_size, slate_length]
        :return: Computed LambdaRank loss.
        """
        device = y_pred.device

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]

        padded_pairs_mask = torch.isfinite(true_diffs)
        padded_pairs_mask &= true_diffs > 0

        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs.masked_fill_(torch.isnan(scores_diffs), 0)

        weighted_probas = torch.sigmoid(self.sigma * scores_diffs).clamp(min=self.eps)

        losses = torch.log(weighted_probas)

        losses = losses * padded_pairs_mask.float()

        if self.reduction == 'sum':
            loss = -torch.sum(losses)
        elif self.reduction == 'mean':
            loss = -torch.mean(losses)
        else:
            raise ValueError("Reduction method can be either 'mean' or 'sum'")

        return loss

