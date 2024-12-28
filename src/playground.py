import torch 
from models.loss import (PairPushLoss, ListOneLoss, ListAllLoss, LambdaLoss, 
                            lambdaRank_scheme, NeuralNDCG, ApproxNDCGLoss)


def main():
    loss_obj = NeuralNDCG()
    y_pred = torch.Tensor([[-0.1, 2, 9, 5]]).to("cuda")
    y_true = torch.Tensor([[1, 2, 5, 9]]).to("cuda")
    print(y_true)
    print(y_true.min())
    print(y_true.min()+(y_true.max()-y_true))
    loss = loss_obj.forward(y_pred, y_true)
    print(loss)
    # y_true[padded_mask] = float("-inf")

if __name__ == "__main__":
    main()