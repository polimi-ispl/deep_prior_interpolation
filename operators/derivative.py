import torch

__all__ = [
    "VerticalGrad",
]


class VerticalGrad(torch.nn.Module):
    def __init__(self):
        super(VerticalGrad, self).__init__()
    
    def forward(self, x):
        y = torch.zeros_like(x)
        y[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
        return y
    
    def adjoint(self, y):
        x = torch.zeros_like(y)
        x[:, :, :-1, :] -= y[:, :, :-1, :]
        x[:, :, 1:, :] += y[:, :, :-1, :]
        return x
