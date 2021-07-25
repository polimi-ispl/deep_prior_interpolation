import torch
import numpy as np


def threshold(in_content: torch.Tensor or np.ndarray, thresh: float = None) -> torch.Tensor or np.ndarray:
    if thresh is None:
        thresh = compute_threshold(in_content)

    if isinstance(in_content, torch.Tensor):
        p = (in_content > thresh).float()
        m = (in_content < -thresh).float()
    else:
        p = (in_content > thresh).astype(np.float)
        m = (in_content < -thresh).astype(np.float)
    return in_content * (p + m)


def compute_threshold(in_content: torch.Tensor or np.ndarray, perc: float = 10):
    return float(in_content.max() * perc/100)


def pocs_fk_fn(out: torch.Tensor or np.ndarray,
               data: torch.Tensor or np.ndarray,
               mask: torch.Tensor or np.ndarray,
               th: float, alp: float = 0.2) -> torch.Tensor or np.ndarray:
    assert type(out) == type(data) == type(mask)
    
    if isinstance(out, torch.Tensor):
        dim = out.ndim - 2
        _ = torch.rfft(out, dim, onesided=False)
        _ = threshold(_, th)
        _ = torch.irfft(_, dim, onesided=False)
    else:
        _ = np.fft.rfftn(out)
        _ = threshold(_, th)
        _ = np.fft.irfftn(_)
    
    pocs = _ * (1 - alp * mask)
    res = alp * data + pocs
    
    return res


class POCS(torch.nn.Module):
    """
    Base implementation of POCS method.
    Arguments:
        data: tensor of data
        mask: binary tensor of mask
        weight: weighting factor between the true data and the POCS one
        forward_fn: transform forward function (from tensor to tensor)
        adjoint_fn: transform adjoint (or inverse) function (from tensor to tensor)
        thresh_perc: percentile for computing the threshold
    """
    
    def __init__(self, data: torch.Tensor, mask: torch.Tensor, weight: float,
                 forward_fn: callable, adjoint_fn: callable, thresh_perc: float = None):
        super(POCS, self).__init__()
        self.weighted_data = weight * data
        self.weighted_mask = torch.ones_like(mask) - weight * mask
        self.weight = weight
        self.forward_fn = forward_fn
        self.adjoint_fn = adjoint_fn
        self.thresh_perc = thresh_perc
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        fn = str(self.forward_fn).replace('<function ', '')
        fn = fn.split('_')[0]
        return "POCS(weight=%.3f, fn=%s)" % (self.weight, fn)
    
    def forward(self, x, thresh: float = None):
        _ = self.forward_fn(x)
        th = compute_threshold(_, self.thresh_perc) if self.thresh_perc is not None else thresh
        _ = threshold(_, th)
        _ = self.adjoint_fn(_)
        return self.weighted_data + self.weighted_mask * _


__all__ = [
    "POCS",
]
