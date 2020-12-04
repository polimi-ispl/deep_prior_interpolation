import torch
import numpy as np


def threshold(in_content: torch.Tensor or np.ndarray, thresh: float = 0.) -> torch.Tensor or np.ndarray:
    if isinstance(in_content, torch.Tensor):
        p = (in_content > thresh).float()
        m = (in_content < -thresh).float()
    else:
        p = (in_content > thresh).astype(np.float)
        m = (in_content < -thresh).astype(np.float)
    return in_content * (p + m)


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
    """
    
    def __init__(self, data: torch.Tensor, mask: torch.Tensor, weight: float,
                 forward_fn: callable, adjoint_fn: callable):
        super(POCS, self).__init__()
        self.weighted_data = weight * data
        self.weighted_mask = 1 - weight * mask
        self.weight = weight
        self.forward_fn = forward_fn
        self.adjoint_fn = adjoint_fn
    
    def __str__(self):
        return "POCS"
    
    def forward(self, x, thresh: float = None):
        _ = self.forward_fn(x)
        _ = threshold(_, thresh)
        _ = self.adjoint_fn(_)
        return self.weighted_data + self.weighted_mask * _


class POCS_FFT(POCS):
    def __init__(self, data: torch.Tensor, mask: torch.Tensor,
                 weight: float = 1., fft_range: tuple = None, dim: tuple = None):
        
        if torch.__version__ == '1.7.0':
            forward_fn = lambda x: torch.rfft(x, signal_ndim=data.ndim - 2, onesided=False)
            adjoint_fn = lambda x: torch.irfft(x, signal_ndim=data.ndim - 2, onesided=False)
        else:
            forward_fn = lambda x: torch.fft.fftn(x, dim=dim, s=fft_range)
            adjoint_fn = lambda x: torch.fft.ifftn(x, dim=dim, s=data.shape)
        super(POCS_FFT, self).__init__(data, mask, weight, forward_fn, adjoint_fn)


__all__ = [
    "POCS",
    "pocs_fk_fn",
]
