import numpy as np
import torch


def snr(output: np.ndarray or torch.Tensor,
        target: np.ndarray or torch.Tensor
        ) -> np.float or torch.Tensor:
    """Compute the Signal-to-Noise Ratio in dB"""
    
    if target.shape != output.shape:
        raise ValueError('There is something wrong with the dimensions!')
    
    if isinstance(output, torch.Tensor) and isinstance(target, torch.Tensor):
        return 10 * torch.log10(torch.sum(target ** 2) / torch.sum((target - output) ** 2))
    else:
        return 10 * np.log10(np.sum(target ** 2) / np.sum((target - output) ** 2))


def pcorr(output: np.ndarray or torch.Tensor,
          target: np.ndarray or torch.Tensor
          ) -> np.float or torch.Tensor:
    """
    Compute the Pearson Correlation Coefficient
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """
    
    if target.shape != output.shape:
        raise ValueError('There is something wrong with the dimensions!')
    
    if isinstance(output, torch.Tensor) and isinstance(target, torch.Tensor):
        mean_tar = torch.mean(target)
        mean_out = torch.mean(output)
        tar_dif = target - mean_tar
        out_dif = output - mean_out
        pcorr_value = torch.sum(tar_dif*out_dif)/(torch.sqrt(torch.sum(tar_dif**2))*torch.sqrt(torch.sum(out_dif**2)))
    else:
        mean_tar = np.mean(target)
        mean_out = np.mean(output)
        tar_dif = target - mean_tar
        out_dif = output - mean_out
        pcorr_value = np.sum(tar_dif*out_dif)/(np.sqrt(np.sum(tar_dif**2))*np.sqrt(np.sum(out_dif**2)))
    
    return pcorr_value


class History:
    def __init__(self):
        self.loss = []
        self.snr = []
        self.pcorr = []
    
    def __getitem__(self, item):
        return self.loss[item], self.snr[item], self.pcorr[item]
    
    def __setitem__(self, idx, values):
        l, s, p = values
        self.loss[idx] = l
        self.snr[idx] = s
        self.pcorr[idx] = p
    
    def append(self, values):
        l, s, p = values
        self.loss.append(l)
        self.snr.append(s)
        self.pcorr.append(p)
    
    def __len__(self):
        assert len(self.loss) == len(self.snr) == len(self.pcorr)
        return len(self.loss)
    
    def __str__(self):
        return "Loss : " + str(self.loss) + "\nSNR  : " + str(self.snr) + "\nPCORR: " + str(self.pcorr)
        
    def __repr__(self):
        return self.__str__()
        
        
__all__ = [
    'snr',
    'pcorr',
    'History',
]
