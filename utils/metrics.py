import numpy as np
import torch
from utils.generic import ten_digit


def snr(output: np.ndarray or torch.Tensor,
        target: np.ndarray or torch.Tensor
        ) -> float or torch.Tensor:
    """Compute the Signal-to-Noise Ratio in dB"""
    
    if target.shape != output.shape:
        raise ValueError('There is something wrong with the dimensions!')
    
    if isinstance(output, torch.Tensor) and isinstance(target, torch.Tensor):
        return 10 * torch.log10(torch.sum(target ** 2) / torch.sum((target - output) ** 2))
    else:
        return 10 * np.log10(np.sum(target ** 2) / np.sum((target - output) ** 2))


def pcorr(output: np.ndarray or torch.Tensor,
          target: np.ndarray or torch.Tensor
          ) -> float or torch.Tensor:
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
    def __init__(self, epochs):
        self.loss = []
        self.snr = []
        self.pcorr = []
        self.lr = []
        self.msg = "Iter %s, Loss = %+.2e, SNR = %+2.2f dB, PCORR = %+.2f %%"
        self.zfill = ten_digit(epochs)
    
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
        assert len(self.loss) == len(self.snr) == len(self.pcorr) == len(self.lr)
        return len(self.loss)
    
    def log_message(self, idx):
        return self.msg % (str(idx + 1).zfill(self.zfill),
                           self.loss[idx], self.snr[idx], self.pcorr[idx] * 100)
        
    def __str__(self):
        return   "Loss : " + str(self.loss) + \
               "\nSNR  : " + str(self.snr) + \
               "\nPCORR: " + str(self.pcorr)
        
    def __repr__(self):
        return self.__str__()


class HistoryReg:
    def __init__(self, epochs):
        self.loss = []
        self.snr = []
        self.pcorr = []
        self.lr = []
        self.df = []
        self.reg = []
        self.msg = "Iter %s, Loss = %+.2e, DF = %.2e, REG = %.2e, SNR = %+.2f dB, PCORR = %+.2f %%"
        self.zfill = ten_digit(epochs)
    
    def __getitem__(self, item):
        return self.loss[item], self.reg[item], self.snr[item], self.pcorr[item]
    
    def __setitem__(self, idx, values):
        l, d, r, s, p = values
        self.loss[idx] = l
        self.df[idx] = d
        self.reg[idx] = r
        self.snr[idx] = s
        self.pcorr[idx] = p
    
    def append(self, values):
        l, d, r, s, p = values
        self.loss.append(l)
        self.df.append(d)
        self.reg.append(r)
        self.snr.append(s)
        self.pcorr.append(p)
    
    def __len__(self):
        assert len(self.loss) == len(self.snr) == len(self.pcorr) == len(self.lr) == len(self.df) == len(self.reg)
        return len(self.loss)
    
    def log_message(self, idx):
        return self.msg % (str(idx + 1).zfill(self.zfill),
                           self.loss[idx],
                           self.df[idx],
                           self.reg[idx],
                           self.snr[idx],
                           self.pcorr[idx] * 100)
    
    def __str__(self):
        return   "Loss : " + str(self.loss) + \
               "\nReg  : " + str(self.reg) + \
               "\nSNR  : " + str(self.snr) + \
               "\nPCORR: " + str(self.pcorr)
    
    def __repr__(self):
        return self.__str__()


class HistoryPOCS:
    def __init__(self, epochs):
        self.loss = []
        self.df = []
        self.reg = []
        self.eps = []
        self.lr = []
        self.snr = []
        self.th = []
        self.msg = "Iter %s, loss=%.2e, df=%.2e, reg=%.2e, eps=%.2e, SNR=%+.2fdB, th=%.2e"
        self.zfill = ten_digit(epochs)
    
    def __getitem__(self, item):
        return self.loss[item], self.reg[item], self.snr[item]
    
    def __setitem__(self, idx, values):
        l, d, r, e, s, t = values
        self.loss[idx] = l
        self.df[idx] = d
        self.reg[idx] = r
        self.snr[idx] = s
        self.eps[idx] = e
        self.th[idx] = t
    
    def append(self, values):
        l, d, r, e, s, t = values
        self.loss.append(l)
        self.df.append(d)
        self.reg.append(r)
        self.snr.append(s)
        self.eps.append(e)
        self.th.append(t)
    
    def __len__(self):
        assert len(self.loss) == len(self.snr) == len(self.eps) == len(self.lr) == len(self.df) == len(self.reg) == len(self.th)
        return len(self.loss)
    
    def log_message(self, idx):
        return self.msg % (str(idx + 1).zfill(self.zfill),
                           self.loss[idx],
                           self.df[idx],
                           self.reg[idx],
                           self.eps[idx],
                           self.snr[idx],
                           self.th[idx])
    
    def __str__(self):
        return "Loss : " + str(self.loss) + \
               "\nReg  : " + str(self.reg) + \
               "\nSNR  : " + str(self.snr)
    
    def __repr__(self):
        return self.__str__()


__all__ = [
    'snr',
    'pcorr',
    'History',
    'HistoryReg',
    'HistoryPOCS',
]
