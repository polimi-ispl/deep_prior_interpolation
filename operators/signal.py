import torch

__all__ = [
    "VerticalConv",
]


class VerticalConv(torch.nn.Module):
    def __init__(self, wavelet):
        super(VerticalConv, self).__init__()
        
        ntwav = len(wavelet)
        
        # forward kernel: true convolution
        kernel = torch.zeros((1, ntwav, ntwav))
        kernel[:, :, kernel.shape[1] // 2] = torch.from_numpy(wavelet[::-1] / 2)
        
        self.C = torch.nn.Conv2d(1, 1, kernel_size=ntwav, stride=1, padding=kernel.shape[-1] // 2,
                                 bias=False, groups=1)
        self.C.weight = torch.nn.Parameter(kernel.float().unsqueeze(0), requires_grad=False)
        
        # adjoint kernel: cross-correlation
        kernelT = torch.zeros((1, ntwav, ntwav))
        kernelT[:, :, kernelT.shape[1] // 2] = torch.from_numpy(wavelet / 2)
        self.CT = torch.nn.Conv2d(1, 1, kernel_size=ntwav, stride=1, padding=kernelT.shape[-1] // 2,
                                  bias=False, groups=1)
        self.CT.weight = torch.nn.Parameter(kernelT.float().unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        
        self.C = self.C.to(x.device)
        
        if x.size(1) == 1:
            return self.C(x)
        else:
            return torch.cat([self.C(d.unsqueeze(0).unsqueeze(0)) for d in x.squeeze(0)], dim=1)
    
    def adjoint(self, y):
        
        self.CT = self.CT.to(y.device)
        
        if y.size(1) == 1:
            return self.CT(y)
        else:
            return torch.cat([self.CT(d.unsqueeze(0).unsqueeze(0)) for d in y.squeeze(0)], dim=1)
