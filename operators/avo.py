import torch


__all__ = [
    "AVOLinearModelling",
]


def _akirichards(theta, vsvp, n=1):
    theta = torch.deg2rad(theta)
    vsvp = vsvp * torch.ones(n) if not isinstance(vsvp, torch.Tensor) else vsvp
    
    theta = theta[:, None] if vsvp.numel() > 1 else theta
    vsvp = vsvp[:, None].T if vsvp.numel() > 1 else vsvp
    
    c2 = torch.cos(theta) ** 2
    s2 = torch.sin(theta) ** 2
    
    G1 = 1. / (2. * c2) + 0 * vsvp
    G2 = -4. * vsvp ** 2 * s2
    G3 = 0.5 - 2. * vsvp ** 2 * s2
    
    return G1, G2, G3


def _fatti(theta, vsvp, n=1):
    theta = torch.deg2rad(theta)
    vsvp = vsvp * torch.ones(n) if not isinstance(vsvp, torch.Tensor) else vsvp
    
    theta = theta[:, None] if vsvp.numel() > 1 else theta
    vsvp = vsvp[:, None].T if vsvp.numel() > 1 else vsvp
    
    t2 = torch.tan(theta) ** 2
    s2 = torch.sin(theta) ** 2
    
    G1 = 0.5 * (1. + t2) + 0 * vsvp
    G2 = -4. * vsvp ** 2 * s2
    G3 = 0.5 * (4 * vsvp ** 2 * s2 - t2)
    
    return G1, G2, G3


class AVOLinearModelling(torch.nn.Module):
    
    def __init__(self, theta, vsvp=0.5, nt0=1, spatdims=None, linearization='akirich'):
        
        super(AVOLinearModelling, self).__init__()
        
        self.nt0 = nt0 if not isinstance(vsvp, torch.Tensor) else len(vsvp)
        self.ntheta = len(theta)
        
        if spatdims is None:
            self.spatdims = ()
        else:
            self.spatdims = spatdims if isinstance(spatdims, tuple) else (spatdims,)
        
        # Compute AVO coefficients
        if linearization == "fatti":
            self.G = torch.stack([gs for gs in _fatti(theta, vsvp, n=self.nt0)], dim=1)
        else:
            self.G = torch.stack([gs for gs in _akirichards(theta, vsvp, n=self.nt0)], dim=1)
        
        # add dimensions to G to account for horizonal axes
        for _ in range(len(self.spatdims)):
            self.G = self.G.unsqueeze(-1)
    
    def forward(self, x):
        """
        from model to data

        3 channels -> ntheta channels
        """
        if self.G.device != x.device:
            self.G = self.G.to(x.device)
        # G is (ntheta, 3, nt0, spatdims)
        # x is (1, 3, nt0, spatdims)
        # the output has to be (1, ntheta, nt0, spatdims)
        y = torch.sum(self.G * x, dim=1).unsqueeze(0)
        
        return y
    
    def adjoint(self, y):
        """
        from data to model

        ntheta channels -> 3 channels
        """
        if self.G.device != y.device:
            self.G = self.G.to(y.device)
        # G is (ntheta, 3, nt0, spatdims)
        # y is (1, ntheta, nt0, spatdims)
        # the output has to be (1, 3, nt0, spatdims)
        x = torch.sum(self.G * y.transpose(0, 1), dim=0).unsqueeze(0)
        
        return x
