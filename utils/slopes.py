from typing import Tuple
import torch
from .processing import first_derivative, GaussianFilter
    
    
def structure_tensor_dips(in_content: torch.Tensor, dv: float = 1., dh: float = 1,
                          smooth: float = 0.) -> Tuple[torch.Tensor, torch.Tensor]:
    """Local dip estimation through structure tensor algorithm.

    :param in_content: tensor to be processed of shape BCHW
    :param dv: sampling along the vertical dimension
    :param dh: sampling along the horizonal dimension
    :param smooth: lenght of smoothing filter to be applied to the estimated gradients

    van Vliet, Lucas and Verbeek, Piet, 1995. Estimators for Orientation and Anisotropy in Digitized Images. In ASCI'95.
    """
    # TODO change axis and go for "normal" dimensionality.
    #  The BC dimensions has to be taken into account by the calling code.
    gv = first_derivative(in_content, spacing=dv, axis=2, stencil='forward')
    gh = first_derivative(in_content, spacing=dh, axis=3, stencil='forward')
    gvv, gvh, ghh = gv * gv, gv * gh, gh * gh
    
    # smoothing
    if smooth > 0:
        G = GaussianFilter(channels=in_content.shape[1],
                           kernel_size=2 * min(in_content.shape[2], in_content.shape[3]) // 2 + 1,
                           ndim=2,
                           std=smooth).to(in_content.device)
        gvv = G(gvv)
        gvh = G(gvh)
        ghh = G(ghh)
    
    # quadratic formula for eigenvalues
    _term1 = 0.5 * (gvv + ghh)
    _term2 = 0.5 * torch.sqrt(torch.pow(gvv - ghh, 2) + 4 * torch.pow(gvh, 2))
    
    eigenvalue1 = _term1 + _term2
    eigenvalue2 = _term1 - _term2
    
    phi1 = torch.atan((eigenvalue1 - gvv) / gvh)
    # phi2 = torch.atan2(gvh, eigenvalue2 - ghh)
    phi1[torch.isnan(phi1)] = 0.
    
    # TODO which one is better?
    anisotropy = 1 - eigenvalue2 / eigenvalue1
    # coherence = torch.pow((eigenvalue1 - eigenvalue2) / (eigenvalue1 + eigenvalue2), 2)
    
    return phi1, anisotropy


def directional_laplacian(in_content: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # orthogonal eigenvector
    u1 = torch.cos(theta)
    u2 = -torch.sin(theta)
    
    # gradient compontents
    grad_v = first_derivative(in_content, axis=2, stencil='forward')
    grad_h = first_derivative(in_content, axis=3, stencil='forward')
    
    # apply rotation
    vvtGrad_1 = u1 * u1 * grad_v + u1 * u2 * grad_h
    vvtGrad_2 = u1 * u2 * grad_v + u2 * u2 * grad_h
    
    # compute gradient adjoint (aka, divergence)
    AtA_1 = first_derivative(vvtGrad_1, axis=3, stencil='forward')
    AtA_2 = first_derivative(vvtGrad_2, axis=2, stencil='forward')
    AtA = AtA_1 + AtA_2
    
    return -AtA


class Hale2D(torch.nn.Module):
    
    def __init__(self, directions: torch.Tensor):
        """Directional Laplacian operator built upon directions tensor.
        It operates on tensors of dimension BCHW.
        
        :param directions: tensor of dimension BCHW
        """
        super(Hale2D, self).__init__()
        
        # orthogonal eigenvector
        u1 = torch.cos(directions)
        u2 = -torch.sin(directions)
        with torch.no_grad():
            self.a = u1 * u1
            self.b = u1 * u2
            self.c = u2 * u2
            self.dips = directions
    
    def forward(self, inputs):
        # gradient compontents
        grad_v = first_derivative(inputs, axis=2, stencil='forward')
        grad_h = first_derivative(inputs, axis=3, stencil='forward')
        
        # apply rotation
        vvtGrad_1 = self.a * grad_v + self.b * grad_h
        vvtGrad_2 = self.b * grad_v + self.c * grad_h
        
        # compute gradient adjoint (aka, divergence)
        AtA_1 = first_derivative(vvtGrad_1, axis=3, stencil='forward')
        AtA_2 = first_derivative(vvtGrad_2, axis=2, stencil='forward')
        AtA = AtA_1 + AtA_2
        
        return -AtA
    

__all__ = [
    "Hale2D",
    "directional_laplacian",
    "structure_tensor_dips",
]
