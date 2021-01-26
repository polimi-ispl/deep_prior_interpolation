import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import filtfilt, butter, firls, freqz
import torch


def normalize(image, time_step, velo):
    nt, nx, ny = image.shape
    step = time_step * velo
    t = np.linspace(step, nt * step, nt)
    t = np.tile(t, (nx, ny, 1)).transpose(-1, 0, 1)
    gain = np.sqrt(t)
    
    return image * gain


def denormalize(image, time_step, velo):
    nt, nx, ny = image.shape
    step = time_step * velo
    t = np.linspace(step, nt * step, nt)
    t = np.tile(t, (nx, ny, 1)).transpose(-1, 0, 1)
    gain = np.sqrt(t)
    
    return image / gain


def bool2bin(in_content: np.ndarray, logic: bool = True):
    temp = in_content.copy()
    temp[np.isnan(temp) == False] = 1 if logic else 0
    temp[np.isnan(temp) == True] = 0 if logic else 1
    return temp


def filter_noise_traces(in_content: np.ndarray, filt: np.ndarray) -> np.ndarray:
    assert filt.ndim == 1, "filter has to be a 1D array"

    filtered = convolve1d(in_content, filt, axis=2)
    
    return filtered


def filter_noise_butter(in_content, fs, cutoff, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, in_content, axis=2)
 

class LowPassButterworth2D(torch.nn.Module):
    
    def __init__(self, fc, fs=None, ntaps=101, order=4, nfft=1024, dtype=torch.cuda.FloatTensor):
        super(LowPassButterworth2D, self).__init__()
        
        b, a = butter(order, fc, fs=fs, btype='low', analog=False)
        w_iir, h_iir = freqz(b, a, worN=nfft, fs=fs)
        self.taps = firls(ntaps, w_iir, abs(h_iir), fs=fs)
        
        kernel = torch.zeros((ntaps, ntaps))
        kernel[:, kernel.shape[1] // 2] = torch.from_numpy(self.taps)
        
        self.kernel = kernel.float().unsqueeze(0).unsqueeze(0).type(dtype)
        self.pad = kernel.shape[-1] // 2
    
    def forward(self, input):
        return torch.nn.functional.conv_transpose2d(input, self.kernel, padding=self.pad)


def _gaussian_kernel(M: int, std: float, sym=True) -> torch.Tensor:
    assert M > 1
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w


def ricker_wavelet(points: int, a: float) -> torch.Tensor:
    A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    vec = torch.arange(0, points) - (points - 1.0) / 2
    xsq = vec ** 2
    mod = 1 - xsq / wsq
    gauss = torch.exp(-xsq / (2 * wsq))
    total = A * mod * gauss
    return total


def GaussianFilter(channels: int, kernel_size: int, ndim: int, std: float) -> torch.nn.Module:
    """Build an isotropic Gaussian blurring operator

    :param channels: number of tensor channels
    :param kernel_size: number of samples of the gaussian kernel
    :param ndim: number of convolution dimensions
    :param std: standard deviation of the gaussian bell
    """
    w = _gaussian_kernel(M=kernel_size, std=std, sym=True)
    
    if ndim == 1:
        conv = torch.nn.ConvTranspose1d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False)
        w = w
    elif ndim == 2:
        conv = torch.nn.ConvTranspose2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False)
        w = torch.outer(w, w)
    elif ndim == 3:
        conv = torch.nn.ConvTranspose3d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False)
        w = torch.outer(w, torch.outer(w, w))
    else:
        raise ValueError
    
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(w.unsqueeze(0).unsqueeze(0))
    return conv


def first_derivative(in_content: torch.Tensor, spacing: float = 1., axis: int = 0, stencil='forward') -> torch.Tensor:
    """Compute the first derivative with a first order stencil

    :param in_content: input tensor
    :param spacing: sampling step along the derivation axis
    :param axis: direction along with the derivative is computed
    :param stencil: centered, forward or backward
    """
    if axis != 0:
        in_content = torch.transpose(in_content.clone(), 0, axis)
    
    grad = torch.zeros_like(in_content)
    if stencil == 'centered':
        grad[1:-1] = (.5 * in_content[2:] - .5 * in_content[:-2, ]) / spacing
    elif stencil == 'forward':
        grad[:-1] = (in_content[1:] - in_content[:-1]) / spacing
    elif stencil == 'backward':
        grad[1:] = (in_content[1:] - in_content[:-1]) / spacing
    else:
        raise ValueError('Stencil has to be centered, forward or backward')
    if axis != 0:
        grad = torch.transpose(grad, 0, axis)
    
    return grad


def second_derivative(in_content: torch.Tensor, spacing: float = 1., axis: int = 0) -> torch.Tensor:
    """Compute the second derivative with a first order centered stencil

    :param in_content: input tensor
    :param spacing: sampling step along the derivation axis
    :param axis: direction along with the derivative is computed
    """
    if axis != 0:
        in_content = torch.transpose(in_content.clone(), 0, axis)
    
    grad = torch.zeros_like(in_content)
    grad[1:-1] = (in_content[2:] - 2 * in_content[1:-1] + in_content[:-2]) / (spacing ** 2)
    
    if axis != 0:
        grad = torch.transpose(grad, 0, axis)
    
    return grad


__all__ = [
    "normalize",
    "denormalize",
    "bool2bin",
    "filter_noise_traces",
    "filter_noise_butter",
    "LowPassButterworth2D",
    "GaussianFilter",
    "ricker_wavelet",
    "first_derivative",
    "second_derivative",
]
