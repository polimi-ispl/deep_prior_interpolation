import numpy as np


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


__all__ = [
    "normalize",
    "denormalize",
]