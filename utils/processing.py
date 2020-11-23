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


def bool2bin(in_content: np.ndarray, logic: bool = True):
    temp = in_content.copy()
    temp[np.isnan(temp) == False] = 1 if logic else 0
    temp[np.isnan(temp) == True] = 0 if logic else 1
    return temp


__all__ = [
    "normalize",
    "denormalize",
    "bool2bin",
]
