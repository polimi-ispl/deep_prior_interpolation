import numpy as np
import torch
from cv2 import dilate


def build_mask(data: np.ndarray, rate: float, regular: bool = False) -> np.ndarray:
    """
    Build a binary mask related to the subsampling operator
    
    Please, pay attention to the regular 3D case, if you need to simulate a specific geometry.
    Args:
        data: original data cube ordered as (t,x,[y])
        rate: rate of missing traces in [0, 1]
        regular: whether the mask has to be regular or random

    Returns:
        binary mask
    """
    if data.ndim == 2:
        nt, nx = data.shape
        ny = 1
    elif data.ndim == 3:
        nt, nx, ny = data.shape
        data = data.reshape((nt, -1))
    else:
        raise ValueError("data volume has to be either 2D or 3D")
    
    num_traces = nx * ny
    
    num_deletion = int(num_traces * rate)
    
    if regular:
        if rate >= .5:
            mask = np.ones_like(data)
            remain_num = num_traces - num_deletion
            m = int(np.ceil(num_traces / remain_num))
            
            for i in range(remain_num):
                mask[:, i * m + 1:i * m + m] = 0
        
        if rate < .5:
            mask = np.zeros_like(data)
            remain_num = num_deletion
            m = int(np.ceil(num_traces / remain_num))
            
            for i in range(remain_num):
                mask[:, i * m + 1:i * m + m] = 1
    else:
        del_idx = np.random.choice(np.arange(num_traces), num_deletion, replace=False)
        mask = np.ones_like(data)
        mask[:, del_idx] = 0
    
    return mask.reshape((nt, nx, ny)).squeeze()


def add_rand_mask(mask, perc=0.3):
    """
        add the addictive random missing points to the mask.
        parameter:
            mask -- the mask(2D or 3D), which should be (nt, nx), (nt, nx, ny)
            perc -- the percent of addictive deleting samples

        return:
            the processed new makk
    """
    m = mask.copy()
    points = np.argwhere(m[0] == 1)
    rr = np.random.choice(np.arange(points.shape[0]), int(points.shape[0] * perc), replace=False)
    if m.ndim == 2:
        for p in points[rr]:
            m[:, p[0]] = 0
    else:
        for p in points[rr]:
            m[:, p[0], p[1]] = 0
    return m


def _dilate_mask(mask: torch.Tensor, iterations: int = 1):
    kernel = np.ones((2, 2), np.uint8)
    ddtype = mask.dtype
    device = mask.device
    mask_np = mask.clone().detach().cpu().numpy()
    shape = mask_np.shape
    mask_np = mask_np.squeeze()
    mask_res = np.empty_like(mask_np)
    for i in range(mask_np.shape[0]):
        m = mask_np[i].astype(np.uint8)
        mask_res[i] = dilate(m, kernel, iterations=iterations)
    
    mask_res = torch.from_numpy(mask_res.reshape(shape)).type(ddtype).to(device)
    return mask_res


class MaskUpdate:
    def __init__(self, mask: torch.Tensor, threshold: int, step: int) -> None:
        self.threshold = threshold
        self.step = step
        self.iter = 0
        self.new_mask = mask
        self.old_mask = mask
    
    def update(self, iiter):
        mask_return = self.old_mask
        if iiter > self.threshold:
            iiter_dil = (iiter - self.threshold) // self.step + 1
            if iiter_dil > self.iter:
                self.old_mask = self.new_mask
                self.new_mask = _dilate_mask(self.old_mask)
                self.iter = iiter_dil
            iter_drop = (iiter - self.threshold) % self.step
            p = 1. - 1. / self.step * (iter_drop + 1)
            mask_add = torch.nn.functional.dropout(self.new_mask - self.old_mask, p)
            mask_add[mask_add != 0] = 1
            
            mask_return = self.old_mask + mask_add
        return mask_return


__all__ = [
    "build_mask",
    "add_rand_mask",
    "_dilate_mask",
    "MaskUpdate",
]
