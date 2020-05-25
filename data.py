import torch
import torch.nn as nn
import torchvision
import sys
import os
import matplotlib.image as mpimg
import numpy as np


def get_patch_2d(args):
    """The funcion used for get the patchs of the 2D data.
        Parameters:
            args -- the parameters must include below:
            img_dir (str) -- the dir where saved corrupted data for processing.
            img_name (list) -- the name of datas for processing.
            gain (list) -- the gain values multiplying to the data.
        Return:
            A list of the dict including the corrupted data, mask and it's name.
        note:
            If the gain value is set as only one value, then all data share the same gain given.
            Tthe data type should be '.npy', and np.NaN represents the missing data.

    """
    img_dir = args.imgdir
    imgnames = args.imgname
    masknames = args.maskname
    
    assert len(imgnames) == len(masknames)
    
    outputs = []
    if len(args.gain) != len(imgnames):
        args.gain = [args.gain[0]] * len(imgnames)
    i = 0
    for imgname, maskname in zip(imgnames, masknames):
        out = {}
        img = np.load(os.path.join(img_dir, imgname))
        mask = np.load(os.path.join(img_dir, maskname))
        if len(img.shape) == 2:
            img = img[..., np.newaxis]
        if len(mask.shape) == 2:
            mask = mask[..., np.newaxis]
        
        mask[np.isnan(mask) == False] = 1
        mask[np.isnan(mask) == True] = 0
        
        out['img'] = img * args.gain[i]
        out['name'] = maskname
        out['mask'] = mask
        outputs.append(out)
        i += 1
    return outputs


def get_patch_2_5d(args):
    """The funcion used for get the slices of the 3D data.
        Parameters:
            args -- the parameters must include below:
            img_dir (str) -- the dir where saved corrupted data for processing.
            img_name (list) -- the name of datas for processing.
            gain (list) -- the gain values multiplying to the data.
            slice (str) -- the kind of slice including XT(nt, nx), XY (nx,ny), YT(nt, ny).
            imgchannel (int) -- the number of slices for processing in the same time.
        Return:
            A list of the dict including the slices of corrupted data, mask and it's index.
        note:
            The 3D data is restored slice by slice through 2D convolutional filters, so the funcion is named 2.5D.
            The problem could only process only one 3D data in one time. So the length of img_name and gain should be 1.
            The dimension of input 3D data should be (nt, nx, ny).
    """
    img_dir = args.imgdir
    imgname = args.imgname[0]
    maskname = args.maskname[0]
    
    perc = args.imgchannel
    
    img = np.load(os.path.join(img_dir, imgname)) * args.gain[0]
    mask = np.load(os.path.join(img_dir, maskname))
    
    if args.slice == 'XY':
        img_np = img.transpose(1, 2, 0)
        mask_np = mask.transpose(1, 2, 0)
    elif args.slice == 'YT':
        img_np = np.swapaxes(img, 2, 1)
        mask_np = np.swapaxes(mask, 2, 1)
    elif args.slice == 'XT':
        img_np = img
        mask_np = mask
    
    mask_np[np.isnan(mask_np) == False] = 1
    mask_np[np.isnan(mask_np) == True] = 0
    
    outputs = [{'img' : img_np[..., i: i + perc], 'mask': mask_np[..., i: i + perc],
                'name': str(i) + '_' + args.slice + '.npy'}
               for i in range(0, img_np.shape[-1], perc)]
    
    return outputs


def get_patch_3d(args):
    """The function used for get the 3D data.
        Parameters:
            args -- the parameters must include below:
            img_dir (str) -- the dir where saved corrupted data for processing.
            img_name (list) -- the name of datas for processing.
            gain (list) -- the gain values multiplying to the data.
        Return:
            A dict including the corrupted data, mask and it's index.
        note:
            The 3D data is restored through 3D convolution filters
            The problem could only process only one 3D data in one time. So the length of img_name and gain should be 1.
            The dimension of input 3D data should be (nt, nx, ny).
    """
    img_dir = args.imgdir
    imgname = args.imgname[0]
    maskname = args.maskname[0]
    
    img = np.load(os.path.join(img_dir, imgname))
    mask = np.load(os.path.join(img_dir, maskname))
    
    mask[np.isnan(mask) == False] = 1
    mask[np.isnan(mask) == True] = 0
    
    if 'hyperbolic3d' in imgname:
        segt = [(0, 250)]
        segx = [(0, 80)]
        segy = [(0, 80)]
    elif 'Marmousi' in imgname:
        segt = [(0, 104), (100, 204), (196, 300)]
        segx = [(0, 132), (124, 256)]
        segy = [(0, 132), (124, 256)]
    elif 'netherlandsf3' in imgname:
        segt = [(0, 128), (112, 240), (224, 352), (320, 448)]
        segx = [(0, 128)]
        segy = [(0, 128)]
    
    segs = [[t[0], t[1], x[0], x[1], y[0], y[1]] for t in segt for x in segx for y in segy]
    
    if len(args.gain) == 1 and len(segs) != len(args.gain):
        args.gain = [args.gain[0]] * len(segs)
    
    imgs = [img[x[0]:x[1], x[2]:x[3], x[4]:x[5]] * y for x, y in zip(segs, args.gain)]
    masks = [mask[x[0]:x[1], x[2]:x[3], x[4]:x[5]] for x in segs]
    
    outputs = [{'img' : imgs[i][..., np.newaxis], 'mask': masks[i][..., np.newaxis],
                'name': str(i) + '.npy'} for i in range(len(imgs))]
    print('The number of patches is %d' % len(imgs))
    
    return outputs


def get_patch(args):
    if args.datadim == '3d':
        outputs = get_patch_3d(args)
    elif args.datadim == '2d':
        outputs = get_patch_2d(args)
    elif args.datadim == '2.5d':
        outputs = get_patch_2_5d(args)
    else:
        raise ValueError('The datadim in the args must be 2d, 3d, 4d')
    return outputs
