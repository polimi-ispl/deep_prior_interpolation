from typing import Dict, List, Optional, Union
import os
import numpy as np
from utils import add_rand_mask


def _get_patch_2d(args) -> List[dict]:

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
        img = np.load(os.path.join(img_dir, imgname), allow_pickle=True)
        mask = np.load(os.path.join(img_dir, maskname), allow_pickle=True)
        height = img.shape[0] // 64 * 64
        img = img[:height]
        mask = mask[:height]
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, -1)
        
        mask[np.isnan(mask) == False] = 1
        mask[np.isnan(mask) == True] = 0
        
        if args.adirandel > 0:
            mask = add_rand_mask(mask, args.adirandel)
        
        out['img'] = img * args.gain[i]
        out['name'] = maskname
        out['mask'] = mask
        outputs.append(out)
        i += 1
    return outputs


def _get_patch_2_5d(args) -> List[dict]:

    img_dir = args.imgdir
    imgname = args.imgname[0]
    maskname = args.maskname[0]
    
    perc = args.imgchannel
    
    img = np.load(os.path.join(img_dir, imgname), allow_pickle=True) * args.gain[0]
    mask = np.load(os.path.join(img_dir, maskname), allow_pickle=True)
    
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
    
    if args.adirandel > 0:
        mask_np = add_rand_mask(mask_np, args.adirandel)
    
    outputs = [{'img' : img_np[..., i: i + perc],
                'mask': mask_np[..., i: i + perc],
                'name': str(i) + '_' + args.slice + '.npy'}
               for i in range(0, img_np.shape[-1], perc)]
    
    return outputs


def _get_patch_3d(args) -> List[dict]:

    img_dir = args.imgdir
    imgname = args.imgname[0]
    maskname = args.maskname[0]
    
    img = np.load(os.path.join(img_dir, imgname), allow_pickle=True)
    mask = np.load(os.path.join(img_dir, maskname), allow_pickle=True)
    
    mask[np.isnan(mask) == False] = 1
    mask[np.isnan(mask) == True] = 0
    
    if args.adirandel > 0:
        mask = add_rand_mask(mask, args.adirandel)
    pad_w = args.padwidth
    
    if pad_w > 0:
        img = np.pad(img, ((pad_w, pad_w), (pad_w, pad_w), (pad_w, pad_w)), mode='edge')
        mask = np.pad(mask, ((pad_w, pad_w), (pad_w, pad_w), (pad_w, pad_w)), mode='constant', constant_values=1)
    
    if 'hyperbolic' in imgname.lower():
        segt = [(0, 256), (128, 384), (256, 512)]
        segx = [(0, 128 + 2 * pad_w)]
        segy = [(0, 128 + 2 * pad_w)]
    elif 'marmousi' in imgname.lower():
        segt = [(0, 104), (100, 204), (196, 300)]
        segx = [(0, 132), (124, 256)]
        segy = [(0, 132), (124, 256)]
    elif 'f3' in imgname.lower():
        segt = [(0, 80), (64, 144), (128, 208), (192, 272), (256, 336), (320, 400), (368, 448)]
        segx = [(0, 80), (48, 128)]
        segy = [(0, 80), (48, 128)]
    else:
        segt = [(i * 224, i * 224 + 256) for i in range(9)]
        segx = [(0, img.shape[1])]
        segy = [(0, img.shape[2])]
    
    segs = [[t[0], t[1], x[0], x[1], y[0], y[1]] for t in segt for x in segx for y in segy]
    
    if len(args.gain) == 1 and len(segs) != len(args.gain):
        args.gain = [args.gain[0]] * len(segs)
    
    imgs = [img[x[0]:x[1], x[2]:x[3], x[4]:x[5]] * y for x, y in zip(segs, args.gain)]
    masks = [mask[x[0]:x[1], x[2]:x[3], x[4]:x[5]] for x in segs]
    
    outputs = [{'img' : np.expand_dims(imgs[i], -1),
                'mask': np.expand_dims(masks[i], -1),
                'name': str(i) + '.npy'}
               for i in range(len(imgs))]
    print('The number of patches is %d' % len(imgs))
    
    return outputs


def _get_logging(args) -> dict:
    
    img_dir = args.imgdir
    imgnames = args.imgname
    masknames = args.maskname
    
    assert len(imgnames) == len(masknames)
    
    imgs = []
    masks = []
    if len(args.gain) != len(imgnames):
        args.gain = [args.gain[0]] * len(imgnames)
    i = 0
    for imgname, maskname in zip(imgnames, masknames):
        
        img = np.load(os.path.join(img_dir, imgname), allow_pickle=True)[512: 1024] * args.gain[i]
        mask = np.load(os.path.join(img_dir, maskname), allow_pickle=True)[512: 1024]
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, -1)
        
        mask[np.isnan(mask) == False] = 1
        mask[np.isnan(mask) == True] = 0
        
        imgs.append(img)
        masks.append(mask)
        i += 1
    output = {'img_name': img_dir.split('/')[-1],
              'img'     : np.stack(imgs, axis=0),
              'mask'    : np.stack(masks, axis=0)}
    return output


def get_patch(args) -> List[dict] or dict:
    """Extract patches out of data.
        Parameters:
            args: arguments Namespace with:
                img_dir (str) -- directory of the data
                img_name (list) -- .npy data file (shape is `nt \times nx \times ny`)
                gain (list) -- the gain values
        Return:
            A list of the dict including the corrupted data, mask and it's name.
        Note:
            If the gain value is set as only one value, then all data share the same gain given.
            Tthe data type should be '.npy', and np.NaN represents the missing data.

    """
    if args.datadim == '3d':
        outputs = _get_patch_3d(args)
    elif args.datadim == '2d':
        outputs = _get_patch_2d(args)
    elif args.datadim == '2.5d':
        outputs = _get_patch_2_5d(args)
    elif args.datadim == 'log':
        outputs = _get_logging(args)
    else:
        raise ValueError('The datadim in the args must be 2d, 3d, 4d')
    return outputs
