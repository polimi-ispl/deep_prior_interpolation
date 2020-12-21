from typing import List
import os
import numpy as np
from utils import ten_digit, bool2bin, add_rand_mask, PatchExtractor


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
        
        out['image'] = img * args.gain[i]
        out['name'] = maskname
        out['mask'] = mask
        outputs.append(out)
        i += 1
    return outputs


def _get_patch_2_5d(args) -> List[dict]:
    img_dir = args.imgdir
    imgname = args.imgname
    maskname = args.maskname
    
    perc = args.imgchannel
    
    img = np.load(os.path.join(img_dir, imgname), allow_pickle=True) * args.gain
    mask = np.load(os.path.join(img_dir, maskname), allow_pickle=True)
    
    if args.slice == 'XY':
        img_np = img.transpose(1, 2, 0)
        mask_np = mask.transpose(1, 2, 0)
    elif args.slice == 'TY':
        img_np = np.swapaxes(img, 2, 1)
        mask_np = np.swapaxes(mask, 2, 1)
    elif args.slice == 'TX':
        img_np = img
        mask_np = mask
    
    mask_np[np.isnan(mask_np) == False] = 1
    mask_np[np.isnan(mask_np) == True] = 0
    
    if args.adirandel > 0:
        mask_np = add_rand_mask(mask_np, args.adirandel)
    
    outputs = [{'image' : img_np[..., i: i + perc],
                'mask': mask_np[..., i: i + perc],
                'name': str(i) + '_' + args.slice + '.npy'}
               for i in range(0, img_np.shape[-1], perc)]
    
    return outputs


def _get_patch_3d(args) -> List[dict]:
    _dataset = args.imgdir + args.imgname + args.maskname
    img_dir = args.imgdir
    imgname = args.imgname
    maskname = args.maskname
    
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
    
    if 'hyperbolic' in _dataset.lower():
        segt = [(0, 256), (128, 384), (256, 512)]
        segx = [(0, 128 + 2 * pad_w)]
        segy = [(0, 128 + 2 * pad_w)]
    elif 'marmousi' in _dataset.lower():
        segt = [(0, 104), (100, 204), (196, 300)]
        segx = [(0, 132), (124, 256)]
        segy = [(0, 132), (124, 256)]
    elif 'f3' in _dataset.lower():
        segt = [(0, 80), (64, 144), (128, 208), (192, 272), (256, 336), (320, 400), (368, 448)]
        segx = [(0, 80), (48, 128)]
        segy = [(0, 80), (48, 128)]
    else:
        segt = [(i * 224, i * 224 + 256) for i in range(9)]
        segx = [(0, img.shape[1])]
        segy = [(0, img.shape[2])]
    
    segs = [[t[0], t[1], x[0], x[1], y[0], y[1]] for t in segt for x in segx for y in segy]
    
    gains = [args.gain] * len(segs)
    
    imgs = [img[x[0]:x[1], x[2]:x[3], x[4]:x[5]] * y for x, y in zip(segs, gains)]
    masks = [mask[x[0]:x[1], x[2]:x[3], x[4]:x[5]] for x in segs]
    
    outputs = [{'image' : np.expand_dims(imgs[i], -1),
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
              'image'   : np.stack(imgs, axis=0),
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


def _get_patch_extractor(in_shape: tuple, patch_shape: tuple, patch_stride: tuple,
                         datadim: str, imgchannel: int = None) -> PatchExtractor:
    ndim = len(in_shape)
    patch_shape = [patch_shape[d] if patch_shape[d] != -1 else in_shape[d] for d in range(ndim)]
    if datadim == "2.5d" and imgchannel is not None:
        patch_shape[-1] = imgchannel
    
    patch_stride = [patch_stride[d] if patch_stride[d] != -1 else patch_shape[d] for d in range(len(patch_shape))]
    
    return PatchExtractor(dim=tuple(patch_shape), stride=tuple(patch_stride))


def _transpose_patches_25d(in_content: np.ndarray, slice: str = 'XY', adj: bool = False):
    slice = slice.lower()
    if slice == "xt":
        slice = "tx"
    if slice == "yt":
        slice = "ty"
    
    if adj:
        if slice == 'xy':  # BXYT -> BTXY
            in_content = in_content.transpose((0, 3, 1, 2))
        elif slice == 'ty':  # BTYX -> BTXY
            in_content = in_content.transpose((0, 1, 3, 2))
        else:  # we already are in (t, x, y), great!
            pass
    else:
        if slice == 'xy':  # BTXY -> BXYT
            in_content = in_content.transpose((0, 2, 3, 1))
        elif slice == 'ty':  # BTXY -> BTYX
            in_content = in_content.transpose((0, 1, 3, 2))
        else:
            pass
    return in_content


def extract_patches(args) -> List[dict]:
    original = np.load(os.path.join(args.imgdir, args.imgname), allow_pickle=True)
    corrupted = np.load(os.path.join(args.imgdir, args.maskname), allow_pickle=True)
    
    assert original.shape == corrupted.shape, "Original and Corrupted data must have the same dimension"
    assert original.ndim in [2, 3], "Data volumes have to be 2D or 3D"
    
    # we have created masks in two ways: binary value (0 or 1) or a copy of the data with NaN traces
    # adopt the binary representation
    if np.isnan(corrupted).any():
        corrupted = bool2bin(corrupted)

    pe = _get_patch_extractor(original.shape, args.patch_shape, args.patch_stride, args.datadim, args.imgchannel)
    
    if args.datadim == "2.5d":
        final_shape = (-1,) + pe.dim
    else:
        final_shape = (-1,) + pe.dim + (1,)
    
    patches_img = pe.extract(original).reshape(final_shape)
    patches_msk = pe.extract(corrupted).reshape(final_shape)
    
    if args.datadim == '2.5d':
        patches_img = _transpose_patches_25d(patches_img, args.slice)
        patches_msk = _transpose_patches_25d(patches_msk, args.slice)
    
    outputs = []
    num_patches = patches_img.shape[0]
    _zeros = ten_digit(num_patches)
    
    for p in range(num_patches):
        
        i = patches_img[p]
        m = patches_msk[p]
        
        if args.adirandel > 0:
            m = add_rand_mask(m, args.adirandel)
        
        outputs.append({'image': i * args.gain, 'mask': m, 'name': str(p).zfill(_zeros)})
    
    return outputs
