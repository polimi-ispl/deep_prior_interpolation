from typing import List, Tuple
import os
import numpy as np
from glob import glob
import utils as u


def _get_patch_extractor(in_shape: tuple, patch_shape: tuple, patch_stride: tuple,
                         datadim: str, imgchannel: int = None) -> u.PatchExtractor:
    ndim = len(in_shape)
    patch_shape = [patch_shape[d] if patch_shape[d] != -1 else in_shape[d] for d in range(ndim)]
    if datadim == "2.5d" and imgchannel is not None:
        patch_shape[-1] = imgchannel
    
    patch_stride = [patch_stride[d] if patch_stride[d] != -1 else patch_shape[d] for d in range(len(patch_shape))]
    
    return u.PatchExtractor(dim=tuple(patch_shape), stride=tuple(patch_stride))


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
        corrupted = u.bool2bin(corrupted)

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
    _zeros = u.ten_digit(num_patches)
    
    for p in range(num_patches):
        
        i = patches_img[p]
        m = patches_msk[p]
        
        if args.adirandel > 0:
            m = u.add_rand_mask(m, args.adirandel)
        
        outputs.append({'image': i * args.gain, 'mask': m, 'name': str(p).zfill(_zeros)})
    
    return outputs


def reconstruct_patches(args, return_history=False, verbose=False) -> Tuple[np.ndarray, list]:
    
    inputs = np.load(os.path.join(args.imgdir, args.imgname), allow_pickle=True)
   
    pe = _get_patch_extractor(inputs.shape, args.patch_shape, args.patch_stride, args.datadim, args.imgchannel)
    # this is necessary for setting pe attributes
    _ = pe.extract(inputs)
    patch_array_shape = u.patch_array_shape(inputs.shape, pe.dim, pe.stride)
    
    patches_out = []
    elapsed = []
    history = []
    for path in glob(os.path.join('./results', args.outdir) + '/*.npy'):
        try:
            out = np.load(path, allow_pickle=True).item()
        except AttributeError:
            out = np.load(path, allow_pickle=True).item()
        patches_out.append(out['output'])
        try:
            elapsed.append(out['elapsed'])
        except KeyError:
            elapsed.append(out['elapsed time'])
        history.append(out['history'])
    
    patches_out = np.asarray(patches_out)
    if args.datadim == '2.5d':
        patches_out = _transpose_patches_25d(patches_out, args.slice, adj=True)
    outputs = pe.reconstruct(patches_out.reshape(patch_array_shape)) / args.gain
    
    try:
        gpu_ = u.get_gpu_name(int(out['device']))
    except:
        gpu_ = out['device']
    
    if verbose:
        print('\n%d patches; total elapsed time on %s: %s'
              % (len(history), gpu_, u.sec2time(sum([u.time2sec(e) for e in elapsed]))))
    
    if return_history:
        return outputs, history
    else:
        return outputs
