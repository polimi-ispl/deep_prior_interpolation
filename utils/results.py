from pathlib import Path
import os
import numpy as np
from glob import glob
import utils as u
from data import _get_patch_extractor, _transpose_patches_25d
import matplotlib.pyplot as plt
from collections import namedtuple
from random import sample

OldHistory = namedtuple("History", ['loss', 'snr', 'pcorr'])


def show_results(res_dir: Path or str, opts: dict = None, num: int = 6, savefig=False):
    res_dir = Path(res_dir)
    args = u.read_args(res_dir / "args.txt")
    print(args.__dict__)
    
    images = np.load(os.path.join(args.imgdir, args.imgname), allow_pickle=True)
    if opts is None:
        opts = dict()
    if 'clipval' not in opts.keys():
        opts['clipval'] = u.clim(images, 98)
    if 'save_opts' not in opts.keys():
        opts['save_opts'] = {'format': 'png', 'dpi': 150, 'bbox_inches': 'tight'}
    
    # if args.imgchannel is not None and args.patch_stride[-1] == -1:
    #     patch_stride = list(args.patch_stride)
    #     patch_stride[-1] = args.imgchannel
    #     args.patch_stride = tuple(patch_stride)
    
    pe = _get_patch_extractor(images.shape, args.patch_shape, args.patch_stride, args.datadim, args.imgchannel)
    # this is necessary for setting pe attributes
    _ = pe.extract(images)
    patch_array_shape = u.patch_array_shape(images.shape, pe.dim, pe.stride)
    
    patches_out = []
    elapsed = []
    hist = []
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
        hist.append(out['history'])
    
    patches_out = np.asarray(patches_out)
    if args.datadim == '2.5d':
        patches_out = _transpose_patches_25d(patches_out, args.slice, adj=True)
    outputs = pe.reconstruct(patches_out.reshape(patch_array_shape)) / args.gain
    
    if outputs.shape != images.shape:
        print("\n\tWarning! Outputs and Inputs have different shape!")
        images = images[:outputs.shape[0], :outputs.shape[1]]
        if images.ndim == 3:
            images = images[:, :, :outputs.shape[2]]
    
    try:
        gpu_ = u.get_gpu_name(int(out['device']))
    except:
        gpu_ = out['device']
    print('\n%d patches; total elapsed time on %s: %s'
          % (len(hist), gpu_, u.sec2time(sum([u.time2sec(e) for e in elapsed]))))
    
    # plot output volume
    u.explode_volume(outputs, filename=res_dir / "output", **opts)
    
    # plot curves
    if len(hist) <= num:
        idx = range(len(hist))
    else:
        idx = sample(range(len(hist)), num)
        idx.sort()
    
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    
    for i in idx:
        axs[0].plot(hist[i].loss, label='patch %d' % i)
        axs[1].plot(hist[i].snr, label='patch %d' % i)
        axs[2].plot(hist[i].pcorr, label='patch %d' % i)
        try:
            axs[3].plot(hist[i].lr, label='patch %d' % i)
        except AttributeError:
            pass
    
    try:
        axs[0].set_title('LOSS %s' % args.loss)
    except AttributeError:
        axs[0].set_title('LOSS mae')
    axs[1].set_title('SNR = %.2f dB' % u.snr(outputs, images))
    axs[2].set_title('PCORR = %.2f %%' % (u.pcorr(outputs, images) * 100))
    axs[3].set_title('Learning Rate')
    
    for a in axs:
        a.legend()
        a.set_xlim(0, args.epochs)
        a.grid()
    
    axs[0].set_ylim(0)
    axs[1].set_ylim(0)
    axs[2].set_ylim(0, 1)
    axs[3].set_ylim(0, args.lr * 10)
    
    plt.suptitle(res_dir)
    plt.tight_layout(pad=.5)
    if savefig:
        plt.savefig(res_dir / f"curves.{opts['save_opts']['format']}", **opts['save_opts'])
    plt.show()


__all__ = [
    "show_results",
]
