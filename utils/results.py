from pathlib import Path
import os
import numpy as np
import utils as u
from data import reconstruct_patches
import matplotlib.pyplot as plt
from collections import namedtuple
from random import sample

OldHistory = namedtuple("History", ['loss', 'snr', 'pcorr'])


def show_results(res_dir: Path or str, opts: dict = None, curves: int = 0, savefig=False):
    res_dir = Path(res_dir)
    args = u.read_args(res_dir / "args.txt")
    print(args.__dict__)

    inputs = np.load(os.path.join(args.imgdir, args.imgname), allow_pickle=True)
    
    if opts is None:
        opts = dict()
    if 'clipval' not in opts.keys():
        opts['clipval'] = u.clim(inputs, 98)
    if 'save_opts' not in opts.keys():
        opts['save_opts'] = {'format': 'png', 'dpi': 150, 'bbox_inches': 'tight'}
    
    outputs, hist = reconstruct_patches(args, return_history=True, verbose=True)
    if outputs.shape != inputs.shape:
        print("\n\tWarning! Outputs and Inputs have different shape! %s - %s" % (outputs.shape, inputs.shape))
        inputs = inputs[:outputs.shape[0], :outputs.shape[1]]
        if inputs.ndim == 3:
            inputs = inputs[:, :, :outputs.shape[2]]
   
    # plot output volume
    if savefig:
        u.explode_volume(outputs, filename=res_dir / "output", **opts)
    else:
        u.explode_volume(outputs, **opts)
    
    # plot curves
    if curves > 0:
        if len(hist) <= curves:
            idx = range(len(hist))
        else:
            idx = sample(range(len(hist)), curves)
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
        axs[1].set_title('SNR = %.2f dB' % u.snr(outputs, inputs))
        axs[2].set_title('PCORR = %.2f %%' % (u.pcorr(outputs, inputs) * 100))
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
