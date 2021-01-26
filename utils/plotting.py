from typing import Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from imageio import mimsave
from cv2 import resize


def clim(in_content: np.ndarray, ratio: float = 95) -> Tuple[float, float]:
    c = np.percentile(np.absolute(in_content), ratio)
    return -c, c


def explode_volume(volume: np.ndarray, t: int = None, x: int = None, y: int = None,
                   figsize: tuple = (8, 8), cmap: str = 'bone', clipval: tuple = None, p: int = 98,
                   tlim: tuple = None, xlim: tuple = None, ylim: tuple = None, labels : list = ('[s]', '[km]', '[km]'),
                   ratio: tuple = None, linespec: dict = None,
                   filename: str or Path = None, save_opts: dict = None) -> plt.figure:
    if linespec is None:
        linespec = dict(ls='-', lw=1, color='orange')
    nt, nx, ny = volume.shape
    t_label, x_label, y_label = labels
    
    t = t if t is not None else nt//2
    x = x if x is not None else nx//2
    y = y if y is not None else ny//2

    if tlim is None:
        t_label = "samples"
        tlim = (0, volume.shape[0])
    if xlim is None:
        x_label = "samples"
        xlim = (0, volume.shape[1])
    if ylim is None:
        y_label = "samples"
        ylim = (0, volume.shape[2])
    
    # vertical lines for coordinates reference
    tline = (tlim[1] - tlim[0]) / nt * t + tlim[0]
    xline = (xlim[1] - xlim[0]) / nx * x + xlim[0]
    yline = (ylim[1] - ylim[0]) / ny * y + ylim[0]
    
    # instantiate plots
    fig = plt.figure(figsize=figsize)
    if ratio is None:
        wr = (nx, ny)
        hr = (ny, nx)
    else:
        wr = ratio[0]
        hr = ratio[1]
    opts = dict(cmap=cmap, clim=clipval if clipval is not None else clim(volume, p), aspect='auto')
    gs = fig.add_gridspec(2, 2, width_ratios=wr, height_ratios=hr,
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    
    # central plot
    ax.imshow(volume[:, :, y], extent=[xlim[0], xlim[1], tlim[1], tlim[0]], **opts)
    ax.axvline(x=xline, **linespec)
    ax.axhline(y=tline, **linespec)
    
    # top plot
    ax_top.imshow(volume[t].T, extent=[xlim[0], xlim[1], ylim[1], ylim[0]], **opts)
    ax_top.axvline(x=xline, **linespec)
    ax_top.axhline(y=yline, **linespec)
    ax_top.invert_yaxis()
    
    # right plot
    ax_right.imshow(volume[:, x], extent=[ylim[0], ylim[1], tlim[1], tlim[0]], **opts)
    ax_right.axvline(x=yline, **linespec)
    ax_right.axhline(y=tline, **linespec)
    
    # labels
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_right.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("x " + x_label)
    ax.set_ylabel("t " + t_label)
    ax_right.set_xlabel("y " + y_label)
    ax_top.set_ylabel("y " + y_label)
    
    if filename is not None:
        if save_opts is None:
            save_opts = {'format': 'png', 'dpi': 150, 'bbox_inches': 'tight'}
        plt.savefig(f"{filename}.{save_opts['format']}", **save_opts)
    plt.show()


def gif_from_array(in_content: np.ndarray, filename: str or Path,
                   clipval: tuple = None, p: int = 98, axis: int = 0,
                   width: int = None, height: int = None, **kwargs) -> None:
    if clipval is None:
        clipval = clim(in_content, p)
    if axis > in_content.ndim:
        raise ValueError("Provided dir has to be a in_content dimension")
    
    in_content = np.clip(in_content, clipval[0], clipval[1])
    in_content = (in_content - clipval[0]) / (clipval[1] - clipval[0])
    in_content = (in_content * 255).astype(np.uint8)
    
    if axis != 0:
        in_content = np.swapaxes(in_content, axis, 0)
    
    frames = [in_content[_].T for _ in range(in_content.shape[0])]
    
    if width is not None and height is not None:
        dim = (width, height)
        frames = [resize(f, dim) for f in frames]
    
    mimsave(filename, frames, 'GIF', **kwargs)


def seismograms(in_content: np.ndarray, ax, tlim: tuple = None, xlim: tuple = None,
                gain: float = 1., color: Union[str, Tuple[str]] = 'black') -> None:
    if isinstance(color, str):
        color = (color, color)
    elif isinstance(color, tuple):
        assert len(color) == 2, "color has to be a tuple of 2 elements"
    else:
        raise ValueError("color has to be a tuple of 2 elements")
    
    tlim_ = tlim if tlim is not None else (0, in_content.shape[0])
    xlim_ = xlim if xlim is not None else (1, in_content.shape[1])
    
    t_axis = np.linspace(tlim_[0], tlim_[1], in_content.shape[0])
    x_axis = np.linspace(xlim_[0], xlim_[1], in_content.shape[1])
    
    for idx, x in enumerate(x_axis):
        trace = in_content[:, idx] * gain + x
        ax.fill_betweenx(t_axis, trace, x, where=trace >= x, facecolor=color[0])
        ax.fill_betweenx(t_axis, trace, x, where=trace <= x, facecolor=color[1])
    
    ax.set_ylim(tlim_[0], tlim_[1])
    ax.invert_yaxis()

    ax.set_xticks(x_axis)
    ax.tick_params(axis='x', size=2, width=1)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')

    ax.grid(b=True, which='major', axis='y')


def plot_gather(gather: np.ndarray, figsize: tuple = (8, 8), cmap: str = 'bone',
                clipval: tuple = None, p: int = 98, tlim: tuple = None, xlim: tuple = None,
                labels: list = ('[s]', '[km]'), filename: str or Path = None) -> plt.figure:
   
    t_label, x_label = labels
    
    if tlim is None:
        t_label = "samples"
        tlim = (0, gather.shape[0])
    if xlim is None:
        x_label = "samples"
        xlim = (0, gather.shape[1])

    # instantiate plots
    plt.figure(figsize=figsize)

    plt.imshow(gather, cmap=cmap, aspect='auto',
               clim=clipval if clipval is not None else clim(gather, p),
               extent=[xlim[0], xlim[1], tlim[1], tlim[0]])

    plt.xlabel("x " + x_label)
    plt.ylabel("t " + t_label)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.show()


__all__ = [
    "clim",
    "explode_volume",
    "gif_from_array",
    "seismograms",
    "plot_gather",
]
