import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from imageio import mimsave
from cv2 import resize


def clim(in_content, ratio=95):
    c = np.percentile(np.absolute(in_content), ratio)
    return -c, c


def explode_volume(volume: np.ndarray, t: int = 0, x: int = 0, y: int = 0,
                   figsize: tuple = (8, 8), cmap: str = 'seismic', clipval: tuple = None, p: int = 98,
                   tlim: tuple = None, xlim: tuple = None, ylim: tuple = None,
                   linespec: dict = None, filename: str or Path = None) -> plt.figure:
    if linespec is None:
        linespec = dict(ls='-', lw=1, color='gold')
    nt, nx, ny = volume.shape
    
    # vertical lines for coordinates reference
    tline = (tlim[1] - tlim[0]) / nt * t + tlim[0]
    xline = (xlim[1] - xlim[0]) / nx * x + xlim[0]
    yline = (ylim[1] - ylim[0]) / ny * y + ylim[0]
    
    # instantiate plots
    fig = plt.figure(figsize=figsize)
    opts = dict(cmap=cmap, clim=clipval if clipval is not None else clim(volume, p), aspect='auto')
    gs = fig.add_gridspec(2, 2, width_ratios=(nx, ny), height_ratios=(ny, nx),
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
    ax.set_xlabel("x [m]")
    ax.set_ylabel("t [ms]")
    ax_right.set_xlabel("y [m]")
    ax_top.set_ylabel("y [m]")
    
    if filename is not None:
        plt.savefig(filename)
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


__all__ = [
    "clim",
    "explode_volume",
    "gif_from_array",
]
