"""
Plotting utilities for sabench sensitivity index maps.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_s1_maps_2d(
    S1: np.ndarray,
    z1_vals: np.ndarray,
    z2_vals: np.ndarray,
    input_names: list[str] | None = None,
    title: str = "First-order Sobol indices $S_1(z_1, z_2)$",
    figsize: tuple = (14, 8),
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Plot S1 maps for all inputs on a 2D spatial grid.

    Parameters
    ----------
    S1 : (d, n_z1, n_z2)  first-order index maps
    """
    d = S1.shape[0]
    ncols = min(d, 4)
    nrows = (d + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             squeeze=False,
                             subplot_kw={'aspect': 'equal'})
    fig.suptitle(title, fontsize=12)
    ext = [z1_vals[0], z1_vals[-1], z2_vals[0], z2_vals[-1]]

    for i in range(d):
        row, col = divmod(i, ncols)
        ax  = axes[row][col]
        im  = ax.imshow(S1[i].T, origin='lower', extent=ext,
                        vmin=0, vmax=1, cmap=cmap, aspect='auto')
        lbl = input_names[i] if input_names else f"$X_{{{i+1}}}$"
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("$z_1$", fontsize=8)
        ax.set_ylabel("$z_2$", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for i in range(d, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    return fig


def plot_s1_slices_3d(
    S1: np.ndarray,
    z1_vals: np.ndarray,
    z2_vals: np.ndarray,
    z3_slices: np.ndarray,
    input_indices: list[int] | None = None,
    input_names: list[str] | None = None,
    title: str = "S1 maps at fixed $z_3$ slices",
    figsize: tuple = (16, 10),
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Plot S1 maps for selected inputs at each z3 slice.

    Parameters
    ----------
    S1 : (d, n_z1, n_z2, n_z3)
    z3_slices : values used as column labels
    input_indices : which inputs to show (default: first 4)
    """
    d   = S1.shape[0]
    nz3 = len(z3_slices)
    idx = input_indices if input_indices is not None else list(range(min(d, 4)))

    fig, axes = plt.subplots(len(idx), nz3, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=12)
    ext = [z1_vals[0], z1_vals[-1], z2_vals[0], z2_vals[-1]]

    for ri, xi in enumerate(idx):
        for ci in range(nz3):
            ax  = axes[ri][ci]
            im  = ax.imshow(S1[xi, :, :, ci].T, origin='lower', extent=ext,
                            vmin=0, vmax=1, cmap=cmap, aspect='auto')
            if ri == 0:
                ax.set_title(f"$z_3 = {z3_slices[ci]:.0f}°$", fontsize=10)
            lbl = (input_names[xi] if input_names else f"$X_{{{xi+1}}}$")
            if ci == 0:
                ax.set_ylabel(lbl, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def plot_functional_s1(
    S1: np.ndarray,
    t_vals: np.ndarray,
    input_names: list[str] | None = None,
    title: str = "First-order Sobol indices over time",
    figsize: tuple = (10, 5),
    fill_alpha: float = 0.15,
) -> plt.Figure:
    """
    Line plot of S1(t) for functional outputs.

    Parameters
    ----------
    S1 : (d, n_t) — time-varying first-order indices
    """
    d    = S1.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.get_cmap("tab10", d)

    for i in range(d):
        lbl = input_names[i] if input_names else f"$X_{{{i+1}}}$"
        c   = cmap(i)
        ax.plot(t_vals, S1[i], color=c, lw=2, label=lbl)
        ax.fill_between(t_vals, 0, S1[i], color=c, alpha=fill_alpha)

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("$S_1(t)$", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, ncol=2)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
