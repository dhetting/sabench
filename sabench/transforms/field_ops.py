"""Field-operation transforms extracted from the legacy transform monolith."""

from __future__ import annotations

import numpy as np


def t_gradient_magnitude(Y: np.ndarray) -> np.ndarray:
    """Per-sample spatial gradient magnitude field |∇Y|."""
    if Y.ndim < 3:
        return np.zeros_like(Y)
    y_out = np.empty_like(Y, dtype=float)
    for sample_idx in range(len(Y)):
        gradients = np.gradient(Y[sample_idx].astype(float))
        y_out[sample_idx] = np.sqrt(sum(gradient**2 for gradient in gradients))
    return y_out


def t_matern_smooth(Y, length_scale=0.15, nu=1.5):
    """Matern-1.5 separable smoothing; works for any number of spatial axes."""
    if Y.ndim < 3:
        return Y.copy()
    n = Y.shape[0]
    spatial_shape = Y.shape[1:]

    def matern15_kernel(x, kernel_length_scale):
        r = np.abs(x)
        sr = np.sqrt(3.0) * r / kernel_length_scale
        return (1.0 + sr) * np.exp(-sr)

    def conv1d(arr, k, axis):
        pad = len(k) // 2
        sl = [slice(None)] * arr.ndim
        sl_lo = list(sl)
        sl_lo[axis] = slice(pad, 0, -1)
        sl_hi = list(sl)
        sl_hi[axis] = slice(-2, -2 - pad, -1)
        padded = np.concatenate([arr[tuple(sl_lo)], arr, arr[tuple(sl_hi)]], axis=axis)
        out = np.zeros_like(arr)
        for ki, kv in enumerate(k):
            s2 = list(sl)
            s2[axis] = slice(ki, ki + arr.shape[axis])
            out += kv * padded[tuple(s2)]
        return out

    kernels = []
    for ax_size in spatial_shape:
        dx = 1.0 / ax_size
        w = max(1, int(np.ceil(3.0 * length_scale / dx)))
        k = matern15_kernel(np.arange(-w, w + 1) * dx, length_scale)
        k /= k.sum()
        kernels.append(k)

    Yout = np.empty_like(Y)
    for s in range(n):
        tmp = Y[s].copy()
        for ax, k in enumerate(kernels):
            tmp = conv1d(tmp, k, ax)
        Yout[s] = tmp
    return Yout


def t_laplacian_roughness(Y):
    """Discrete Laplacian |∇²Y| via central differences, works for any ndim≥3."""
    if Y.ndim < 3:
        return np.zeros_like(Y)
    Yout = np.empty_like(Y, dtype=float)
    n_spatial = Y.ndim - 1
    coeff = 2.0 * n_spatial
    for s in range(len(Y)):
        f = Y[s].astype(float)
        lap = -coeff * f
        for ax in range(n_spatial):
            lap += np.roll(f, 1, ax) + np.roll(f, -1, ax)
        Yout[s] = np.abs(lap)
    return Yout


def t_contour_exceedance(Y, quantile=0.75):
    """Signed distance to an exceedance contour at the given quantile.

    For each sample, compute the contour level c = Q_q(Y_flat), then return
    (Y - c) / range(Y) clipped to [-1, 1]. This is a smooth spatial scalar
    that encodes contour crossing information without a hard binary threshold.
    On spatial fields this is interpretable as 'how far is each cell from the
    design flood contour'; on temporal fields it is a signed departure from
    a critical level.

    Physical motivation: many spatial risk assessments define zones relative to
    a contour (FEMA flood zones, seismic hazard contours), so operators that
    express output relative to such contours arise naturally.
    """
    flat = Y.reshape(len(Y), -1)
    c = np.quantile(flat, quantile, axis=1, keepdims=True)
    r = (flat.max(axis=1, keepdims=True) - flat.min(axis=1, keepdims=True)).clip(min=1e-12)
    return np.clip((flat - c) / r, -1.0, 1.0).reshape(Y.shape)


def t_isoline_length(Y, quantile=0.75):
    """Approximate length of the exceedance isoline as a scalar summary.

    Counts the number of spatial-cell boundary crossings of the threshold
    contour (Euler-characteristic-like perimeter proxy).  Returns a scalar
    per sample, broadcast to match Y's output shape for compatibility with
    the Jansen estimator.  Only meaningful for spatial outputs; returns zeros
    for scalar/1D outputs.
    """
    if Y.ndim < 3:
        return np.zeros_like(Y)
    flat = Y.reshape(len(Y), -1)
    c = np.quantile(flat, quantile, axis=1)  # (n,)
    # For 2D spatial: count sign changes along rows and columns
    n = len(Y)
    out = np.zeros(n, dtype=float)
    for s in range(n):
        field = Y[s]
        thresh = c[s]
        above = (field > thresh).astype(int)
        # Count transitions along each spatial axis
        crossings = 0
        for ax in range(field.ndim):
            diff = np.diff(above, axis=ax)
            crossings += np.abs(diff).sum()
        out[s] = float(crossings)
    # Normalise to [0,1] range across the sample
    rng = out.max() - out.min()
    if rng > 1e-12:
        out = (out - out.min()) / rng
    # Broadcast back to spatial shape
    return (out[:, None] * np.ones_like(Y.reshape(n, -1))).reshape(Y.shape)
