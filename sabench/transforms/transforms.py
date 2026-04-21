"""
transforms.py  —  Registry of physically motivated nonlinear output transformations
for empirical investigation of Sobol index non-commutativity.

Categories
----------
environmental  : log/power/Box-Cox/exceedance transforms (hydrology, flood)
engineering    : Arrhenius, Hill, Weibull, fatigue, Carnot (physical science)
spatial        : COS block-avg, Matérn, Laplacian, gradient, contour (geostatistics)
temporal       : temporal aggregation, cumulative, peak/duration, spectral (time-series SA)
statistical    : rank/PIT, standardised anomaly, entropy proxy
mathematical   : convex/concave/monotone/oscillatory/threshold/fractal transforms
                 designed for systematic mathematical-property sweeps
"""

from __future__ import annotations

import numpy as np

from sabench.transforms.aggregation import t_temporal_peak
from sabench.transforms.field_ops import t_gradient_magnitude
from sabench.transforms.linear import t_affine
from sabench.transforms.nonlinear import t_softplus_pointwise
from sabench.transforms.pointwise import (
    t_abs_pointwise,
    t_exp_pointwise,
    t_log1p_abs,
    t_relu_pointwise,
    t_sqrt_abs,
    t_square_pointwise,
    t_tanh_pointwise,
)
from sabench.transforms.samplewise import t_temporal_cumsum
from sabench.transforms.utilities import _bc, _safe_range, _ymin

# ══════════════════════════════════════════════════════════════════════════════
# Environmental / Hydrological
# ══════════════════════════════════════════════════════════════════════════════


def t_log_shift(Y, eps=1.0):
    return np.log(Y - _bc(_ymin(Y), Y) + eps)


def t_power_law(Y, beta=2.0):
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    return ((Y - s) / r) ** beta


def t_box_cox(Y, lam=0.5):
    s = _bc(_ymin(Y), Y)
    Ypos = Y - s + 1.0
    return np.log(Ypos) if abs(lam) < 1e-8 else (Ypos**lam - 1.0) / lam


def t_clipped_excess(Y, quantile=0.90):
    u = _bc(np.quantile(Y.reshape(len(Y), -1), quantile, axis=1), Y)
    return np.maximum(Y - u, 0.0)


def _exceed(Y, q):
    t = _bc(np.quantile(Y.reshape(len(Y), -1), q, axis=1), Y)
    return (Y > t).astype(float)


def t_exceed_q75(Y):
    return _exceed(Y, 0.75)


def t_exceed_q90(Y):
    return _exceed(Y, 0.90)


def t_exceed_q95(Y):
    return _exceed(Y, 0.95)


def t_exceed_q99(Y):
    return _exceed(Y, 0.99)


# ══════════════════════════════════════════════════════════════════════════════
# Engineering / Physical
# ══════════════════════════════════════════════════════════════════════════════


def t_carnot_quadratic(Y, delta=1.0):
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    return ((Y - s + delta) / (r + delta)) ** 2


def t_sigmoid_dose(Y, EC50_q=0.5, n_hill=4.0):
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    Yn = (Y - s) / r
    return Yn**n_hill / (EC50_q**n_hill + Yn**n_hill + 1e-30)


def t_arrhenius(Y, Ea_over_R=2.0):
    """Arrhenius-type exponential: exp(-Ea/R / Y_pos).
    Physical context: converts a temperature-like field to a reaction-rate field
    (e.g., atmospheric chemistry, thermal diffusivity in geothermal models).
    The strong nonlinearity amplifies high-value regions of the field.
    NOTE: This transform is used throughout for illustration precisely because
    its nonlinearity is parametric, physically motivated, and produces large D
    scores, making it an informative worst-case reference point.
    """
    s = _bc(_ymin(Y), Y)
    Ypos = Y - s + 1.0
    return np.exp(-Ea_over_R / Ypos)


def t_normalised_stress(Y, yield_q=0.80):
    s = _bc(_ymin(Y), Y)
    yld = _bc(np.quantile(Y.reshape(len(Y), -1), yield_q, axis=1), Y)
    return np.clip((Y - s) / (yld - s + 1e-12), 0.0, 1.0)


def t_weibull_reliability(Y, shape=2.0):
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    return 1.0 - np.exp(-(((Y - s) / r) ** shape))


# ══════════════════════════════════════════════════════════════════════════════
# Spatial / Geostatistical
# ══════════════════════════════════════════════════════════════════════════════


def t_regional_mean(Y):
    mu = Y.reshape(len(Y), -1).mean(axis=1)
    return np.ones_like(Y) * _bc(mu, Y)


def _block_avg(Y, block):
    """Return block-aggregated output on the COARSENED support (no upsample).

    Handles 2D spatial grids (n, nz1, nz2) → (n, nb1, nb2)
    and 3D spatial volumes (n, nz1, nz2, nz3) → (n, nb1, nb2, nb3).
    This is the geostatistically correct change-of-support (COS): the
    sensitivity analysis is performed on the aggregated output space.
    If Y is 1-D or 2-D (scalar benchmark or temporal benchmark), falls back
    to a coarsened temporal average.
    """
    if Y.ndim < 3:
        # For scalar or 1D temporal: block-average along the output axis
        n = Y.shape[0]
        vals = Y.reshape(n, -1)
        m = vals.shape[1]
        nb = m // block
        if nb < 1:
            return vals.mean(axis=1, keepdims=True)
        trimmed = vals[:, : nb * block]
        return trimmed.reshape(n, nb, block).mean(axis=2)
    n = Y.shape[0]
    spatial_dims = Y.shape[1:]
    slices = [slice(None)]
    trimmed = []
    for s in spatial_dims:
        st = (s // block) * block
        slices.append(slice(0, st))
        trimmed.append(st)
    Yt = Y[tuple(slices)]
    new_shape = [n]
    for st in trimmed:
        new_shape += [st // block, block]
    Yr = Yt.reshape(new_shape)
    block_axes = tuple(range(2, 2 + 2 * len(trimmed), 2))
    return Yr.mean(axis=block_axes)


def t_block_2x2(Y):
    return _block_avg(Y, 2)


def t_block_4x4(Y):
    return _block_avg(Y, 4)


def t_block_8x8(Y):
    return _block_avg(Y, 8)


def t_exceedance_area(Y, quantile=0.75):
    flat = Y.reshape(len(Y), -1)
    t = np.quantile(flat, quantile, axis=1, keepdims=True)
    frac = (flat > t).mean(axis=1)
    return np.ones_like(Y) * _bc(frac, Y)


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


# ══════════════════════════════════════════════════════════════════════════════
# Temporal  (1-D functional outputs: Y shape (n_eval, n_t))
# ══════════════════════════════════════════════════════════════════════════════


def t_temporal_log_cumsum(Y, eps=1.0):
    """Log of the cumulative sum: Z(t) = log(sum_{s<=t} Y(s) + eps).

    Nonlinear composed transform of a cumulative integral.  Arises in
    log-transformed streamflow volume analysis and pharmacokinetic
    log-AUC calculations. Combines the nonlinearity of log with the
    time-integration operator.
    """
    flat = Y.reshape(len(Y), -1)
    flat_pos = flat - flat.min(axis=1, keepdims=True) + eps
    return np.log(np.cumsum(flat_pos, axis=1)).reshape(Y.shape)


def t_temporal_exceedance_duration(Y, quantile=0.75):
    """Cumulative duration spent above threshold: Z(t) = sum_{s<=t} 1[Y(s) > c].

    Standard metric in flood frequency analysis (flood duration), ecology
    (duration above thermal stress threshold), and climate science (heat-wave
    duration). Returns a monotone non-decreasing step function; the nonlinearity
    arises from the indicator threshold.
    """
    flat = Y.reshape(len(Y), -1)
    c = np.quantile(flat, quantile, axis=1, keepdims=True)
    above = (flat > c).astype(float)
    return np.cumsum(above, axis=1).reshape(Y.shape)


def t_temporal_envelope(Y):
    """Running maximum envelope: Z(t) = max_{s<=t} Y(s).

    The running maximum envelope of a time series is used in structural
    dynamics (maximum response up to time t), extreme value analysis, and
    signal processing. It is a nonlinear (idempotent, monotone) operator that
    strongly compresses the sensitivity of late-time inputs.
    """
    flat = Y.reshape(len(Y), -1)
    return np.maximum.accumulate(flat, axis=1).reshape(Y.shape)


def t_temporal_bandpass(Y, low_frac=0.05, high_frac=0.30):
    """Bandpass filter retaining frequencies in [low_frac, high_frac] * Nyquist.

    Isolates a frequency band of the time series via zero-phase FFT filtering.
    Arises in spectral analysis of climate variables, seismology (body-wave
    extraction), and signal conditioning. The filtering operator is linear in
    the frequency domain but compresses the contributions of inputs that drive
    the suppressed frequency components.
    NOTE: This transform IS linear (convolution), providing another negative
    control alongside cumsum.
    """
    flat = Y.reshape(len(Y), -1)
    n_t = flat.shape[1]
    freq = np.fft.rfftfreq(n_t)  # [0, 1/(2)]
    fft_Y = np.fft.rfft(flat, axis=1)
    mask = (freq >= low_frac) & (freq <= high_frac)
    fft_Y[:, ~mask] = 0.0
    return np.fft.irfft(fft_Y, n=n_t, axis=1).reshape(Y.shape)


def t_temporal_block_avg(Y, block=10):
    """Temporal block average: coarsen n_t time steps to n_t//block steps.

    Temporal analogue of spatial change-of-support.  Arises when a fine-scale
    daily model output is compared against coarser monthly or annual observations.
    Reduces temporal resolution and potentially changes which inputs drive the
    averaged-output variance.  This is the temporal COS operator.
    """
    return _block_avg(Y, block)


# ══════════════════════════════════════════════════════════════════════════════
# Statistical / Information-Theoretic
# ══════════════════════════════════════════════════════════════════════════════


def t_rank_transform(Y):
    flat = Y.reshape(len(Y), -1)
    ranks = np.argsort(np.argsort(flat, axis=0), axis=0).astype(float) / (len(Y) - 1.0)
    return ranks.reshape(Y.shape)


def t_standardised_anomaly(Y):
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sig = flat.std(axis=1, keepdims=True).clip(min=1e-12)
    return ((flat - mu) / sig).reshape(Y.shape)


def t_entropy_proxy(Y):
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    return -((flat - mu) ** 2).reshape(Y.shape)


# ══════════════════════════════════════════════════════════════════════════════
# Mathematical Transforms — pointwise, designed for systematic property sweeps
# ══════════════════════════════════════════════════════════════════════════════

# ── Convex pointwise ──────────────────────────────────────────────────────────


def t_cube_pointwise(Y):
    """Cube: φ(y) = y³.
    Odd function; convex for y>0, concave for y<0 (inflection at 0).
    C∞, monotone increasing, φ'=3y²≥0, φ''=6y changes sign.
    """
    return Y**3


def t_cosh_pointwise(Y, scale=0.1):
    """Hyperbolic cosine: φ(y) = cosh(scale·y).
    Strictly convex, even (symmetric), C∞, non-monotone.
    Grows exponentially; models symmetric amplification.
    """
    return np.cosh(np.clip(scale * Y, -100, 100))


def t_softmax_shift(Y):
    """Shifted softmax normalisation: Z_k = exp(Y_k) / sum_k exp(Y_k).
    Nonlocal (uses sum across outputs), always produces outputs in (0,1).
    Smoothly normalises any field; often used in classification outputs.
    """
    flat = Y.reshape(len(Y), -1)
    shifted = flat - flat.max(axis=1, keepdims=True)
    ex = np.exp(shifted)
    return (ex / ex.sum(axis=1, keepdims=True)).reshape(Y.shape)


# ── Concave pointwise ─────────────────────────────────────────────────────────


def t_cbrt_pointwise(Y):
    """Cube root: φ(y) = y^(1/3) = cbrt(y).
    Monotone, odd, concave for y>0, convex for y<0.
    C∞ except at y=0 (derivative→∞). Strongly compresses tails.
    """
    return np.cbrt(Y)


def t_neg_square(Y):
    """Negative square: φ(y) = −y².
    Strictly concave (φ''=−2<0), even, non-monotone.
    Symmetric about 0; maps all values to (−∞, 0].
    """
    return -(Y**2)


# ── Monotone S-shaped ────────────────────────────────────────────────────────


def t_logistic_pointwise(Y, k=1.0):
    """Logistic: φ(y) = 1/(1+exp(−k·y)).
    Monotone increasing, bounded in (0,1), C∞.
    Concave for y>0, convex for y<0; inflection at y=0.
    Used in probability models and neural-network output layers.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(k * Y, -100, 100)))


def t_arctan_pointwise(Y, scale=1.0):
    """Arctan: φ(y) = arctan(scale·y).
    Monotone increasing, bounded in (−π/2, π/2), C∞.
    Concave for y>0, convex for y<0; symmetric saturation.
    """
    return np.arctan(scale * Y)


def t_erf_pointwise(Y, scale=0.5):
    """Error function: φ(y) = erf(scale·y).
    Monotone, bounded in (−1,1), C∞, odd function.
    Concave for y>0; related to normal CDF by erf(y)=2Φ(√2 y)−1.
    """
    from math import erf as _erf

    return np.vectorize(lambda y: _erf(scale * y))(Y).astype(float)


def t_sinh_pointwise(Y, scale=0.1):
    """Sinh: φ(y) = sinh(scale·y).
    Monotone, odd, convex for y>0, concave for y<0 (inflection at 0).
    Grows exponentially; complementary to tanh.
    """
    return np.sinh(np.clip(scale * Y, -100, 100))


# ── Oscillatory / non-monotone ────────────────────────────────────────────────


def t_sin_pointwise(Y, freq=0.5):
    """Sine: φ(y) = sin(freq·y).
    Periodic (T=2π/freq), bounded in [−1,1], C∞.
    Alternately convex and concave; zero mean for uniform Y.
    Alternates sensitivity signs with period: high-frequency → strong noncommutativity.
    """
    return np.sin(freq * Y)


def t_cos_pointwise(Y, freq=0.5):
    """Cosine: φ(y) = cos(freq·y).
    Periodic, even, bounded in [−1,1], C∞.
    Like sine but with maximum at origin.
    """
    return np.cos(freq * Y)


def t_step_pointwise(Y, threshold=0.0):
    """Heaviside step: φ(y) = 1[y > threshold].
    Discontinuous (C⁻¹), non-monotone in the sense of being constant piecewise.
    Extreme noncommutativity expected; destroys all metric information.
    """
    return (Y > threshold).astype(float)


def t_triangle_wave(Y, period=4.0):
    """Triangle wave: piecewise linear periodic function with period 'period'.
    Bounded, periodic, continuous but non-differentiable at peaks.
    Models periodic clipping in signal processing.
    """
    t = (Y % period) / period  # [0, 1)
    return 2.0 * np.abs(2.0 * t - 1.0) - 1.0


# ── Higher-order derivative structure ────────────────────────────────────────


def t_smooth_bump(Y, width=3.0):
    """C∞ smooth bump (mollifier): φ(y) = exp(−width/(width²−y²)) if |y|<width, else 0.
    Has compact support; zero outside ±width, C∞ everywhere including boundary.
    Canonical example of C∞ function with compact support; all derivatives exist.
    """
    arg = width**2 - Y**2
    out = np.where(arg > 0, np.exp(-width / np.maximum(arg, 1e-20)), 0.0)
    return out


def t_rational_quadratic(Y):
    """Rational quadratic: φ(y) = 1/(1+y²).
    Bounded in (0,1], convex near y=0, decreasing from centre.
    Related to Cauchy distribution; heavy-tail suppression.
    φ''(0) = −2 (concave); φ''(±1) = 0; non-monotone.
    """
    return 1.0 / (1.0 + Y**2)


def t_inverse_abs(Y, eps=1.0):
    """Inverse: φ(y) = 1/(|y| + eps).
    Convex, strictly decreasing in |y|, bounded above by 1/eps.
    Singular at y=0 for eps=0; eps provides regularisation.
    """
    return 1.0 / (np.abs(Y) + eps)


def t_log_abs(Y, eps=1.0):
    """Log of shifted absolute value: φ(y) = log(|y| + eps).
    Concave for |y|>0, symmetric (even), grows logarithmically.
    """
    return np.log(np.abs(Y) + eps)


# ── Normalisation / standardisation ──────────────────────────────────────────


def t_min_max_normalise(Y):
    """Min-max normalisation: Z = (Y − min) / (max − min).
    Affine per-sample (preserves Sobol indices within a single batch);
    but across samples it is nonlocal.
    """
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    return (Y - s) / r


def t_robust_scale(Y):
    """Robust IQR scaling: Z = (Y − Q50) / (Q75 − Q25).
    Nonlocal, outlier-resistant standardisation.
    """
    flat = Y.reshape(len(Y), -1)
    q25 = np.quantile(flat, 0.25, axis=1)
    q50 = np.quantile(flat, 0.50, axis=1)
    q75 = np.quantile(flat, 0.75, axis=1)
    iqr = (q75 - q25).clip(min=1e-12)
    return ((flat - q50[:, None]) / iqr[:, None]).reshape(Y.shape)


def t_clamp_sigma(Y, n_sigma=2.0):
    """Clamp to ±n_sigma standard deviations: soft Winsorisation.
    Nonlocal; clips extreme values without hard quantile boundaries.
    """
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sig = flat.std(axis=1, keepdims=True).clip(min=1e-12)
    lo, hi = mu - n_sigma * sig, mu + n_sigma * sig
    return np.clip(flat, lo, hi).reshape(Y.shape)


# ══════════════════════════════════════════════════════════════════════════════
# Environmental additions
# ══════════════════════════════════════════════════════════════════════════════


def t_gumbel_cdf(Y):
    """Gumbel (EV Type-I) CDF: F(y) = exp(−exp(−(y−μ)/β)) with per-sample μ,β.
    Models the probability of non-exceedance in extreme value analysis (floods,
    wind speeds). The double-exponential structure creates very strong nonlinearity
    at both tails.
    """
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sig = flat.std(axis=1, keepdims=True).clip(min=1e-12)
    beta = sig * np.sqrt(6.0) / np.pi
    loc = mu - 0.5772 * beta
    z = (flat - loc) / beta
    return np.exp(-np.exp(-z)).reshape(Y.shape)


def t_frechet_cdf(Y, shape=2.0):
    """Fréchet (EV Type-II) CDF: F(y) = exp(−(y/s)^{-shape}) for y>0.
    Models heavy-tailed maxima (ocean waves, insurance losses).
    """
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    Ypos = (Y - s) / r + 1e-6
    return np.exp(-(Ypos ** (-shape)))


def t_log_normal_cdf(Y, sigma=0.5):
    """Log-normal CDF: Φ((log(Ypos) − μ)/σ), with μ set so median=1.
    Models multiplicative processes (precipitation, groundwater).
    """
    from math import erfc as _erfc

    s = _bc(_ymin(Y), Y)
    Ypos = Y - s + 1.0
    log_y = np.log(Ypos)
    mu_ln = log_y.reshape(len(Y), -1).mean(axis=1, keepdims=True)
    z = (log_y.reshape(len(Y), -1) - mu_ln) / sigma
    cdf = 0.5 * np.vectorize(lambda zi: 1.0 - _erfc(zi / np.sqrt(2)) / 2)(z).astype(float)
    return cdf.reshape(Y.shape)


def t_return_period(Y):
    """Return period: T = 1/(1 − F(y)), where F is empirical per-sample CDF.
    Transforms a field to the expected return period (in units of sample counts).
    Diverges to infinity at the sample maximum → strong nonlinearity.
    """
    flat = Y.reshape(len(Y), -1)
    n_out = flat.shape[1]
    ranks = np.argsort(np.argsort(flat, axis=1), axis=1).astype(float) + 1.0
    F = ranks / (n_out + 1.0)  # Weibull plotting position
    T = 1.0 / (1.0 - F).clip(min=1.0 / (n_out + 2.0))
    return T.reshape(Y.shape)


# ══════════════════════════════════════════════════════════════════════════════
# Engineering additions
# ══════════════════════════════════════════════════════════════════════════════


def t_johnson_su(Y):
    """Johnson SU normalisation: Z = γ + δ·arcsinh((Y−ξ)/λ).
    Fit parameters estimated from the first 4 sample moments.
    Used in reliability and insurance to normalise heavily skewed data.
    """
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sig = flat.std(axis=1, keepdims=True).clip(min=1e-12)
    return np.arcsinh((flat - mu) / sig).reshape(Y.shape)


def t_fatigue_miner(Y, m=3.0):
    """Palmgren-Miner damage accumulation: D = Σ (ΔS_i / N_f_i) ~ S^m / C.
    Raises normalised amplitude to the Paris-law exponent m (steel: m≈3).
    Models structural fatigue damage accumulation in fracture mechanics.
    """
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    Yn = (Y - s) / r + 1e-6
    return Yn**m


def t_rankine_failure(Y):
    """Rankine failure: Z = max(Y_flat, 0) — principal stress criterion.
    Models the maximum tensile stress component; zeros out compressive stresses.
    Non-smooth (C⁰), non-negative, nonlocal (uses max across outputs).
    """
    flat = Y.reshape(len(Y), -1)
    return np.maximum(flat, 0.0).reshape(Y.shape)


# ══════════════════════════════════════════════════════════════════════════════
# Temporal additions
# ══════════════════════════════════════════════════════════════════════════════


def t_temporal_rms(Y):
    """Root mean square: Z = sqrt(mean(Y^2)) broadcast to Y's shape.
    Scalar summary of temporal energy; common in structural dynamics.
    """
    flat = Y.reshape(len(Y), -1)
    rms = np.sqrt((flat**2).mean(axis=1))
    return (rms[:, None] * np.ones_like(flat)).reshape(Y.shape)


def t_temporal_range(Y):
    """Temporal range: Z = max(Y) − min(Y) broadcast.
    Measures amplitude spread; sensitive to extremes, ignores mean.
    """
    flat = Y.reshape(len(Y), -1)
    rng = flat.max(axis=1) - flat.min(axis=1)
    return (rng[:, None] * np.ones_like(flat)).reshape(Y.shape)


def t_temporal_autocorr(Y):
    """Lag-1 autocorrelation: ρ₁ = Cov(Y_t, Y_{t+1}) / Var(Y).
    Scalar temporal dependency metric, broadcast. In (−1,1).
    Low for white noise, high for smooth time series.
    """
    flat = Y.reshape(len(Y), -1)
    n = flat.shape[1]
    if n < 3:
        return np.zeros_like(Y)
    mu = flat.mean(axis=1, keepdims=True)
    sig2 = flat.var(axis=1).clip(min=1e-12)
    ac = ((flat[:, :-1] - mu) * (flat[:, 1:] - mu)).mean(axis=1) / sig2
    return (ac[:, None] * np.ones_like(flat)).reshape(Y.shape)


def t_temporal_quantile(Y, q=0.50):
    """Temporal q-quantile: scalar summary broadcast to Y's shape."""
    flat = Y.reshape(len(Y), -1)
    qv = np.quantile(flat, q, axis=1)
    return (qv[:, None] * np.ones_like(flat)).reshape(Y.shape)


# ══════════════════════════════════════════════════════════════════════════════
# Statistical additions
# ══════════════════════════════════════════════════════════════════════════════


def t_quantile_normalise(Y):
    """Empirical CDF quantile normalisation: Z = rank/(n+1).
    Maps each value to its empirical quantile. Uniform marginals guaranteed.
    Smoothed version of rank transform; equivalent to nonparametric CDF.
    """
    flat = Y.reshape(len(Y), -1)
    n = flat.shape[0]
    # Per-output quantile normalisation across samples
    ranks = np.argsort(np.argsort(flat, axis=0), axis=0).astype(float) + 1.0
    return (ranks / (n + 1.0)).reshape(Y.shape)


def t_winsorise(Y, low=0.10, high=0.90):
    """Winsorise: clip values below q_low and above q_high per sample."""
    flat = Y.reshape(len(Y), -1)
    lo = np.quantile(flat, low, axis=1, keepdims=True)
    hi = np.quantile(flat, high, axis=1, keepdims=True)
    return np.clip(flat, lo, hi).reshape(Y.shape)


def t_yeo_johnson(Y, lam=0.5):
    """Yeo-Johnson transform — extends Box-Cox to all reals.

    For y ≥ 0:  ((y+1)^λ − 1) / λ         if λ ≠ 0
                log(y+1)                    if λ = 0
    For y < 0:  −((−y+1)^{2−λ} − 1)/(2−λ)  if λ ≠ 2
                −log(−y+1)                  if λ = 2
    """
    out = np.empty_like(Y, dtype=float)
    pos = Y >= 0
    neg = ~pos
    if abs(lam) < 1e-8:
        out[pos] = np.log(Y[pos] + 1.0)
    else:
        out[pos] = ((Y[pos] + 1.0) ** lam - 1.0) / lam
    lam2 = 2.0 - lam
    if abs(lam2) < 1e-8:
        out[neg] = -np.log(-Y[neg] + 1.0)
    else:
        out[neg] = -((-Y[neg] + 1.0) ** lam2 - 1.0) / lam2
    return out


def t_inverse_normal(Y):
    """Inverse normal (probit): Z = Φ⁻¹(F(y)) where F is empirical CDF.
    Maps uniform marginals to standard normal. Equivalent to normal scores.
    Blom (1958) correction: rank (r − 3/8) / (n + 1/4).
    """
    flat = Y.reshape(len(Y), -1)
    n = flat.shape[0]
    ranks = np.argsort(np.argsort(flat, axis=0), axis=0).astype(float) + 1.0
    p = (ranks - 0.375) / (n + 0.25)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    # Rational approximation to Φ⁻¹
    # Beasley-Springer-Moro algorithm
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511349,
        3.21767881768e-5,
        2.888167364e-7,
        3.960315187e-7,
    ]
    q = p - 0.5
    out = np.empty_like(p)
    m = np.abs(q) <= 0.42
    r = q[m] ** 2
    num = ((a[3] * r + a[2]) * r + a[1]) * r + a[0]
    den = (((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0
    out[m] = q[m] * num / den
    lp = p[~m]
    lp = np.where(q[~m] > 0, 1.0 - lp, lp)
    r2 = np.sqrt(-np.log(lp))
    ppf = c[0] + r2 * (
        c[1]
        + r2
        * (c[2] + r2 * (c[3] + r2 * (c[4] + r2 * (c[5] + r2 * (c[6] + r2 * (c[7] + r2 * c[8]))))))
    )
    out[~m] = np.where(q[~m] > 0, ppf, -ppf)
    return out.reshape(Y.shape)


# ══════════════════════════════════════════════════════════════════════════════
# Transform Registry
# ══════════════════════════════════════════════════════════════════════════════


# ============================================================================
# POLYNOMIAL FAMILY  (higher-order, orthogonal, signed)
# ============================================================================


def t_poly4(Y, scale=0.05):
    """phi(y) = (scale*y)^4 -- even, C-inf, convex, quartic."""
    return (scale * Y) ** 4


def t_poly5(Y, scale=0.05):
    """phi(y) = (scale*y)^5 -- odd, C-inf, inflection at 0."""
    return (scale * Y) ** 5


def t_poly6(Y, scale=0.05):
    """phi(y) = (scale*y)^6 -- even, C-inf, strictly convex."""
    return (scale * Y) ** 6


def t_signed_power(Y, p=1.5, scale=0.2):
    """phi(y) = sign(y)*|scale*y|^p -- odd, monotone, C1 for p>=1."""
    u = scale * Y
    return np.sign(u) * (np.abs(u) ** p)


def t_legendre_p3(Y, scale=0.3):
    """Legendre P3: phi(u) = (5u^3 - 3u)/2 -- odd, oscillatory orthogonal polynomial."""
    u = np.clip(scale * Y, -1.0, 1.0)
    return 0.5 * (5.0 * u**3 - 3.0 * u)


def t_chebyshev_t4(Y, scale=0.2):
    """Chebyshev T4: phi(u) = 8u^4 - 8u^2 + 1 -- even, C-inf, 3 extrema."""
    u = np.clip(scale * Y, -1.0, 1.0)
    return 8.0 * u**4 - 8.0 * u**2 + 1.0


def t_hermite_he2(Y, scale=0.3):
    """Probabilist Hermite He2: phi(u) = u^2 - 1 -- even, convex, parabola shifted."""
    u = scale * Y
    return u**2 - 1.0


def t_hermite_he3(Y, scale=0.3):
    """Probabilist Hermite He3: phi(u) = u^3 - 3u -- odd, two local extrema."""
    u = scale * Y
    return u**3 - 3.0 * u


def t_bernstein_b3(Y):
    """Bernstein B3 basis: phi(u) = 3u^2*(1-u) on [0,1] -- hump shape, nonmonotone."""
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    u = np.clip((Y - s) / r, 0.0, 1.0)
    return 3.0 * u**2 * (1.0 - u)


# ============================================================================
# SIGMOID / ACTIVATION FAMILY  (neural, biochemical, bounded)
# ============================================================================


def t_atan2pi(Y, scale=1.0):
    """phi(y) = (2/pi)*arctan(scale*y) -- maps R -> (-1,1), monotone, C-inf."""
    return (2.0 / np.pi) * np.arctan(scale * Y)


def t_gompertz(Y, b=1.0, c=0.5):
    """Gompertz CDF: phi(y) = exp(-exp(-b*(y-c))) -- S-shaped, asymmetric."""
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    u = (Y - s) / r - 0.5
    return np.exp(-np.exp(-b * u))


def t_algebraic_sigmoid(Y, scale=0.5):
    """Algebraic sigmoid: phi(y) = y/sqrt(1+y^2) -- monotone, bounded (-1,1)."""
    u = scale * Y
    return u / np.sqrt(1.0 + u**2)


def t_swish(Y, beta=1.0):
    """Swish activation: phi(y) = y * sigmoid(beta*y) -- non-monotone for beta>0."""
    return Y * (1.0 / (1.0 + np.exp(-beta * Y)))


def t_mish(Y):
    """Mish activation: phi(y) = y * tanh(softplus(y)) -- smooth non-monotone."""
    sp = np.log1p(np.exp(np.clip(Y, -500, 500)))
    return Y * np.tanh(sp)


def t_selu(Y, alpha=1.6733, lam=1.0507):
    """SELU activation: scaled ELU -- piecewise, C1, self-normalising."""
    return lam * np.where(Y >= 0.0, Y, alpha * (np.exp(Y) - 1.0))


def t_softsign(Y, scale=1.0):
    """Softsign: phi(y) = y/(1+|y|) -- monotone, bounded (-1,1), C1."""
    return scale * Y / (1.0 + np.abs(Y))


def t_bent_identity(Y, scale=0.5):
    """Bent identity: phi(y) = (sqrt(y^2+1)-1)/2 + y -- monotone, near-linear."""
    u = scale * Y
    return (np.sqrt(u**2 + 1.0) - 1.0) / 2.0 + u


def t_hard_sigmoid(Y, scale=0.5):
    """Hard sigmoid: phi(y) = clip(0.2*y+0.5, 0, 1) -- piecewise linear, C0."""
    return np.clip(0.2 * scale * Y + 0.5, 0.0, 1.0)


def t_hard_tanh(Y, scale=0.3):
    """Hard tanh: phi(y) = clip(y, -1, 1) -- piecewise linear, bounded, C0."""
    return np.clip(scale * Y, -1.0, 1.0)


# ============================================================================
# OSCILLATORY / PERIODIC FAMILY
# ============================================================================


def t_sinc(Y, scale=0.5):
    """Normalised sinc: phi(y) = sin(pi*scale*y)/(pi*scale*y) -- C-inf, decaying osc."""
    u = scale * Y
    return np.sinc(u)  # numpy sinc is normalised: sin(pi*x)/(pi*x)


def t_sin_squared(Y, freq=0.5):
    """phi(y) = sin^2(freq*y) -- non-negative, bounded [0,1], even, periodic."""
    return np.sin(freq * Y) ** 2


def t_cos_squared(Y, freq=0.5):
    """phi(y) = cos^2(freq*y) -- non-negative, bounded [0,1], even, periodic."""
    return np.cos(freq * Y) ** 2


def t_damped_sin(Y, freq=0.5, decay=0.1):
    """phi(y) = exp(-decay*|y|)*sin(freq*y) -- decaying oscillation, odd, C-inf."""
    return np.exp(-decay * np.abs(Y)) * np.sin(freq * Y)


def t_sawtooth(Y, period=4.0):
    """Sawtooth wave: phi(y) = 2*(y/period - floor(y/period+0.5)) -- C0 except jumps."""
    return 2.0 * (Y / period - np.floor(Y / period + 0.5))


def t_square_wave(Y, period=4.0):
    """Square wave: phi(y) = sign(sin(2*pi*y/period)) -- discontinuous, periodic."""
    return np.sign(np.sin(2.0 * np.pi * Y / period))


def t_double_sin(Y, freq1=0.3, freq2=0.7):
    """phi(y) = sin(freq1*y) + sin(freq2*y) -- two-frequency interference pattern."""
    return np.sin(freq1 * Y) + np.sin(freq2 * Y)


def t_sin_cos_product(Y, freq=0.5):
    """phi(y) = sin(freq*y)*cos(freq*y) = 0.5*sin(2*freq*y) -- harmonic product."""
    return np.sin(freq * Y) * np.cos(freq * Y)


# ============================================================================
# THRESHOLD / PIECEWISE FAMILY
# ============================================================================


def t_soft_threshold(Y, lam=1.0):
    """Soft threshold (lasso shrinkage): phi(y) = sign(y)*max(|y|-lam, 0) -- C0."""
    return np.sign(Y) * np.maximum(np.abs(Y) - lam, 0.0)


def t_hard_threshold(Y, lam=1.0):
    """Hard threshold: phi(y) = y*(|y|>=lam) -- discontinuous at +-lam."""
    return Y * (np.abs(Y) >= lam).astype(float)


def t_ramp(Y, lo=-1.0, hi=1.0):
    """Ramp / leaky clip: phi(y) = clip(y, lo, hi) -- piecewise linear, C0."""
    return np.clip(Y, lo, hi)


def t_spike(Y, center=0.0, width=1.0):
    """Spike indicator: phi(y) = exp(-(y-center)^2/(2*width^2)) -- C-inf bump."""
    return np.exp(-0.5 * ((Y - center) / width) ** 2)


def t_breakpoint(Y, bp=0.0, slope_lo=0.5, slope_hi=2.0):
    """Piecewise linear breakpoint: slope_lo below bp, slope_hi above -- C0 kink."""
    return np.where(Y < bp, slope_lo * (Y - bp), slope_hi * (Y - bp))


def t_hockey_stick(Y, bp=0.0):
    """Hockey stick: phi(y) = 0 for y<bp, y-bp for y>=bp -- convex, C0, ReLU-shift."""
    return np.maximum(Y - bp, 0.0)


def t_deadzone(Y, half_width=1.0):
    """Deadzone: phi(y)=0 for |y|<half_width, y-sign(y)*hw otherwise -- C0."""
    return np.sign(Y) * np.maximum(np.abs(Y) - half_width, 0.0)


def t_bimodal_flip(Y):
    """Bimodal sign flip: phi(u) = 4*u*(1-u)*(2*u-1) on [0,1] -- zero at 0,0.5,1."""
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    u = np.clip((Y - s) / r, 0.0, 1.0)
    return 4.0 * u * (1.0 - u) * (2.0 * u - 1.0)


def t_donut(Y, center=0.0, radius=1.5, width=0.5):
    """Donut/ring indicator: phi(y) = exp(-((|y|-radius)/width)^2) -- C-inf."""
    return np.exp(-(((np.abs(Y - center) - radius) / width) ** 2))


# ============================================================================
# VARIANCE-STABILISING FAMILY
# ============================================================================


def t_anscombe(Y):
    """Anscombe transform: phi(y) = 2*sqrt(y+3/8) -- variance-stabilises Poisson counts."""
    s = _bc(_ymin(Y), Y)
    Ypos = Y - s
    return 2.0 * np.sqrt(np.maximum(Ypos + 0.375, 0.0))


def t_freeman_tukey(Y):
    """Freeman-Tukey: phi(y) = sqrt(y) + sqrt(y+1) -- variance-stabilises counts."""
    s = _bc(_ymin(Y), Y)
    Ypos = np.maximum(Y - s, 0.0)
    return np.sqrt(Ypos) + np.sqrt(Ypos + 1.0)


def t_asinh_vs(Y, scale=0.5):
    """Inverse hyperbolic sine (arcsinh): phi(y) = asinh(scale*y) -- C-inf, VST."""
    return np.arcsinh(scale * Y)


def t_modulus(Y, lam=0.5):
    """Modulus transform: phi(y) = sign(y)*(|y|+1)^lam - 1) -- C1, generalized VST."""
    return np.sign(Y) * ((np.abs(Y) + 1.0) ** lam - 1.0)


def t_dual_power(Y, lam=0.3):
    """Dual power (Yeo-Johnson lam=0.3 special case): maps all reals, C1."""
    lam = float(lam)
    pos = Y >= 0
    Z = np.empty_like(Y, dtype=float)
    Z[pos] = ((Y[pos] + 1.0) ** lam - 1.0) / lam if abs(lam) > 1e-8 else np.log(Y[pos] + 1.0)
    Z[~pos] = (
        -((-Y[~pos] + 1.0) ** (2.0 - lam) - 1.0) / (2.0 - lam)
        if abs(2.0 - lam) > 1e-8
        else -np.log(-Y[~pos] + 1.0)
    )
    return Z


def t_log2_shift(Y, eps=1.0):
    """Log base-2 shift: phi(y) = log2(y - ymin + eps) -- decibel-like scaling."""
    s = _bc(_ymin(Y), Y)
    return np.log2(Y - s + eps)


def t_log10_shift(Y, eps=1.0):
    """Log base-10 shift: phi(y) = log10(y - ymin + eps) -- order-of-magnitude scale."""
    s = _bc(_ymin(Y), Y)
    return np.log10(Y - s + eps)


# ============================================================================
# CURVATURE EXTREMES / SPECIAL SHAPES
# ============================================================================


def t_exp_neg_sq(Y, scale=0.3):
    """Gaussian kernel: phi(y) = exp(-scale^2*y^2) -- C-inf, non-monotone, even."""
    return np.exp(-((scale * Y) ** 2))


def t_exp_pos_sq(Y, scale=0.2):
    """Anti-Gaussian: phi(y) = exp(+(scale*y)^2) -- convex, even, superexponential."""
    return np.exp(np.minimum((scale * Y) ** 2, 700.0))


def t_inverse_sq(Y, eps=1.0):
    """phi(y) = 1/(y^2 + eps) -- convex, even, singularity regularised at 0."""
    return 1.0 / (Y**2 + eps)


def t_log_log(Y, eps=1.0):
    """Double logarithm: phi(y) = log(1 + log(y - ymin + eps)) -- extreme compression."""
    s = _bc(_ymin(Y), Y)
    return np.log1p(np.log(Y - s + eps))


def t_power_exp(Y, scale=0.1):
    """phi(y) = y^2 * exp(-|y|*scale) -- hump, convex near 0, decaying tails."""
    return Y**2 * np.exp(-np.abs(Y) * scale)


def t_gev_cdf(Y, xi=0.3):
    """GEV CDF (xi>0 Frechet family): phi(y) = exp(-(1+xi*u)^(-1/xi)) on u in (-1/xi, inf)."""
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    u = (Y - s) / r
    t_val = np.maximum(1.0 + xi * u, 1e-10)
    return np.exp(-(t_val ** (-1.0 / xi)))


def t_pareto_tail(Y, alpha=1.5):
    """Pareto tail transform: phi(u) = 1 - (1-u)^alpha on [0,1] -- heavy tail."""
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    u = np.clip((Y - s) / r, 0.0, 1.0 - 1e-9)
    return 1.0 - (1.0 - u) ** alpha


def t_log_logistic_cdf(Y, beta=2.0):
    """Log-logistic CDF: phi(u) = u^beta/(1+u^beta) -- S-shaped with heavy tail."""
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    u = np.maximum((Y - s) / r, 0.0)
    ub = np.minimum(u**beta, 1e15)
    return ub / (1.0 + ub)


# ============================================================================
# FINANCIAL TRANSFORMS (scalar output -> risk/return metric)
# ============================================================================


def t_var_proxy(Y, q=0.95):
    """Value-at-Risk proxy: phi(sample) = quantile_q -- nonlocal threshold metric."""
    return _bc(np.quantile(Y.reshape(len(Y), -1), q, axis=1), Y) * np.ones_like(Y)


def t_cvar(Y, q=0.95):
    """CVaR/Expected Shortfall: phi = mean of values above VaR(q) -- nonlocal."""
    flat = Y.reshape(len(Y), -1)
    thresholds = np.quantile(flat, q, axis=1)
    out = np.empty(len(Y))
    for i in range(len(Y)):
        tail = flat[i][flat[i] >= thresholds[i]]
        out[i] = float(tail.mean()) if len(tail) > 0 else thresholds[i]
    return out.reshape(Y.shape[0:1] + (1,) * max(Y.ndim - 1, 0)) * np.ones_like(Y)


def t_sharpe_proxy(Y, rf=0.0):
    """Sharpe ratio proxy: phi = (mean - rf)/std -- nonlocal risk-adjusted return."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1)
    sg = flat.std(axis=1) + 1e-10
    return _bc((mu - rf) / sg, Y) * np.ones_like(Y)


def t_drawdown(Y):
    """Max drawdown: phi(i,t) = Y(i,t) - running_max(Y(i,:t)) -- temporal, nonlocal."""
    if Y.ndim < 2:
        return np.zeros_like(Y)
    cummax = np.maximum.accumulate(Y, axis=1)
    return Y - cummax


def t_fold_change(Y, eps=1.0):
    """Log2 fold-change from sample mean: phi = log2(Y/mean) -- genomics / finance."""
    s = _bc(_ymin(Y), Y)
    Ypos = Y - s + eps
    mu = _bc(Ypos.reshape(len(Y), -1).mean(axis=1), Y)
    return np.log2(Ypos / mu)


def t_excess_return(Y):
    """Excess return over sample mean: phi = Y - mean(Y) -- mean-centring."""
    mu = _bc(Y.reshape(len(Y), -1).mean(axis=1), Y)
    return Y - mu


# ============================================================================
# ECOLOGICAL / BIODIVERSITY TRANSFORMS
# ============================================================================


def t_hellinger(Y):
    """Hellinger transform: phi(u) = sqrt(u/sum(u)) -- chi^2 distance equaliser."""
    flat = np.maximum(Y.reshape(len(Y), -1), 0.0)
    row_sum = flat.sum(axis=1, keepdims=True) + 1e-12
    return np.sqrt(flat / row_sum).reshape(Y.shape)


def t_chord_dist(Y):
    """Chord normalisation: phi(u) = u/||u|| -- L2 unit sphere projection."""
    flat = Y.reshape(len(Y), -1)
    norm = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    return (flat / norm).reshape(Y.shape)


def t_relative_abundance(Y):
    """Relative abundance: phi(u) = u/sum(u) -- simplex projection, sums to 1."""
    flat = np.maximum(Y.reshape(len(Y), -1), 0.0)
    row_sum = flat.sum(axis=1, keepdims=True) + 1e-12
    return (flat / row_sum).reshape(Y.shape)


def t_log_ratio(Y, eps=1.0):
    """Log-ratio (isometric log-ratio like): phi(y) = log(y - ymin + eps) - mean_log."""
    s = _bc(_ymin(Y), Y)
    log_y = np.log(Y - s + eps)
    mu = _bc(log_y.reshape(len(Y), -1).mean(axis=1), Y)
    return log_y - mu


# ============================================================================
# CLIMATE / ENVIRONMENTAL SCIENCE TRANSFORMS
# ============================================================================


def t_anomaly_pct(Y, eps=1.0):
    """Anomaly percent: phi = (Y - mean)/|mean| -- percentage departure from mean."""
    flat = Y.reshape(len(Y), -1)
    mu = _bc(flat.mean(axis=1), Y)
    denom = np.abs(mu) + eps
    return (Y - mu) / denom


def t_bias_correction(Y):
    """Bias correction (linear scaling): phi = Y * (target_mean/sample_mean)."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1) + 1e-12
    return Y * _bc(1.0 / mu, Y)


def t_quantile_delta(Y, q=0.90):
    """Quantile delta: phi = Y * (q_target/q_sample) -- quantile scaling correction."""
    flat = Y.reshape(len(Y), -1)
    qvals = np.quantile(flat, q, axis=1)
    scale = 1.0 / (qvals + 1e-10)
    return Y * _bc(scale, Y)


def t_growing_degree_days(Y, base=10.0):
    """Growing degree days: phi = max(Y - base, 0) -- threshold accumulation."""
    return np.maximum(Y - base, 0.0)


def t_standardised_precip_idx(Y):
    """Standardised precip index proxy: phi = (Y - mean)/std -- climate anomaly."""
    flat = Y.reshape(len(Y), -1)
    mu = _bc(flat.mean(axis=1), Y)
    sg = _bc(flat.std(axis=1) + 1e-10, Y)
    return (Y - mu) / sg


# ============================================================================
# HYDROLOGY / WATER RESOURCES TRANSFORMS
# ============================================================================


def t_nash_sutcliffe(Y):
    """Nash-Sutcliffe efficiency proxy: phi = 1 - MSE/Var(Y_obs) -- goodness-of-fit."""
    flat = Y.reshape(len(Y), -1)
    mu = _bc(flat.mean(axis=1), Y)
    var_obs = _bc(flat.var(axis=1) + 1e-12, Y)
    mse = (Y - mu) ** 2
    return 1.0 - mse / var_obs


def t_pot_log(Y, q=0.90, eps=1.0):
    """Log-transformed peaks-over-threshold: phi = log(max(Y-threshold, 0)+eps)."""
    threshold = _bc(np.quantile(Y.reshape(len(Y), -1), q, axis=1), Y)
    return np.log(np.maximum(Y - threshold, 0.0) + eps)


def t_log_flow(Y, eps=0.01):
    """Log streamflow: phi = log(Y - ymin + eps) -- standard hydrological transform."""
    s = _bc(_ymin(Y), Y)
    return np.log(Y - s + eps)


# ============================================================================
# MEDICAL / PHARMACOLOGICAL TRANSFORMS
# ============================================================================


def t_hill_response(Y, n=2.0, EC50_q=0.5):
    """Hill equation response: phi = Y^n / (EC50^n + Y^n) -- receptor saturation."""
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    Yn = np.maximum((Y - s) / r, 0.0)
    EC50 = EC50_q
    return Yn**n / (EC50**n + Yn**n + 1e-12)


def t_log_auc(Y, eps=1.0):
    """Log area-under-curve proxy: phi = log(mean(Y) + eps) -- pharmacokinetic."""
    flat = Y.reshape(len(Y), -1)
    auc = flat.mean(axis=1)
    auc_shifted = np.maximum(auc, 0.0)
    return _bc(np.log(auc_shifted + eps), Y) * np.ones_like(Y)


def t_emax_model(Y, Emax=1.0, ED50_q=0.5, n=1.0):
    """Emax model: phi = Emax * Y^n / (ED50^n + Y^n) -- maximum effect model."""
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    Yn = np.maximum((Y - s) / r, 0.0)
    return Emax * Yn**n / (ED50_q**n + Yn**n + 1e-12)


# ============================================================================
# STRUCTURAL / MECHANICAL ENGINEERING TRANSFORMS
# ============================================================================


def t_von_mises(Y):
    """Von Mises stress proxy: phi = sqrt(Y^2 + 0.25*Y^2) = |Y|*sqrt(1.25) -- scaled abs."""
    return np.abs(Y) * np.sqrt(1.25)


def t_safety_factor(Y, capacity=1.0):
    """Safety factor: phi = capacity/(|Y| + eps) -- inverse magnitude."""
    return capacity / (np.abs(Y) + 1e-6)


def t_cumulative_damage(Y, m=3.0):
    """Palmgren-Miner cumulative damage per cycle: phi = |Y|^m -- power law damage."""
    return np.abs(Y) ** m


def t_stress_life(Y, C=1e6, m=3.0):
    """Stress-life (Basquin): phi = C / (|Y| + eps)^m -- S-N curve fatigue life."""
    return C / (np.abs(Y) + 1e-3) ** m


# ============================================================================
# SPATIAL / STATISTICAL SUMMARY TRANSFORMS (nonlocal)
# ============================================================================


def t_sample_variance(Y):
    """Sample variance of outputs across pixels/time: phi_i = Var_t[Y_i]."""
    flat = Y.reshape(len(Y), -1)
    vv = flat.var(axis=1)
    return _bc(vv, Y) * np.ones_like(Y)


def t_sample_skewness(Y):
    """Sample skewness: phi_i = E[(Y-mu)^3]/sigma^3 -- third standardised moment."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sg = flat.std(axis=1, keepdims=True) + 1e-10
    skew = ((flat - mu) ** 3).mean(axis=1) / sg.squeeze() ** 3
    return _bc(skew, Y) * np.ones_like(Y)


def t_sample_kurtosis(Y):
    """Excess kurtosis: phi_i = E[(Y-mu)^4]/sigma^4 - 3 -- tail heaviness."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sg = flat.std(axis=1, keepdims=True) + 1e-10
    kurt = ((flat - mu) ** 4).mean(axis=1) / sg.squeeze() ** 4 - 3.0
    return _bc(kurt, Y) * np.ones_like(Y)


def t_percentile_q10(Y):
    """10th percentile nonlocal: phi_i = Q10(Y_i, :)."""
    flat = Y.reshape(len(Y), -1)
    q = np.percentile(flat, 10, axis=1)
    return _bc(q, Y) * np.ones_like(Y)


def t_percentile_q90(Y):
    """90th percentile nonlocal: phi_i = Q90(Y_i, :)."""
    flat = Y.reshape(len(Y), -1)
    q = np.percentile(flat, 90, axis=1)
    return _bc(q, Y) * np.ones_like(Y)


def t_interquartile_range(Y):
    """IQR: phi_i = Q75(Y_i) - Q25(Y_i) -- spread measure, nonlocal."""
    flat = Y.reshape(len(Y), -1)
    iqr = np.percentile(flat, 75, axis=1) - np.percentile(flat, 25, axis=1)
    return _bc(iqr, Y) * np.ones_like(Y)


# ============================================================================
# INFORMATION-THEORETIC TRANSFORMS
# ============================================================================


def t_negentropy_proxy(Y):
    """Negentropy proxy: phi = (mean^2 + 0.25*(kurt-3)^2)/16 -- ICA contrast fn."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sg = flat.std(axis=1, keepdims=True) + 1e-10
    u = (flat - mu) / sg
    kurt = (u**4).mean(axis=1) - 3.0
    mu_u = u.mean(axis=1)
    neg = (mu_u**2 + 0.25 * kurt**2) / 16.0
    return _bc(neg, Y) * np.ones_like(Y)


def t_wasserstein_proxy(Y):
    """Wasserstein-1 proxy: phi_i = mean |Y_i - Y_median_i| -- mean absolute deviation."""
    flat = Y.reshape(len(Y), -1)
    med = np.median(flat, axis=1, keepdims=True)
    mad = np.abs(flat - med).mean(axis=1)
    return _bc(mad, Y) * np.ones_like(Y)


def t_energy_distance_proxy(Y):
    """Energy distance proxy: phi_i = sqrt(2*E[|X-Y|] - E[|X-X'|]) approximation."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sigma = flat.std(axis=1, keepdims=True) + 1e-10
    u = (flat - mu) / sigma
    # E-statistic: 2*E[|u|] - sqrt(2)*sqrt(pi)*std -- simplified proxy
    ed = np.abs(u).mean(axis=1)
    return _bc(ed, Y) * np.ones_like(Y)


def t_entropy_renyi(Y, alpha=2.0, bins=20):
    """Renyi entropy (alpha=2) proxy via histogram: phi_i = -log(sum(p_k^alpha))."""
    flat = Y.reshape(len(Y), -1)
    out = np.empty(len(Y))
    for i in range(len(Y)):
        counts, _ = np.histogram(flat[i], bins=bins)
        p = counts / (counts.sum() + 1e-12)
        p = p[p > 0]
        out[i] = (
            -(1.0 / (1.0 - alpha)) * np.log(np.sum(p**alpha) + 1e-30)
            if abs(1.0 - alpha) > 1e-10
            else -np.sum(p * np.log(p + 1e-30))
        )
    return _bc(out, Y) * np.ones_like(Y)


TRANSFORMS = {
    # ── Environmental ─────────────────────────────────────────────────────────
    "log_shift": {
        "name": "Log-shift",
        "fn": t_log_shift,
        "params": {"eps": 1.0},
        "category": "environmental",
        "reference": "Vogel & Wilson (1996)",
    },
    "power_law_beta2": {
        "name": "Power-law (β=2, depth-damage)",
        "fn": t_power_law,
        "params": {"beta": 2.0},
        "category": "environmental",
        "reference": "Merz et al. (2010)",
    },
    "power_law_beta05": {
        "name": "Power-law (β=0.5, square-root)",
        "fn": t_power_law,
        "params": {"beta": 0.5},
        "category": "environmental",
        "reference": "Merz et al. (2010)",
    },
    "box_cox_sqrt": {
        "name": "Box-Cox (λ=0.5)",
        "fn": t_box_cox,
        "params": {"lam": 0.5},
        "category": "environmental",
        "reference": "Box & Cox (1964)",
    },
    "box_cox_log": {
        "name": "Box-Cox (λ→0, log)",
        "fn": t_box_cox,
        "params": {"lam": 1e-8},
        "category": "environmental",
        "reference": "Box & Cox (1964)",
    },
    "clipped_excess_q90": {
        "name": "Clipped excess POT (q=0.90)",
        "fn": t_clipped_excess,
        "params": {"quantile": 0.90},
        "category": "environmental",
        "reference": "Coles (2001)",
    },
    "exceedance_q75": {
        "name": "Exceedance indicator (q=0.75)",
        "fn": t_exceed_q75,
        "params": {},
        "category": "environmental",
        "reference": "Saltelli et al. (2008)",
    },
    "exceedance_q90": {
        "name": "Exceedance indicator (q=0.90)",
        "fn": t_exceed_q90,
        "params": {},
        "category": "environmental",
        "reference": "Saltelli et al. (2008)",
    },
    "exceedance_q95": {
        "name": "Exceedance indicator (q=0.95)",
        "fn": t_exceed_q95,
        "params": {},
        "category": "environmental",
        "reference": "Coles (2001)",
    },
    "exceedance_q99": {
        "name": "Exceedance indicator (q=0.99)",
        "fn": t_exceed_q99,
        "params": {},
        "category": "environmental",
        "reference": "Coles (2001)",
    },
    # ── Affine (commutative reference / negative control) ────────────────────
    "affine_a2_b1": {
        "name": "Affine (a=2, b=1) — commutative reference",
        "fn": t_affine,
        "params": {"a": 2.0, "b": 1.0},
        "category": "engineering",
        "reference": "Saltelli et al. (2008)",
    },
    "affine_a05_bm3": {
        "name": "Affine (a=0.5, b=−3) — unit conversion",
        "fn": t_affine,
        "params": {"a": 0.5, "b": -3.0},
        "category": "engineering",
        "reference": "Saltelli et al. (2008)",
    },
    # ── Pointwise saturation (genuinely pointwise: no _ymin/_safe_range) ─────
    "tanh_a03": {
        "name": "Tanh saturation (α=0.3)",
        "fn": t_tanh_pointwise,
        "params": {"alpha": 0.3},
        "category": "engineering",
        "reference": "Amari (1977); Wilson & Cowan (1972)",
    },
    "tanh_a10": {
        "name": "Tanh saturation (α=1.0)",
        "fn": t_tanh_pointwise,
        "params": {"alpha": 1.0},
        "category": "engineering",
        "reference": "Amari (1977)",
    },
    "tanh_a005": {
        "name": "Tanh saturation (α=0.05, near-linear)",
        "fn": t_tanh_pointwise,
        "params": {"alpha": 0.05},
        "category": "engineering",
        "reference": "Amari (1977)",
    },
    "softplus_b01": {
        "name": "Softplus (β=0.1, smooth ReLU)",
        "fn": t_softplus_pointwise,
        "params": {"beta": 0.1},
        "category": "engineering",
        "reference": "Glorot et al. (2011)",
    },
    "softplus_b10": {
        "name": "Softplus (β=1.0)",
        "fn": t_softplus_pointwise,
        "params": {"beta": 1.0},
        "category": "engineering",
        "reference": "Glorot et al. (2011)",
    },
    # ── Engineering ───────────────────────────────────────────────────────────
    "carnot_quadratic": {
        "name": "Carnot quadratic efficiency",
        "fn": t_carnot_quadratic,
        "params": {"delta": 1.0},
        "category": "engineering",
        "reference": "Williams et al. (2008)",
    },
    "sigmoid_dose": {
        "name": "Hill dose-response (n=4)",
        "fn": t_sigmoid_dose,
        "params": {"EC50_q": 0.5, "n_hill": 4.0},
        "category": "engineering",
        "reference": "Hill (1910)",
    },
    "arrhenius": {
        "name": "Arrhenius reaction rate",
        "fn": t_arrhenius,
        "params": {"Ea_over_R": 2.0},
        "category": "engineering",
        "reference": "Saltelli & Tarantola (2002)",
    },
    "normalised_stress": {
        "name": "Goodman normalised stress",
        "fn": t_normalised_stress,
        "params": {"yield_q": 0.80},
        "category": "engineering",
        "reference": "Socie & Marquis (2000)",
    },
    "weibull_reliability": {
        "name": "Weibull failure probability",
        "fn": t_weibull_reliability,
        "params": {"shape": 2.0},
        "category": "engineering",
        "reference": "Haldar & Mahadevan (2000)",
    },
    # ── Spatial ───────────────────────────────────────────────────────────────
    "regional_mean": {
        "name": "Regional mean",
        "fn": t_regional_mean,
        "params": {},
        "category": "spatial",
        "reference": "Gotway & Young (2002)",
    },
    "block_2x2": {
        "name": "Block average 2×2 (COS)",
        "fn": t_block_2x2,
        "params": {},
        "category": "spatial",
        "reference": "Gotway & Young (2002)",
    },
    "block_4x4": {
        "name": "Block average 4×4 (COS)",
        "fn": t_block_4x4,
        "params": {},
        "category": "spatial",
        "reference": "Gotway & Young (2002)",
    },
    "block_8x8": {
        "name": "Block average 8×8 (COS)",
        "fn": t_block_8x8,
        "params": {},
        "category": "spatial",
        "reference": "Cressie (1993)",
    },
    "exceedance_area": {
        "name": "Exceedance area fraction (q=0.75)",
        "fn": t_exceedance_area,
        "params": {"quantile": 0.75},
        "category": "spatial",
        "reference": "Beven & Binley (1992)",
    },
    "matern_smooth": {
        "name": "Matern-1.5 spatial smoothing",
        "fn": t_matern_smooth,
        "params": {"length_scale": 0.15, "nu": 1.5},
        "category": "spatial",
        "reference": "Stein (1999)",
    },
    "laplacian_roughness": {
        "name": "Laplacian roughness |∇²Y|",
        "fn": t_laplacian_roughness,
        "params": {},
        "category": "spatial",
        "reference": "Cressie (1993)",
    },
    "gradient_magnitude": {
        "name": "Gradient magnitude |∇Y|",
        "fn": t_gradient_magnitude,
        "params": {},
        "category": "spatial",
        "reference": "Cressie (1993)",
    },
    "contour_exceedance": {
        "name": "Signed contour distance (q=0.75)",
        "fn": t_contour_exceedance,
        "params": {"quantile": 0.75},
        "category": "spatial",
        "reference": "Openshaw (1984)",
    },
    "isoline_length": {
        "name": "Isoline perimeter (q=0.75)",
        "fn": t_isoline_length,
        "params": {"quantile": 0.75},
        "category": "spatial",
        "reference": "Openshaw (1984)",
    },
    # ── Temporal ──────────────────────────────────────────────────────────────
    "temporal_peak": {
        "name": "Temporal peak (max_t)",
        "fn": t_temporal_peak,
        "params": {},
        "category": "temporal",
        "reference": "Sudret (2008)",
    },
    "temporal_cumsum": {
        "name": "Cumulative sum (linear)",
        "fn": t_temporal_cumsum,
        "params": {},
        "category": "temporal",
        "reference": "Saltelli et al. (2005)",
    },
    "temporal_log_cumsum": {
        "name": "Log cumulative sum",
        "fn": t_temporal_log_cumsum,
        "params": {"eps": 1.0},
        "category": "temporal",
        "reference": "Saltelli et al. (2005)",
    },
    "temporal_exceedance_duration": {
        "name": "Exceedance duration (q=0.75)",
        "fn": t_temporal_exceedance_duration,
        "params": {"quantile": 0.75},
        "category": "temporal",
        "reference": "Coles (2001)",
    },
    "temporal_envelope": {
        "name": "Running maximum envelope",
        "fn": t_temporal_envelope,
        "params": {},
        "category": "temporal",
        "reference": "Sudret (2008)",
    },
    "temporal_bandpass": {
        "name": "Bandpass filter (linear)",
        "fn": t_temporal_bandpass,
        "params": {"low_frac": 0.05, "high_frac": 0.30},
        "category": "temporal",
        "reference": "Rabiner & Gold (1975)",
    },
    "temporal_block_avg": {
        "name": "Temporal block avg (temporal COS)",
        "fn": t_temporal_block_avg,
        "params": {"block": 10},
        "category": "temporal",
        "reference": "Gotway & Young (2002)",
    },
    # ── Statistical ───────────────────────────────────────────────────────────
    "rank_transform": {
        "name": "Rank / PIT transform",
        "fn": t_rank_transform,
        "params": {},
        "category": "statistical",
        "reference": "Saltelli et al. (2008)",
    },
    "standardised_anomaly": {
        "name": "Standardised anomaly (Z-score)",
        "fn": t_standardised_anomaly,
        "params": {},
        "category": "statistical",
        "reference": "Saltelli et al. (2008)",
    },
    "entropy_proxy": {
        "name": "Entropy proxy −(Y−mean)²",
        "fn": t_entropy_proxy,
        "params": {},
        "category": "statistical",
        "reference": "Shannon (1948)",
    },
    # ── Mathematical: Convex pointwise ───────────────────────────────────────
    "square_pointwise": {
        "name": "Square (convex, even)",
        "fn": t_square_pointwise,
        "params": {},
        "category": "mathematical",
        "reference": "Hardy et al. (1934)",
    },
    "exp_pointwise": {
        "name": "Exponential (convex, monotone)",
        "fn": t_exp_pointwise,
        "params": {"scale": 0.1},
        "category": "mathematical",
        "reference": "Hardy et al. (1934)",
    },
    "cube_pointwise": {
        "name": "Cube (odd, convex x>0 / concave x<0)",
        "fn": t_cube_pointwise,
        "params": {},
        "category": "mathematical",
        "reference": "Borwein & Lewis (2000)",
    },
    "cosh_pointwise": {
        "name": "Cosh (convex, even, C∞)",
        "fn": t_cosh_pointwise,
        "params": {"scale": 0.1},
        "category": "mathematical",
        "reference": "Abramowitz & Stegun (1972)",
    },
    "relu_pointwise": {
        "name": "ReLU max(y,0) (convex, non-smooth)",
        "fn": t_relu_pointwise,
        "params": {},
        "category": "mathematical",
        "reference": "Glorot et al. (2011)",
    },
    "softmax_shift": {
        "name": "Shifted softmax normalisation",
        "fn": t_softmax_shift,
        "params": {},
        "category": "mathematical",
        "reference": "Goodfellow et al. (2016)",
    },
    "log1p_positive": {
        "name": "log(1+|y|) (concave for y>0)",
        "fn": t_log1p_abs,
        "params": {},
        "category": "mathematical",
        "reference": "Box & Cox (1964)",
    },
    # ── Mathematical: Concave pointwise ──────────────────────────────────────
    "sqrt_abs": {
        "name": "√|y| (concave, non-smooth at 0)",
        "fn": t_sqrt_abs,
        "params": {},
        "category": "mathematical",
        "reference": "Hardy et al. (1934)",
    },
    "cbrt_pointwise": {
        "name": "Cube root (monotone, concave x>0)",
        "fn": t_cbrt_pointwise,
        "params": {},
        "category": "mathematical",
        "reference": "Borwein & Lewis (2000)",
    },
    "neg_square": {
        "name": "−y² (concave, even)",
        "fn": t_neg_square,
        "params": {},
        "category": "mathematical",
        "reference": "Hardy et al. (1934)",
    },
    # ── Mathematical: Monotone non-convex ────────────────────────────────────
    "logistic_pointwise": {
        "name": "Logistic σ(y) (monotone, S-shaped)",
        "fn": t_logistic_pointwise,
        "params": {"k": 1.0},
        "category": "mathematical",
        "reference": "Verhulst (1838)",
    },
    "arctan_pointwise": {
        "name": "arctan(y) (monotone, bounded)",
        "fn": t_arctan_pointwise,
        "params": {"scale": 1.0},
        "category": "mathematical",
        "reference": "Abramowitz & Stegun (1972)",
    },
    "erf_pointwise": {
        "name": "erf(y) (monotone, S-shaped, C∞)",
        "fn": t_erf_pointwise,
        "params": {"scale": 0.5},
        "category": "mathematical",
        "reference": "Abramowitz & Stegun (1972)",
    },
    "sinh_pointwise": {
        "name": "sinh(y) (monotone, convex x>0)",
        "fn": t_sinh_pointwise,
        "params": {"scale": 0.1},
        "category": "mathematical",
        "reference": "Abramowitz & Stegun (1972)",
    },
    # ── Mathematical: Non-monotone / oscillatory ─────────────────────────────
    "sin_pointwise": {
        "name": "sin(y) (periodic, bounded, C∞)",
        "fn": t_sin_pointwise,
        "params": {"freq": 0.5},
        "category": "mathematical",
        "reference": "Saltelli et al. (2010)",
    },
    "cos_pointwise": {
        "name": "cos(y) (periodic, even, bounded)",
        "fn": t_cos_pointwise,
        "params": {"freq": 0.5},
        "category": "mathematical",
        "reference": "Saltelli et al. (2010)",
    },
    "abs_pointwise": {
        "name": "|y| (convex, non-smooth at 0)",
        "fn": t_abs_pointwise,
        "params": {},
        "category": "mathematical",
        "reference": "Hardy et al. (1934)",
    },
    "step_pointwise": {
        "name": "Heaviside step (discontinuous)",
        "fn": t_step_pointwise,
        "params": {"threshold": 0.0},
        "category": "mathematical",
        "reference": "Saltelli et al. (2010)",
    },
    "triangle_wave": {
        "name": "Triangle wave (periodic, non-smooth)",
        "fn": t_triangle_wave,
        "params": {"period": 4.0},
        "category": "mathematical",
        "reference": "Stein (1999)",
    },
    # ── Mathematical: Higher-order derivative structure ───────────────────────
    "smooth_bump": {
        "name": "Smooth bump (C∞, compact support)",
        "fn": t_smooth_bump,
        "params": {"width": 3.0},
        "category": "mathematical",
        "reference": "Wendland (2004)",
    },
    "rational_quadratic": {
        "name": "Rational quadratic 1/(1+y²) (convex)",
        "fn": t_rational_quadratic,
        "params": {},
        "category": "mathematical",
        "reference": "Rasmussen & Williams (2006)",
    },
    "inverse_pointwise": {
        "name": "1/|y| (convex, singular at 0)",
        "fn": t_inverse_abs,
        "params": {"eps": 1.0},
        "category": "mathematical",
        "reference": "Hardy et al. (1934)",
    },
    "log_abs_pointwise": {
        "name": "log|y| (concave for |y|>1, singular)",
        "fn": t_log_abs,
        "params": {"eps": 1.0},
        "category": "mathematical",
        "reference": "Box & Cox (1964)",
    },
    # ── Mathematical: Normalisation / standardisation ─────────────────────────
    "min_max_normalise": {
        "name": "Min-max normalisation [0,1]",
        "fn": t_min_max_normalise,
        "params": {},
        "category": "mathematical",
        "reference": "Saltelli et al. (2008)",
    },
    "robust_scale": {
        "name": "Robust scaling (IQR-based)",
        "fn": t_robust_scale,
        "params": {},
        "category": "mathematical",
        "reference": "Huber (1981)",
    },
    "clamp_sigma": {
        "name": "Clamp to ±2σ (soft winsorisation)",
        "fn": t_clamp_sigma,
        "params": {"n_sigma": 2.0},
        "category": "mathematical",
        "reference": "Tukey (1977)",
    },
    # ── Environmental additional ──────────────────────────────────────────────
    "gumbel_cdf": {
        "name": "Gumbel CDF (EV type-I)",
        "fn": t_gumbel_cdf,
        "params": {},
        "category": "environmental",
        "reference": "Gumbel (1958)",
    },
    "frechet_cdf": {
        "name": "Fréchet CDF (EV type-II, shape=2)",
        "fn": t_frechet_cdf,
        "params": {"shape": 2.0},
        "category": "environmental",
        "reference": "Fréchet (1927)",
    },
    "log_normal_cdf": {
        "name": "Log-normal CDF (σ=0.5)",
        "fn": t_log_normal_cdf,
        "params": {"sigma": 0.5},
        "category": "environmental",
        "reference": "Limpert et al. (2001)",
    },
    "return_period": {
        "name": "Return period T=1/(1-CDF)",
        "fn": t_return_period,
        "params": {},
        "category": "environmental",
        "reference": "Coles (2001)",
    },
    # ── Engineering additional ────────────────────────────────────────────────
    "johnson_su": {
        "name": "Johnson SU (4-parameter normalisation)",
        "fn": t_johnson_su,
        "params": {},
        "category": "engineering",
        "reference": "Johnson (1949)",
    },
    "fatigue_miner": {
        "name": "Palmgren-Miner fatigue accumulation",
        "fn": t_fatigue_miner,
        "params": {"m": 3.0},
        "category": "engineering",
        "reference": "Miner (1945)",
    },
    "rankine_failure": {
        "name": "Rankine failure criterion (max principal)",
        "fn": t_rankine_failure,
        "params": {},
        "category": "engineering",
        "reference": "Timoshenko (1951)",
    },
    # ── Temporal additional ───────────────────────────────────────────────────
    "temporal_rms": {
        "name": "Temporal RMS (root mean square)",
        "fn": t_temporal_rms,
        "params": {},
        "category": "temporal",
        "reference": "Sudret (2008)",
    },
    "temporal_range": {
        "name": "Temporal range max−min",
        "fn": t_temporal_range,
        "params": {},
        "category": "temporal",
        "reference": "Sudret (2008)",
    },
    "temporal_autocorr": {
        "name": "Lag-1 temporal autocorrelation",
        "fn": t_temporal_autocorr,
        "params": {},
        "category": "temporal",
        "reference": "Box et al. (2015)",
    },
    "temporal_quantile_q50": {
        "name": "Temporal median (q=0.50)",
        "fn": t_temporal_quantile,
        "params": {"q": 0.50},
        "category": "temporal",
        "reference": "Saltelli et al. (2008)",
    },
    "temporal_quantile_q10": {
        "name": "Temporal lower decile (q=0.10)",
        "fn": t_temporal_quantile,
        "params": {"q": 0.10},
        "category": "temporal",
        "reference": "Saltelli et al. (2008)",
    },
    "temporal_quantile_q90": {
        "name": "Temporal upper decile (q=0.90)",
        "fn": t_temporal_quantile,
        "params": {"q": 0.90},
        "category": "temporal",
        "reference": "Saltelli et al. (2008)",
    },
    # ── Statistical additional ────────────────────────────────────────────────
    "quantile_transform": {
        "name": "Quantile normalise (empirical CDF)",
        "fn": t_quantile_normalise,
        "params": {},
        "category": "statistical",
        "reference": "Blom (1958)",
    },
    "winsorise_q10_q90": {
        "name": "Winsorise to [q10, q90]",
        "fn": t_winsorise,
        "params": {"low": 0.10, "high": 0.90},
        "category": "statistical",
        "reference": "Tukey (1977)",
    },
    "yeo_johnson": {
        "name": "Yeo-Johnson transform (λ=0.5)",
        "fn": t_yeo_johnson,
        "params": {"lam": 0.5},
        "category": "statistical",
        "reference": "Yeo & Johnson (2000)",
    },
    "inverse_normal": {
        "name": "Inverse normal score (probit)",
        "fn": t_inverse_normal,
        "params": {},
        "category": "statistical",
        "reference": "Blom (1958)",
    },
    # ── Polynomial family ────────────────────────────────────────────────────
    "poly4": {
        "name": "y^4 (quartic, even, convex)",
        "fn": t_poly4,
        "params": {"scale": 0.05},
        "category": "mathematical",
        "reference": "Hardy et al. (1934)",
    },
    "poly5": {
        "name": "y^5 (quintic, odd, C-inf)",
        "fn": t_poly5,
        "params": {"scale": 0.05},
        "category": "mathematical",
        "reference": "Hardy et al. (1934)",
    },
    "poly6": {
        "name": "y^6 (sextic, even, strictly convex)",
        "fn": t_poly6,
        "params": {"scale": 0.05},
        "category": "mathematical",
        "reference": "Hardy et al. (1934)",
    },
    "signed_power_p15": {
        "name": "Signed power p=1.5 (monotone, C1)",
        "fn": t_signed_power,
        "params": {"p": 1.5, "scale": 0.2},
        "category": "mathematical",
        "reference": "Sklar (1959)",
    },
    "signed_power_p05": {
        "name": "Signed power p=0.5 (monotone, C1)",
        "fn": t_signed_power,
        "params": {"p": 0.5, "scale": 0.2},
        "category": "mathematical",
        "reference": "Sklar (1959)",
    },
    "legendre_p3": {
        "name": "Legendre P3 polynomial",
        "fn": t_legendre_p3,
        "params": {"scale": 0.3},
        "category": "mathematical",
        "reference": "Szego (1939)",
    },
    "chebyshev_t4": {
        "name": "Chebyshev T4 polynomial",
        "fn": t_chebyshev_t4,
        "params": {"scale": 0.2},
        "category": "mathematical",
        "reference": "Szego (1939)",
    },
    "hermite_he2": {
        "name": "Hermite He2 polynomial",
        "fn": t_hermite_he2,
        "params": {"scale": 0.3},
        "category": "mathematical",
        "reference": "Szego (1939)",
    },
    "hermite_he3": {
        "name": "Hermite He3 polynomial",
        "fn": t_hermite_he3,
        "params": {"scale": 0.3},
        "category": "mathematical",
        "reference": "Szego (1939)",
    },
    "bernstein_b3": {
        "name": "Bernstein B3 basis polynomial",
        "fn": t_bernstein_b3,
        "params": {},
        "category": "mathematical",
        "reference": "Bernstein (1912)",
    },
    # ── Sigmoid/activation family ─────────────────────────────────────────
    "atan2pi": {
        "name": "(2/pi)*arctan (bounded monotone)",
        "fn": t_atan2pi,
        "params": {"scale": 1.0},
        "category": "mathematical",
        "reference": "Abramowitz & Stegun (1972)",
    },
    "gompertz_cdf": {
        "name": "Gompertz CDF (asymmetric sigmoid)",
        "fn": t_gompertz,
        "params": {"b": 1.0, "c": 0.5},
        "category": "mathematical",
        "reference": "Gompertz (1825)",
    },
    "algebraic_sigmoid": {
        "name": "Algebraic sigmoid y/sqrt(1+y^2)",
        "fn": t_algebraic_sigmoid,
        "params": {"scale": 0.5},
        "category": "mathematical",
        "reference": "Glorot et al. (2011)",
    },
    "swish": {
        "name": "Swish activation (non-monotone)",
        "fn": t_swish,
        "params": {"beta": 1.0},
        "category": "mathematical",
        "reference": "Ramachandran et al. (2017)",
    },
    "mish": {
        "name": "Mish activation (smooth non-monotone)",
        "fn": t_mish,
        "params": {},
        "category": "mathematical",
        "reference": "Misra (2020)",
    },
    "selu": {
        "name": "SELU activation (C1, piecewise)",
        "fn": t_selu,
        "params": {"alpha": 1.6733, "lam": 1.0507},
        "category": "mathematical",
        "reference": "Klambauer et al. (2017)",
    },
    "softsign": {
        "name": "Softsign y/(1+|y|) (monotone, C1)",
        "fn": t_softsign,
        "params": {"scale": 1.0},
        "category": "mathematical",
        "reference": "Glorot & Bengio (2010)",
    },
    "bent_identity": {
        "name": "Bent identity (monotone, near-linear)",
        "fn": t_bent_identity,
        "params": {"scale": 0.5},
        "category": "mathematical",
        "reference": "Glorot et al. (2011)",
    },
    "hard_sigmoid": {
        "name": "Hard sigmoid (piecewise linear, C0)",
        "fn": t_hard_sigmoid,
        "params": {"scale": 0.5},
        "category": "mathematical",
        "reference": "Goodfellow et al. (2016)",
    },
    "hard_tanh": {
        "name": "Hard tanh clip(-1,1) (piecewise, C0)",
        "fn": t_hard_tanh,
        "params": {"scale": 0.3},
        "category": "mathematical",
        "reference": "Goodfellow et al. (2016)",
    },
    # ── Oscillatory/periodic family ───────────────────────────────────────
    "sinc": {
        "name": "Normalised sinc (decaying oscillation)",
        "fn": t_sinc,
        "params": {"scale": 0.5},
        "category": "mathematical",
        "reference": "Whittaker (1915)",
    },
    "sin_squared": {
        "name": "sin^2(fy) (bounded, non-negative)",
        "fn": t_sin_squared,
        "params": {"freq": 0.5},
        "category": "mathematical",
        "reference": "Saltelli et al. (2010)",
    },
    "cos_squared": {
        "name": "cos^2(fy) (even, non-negative)",
        "fn": t_cos_squared,
        "params": {"freq": 0.5},
        "category": "mathematical",
        "reference": "Saltelli et al. (2010)",
    },
    "damped_sin": {
        "name": "Damped sine (decaying oscillation)",
        "fn": t_damped_sin,
        "params": {"freq": 0.5, "decay": 0.1},
        "category": "mathematical",
        "reference": "Stein (1999)",
    },
    "sawtooth": {
        "name": "Sawtooth wave (C0, periodic)",
        "fn": t_sawtooth,
        "params": {"period": 4.0},
        "category": "mathematical",
        "reference": "Oppenheim & Schafer (1989)",
    },
    "square_wave": {
        "name": "Square wave (discontinuous, periodic)",
        "fn": t_square_wave,
        "params": {"period": 4.0},
        "category": "mathematical",
        "reference": "Oppenheim & Schafer (1989)",
    },
    "double_sin": {
        "name": "Double-frequency sin+sin (interference)",
        "fn": t_double_sin,
        "params": {"freq1": 0.3, "freq2": 0.7},
        "category": "mathematical",
        "reference": "Saltelli et al. (2010)",
    },
    "sin_cos_product": {
        "name": "sin*cos = 0.5*sin(2f) (harmonic)",
        "fn": t_sin_cos_product,
        "params": {"freq": 0.5},
        "category": "mathematical",
        "reference": "Abramowitz & Stegun (1972)",
    },
    # ── Threshold/piecewise family ────────────────────────────────────────
    "soft_threshold": {
        "name": "Soft threshold (lasso/shrinkage, C0)",
        "fn": t_soft_threshold,
        "params": {"lam": 1.0},
        "category": "mathematical",
        "reference": "Donoho & Johnstone (1994)",
    },
    "hard_threshold": {
        "name": "Hard threshold (discontinuous)",
        "fn": t_hard_threshold,
        "params": {"lam": 1.0},
        "category": "mathematical",
        "reference": "Donoho & Johnstone (1994)",
    },
    "ramp": {
        "name": "Ramp/clip (piecewise linear, C0)",
        "fn": t_ramp,
        "params": {"lo": -1.0, "hi": 1.0},
        "category": "mathematical",
        "reference": "Saltelli et al. (2008)",
    },
    "spike_gaussian": {
        "name": "Gaussian spike (C-inf, compact-ish)",
        "fn": t_spike,
        "params": {"center": 0.0, "width": 1.0},
        "category": "mathematical",
        "reference": "Wendland (2004)",
    },
    "breakpoint": {
        "name": "Piecewise linear breakpoint (C0)",
        "fn": t_breakpoint,
        "params": {"bp": 0.0, "slope_lo": 0.5, "slope_hi": 2.0},
        "category": "mathematical",
        "reference": "Hansen (2000)",
    },
    "hockey_stick": {
        "name": "Hockey stick ReLU-shift (convex, C0)",
        "fn": t_hockey_stick,
        "params": {"bp": 0.0},
        "category": "mathematical",
        "reference": "Glorot et al. (2011)",
    },
    "deadzone": {
        "name": "Deadzone (zero band, C0)",
        "fn": t_deadzone,
        "params": {"half_width": 1.0},
        "category": "mathematical",
        "reference": "Donoho & Johnstone (1994)",
    },
    "bimodal_flip": {
        "name": "Bimodal flip (zero crossings at 0,0.5,1)",
        "fn": t_bimodal_flip,
        "params": {},
        "category": "mathematical",
        "reference": "Saltelli et al. (2010)",
    },
    "donut": {
        "name": "Donut/ring indicator (C-inf)",
        "fn": t_donut,
        "params": {"center": 0.0, "radius": 1.5, "width": 0.5},
        "category": "mathematical",
        "reference": "Wendland (2004)",
    },
    # ── Variance-stabilising family ───────────────────────────────────────
    "anscombe": {
        "name": "Anscombe VST (Poisson counts)",
        "fn": t_anscombe,
        "params": {},
        "category": "statistical",
        "reference": "Anscombe (1948)",
    },
    "freeman_tukey": {
        "name": "Freeman-Tukey VST (count data)",
        "fn": t_freeman_tukey,
        "params": {},
        "category": "statistical",
        "reference": "Freeman & Tukey (1950)",
    },
    "asinh_vst": {
        "name": "Inverse hyperbolic sine (VST, C-inf)",
        "fn": t_asinh_vs,
        "params": {"scale": 0.5},
        "category": "statistical",
        "reference": "Johnson (1949)",
    },
    "modulus_lam05": {
        "name": "Modulus transform lam=0.5 (C1 VST)",
        "fn": t_modulus,
        "params": {"lam": 0.5},
        "category": "statistical",
        "reference": "John & Draper (1980)",
    },
    "dual_power_lam03": {
        "name": "Dual power lam=0.3 (C1, all-real)",
        "fn": t_dual_power,
        "params": {"lam": 0.3},
        "category": "statistical",
        "reference": "Yeo & Johnson (2000)",
    },
    "log2_shift": {
        "name": "Log base-2 (bit/octave scaling)",
        "fn": t_log2_shift,
        "params": {"eps": 1.0},
        "category": "environmental",
        "reference": "Shannon (1948)",
    },
    "log10_shift": {
        "name": "Log base-10 (order-of-magnitude)",
        "fn": t_log10_shift,
        "params": {"eps": 1.0},
        "category": "environmental",
        "reference": "Richter (1935)",
    },
    # ── Curvature extremes/special ────────────────────────────────────────
    "exp_neg_sq": {
        "name": "Gaussian kernel exp(-s^2*y^2) (C-inf)",
        "fn": t_exp_neg_sq,
        "params": {"scale": 0.3},
        "category": "mathematical",
        "reference": "Rasmussen & Williams (2006)",
    },
    "exp_pos_sq": {
        "name": "Anti-Gaussian exp(+s^2*y^2) (convex)",
        "fn": t_exp_pos_sq,
        "params": {"scale": 0.2},
        "category": "mathematical",
        "reference": "Hardy et al. (1934)",
    },
    "inverse_sq": {
        "name": "1/(y^2+eps) (convex, even, regularised)",
        "fn": t_inverse_sq,
        "params": {"eps": 1.0},
        "category": "mathematical",
        "reference": "Hardy et al. (1934)",
    },
    "log_log": {
        "name": "Double log log(1+log(y)) (extreme compr)",
        "fn": t_log_log,
        "params": {"eps": 1.0},
        "category": "environmental",
        "reference": "Box & Cox (1964)",
    },
    "power_exp": {
        "name": "y^2*exp(-|y|) (hump, C-inf)",
        "fn": t_power_exp,
        "params": {"scale": 0.1},
        "category": "mathematical",
        "reference": "Stein (1999)",
    },
    "gev_cdf": {
        "name": "GEV CDF xi=0.3 (Frechet type)",
        "fn": t_gev_cdf,
        "params": {"xi": 0.3},
        "category": "environmental",
        "reference": "Jenkinson (1955)",
    },
    "pareto_tail": {
        "name": "Pareto tail transform alpha=1.5",
        "fn": t_pareto_tail,
        "params": {"alpha": 1.5},
        "category": "environmental",
        "reference": "Pickands (1975)",
    },
    "log_logistic_cdf": {
        "name": "Log-logistic CDF beta=2 (S-shaped tail)",
        "fn": t_log_logistic_cdf,
        "params": {"beta": 2.0},
        "category": "environmental",
        "reference": "Tadikamalla (1980)",
    },
    # ── Financial transforms ──────────────────────────────────────────────
    "var_q95": {
        "name": "VaR proxy q=0.95 (nonlocal quantile)",
        "fn": t_var_proxy,
        "params": {"q": 0.95},
        "category": "financial",
        "reference": "Artzner et al. (1999)",
    },
    "cvar_q95": {
        "name": "CVaR/ES q=0.95 (nonlocal tail mean)",
        "fn": t_cvar,
        "params": {"q": 0.95},
        "category": "financial",
        "reference": "Artzner et al. (1999)",
    },
    "sharpe_proxy": {
        "name": "Sharpe ratio proxy (mean/std)",
        "fn": t_sharpe_proxy,
        "params": {"rf": 0.0},
        "category": "financial",
        "reference": "Sharpe (1966)",
    },
    "drawdown": {
        "name": "Max drawdown (temporal, nonlocal)",
        "fn": t_drawdown,
        "params": {},
        "category": "financial",
        "reference": "Magdon-Ismail et al. (2004)",
    },
    "fold_change": {
        "name": "Log2 fold-change from sample mean",
        "fn": t_fold_change,
        "params": {"eps": 1.0},
        "category": "financial",
        "reference": "Pfaffl (2001)",
    },
    "excess_return": {
        "name": "Excess return (mean-centred)",
        "fn": t_excess_return,
        "params": {},
        "category": "financial",
        "reference": "Sharpe (1966)",
    },
    # ── Ecological transforms ─────────────────────────────────────────────
    "hellinger": {
        "name": "Hellinger transform (chi^2 equaliser)",
        "fn": t_hellinger,
        "params": {},
        "category": "ecological",
        "reference": "Legendre & Gallagher (2001)",
    },
    "chord_normalise": {
        "name": "Chord normalisation (L2 sphere proj)",
        "fn": t_chord_dist,
        "params": {},
        "category": "ecological",
        "reference": "Orloci (1967)",
    },
    "relative_abundance": {
        "name": "Relative abundance (simplex projection)",
        "fn": t_relative_abundance,
        "params": {},
        "category": "ecological",
        "reference": "Legendre & Legendre (1998)",
    },
    "log_ratio": {
        "name": "Log-ratio (ILR-like, mean-centred)",
        "fn": t_log_ratio,
        "params": {"eps": 1.0},
        "category": "ecological",
        "reference": "Aitchison (1986)",
    },
    # ── Climate transforms ────────────────────────────────────────────────
    "anomaly_pct": {
        "name": "Anomaly percent departure from mean",
        "fn": t_anomaly_pct,
        "params": {"eps": 1.0},
        "category": "climate",
        "reference": "Jones et al. (1999)",
    },
    "bias_correction": {
        "name": "Bias correction (linear scaling to 1)",
        "fn": t_bias_correction,
        "params": {},
        "category": "climate",
        "reference": "Piani et al. (2010)",
    },
    "quantile_delta": {
        "name": "Quantile delta mapping q=0.90",
        "fn": t_quantile_delta,
        "params": {"q": 0.90},
        "category": "climate",
        "reference": "Cannon et al. (2015)",
    },
    "growing_degree_days": {
        "name": "Growing degree days base=10",
        "fn": t_growing_degree_days,
        "params": {"base": 10.0},
        "category": "climate",
        "reference": "McMaster & Wilhelm (1997)",
    },
    "std_precip_idx": {
        "name": "Standardised precip index (Z-score)",
        "fn": t_standardised_precip_idx,
        "params": {},
        "category": "climate",
        "reference": "McKee et al. (1993)",
    },
    # ── Hydrology transforms ──────────────────────────────────────────────
    "nash_sutcliffe": {
        "name": "Nash-Sutcliffe efficiency proxy",
        "fn": t_nash_sutcliffe,
        "params": {},
        "category": "hydrology",
        "reference": "Nash & Sutcliffe (1970)",
    },
    "pot_log": {
        "name": "Log peaks-over-threshold q=0.90",
        "fn": t_pot_log,
        "params": {"q": 0.90, "eps": 1.0},
        "category": "hydrology",
        "reference": "Coles (2001)",
    },
    "log_flow": {
        "name": "Log streamflow (hydrological VST)",
        "fn": t_log_flow,
        "params": {"eps": 0.01},
        "category": "hydrology",
        "reference": "Vogel & Wilson (1996)",
    },
    # ── Medical/pharmacological transforms ───────────────────────────────
    "hill_response": {
        "name": "Hill equation response n=2 (saturation)",
        "fn": t_hill_response,
        "params": {"n": 2.0, "EC50_q": 0.5},
        "category": "medical",
        "reference": "Hill (1910)",
    },
    "log_auc": {
        "name": "Log area-under-curve proxy",
        "fn": t_log_auc,
        "params": {"eps": 1.0},
        "category": "medical",
        "reference": "Gabrielsson & Weiner (2000)",
    },
    "emax_model": {
        "name": "Emax pharmacodynamic model n=1",
        "fn": t_emax_model,
        "params": {"Emax": 1.0, "ED50_q": 0.5, "n": 1.0},
        "category": "medical",
        "reference": "Holford & Sheiner (1981)",
    },
    # ── Structural engineering transforms ─────────────────────────────────
    "von_mises_stress": {
        "name": "Von Mises stress proxy (scaled |y|)",
        "fn": t_von_mises,
        "params": {},
        "category": "engineering",
        "reference": "von Mises (1913)",
    },
    "safety_factor": {
        "name": "Safety factor (inverse magnitude)",
        "fn": t_safety_factor,
        "params": {"capacity": 1.0},
        "category": "engineering",
        "reference": "Haldar & Mahadevan (2000)",
    },
    "cumulative_damage": {
        "name": "Cumulative damage |y|^m m=3 (Miner)",
        "fn": t_cumulative_damage,
        "params": {"m": 3.0},
        "category": "engineering",
        "reference": "Miner (1945)",
    },
    "stress_life": {
        "name": "S-N Basquin curve (fatigue life)",
        "fn": t_stress_life,
        "params": {"C": 1e6, "m": 3.0},
        "category": "engineering",
        "reference": "Basquin (1910)",
    },
    # ── Spatial/statistical summary transforms ────────────────────────────
    "sample_variance": {
        "name": "Sample variance (nonlocal spread)",
        "fn": t_sample_variance,
        "params": {},
        "category": "statistical",
        "reference": "Fisher (1925)",
    },
    "sample_skewness": {
        "name": "Sample skewness (3rd moment)",
        "fn": t_sample_skewness,
        "params": {},
        "category": "statistical",
        "reference": "Fisher (1925)",
    },
    "sample_kurtosis": {
        "name": "Excess kurtosis (4th moment - 3)",
        "fn": t_sample_kurtosis,
        "params": {},
        "category": "statistical",
        "reference": "Fisher (1925)",
    },
    "percentile_q10": {
        "name": "10th percentile (nonlocal quantile)",
        "fn": t_percentile_q10,
        "params": {},
        "category": "statistical",
        "reference": "Hyndman & Fan (1996)",
    },
    "percentile_q90": {
        "name": "90th percentile (nonlocal quantile)",
        "fn": t_percentile_q90,
        "params": {},
        "category": "statistical",
        "reference": "Hyndman & Fan (1996)",
    },
    "iqr": {
        "name": "Interquartile range (nonlocal spread)",
        "fn": t_interquartile_range,
        "params": {},
        "category": "statistical",
        "reference": "Tukey (1977)",
    },
    # ── Information-theoretic transforms ──────────────────────────────────
    "negentropy_proxy": {
        "name": "Negentropy proxy (ICA contrast function)",
        "fn": t_negentropy_proxy,
        "params": {},
        "category": "information",
        "reference": "Hyvarinen & Oja (2000)",
    },
    "wasserstein_proxy": {
        "name": "Wasserstein-1 proxy (MAD)",
        "fn": t_wasserstein_proxy,
        "params": {},
        "category": "information",
        "reference": "Villani (2009)",
    },
    "energy_distance": {
        "name": "Energy distance proxy (E-statistic)",
        "fn": t_energy_distance_proxy,
        "params": {},
        "category": "information",
        "reference": "Szekely & Rizzo (2013)",
    },
    "renyi_entropy_a2": {
        "name": "Renyi entropy alpha=2 (collision entropy)",
        "fn": t_entropy_renyi,
        "params": {"alpha": 2.0, "bins": 20},
        "category": "information",
        "reference": "Renyi (1961)",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Set-theoretic property classification of all registered transforms
# Used for regression analysis against categorical metadata
# ─────────────────────────────────────────────────────────────────────────────

# ── Linear-transform marker (for paper Section 3.3 negative controls) ─────────
LINEAR_TRANSFORMS = {"temporal_cumsum", "temporal_bandpass"}

# ── Genuinely pointwise transforms (φ(y) depends only on scalar y) ─────────────
POINTWISE_TRANSFORMS = {
    "affine_a2_b1",
    "affine_a05_bm3",
    "tanh_a03",
    "tanh_a10",
    "tanh_a005",
    "softplus_b01",
    "softplus_b10",
    # Mathematical transforms — all genuinely pointwise
    "square_pointwise",
    "exp_pointwise",
    "cube_pointwise",
    "cosh_pointwise",
    "relu_pointwise",
    "log1p_positive",
    "sqrt_abs",
    "cbrt_pointwise",
    "neg_square",
    "logistic_pointwise",
    "arctan_pointwise",
    "erf_pointwise",
    "sinh_pointwise",
    "sin_pointwise",
    "cos_pointwise",
    "abs_pointwise",
    "step_pointwise",
    "triangle_wave",
    "smooth_bump",
    "rational_quadratic",
    "inverse_pointwise",
    "log_abs_pointwise",
    "softmax_shift",
}

# ── Affine-transform marker (commutative subset of pointwise) ───────────────────
AFFINE_TRANSFORMS = {"affine_a2_b1", "affine_a05_bm3"}

# ── Nonlocal-transform marker (use per-sample statistics) ───────────────────────
NONLOCAL_TRANSFORMS = set(TRANSFORMS.keys()) - POINTWISE_TRANSFORMS

# ── Convex transforms (φ''(y) ≥ 0 globally, or after shift) ────────────────────
CONVEX_TRANSFORMS = {
    # Strictly convex pointwise
    "square_pointwise",  # φ=y², φ''=2>0
    "exp_pointwise",  # φ=exp(αy), φ''=α²exp>0
    "cosh_pointwise",  # φ=cosh(αy), φ''=α²cosh>0
    "relu_pointwise",  # max(0,y): convex (piecewise linear)
    "rational_quadratic",  # 1/(1+y²): convex near y=0
    "abs_pointwise",  # |y|: convex (triangle inequality)
    "smooth_bump",  # φ: convex on tails, concave in middle (mixed)
    # Nonlocal convex
    "power_law_beta2",  # y² after shift/scale
    "carnot_quadratic",  # quadratic after normalisation
    "weibull_reliability",  # 1-exp(-y^2): convex in normalised domain
}

# ── Concave transforms (φ''(y) ≤ 0 globally, or after shift) ───────────────────
CONCAVE_TRANSFORMS = {
    "log_shift",  # log(y+ε): φ''=-1/(y+ε)²<0
    "box_cox_sqrt",  # y^0.5: φ''=-0.25*y^(-3/2)<0
    "power_law_beta05",  # y^0.5 after normalisation
    "sqrt_abs",  # √|y|: concave for y>0
    "cbrt_pointwise",  # y^(1/3): concave for y>0
    "log1p_positive",  # log(1+|y|): concave
    "log_abs_pointwise",  # log|y|: concave for |y|>0
    "arctan_pointwise",  # arctan: concave for y>0
    "erf_pointwise",  # erf: concave for y>0
    "neg_square",  # -y²: concave
    "tanh_a03",  # tanh: concave for y>0
    "tanh_a10",
    "tanh_a005",
    "softplus_b01",  # softplus: concave-ish for large y
    "softplus_b10",
    "arrhenius",  # exp(-Ea/y): concave in y
}

# ── Monotone transforms (φ strictly increasing or decreasing) ────────────────────
MONOTONE_TRANSFORMS = {
    "log_shift",
    "box_cox_sqrt",
    "box_cox_log",
    "clipped_excess_q90",
    "affine_a2_b1",
    "affine_a05_bm3",
    "tanh_a03",
    "tanh_a10",
    "tanh_a005",
    "softplus_b01",
    "softplus_b10",
    "exp_pointwise",
    "cube_pointwise",
    "logistic_pointwise",
    "arctan_pointwise",
    "erf_pointwise",
    "sinh_pointwise",
    "cbrt_pointwise",
    "sqrt_abs",
    "log1p_positive",
    "log_abs_pointwise",
    "temporal_cumsum",  # monotone in cumulative sense
    "temporal_envelope",  # non-decreasing running max
    "weibull_reliability",
    "arrhenius",
    "gumbel_cdf",
    "frechet_cdf",
    "log_normal_cdf",
    "return_period",
    "yeo_johnson",
    "fatigue_miner",
}

# ── Non-monotone transforms ───────────────────────────────────────────────────
NONMONOTONE_TRANSFORMS = set(TRANSFORMS.keys()) - MONOTONE_TRANSFORMS

# ── Smooth transforms (C∞ or Cᵏ for k≥2) ────────────────────────────────────
SMOOTH_TRANSFORMS = {
    "log_shift",
    "box_cox_sqrt",
    "box_cox_log",
    "affine_a2_b1",
    "affine_a05_bm3",
    "tanh_a03",
    "tanh_a10",
    "tanh_a005",
    "softplus_b01",
    "softplus_b10",
    "exp_pointwise",
    "cube_pointwise",
    "cosh_pointwise",
    "logistic_pointwise",
    "arctan_pointwise",
    "erf_pointwise",
    "sinh_pointwise",
    "sin_pointwise",
    "cos_pointwise",
    "smooth_bump",
    "rational_quadratic",
    "carnot_quadratic",
    "sigmoid_dose",
    "arrhenius",
    "weibull_reliability",
    "matern_smooth",
    "contour_exceedance",
    "temporal_cumsum",
    "temporal_bandpass",
    "temporal_log_cumsum",
    "gumbel_cdf",
    "frechet_cdf",
    "log_normal_cdf",
    "johnson_su",
    "yeo_johnson",
    "inverse_normal",
}

# ── Non-smooth transforms (discontinuous or only C⁰) ────────────────────────
NONSMOOTH_TRANSFORMS = {
    "exceedance_q75",
    "exceedance_q90",
    "exceedance_q95",
    "exceedance_q99",
    "step_pointwise",
    "relu_pointwise",
    "abs_pointwise",
    "rank_transform",
    "inverse_pointwise",
    "clamp_sigma",
    "winsorise_q10_q90",
    "temporal_peak",
    "temporal_envelope",
    "exceedance_area",
    "laplacian_roughness",
}


# ══════════════════════════════════════════════════════════════════════════════
# Estimation helpers
# ══════════════════════════════════════════════════════════════════════════════


def _vw_s1(S1, Y_flat):
    """Variance-weighted aggregate of per-output first-order indices.

    S1 : (d,) pre-computed aggregate indices, or (d, n_outputs) per-output map.
    Y_flat : (n_samples, ...) field used only for variance weighting when S1 is 2D.
    Returns 1D array of shape (d,).
    """
    S1 = np.asarray(S1)
    if S1.ndim == 1:
        # Already aggregated; return directly
        return S1.copy()
    var_px = Y_flat.var(axis=0).ravel()
    total = var_px.sum()
    S1_2d = S1.reshape(S1.shape[0], -1)
    if total < 1e-30:
        return S1_2d.mean(axis=1)
    return (S1_2d * var_px[None, :]).sum(axis=1) / total


def apply_transform(Y, key):
    t = TRANSFORMS[key]
    return t["fn"](Y, **t["params"])


def score_transform(S1_orig, S1_trans, Y_orig, Y_trans, top_k=3, threshold=0.05):
    """Compute two independent, bounded, cross-benchmark-comparable non-commutativity scores.

    Metric 1 — Decision Score D ∈ [0, 1]  (decision-relevance)
    -----------------------------------------------------------
    Uses a soft sigmoid threshold to measure how much the transform moves inputs
    across the keep/discard boundary at `threshold`.  For each input i:

        soft_i(S; τ) = 1 / (1 + exp(−(Sᵢ − τ)/τ))     where τ = threshold

    D is the mean per-input absolute sigmoid difference:
        D = (1/d) Σᵢ |soft_i(Ŝ(Z)) − soft_i(Ŝ(Y))|

    D = 0: no input moved across the significance boundary.
    D → 1: every input flipped from deep inactive to deep active (or vice versa).

    Metric 2 — Sensitivity Shift Δ ∈ [0, 1]  (Bray-Curtis dissimilarity)
    ----------------------------------------------------------------------
    Measures raw redistribution of sensitivity mass:

        Δ = Σᵢ |Ŝᵢ(Z) − Ŝᵢ(Y)| / (Σᵢ Ŝᵢ(Z) + Σᵢ Ŝᵢ(Y))

    Pooled denominator makes Δ robust to near-zero indices (unlike relative L₂,
    which diverges when Ŝᵢ(Y) ≈ 0). This choice was motivated by the observation
    that relative-L₂ diverged to O(10¹¹) on the Borehole benchmark where 6 of 8
    inputs have Ŝᵢ ≈ 0. Bray-Curtis is the standard dissimilarity in community
    ecology (Bray & Curtis 1957) and has been applied to sensitivity index
    comparison in Saltelli et al. (2008).
    """
    agg_o = _vw_s1(S1_orig, Y_orig)
    agg_t = _vw_s1(S1_trans, Y_trans)
    agg_o = np.clip(agg_o, 0.0, 1.0)
    agg_t = np.clip(agg_t, 0.0, 1.0)

    tau = float(threshold)

    def _soft(s):
        return 1.0 / (1.0 + np.exp(-(s - tau) / tau))

    D = float(np.mean(np.abs(_soft(agg_t) - _soft(agg_o))))

    num = np.sum(np.abs(agg_t - agg_o))
    denom = np.sum(agg_t) + np.sum(agg_o)
    delta = float(num / denom) if denom > 1e-12 else 0.0

    # Legacy metrics
    rank_o = len(agg_o) + 1 - np.argsort(np.argsort(agg_o))
    rank_t = len(agg_t) + 1 - np.argsort(np.argsort(agg_t))
    topk_changed = set(np.argsort(agg_o)[-top_k:].tolist()) != set(
        np.argsort(agg_t)[-top_k:].tolist()
    )
    threshold_flip = int(np.sum((agg_o >= threshold) != (agg_t >= threshold)))
    l2_rel = np.abs(agg_t - agg_o) / (np.abs(agg_o) + 1e-12)
    composite = 3.0 * threshold_flip + 2.0 * int(topk_changed) + float(l2_rel.mean())

    return {
        "D": D,
        "delta": delta,
        "agg_orig": agg_o,
        "agg_trans": agg_t,
        "rank_orig": rank_o,
        "rank_trans": rank_t,
        "threshold_flip": threshold_flip,
        "topk_changed": topk_changed,
        "mean_l2": float(l2_rel.mean()),
        "composite": composite,
    }


# ── Update property sets with new transforms ─────────────────────────────────

# Add new pointwise transforms
POINTWISE_TRANSFORMS.update(
    {
        # Polynomial
        "poly4",
        "poly5",
        "poly6",
        "signed_power_p15",
        "signed_power_p05",
        "legendre_p3",
        "chebyshev_t4",
        "hermite_he2",
        "hermite_he3",
        # Sigmoid/activation
        "atan2pi",
        "algebraic_sigmoid",
        "swish",
        "mish",
        "selu",
        "softsign",
        "bent_identity",
        "hard_sigmoid",
        "hard_tanh",
        # Oscillatory
        "sinc",
        "sin_squared",
        "cos_squared",
        "damped_sin",
        "sawtooth",
        "square_wave",
        "double_sin",
        "sin_cos_product",
        # Threshold/piecewise
        "soft_threshold",
        "hard_threshold",
        "ramp",
        "spike_gaussian",
        "breakpoint",
        "hockey_stick",
        "deadzone",
        "bimodal_flip",
        "donut",
        # Curvature extremes
        "exp_neg_sq",
        "exp_pos_sq",
        "inverse_sq",
        "power_exp",
        # Structural
        "von_mises_stress",
        "safety_factor",
        "cumulative_damage",
        "stress_life",
        # Growth/activation
        "growing_degree_days",
    }
)

# Add new convex transforms
CONVEX_TRANSFORMS.update(
    {
        "poly4",  # y^4, even, phi''=12y^2>=0
        "poly6",  # y^6, even, strictly convex
        "exp_pos_sq",  # exp(+s^2*y^2), convex
        # "inverse_sq" removed: mixed convex/concave (concave near 0)
        "hockey_stick",  # max(y-bp,0), convex (ReLU shifted)
        "ramp",  # clip: convex
        "hard_sigmoid",  # piecewise linear, convex
        "hard_tanh",  # clip, convex
        "hermite_he2",  # u^2-1, convex
        "signed_power_p15",  # monotone convex for p>1
        "cumulative_damage",  # |y|^3, convex for y>0
        "stress_life",  # 1/y^3, convex and decreasing
        "von_mises_stress",  # |y|*sqrt(1.25), convex
        "exp_neg_sq",  # convex on tails, concave near 0 (mixed -- tails convex)
    }
)

# Add new concave transforms
CONCAVE_TRANSFORMS.update(
    {
        "signed_power_p05",  # |y|^0.5 sign, concave for y>0
        "atan2pi",  # (2/pi)*arctan: concave for y>0
        "algebraic_sigmoid",  # y/sqrt(1+y^2): concave for y>0
        "softsign",  # y/(1+|y|): concave for y>0
        "anscombe",  # sqrt-family, concave
        "freeman_tukey",  # sqrt+sqrt, concave
        "asinh_vst",  # arcsinh: concave for y>0
        "modulus_lam05",  # |y|^0.5*sign: concave
        "log2_shift",  # concave
        "log10_shift",  # concave
        "log_log",  # doubly concave
        "hill_response",  # Hill equation, S-shaped (concave above EC50)
        "emax_model",  # concave above ED50
        "pareto_tail",  # concave for alpha<1, convex for alpha>1
    }
)

# Add new monotone transforms
MONOTONE_TRANSFORMS.update(
    {
        "signed_power_p15",
        "signed_power_p05",
        "atan2pi",
        "algebraic_sigmoid",
        "softsign",
        "bent_identity",
        "swish",  # non-monotone technically, but increasing for small beta -- excluded below
        "selu",
        "anscombe",
        "freeman_tukey",
        "asinh_vst",
        "modulus_lam05",
        "dual_power_lam03",
        "log2_shift",
        "log10_shift",
        "gompertz_cdf",
        "gev_cdf",
        "pareto_tail",
        "log_logistic_cdf",
        "hockey_stick",  # non-decreasing
        "hill_response",
        "emax_model",
        "log_flow",
        "pot_log",
        "growing_degree_days",  # non-decreasing
        "cumulative_damage",  # monotone in |y|
    }
)

# Swish is NOT globally monotone -- remove if accidentally added
MONOTONE_TRANSFORMS.discard("swish")

# Add new smooth transforms
SMOOTH_TRANSFORMS.update(
    {
        "poly4",
        "poly5",
        "poly6",
        "signed_power_p15",  # C1 for p>=1
        "legendre_p3",
        "chebyshev_t4",
        "hermite_he2",
        "hermite_he3",
        "atan2pi",
        "algebraic_sigmoid",
        "swish",
        "mish",
        "bent_identity",
        "sinc",
        "sin_squared",
        "cos_squared",
        "damped_sin",
        "double_sin",
        "sin_cos_product",
        "soft_threshold",  # C0 not C1 -- exclude
        "spike_gaussian",
        "donut",
        "power_exp",
        "exp_neg_sq",
        "anscombe",
        "freeman_tukey",
        "asinh_vst",
        "modulus_lam05",
        "log2_shift",
        "log10_shift",
        "log_log",
        "gompertz_cdf",
        "gev_cdf",
        "log_logistic_cdf",
        "bernstein_b3",
        "hill_response",
        "emax_model",
        "log_auc",
        "log_flow",
        "anomaly_pct",
        "bias_correction",
        "std_precip_idx",
        "nash_sutcliffe",
        "hellinger",
        "chord_normalise",
        "relative_abundance",
        "sample_variance",
        "sample_skewness",
        "sample_kurtosis",
        "negentropy_proxy",
        "wasserstein_proxy",
        "energy_distance",
        "sharpe_proxy",
        "excess_return",
        "fold_change",
        "var_q95",  # quantile -- piecewise, not smooth
    }
)
# Remove non-smooth items that slipped in
SMOOTH_TRANSFORMS.discard("soft_threshold")
SMOOTH_TRANSFORMS.discard("var_q95")

# Add new non-smooth transforms
NONSMOOTH_TRANSFORMS.update(
    {
        "hard_threshold",
        "square_wave",
        "sawtooth",
        "hard_sigmoid",
        "hard_tanh",
        "ramp",
        "bimodal_flip",  # C0 only
        "selu",  # C1 but not C2 -- borderline; keep in nonsmooth for conservatism
        "cvar_q95",
        "var_q95",  # piecewise quantile
        "drawdown",  # running-max is piecewise smooth
        "soft_threshold",  # C0
    }
)


# ── Recompute derived sets after all updates ─────────────────────────────────
# NONLOCAL is the complement of POINTWISE -- must recompute after POINTWISE.update()
NONLOCAL_TRANSFORMS = set(TRANSFORMS.keys()) - POINTWISE_TRANSFORMS

# NONMONOTONE is the complement of MONOTONE -- must recompute after MONOTONE.update()
NONMONOTONE_TRANSFORMS = set(TRANSFORMS.keys()) - MONOTONE_TRANSFORMS

# Ensure SELU removed from SMOOTH (it is C1 not C2 -- classified as nonsmooth)
SMOOTH_TRANSFORMS.discard("selu")

# Ensure smooth/nonsmooth disjoint
SMOOTH_TRANSFORMS -= NONSMOOTH_TRANSFORMS
