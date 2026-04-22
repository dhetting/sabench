"""Statistical and normalization transforms extracted from the legacy monolith."""

from __future__ import annotations

import numpy as np

from sabench.transforms.utilities import _bc, _safe_range, _ymin


def t_rank_transform(Y: np.ndarray) -> np.ndarray:
    """Rank-transform each output location across samples onto ``[0, 1]``."""
    flat = Y.reshape(len(Y), -1)
    ranks = np.argsort(np.argsort(flat, axis=0), axis=0).astype(float) / (len(Y) - 1.0)
    return ranks.reshape(Y.shape)


def t_standardised_anomaly(Y: np.ndarray) -> np.ndarray:
    """Standardize each sample by its within-sample mean and standard deviation."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sig = flat.std(axis=1, keepdims=True).clip(min=1e-12)
    return ((flat - mu) / sig).reshape(Y.shape)


def t_entropy_proxy(Y: np.ndarray) -> np.ndarray:
    """Quadratic negative spread proxy centered at the within-sample mean."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    return -((flat - mu) ** 2).reshape(Y.shape)


def t_softmax_shift(Y: np.ndarray) -> np.ndarray:
    """Shifted softmax normalization over each sample's output support."""
    shift = _bc(_ymin(Y), Y)
    z = Y - shift
    ez = np.exp(np.clip(z, -50, 50))
    denom = ez.reshape(len(Y), -1).sum(axis=1)
    return ez / _bc(denom, ez)


def t_min_max_normalise(Y: np.ndarray) -> np.ndarray:
    """Per-sample min-max normalization."""
    shift = _bc(_ymin(Y), Y)
    value_range = _bc(_safe_range(Y), Y)
    return (Y - shift) / value_range


def t_robust_scale(Y: np.ndarray) -> np.ndarray:
    """Per-sample interquartile-range scaling."""
    flat = Y.reshape(len(Y), -1)
    q25 = np.quantile(flat, 0.25, axis=1)
    q50 = np.quantile(flat, 0.50, axis=1)
    q75 = np.quantile(flat, 0.75, axis=1)
    iqr = (q75 - q25).clip(min=1e-12)
    return ((flat - q50[:, None]) / iqr[:, None]).reshape(Y.shape)


def t_clamp_sigma(Y: np.ndarray, n_sigma: float = 2.0) -> np.ndarray:
    """Clamp each sample to ``±n_sigma`` standard deviations."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sig = flat.std(axis=1, keepdims=True).clip(min=1e-12)
    lo, hi = mu - n_sigma * sig, mu + n_sigma * sig
    return np.clip(flat, lo, hi).reshape(Y.shape)


def t_quantile_normalise(Y: np.ndarray) -> np.ndarray:
    """Map each output location across samples to its empirical quantile."""
    flat = Y.reshape(len(Y), -1)
    n_samples = flat.shape[0]
    ranks = np.argsort(np.argsort(flat, axis=0), axis=0).astype(float) + 1.0
    return (ranks / (n_samples + 1.0)).reshape(Y.shape)


def t_winsorise(Y: np.ndarray, low: float = 0.10, high: float = 0.90) -> np.ndarray:
    """Clip each sample to its ``[q_low, q_high]`` interval."""
    flat = Y.reshape(len(Y), -1)
    lo = np.quantile(flat, low, axis=1, keepdims=True)
    hi = np.quantile(flat, high, axis=1, keepdims=True)
    return np.clip(flat, lo, hi).reshape(Y.shape)


def t_inverse_normal(Y: np.ndarray) -> np.ndarray:
    """Map empirical quantiles to approximate normal scores."""
    flat = Y.reshape(len(Y), -1)
    n_samples = flat.shape[0]
    ranks = np.argsort(np.argsort(flat, axis=0), axis=0).astype(float) + 1.0
    p = (ranks - 0.375) / (n_samples + 0.25)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)

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
    central = np.abs(q) <= 0.42
    r = q[central] ** 2
    num = ((a[3] * r + a[2]) * r + a[1]) * r + a[0]
    den = (((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0
    out[central] = q[central] * num / den

    tail_p = p[~central]
    tail_p = np.where(q[~central] > 0.0, 1.0 - tail_p, tail_p)
    r2 = np.sqrt(-np.log(tail_p))
    ppf = c[0] + r2 * (
        c[1]
        + r2
        * (c[2] + r2 * (c[3] + r2 * (c[4] + r2 * (c[5] + r2 * (c[6] + r2 * (c[7] + r2 * c[8]))))))
    )
    out[~central] = np.where(q[~central] > 0.0, ppf, -ppf)
    return out.reshape(Y.shape)


def t_gumbel_cdf(Y: np.ndarray) -> np.ndarray:
    """Gumbel (extreme-value type I) non-exceedance CDF fitted per sample."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sig = flat.std(axis=1, keepdims=True).clip(min=1e-12)
    beta = sig * np.sqrt(6.0) / np.pi
    loc = mu - 0.5772 * beta
    z = (flat - loc) / beta
    return np.exp(-np.exp(-z)).reshape(Y.shape)


def t_frechet_cdf(Y: np.ndarray, shape: float = 2.0) -> np.ndarray:
    """Fréchet heavy-tail CDF on per-sample min-max normalized support."""
    shift = _bc(_ymin(Y), Y)
    value_range = _bc(_safe_range(Y), Y)
    y_pos = (Y - shift) / value_range + 1e-6
    return np.exp(-(y_pos ** (-shape)))


def t_log_normal_cdf(Y: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Log-normal CDF with per-sample log-location estimated from the data."""
    from math import erfc

    shift = _bc(_ymin(Y), Y)
    y_pos = Y - shift + 1.0
    log_y = np.log(y_pos)
    mu_ln = log_y.reshape(len(Y), -1).mean(axis=1, keepdims=True)
    z = (log_y.reshape(len(Y), -1) - mu_ln) / sigma
    cdf = 0.5 * np.vectorize(lambda zi: 1.0 - erfc(zi / np.sqrt(2.0)) / 2.0)(z).astype(float)
    return cdf.reshape(Y.shape)


def t_return_period(Y: np.ndarray) -> np.ndarray:
    """Empirical Weibull plotting-position return period within each sample."""
    flat = Y.reshape(len(Y), -1)
    n_out = flat.shape[1]
    ranks = np.argsort(np.argsort(flat, axis=1), axis=1).astype(float) + 1.0
    cdf = ranks / (n_out + 1.0)
    period = 1.0 / (1.0 - cdf).clip(min=1.0 / (n_out + 2.0))
    return period.reshape(Y.shape)


def t_johnson_su(Y: np.ndarray) -> np.ndarray:
    """Johnson SU-style arcsinh normalization using per-sample moments."""
    flat = Y.reshape(len(Y), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sig = flat.std(axis=1, keepdims=True).clip(min=1e-12)
    return np.arcsinh((flat - mu) / sig).reshape(Y.shape)


def t_gev_cdf(Y: np.ndarray, xi: float = 0.3) -> np.ndarray:
    """Generalized extreme-value CDF on min-max normalized support."""
    shift = _bc(_ymin(Y), Y)
    value_range = _bc(_safe_range(Y), Y)
    u = (Y - shift) / value_range
    t_val = np.maximum(1.0 + xi * u, 1e-10)
    return np.exp(-(t_val ** (-1.0 / xi)))


def t_pareto_tail(Y: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """Pareto-tail transform on normalized support."""
    shift = _bc(_ymin(Y), Y)
    value_range = _bc(_safe_range(Y), Y)
    u = np.clip((Y - shift) / value_range, 0.0, 1.0 - 1e-9)
    return 1.0 - (1.0 - u) ** alpha


def t_log_logistic_cdf(Y: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Log-logistic CDF on normalized support."""
    shift = _bc(_ymin(Y), Y)
    value_range = _bc(_safe_range(Y), Y)
    u = np.maximum((Y - shift) / value_range, 0.0)
    u_beta = np.minimum(u**beta, 1e15)
    return u_beta / (1.0 + u_beta)
