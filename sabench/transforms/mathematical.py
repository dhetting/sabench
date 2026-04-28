from __future__ import annotations

import numpy as np

from sabench.transforms.utilities import _bc, _safe_range, _ymin


def t_poly4(Y, scale=0.05):
    """phi(y) = (scale*y)^4 -- even, C-inf, convex, quartic."""
    return (scale * Y) ** 4


def t_poly5(Y, scale=0.05):
    """phi(y) = (scale*y)^5 -- odd, C-inf, inflection at 0."""
    return (scale * Y) ** 5


def t_poly6(Y, scale=0.05):
    """phi(y) = (scale*y)^6 -- even, C-inf, strictly convex."""
    return (scale * Y) ** 6


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


def t_neg_square(Y):
    """Negative square: phi(y) = -y^2."""
    return -(Y**2)


def t_smooth_bump(Y, width=3.0):
    """Smooth compact-support bump response."""
    arg = width**2 - Y**2
    out = np.where(arg > 0, np.exp(-width / np.maximum(arg, 1e-20)), 0.0)
    return out


def t_rational_quadratic(Y):
    """Rational quadratic response phi(y) = 1 / (1 + y^2)."""
    return 1.0 / (1.0 + Y**2)


def t_inverse_abs(Y, eps=1.0):
    """Inverse absolute response phi(y) = 1 / (|y| + eps)."""
    return 1.0 / (np.abs(Y) + eps)


def t_atan2pi(Y, scale=1.0):
    """Bounded arctangent response phi(y) = (2/pi) * arctan(scale*y)."""
    return (2.0 / np.pi) * np.arctan(scale * Y)


def t_exp_neg_sq(Y, scale=0.3):
    """Gaussian-kernel response phi(y) = exp(-(scale*y)^2)."""
    return np.exp(-((scale * Y) ** 2))


def t_exp_pos_sq(Y, scale=0.2):
    """Anti-Gaussian response phi(y) = exp(+(scale*y)^2)."""
    return np.exp(np.minimum((scale * Y) ** 2, 700.0))


def t_inverse_sq(Y, eps=1.0):
    """Inverse-square response phi(y) = 1 / (y^2 + eps)."""
    return 1.0 / (Y**2 + eps)


def t_power_exp(Y, scale=0.1):
    """Power-exponential hump phi(y) = y^2 * exp(-|y| * scale)."""
    return Y**2 * np.exp(-np.abs(Y) * scale)


def t_triangle_wave(Y, period=4.0):
    """Triangle wave: piecewise linear periodic function with period ``period``."""
    t = (Y % period) / period
    return 2.0 * np.abs(2.0 * t - 1.0) - 1.0


def t_signed_power(Y, p=1.5, scale=0.2):
    """Signed power response ``sign(u) * |u|**p`` for ``u = scale * Y``."""
    u = scale * Y
    return np.sign(u) * (np.abs(u) ** p)


def t_bernstein_b3(Y):
    """Bernstein B3 hump ``3u**2 * (1-u)`` on normalized support."""
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    u = np.clip((Y - s) / r, 0.0, 1.0)
    return 3.0 * u**2 * (1.0 - u)


def t_bimodal_flip(Y):
    """Bimodal sign-flip shape on normalized support."""
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    u = np.clip((Y - s) / r, 0.0, 1.0)
    return 4.0 * u * (1.0 - u) * (2.0 * u - 1.0)


def t_donut(Y, center=0.0, radius=1.5, width=0.5):
    """Donut/ring response centered on ``radius`` around ``center``."""
    return np.exp(-(((np.abs(Y - center) - radius) / width) ** 2))
