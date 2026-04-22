from __future__ import annotations

import numpy as np


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
