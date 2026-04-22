from __future__ import annotations

import numpy as np

from sabench.transforms.utilities import _bc, _ymin


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
