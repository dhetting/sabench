from __future__ import annotations

import numpy as np

from sabench.transforms.utilities import _bc, _safe_range, _ymin


def t_sigmoid_dose(Y, EC50_q=0.5, n_hill=4.0):
    s = _bc(_ymin(Y), Y)
    r = _bc(_safe_range(Y), Y)
    Yn = (Y - s) / r
    return Yn**n_hill / (EC50_q**n_hill + Yn**n_hill + 1e-30)


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
