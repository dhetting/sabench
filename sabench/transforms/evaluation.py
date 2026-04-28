"""Transform application and non-commutativity scoring helpers."""

from __future__ import annotations

import numpy as np


def _variance_weighted_s1(S1, Y_flat):
    """Return variance-weighted aggregate first-order indices."""
    s1 = np.asarray(S1)
    if s1.ndim == 1:
        return s1.copy()

    var_px = Y_flat.var(axis=0).ravel()
    total = var_px.sum()
    s1_2d = s1.reshape(s1.shape[0], -1)
    if total < 1e-30:
        return s1_2d.mean(axis=1)
    return (s1_2d * var_px[None, :]).sum(axis=1) / total


def apply_transform(Y, key):
    """Apply a registered transform by key using its default parameters."""
    from sabench.transforms.registry import get_transform

    return get_transform(key)(Y)


def score_transform(S1_orig, S1_trans, Y_orig, Y_trans, top_k=3, threshold=0.05):
    """Compute bounded non-commutativity scores for transformed sensitivity indices."""
    agg_o = _variance_weighted_s1(S1_orig, Y_orig)
    agg_t = _variance_weighted_s1(S1_trans, Y_trans)
    agg_o = np.clip(agg_o, 0.0, 1.0)
    agg_t = np.clip(agg_t, 0.0, 1.0)

    tau = float(threshold)

    def _soft(s):
        return 1.0 / (1.0 + np.exp(-(s - tau) / tau))

    decision_score = float(np.mean(np.abs(_soft(agg_t) - _soft(agg_o))))

    numerator = np.sum(np.abs(agg_t - agg_o))
    denominator = np.sum(agg_t) + np.sum(agg_o)
    delta = float(numerator / denominator) if denominator > 1e-12 else 0.0

    rank_o = len(agg_o) + 1 - np.argsort(np.argsort(agg_o))
    rank_t = len(agg_t) + 1 - np.argsort(np.argsort(agg_t))
    topk_changed = set(np.argsort(agg_o)[-top_k:].tolist()) != set(
        np.argsort(agg_t)[-top_k:].tolist()
    )
    threshold_flip = int(np.sum((agg_o >= threshold) != (agg_t >= threshold)))
    l2_rel = np.abs(agg_t - agg_o) / (np.abs(agg_o) + 1e-12)
    composite = 3.0 * threshold_flip + 2.0 * int(topk_changed) + float(l2_rel.mean())

    return {
        "D": decision_score,
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
