"""Transform application and scoring helpers."""

from __future__ import annotations

import numpy as np


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
    """Apply a registered transform by key using its default parameters."""
    from sabench.transforms.registry import get_transform

    return get_transform(key)(Y)


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
