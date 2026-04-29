"""Metrics for empirical Sobol-profile noncommutativity analysis.

The utilities in this module implement the two primary profile-comparison
metrics used by the noncommutativity analysis notebooks and paper:

* Decision Score, a soft-threshold crossing score around a practical Sobol
  significance threshold.
* Sensitivity Shift, the Bray-Curtis dissimilarity between two nonnegative
  sensitivity-index profiles.

All functions operate on one-dimensional Sobol-index profiles. Higher-level
analysis code is responsible for estimating, aggregating, or selecting those
profiles before calling these metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProfileShiftMetrics:
    """Summary metrics comparing two Sobol sensitivity-index profiles.

    Attributes
    ----------
    decision_score:
        Mean absolute difference in soft threshold classifications.
    sensitivity_shift:
        Bray-Curtis dissimilarity between the two profiles.
    threshold_flip_count:
        Number of inputs that cross the hard threshold in either direction.
    topk_changed:
        Whether the unordered top-k driver set changed.
    max_abs_shift:
        Largest per-input absolute profile change.
    mean_abs_shift:
        Mean per-input absolute profile change.
    spearman:
        Spearman rank correlation between the two profiles, or ``None`` when
        one profile has constant ranks or fewer than two entries.
    top_source_index:
        Zero-based index of the largest source-profile entry.
    top_transformed_index:
        Zero-based index of the largest transformed-profile entry.
    """

    decision_score: float
    sensitivity_shift: float
    threshold_flip_count: int
    topk_changed: bool
    max_abs_shift: float
    mean_abs_shift: float
    spearman: float | None
    top_source_index: int
    top_transformed_index: int

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation for tabular workflows."""
        return asdict(self)


def soft_threshold(values: np.ndarray, tau: float = 0.05) -> np.ndarray:
    """Evaluate the soft sigmoid threshold used by the Decision Score.

    Parameters
    ----------
    values:
        One-dimensional nonnegative Sobol-index profile.
    tau:
        Positive significance threshold. The sigmoid width is also ``tau``.

    Returns
    -------
    ndarray
        Soft activity scores in ``[0, 1]``.
    """
    profile = _as_profile(values, name="values")
    threshold = _positive_threshold(tau)
    scaled = (profile - threshold) / threshold
    return _stable_sigmoid(scaled)


def decision_score(
    source_profile: np.ndarray,
    transformed_profile: np.ndarray,
    tau: float = 0.05,
) -> float:
    """Return the soft threshold-crossing Decision Score ``D``.

    ``D`` is the mean absolute difference between soft activity indicators for
    the transformed and source Sobol-index profiles.
    """
    source, transformed = _paired_profiles(source_profile, transformed_profile)
    source_soft = soft_threshold(source, tau=tau)
    transformed_soft = soft_threshold(transformed, tau=tau)
    return float(np.mean(np.abs(transformed_soft - source_soft)))


def sensitivity_shift(source_profile: np.ndarray, transformed_profile: np.ndarray) -> float:
    """Return the Bray-Curtis Sensitivity Shift ``Δ``.

    The two profiles must be finite, one-dimensional, same-length, and
    nonnegative. At least one profile must have positive total sensitivity mass.
    """
    source, transformed = _paired_profiles(source_profile, transformed_profile)
    denominator = float(source.sum() + transformed.sum())
    if denominator <= 0.0:
        raise ValueError("sensitivity_shift requires positive total sensitivity mass")
    numerator = float(np.abs(transformed - source).sum())
    return numerator / denominator


def threshold_flip_count(
    source_profile: np.ndarray,
    transformed_profile: np.ndarray,
    tau: float = 0.05,
) -> int:
    """Count hard threshold crossings between two profiles."""
    source, transformed = _paired_profiles(source_profile, transformed_profile)
    threshold = _positive_threshold(tau)
    return int(np.count_nonzero((source >= threshold) != (transformed >= threshold)))


def topk_changed(source_profile: np.ndarray, transformed_profile: np.ndarray, k: int = 3) -> bool:
    """Return whether the unordered top-k driver set changed."""
    source, transformed = _paired_profiles(source_profile, transformed_profile)
    if k <= 0:
        raise ValueError("k must be positive")
    k_eff = min(k, source.size)
    return _topk_indices(source, k_eff) != _topk_indices(transformed, k_eff)


def spearman_rank_correlation(
    source_profile: np.ndarray,
    transformed_profile: np.ndarray,
) -> float | None:
    """Return Spearman rank correlation for two profiles.

    ``None`` is returned when the correlation is undefined because there are
    fewer than two entries or either profile has constant ranks.
    """
    source, transformed = _paired_profiles(source_profile, transformed_profile)
    if source.size < 2:
        return None

    source_ranks = _average_ranks(source)
    transformed_ranks = _average_ranks(transformed)
    source_centered = source_ranks - source_ranks.mean()
    transformed_centered = transformed_ranks - transformed_ranks.mean()
    denominator = float(
        np.sqrt(np.sum(source_centered**2)) * np.sqrt(np.sum(transformed_centered**2))
    )
    if denominator == 0.0:
        return None
    return float(np.sum(source_centered * transformed_centered) / denominator)


def profile_shift_summary(
    source_profile: np.ndarray,
    transformed_profile: np.ndarray,
    tau: float = 0.05,
    top_k: int = 3,
) -> ProfileShiftMetrics:
    """Compute the standard noncommutativity metrics for one profile pair."""
    source, transformed = _paired_profiles(source_profile, transformed_profile)
    absolute_shift = np.abs(transformed - source)
    return ProfileShiftMetrics(
        decision_score=decision_score(source, transformed, tau=tau),
        sensitivity_shift=sensitivity_shift(source, transformed),
        threshold_flip_count=threshold_flip_count(source, transformed, tau=tau),
        topk_changed=topk_changed(source, transformed, k=top_k),
        max_abs_shift=float(absolute_shift.max()),
        mean_abs_shift=float(absolute_shift.mean()),
        spearman=spearman_rank_correlation(source, transformed),
        top_source_index=int(np.argmax(source)),
        top_transformed_index=int(np.argmax(transformed)),
    )


def sobol_profile_shift_metrics(
    source_s1: np.ndarray,
    transformed_s1: np.ndarray,
    source_st: np.ndarray,
    transformed_st: np.ndarray,
    tau: float = 0.05,
    top_k: int = 3,
) -> dict[str, Any]:
    """Return flattened profile-shift summaries for first-order and total effects.

    The output keys are suffixed with ``_s1`` and ``_st`` so the dictionary can
    be directly converted into one row of a grid-analysis table.
    """
    first_order = profile_shift_summary(source_s1, transformed_s1, tau=tau, top_k=top_k)
    total_effect = profile_shift_summary(source_st, transformed_st, tau=tau, top_k=top_k)
    return {
        **_suffix_keys(first_order.as_dict(), "s1"),
        **_suffix_keys(total_effect.as_dict(), "st"),
    }


def _as_profile(values: np.ndarray, name: str) -> np.ndarray:
    profile = np.asarray(values, dtype=float)
    if profile.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sensitivity profile")
    if profile.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(profile)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(profile < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return profile


def _paired_profiles(
    source_profile: np.ndarray,
    transformed_profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = _as_profile(source_profile, name="source_profile")
    transformed = _as_profile(transformed_profile, name="transformed_profile")
    if source.shape != transformed.shape:
        raise ValueError(
            "source_profile and transformed_profile must have the same shape; "
            f"got {source.shape} and {transformed.shape}"
        )
    return source, transformed


def _positive_threshold(tau: float) -> float:
    threshold = float(tau)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("tau must be a positive finite threshold")
    return threshold


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0.0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _topk_indices(values: np.ndarray, k: int) -> frozenset[int]:
    order = np.argsort(values, kind="mergesort")
    return frozenset(int(index) for index in order[-k:])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)

    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def _suffix_keys(values: dict[str, Any], suffix: str) -> dict[str, Any]:
    return {f"{key}_{suffix}": value for key, value in values.items()}
