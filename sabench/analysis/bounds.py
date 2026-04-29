"""Taylor-reference perturbation bounds for transformed Sobol profiles.

The utilities in this module implement the reusable numerical pieces behind the
bounds-theorem analysis notebook. They intentionally separate theorem-facing
quantities from notebook rendering:

* projection and local-affine perturbation bounds;
* Taylor-reference and residual diagnostics for ``Z = phi(Y)``;
* a small, explicit derivative registry for smooth pointwise transforms whose
  derivatives match the registered transform definitions.

Rows produced by future grid notebooks should label sample-range calculations as
diagnostics unless genuine almost-sure support bounds are supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import factorial
from typing import Any, Literal

import numpy as np

BoundsApplicabilityStatus = Literal[
    "bounds_supported",
    "bounds_not_scalar_output",
    "bounds_not_pointwise",
    "bounds_not_smooth",
    "bounds_no_derivative_metadata",
]
SupportSource = Literal["sample_range", "provided_support"]
TaylorStatus = Literal["computed", "reference_zero_variance", "eta_ge_one"]
LocalAffineStatus = Literal[
    "computed",
    "zero_output_variance",
    "zero_slope",
    "missing_second_derivative_sup",
    "lambda_ge_two",
]

ArrayTransform = Callable[[np.ndarray], np.ndarray]
Derivative = Callable[[np.ndarray, int], np.ndarray]
DerivativeSupremum = Callable[[int, float, float], float | None]


@dataclass(frozen=True, slots=True)
class SmoothPointwiseTransformAnalysis:
    """Derivative metadata for one smooth pointwise scalar transform."""

    key: str
    name: str
    max_taylor_order: int
    transform: ArrayTransform
    derivative: Derivative
    derivative_supremum: DerivativeSupremum


@dataclass(frozen=True, slots=True)
class BoundsApplicability:
    """Static theorem-bound applicability classification for one pair."""

    status: BoundsApplicabilityStatus
    reason: str

    @property
    def supported(self) -> bool:
        """Return whether theorem-bound diagnostics are supported."""
        return self.status == "bounds_supported"


@dataclass(frozen=True, slots=True)
class TaylorReferenceDiagnostics:
    """Empirical Taylor-reference quantities for ``Z = phi(Y)``."""

    transform_key: str
    order: int
    status: TaylorStatus
    support_source: SupportSource
    support_lower: float
    support_upper: float
    mu_y: float
    sigma_reference: float
    eta_empirical: float | None
    eta_sufficient: float | None
    eta_empirical_lt_one: bool
    eta_sufficient_lt_one: bool | None
    reference_values: np.ndarray
    residual_values: np.ndarray

    def as_summary_dict(self) -> dict[str, Any]:
        """Return scalar diagnostics suitable for tabular output."""
        return {
            "transform_key": self.transform_key,
            "order_m": self.order,
            "bound_status": self.status,
            "support_source": self.support_source,
            "support_lower": self.support_lower,
            "support_upper": self.support_upper,
            "mu_y": self.mu_y,
            "sigma_reference": self.sigma_reference,
            "eta_empirical": self.eta_empirical,
            "eta_sufficient": self.eta_sufficient,
            "eta_empirical_lt_one": self.eta_empirical_lt_one,
            "eta_sufficient_lt_one": self.eta_sufficient_lt_one,
        }


@dataclass(frozen=True, slots=True)
class LocalAffineDiagnostics:
    """Empirical nonzero-slope local-affine quantities."""

    transform_key: str
    status: LocalAffineStatus
    support_source: SupportSource
    support_lower: float
    support_upper: float
    mu_y: float
    sigma_y: float
    mu4: float
    phi_prime_mu: float
    rho2: float | None
    kappa: float | None
    moment_ratio: float | None
    lambda_value: float | None
    eta_upper: float | None
    lambda_lt_two: bool | None

    def as_summary_dict(self) -> dict[str, Any]:
        """Return scalar diagnostics suitable for tabular output."""
        return {
            "transform_key": self.transform_key,
            "local_affine_status": self.status,
            "support_source": self.support_source,
            "support_lower": self.support_lower,
            "support_upper": self.support_upper,
            "mu_y": self.mu_y,
            "sigma_y": self.sigma_y,
            "mu4": self.mu4,
            "phi_prime_mu": self.phi_prime_mu,
            "rho2": self.rho2,
            "kappa": self.kappa,
            "moment_ratio": self.moment_ratio,
            "lambda_value": self.lambda_value,
            "eta_upper": self.eta_upper,
            "lambda_lt_two": self.lambda_lt_two,
        }


def projection_perturbation_bound(
    eta: float,
    p: float | np.ndarray,
    *,
    cap: bool = True,
) -> float | np.ndarray:
    """Return the abstract projection perturbation bound.

    The bound is defined for ``0 <= eta < 1`` and ``0 <= p <= 1`` as

    ``[2 eta sqrt(p)(1 + sqrt(p)) + eta^2(1 + p)] / (1 - eta)^2``.
    """
    eta_value = _validate_eta(eta, upper=1.0, name="eta")
    p_values = _validate_probability_like(p, name="p")
    sqrt_p = np.sqrt(p_values)
    bound = (2.0 * eta_value * sqrt_p * (1.0 + sqrt_p) + eta_value**2 * (1.0 + p_values)) / (
        1.0 - eta_value
    ) ** 2
    result = np.minimum(bound, 1.0) if cap else bound
    return _scalar_or_array(result, p)


def local_affine_perturbation_bound(
    lambda_value: float,
    p: float | np.ndarray,
    *,
    cap: bool = True,
) -> float | np.ndarray:
    """Return the nonzero-slope local-affine perturbation bound.

    The bound is defined for ``0 <= lambda < 2`` and ``0 <= p <= 1`` as

    ``[lambda sqrt(p)(1 + sqrt(p)) + lambda^2(1 + p)/4] / (1 - lambda/2)^2``.
    """
    lam = _validate_eta(lambda_value, upper=2.0, name="lambda_value")
    p_values = _validate_probability_like(p, name="p")
    sqrt_p = np.sqrt(p_values)
    bound = (lam * sqrt_p * (1.0 + sqrt_p) + 0.25 * lam**2 * (1.0 + p_values)) / (
        1.0 - 0.5 * lam
    ) ** 2
    result = np.minimum(bound, 1.0) if cap else bound
    return _scalar_or_array(result, p)


def supported_smooth_pointwise_transform_keys() -> tuple[str, ...]:
    """Return transform keys with explicit derivative metadata."""
    return tuple(sorted(_SMOOTH_POINTWISE_ANALYSES))


def get_smooth_pointwise_analysis(key: str) -> SmoothPointwiseTransformAnalysis:
    """Return derivative metadata for a supported smooth pointwise transform."""
    try:
        return _SMOOTH_POINTWISE_ANALYSES[key]
    except KeyError as exc:
        raise KeyError(f"no smooth pointwise derivative metadata for transform {key!r}") from exc


def classify_bounds_applicability(
    *,
    output_kind: str,
    mechanism: str,
    tags: Iterable[str],
    transform_key: str,
) -> BoundsApplicability:
    """Classify whether theorem-bound diagnostics are supported for a pair."""
    tag_set = set(tags)
    if output_kind != "scalar":
        return BoundsApplicability(
            status="bounds_not_scalar_output",
            reason="Taylor-reference bounds currently require scalar benchmark outputs.",
        )
    if mechanism != "pointwise":
        return BoundsApplicability(
            status="bounds_not_pointwise",
            reason="Taylor-reference bounds require pointwise scalar output transforms.",
        )
    if "smooth" not in tag_set:
        return BoundsApplicability(
            status="bounds_not_smooth",
            reason="Taylor-reference bounds require smooth transform metadata.",
        )
    if transform_key not in _SMOOTH_POINTWISE_ANALYSES:
        return BoundsApplicability(
            status="bounds_no_derivative_metadata",
            reason="No explicit derivative metadata is registered for this transform.",
        )
    return BoundsApplicability(
        status="bounds_supported",
        reason="Smooth scalar pointwise transform with registered derivative metadata.",
    )


def taylor_reference_values(
    y: np.ndarray,
    analysis: SmoothPointwiseTransformAnalysis,
    order: int,
) -> np.ndarray:
    """Return ``V_m = sum_k phi^(k)(mu) (Y - mu)^k / k!``."""
    values = _as_output_vector(y)
    order_value = _validate_order(order, max_order=analysis.max_taylor_order)
    mu_y = float(values.mean())
    centered = values - mu_y
    reference = np.zeros_like(values, dtype=float)
    mu_array = np.array([mu_y], dtype=float)
    for k in range(1, order_value + 1):
        derivative_at_mu = float(analysis.derivative(mu_array, k)[0])
        reference += derivative_at_mu * centered**k / float(factorial(k))
    return reference


def taylor_residual_values(
    y: np.ndarray,
    analysis: SmoothPointwiseTransformAnalysis,
    order: int,
) -> np.ndarray:
    """Return ``R_m = phi(Y) - phi(mu_Y) - V_m``."""
    values = _as_output_vector(y)
    reference = taylor_reference_values(values, analysis, order)
    mu_y = float(values.mean())
    phi_values = analysis.transform(values)
    phi_mu = float(analysis.transform(np.array([mu_y], dtype=float))[0])
    return phi_values - phi_mu - reference


def taylor_reference_diagnostics(
    y: np.ndarray,
    analysis: SmoothPointwiseTransformAnalysis,
    order: int,
    *,
    support: tuple[float, float] | None = None,
    atol: float = 1e-12,
) -> TaylorReferenceDiagnostics:
    """Compute empirical Taylor-reference diagnostics for one transform/order."""
    values = _as_output_vector(y)
    support_lower, support_upper, support_source = _resolve_support(values, support)
    reference = taylor_reference_values(values, analysis, order)
    residual = taylor_residual_values(values, analysis, order)
    mu_y = float(values.mean())
    sigma_reference = _empirical_sd(reference)
    if sigma_reference <= atol:
        return TaylorReferenceDiagnostics(
            transform_key=analysis.key,
            order=order,
            status="reference_zero_variance",
            support_source=support_source,
            support_lower=support_lower,
            support_upper=support_upper,
            mu_y=mu_y,
            sigma_reference=sigma_reference,
            eta_empirical=None,
            eta_sufficient=None,
            eta_empirical_lt_one=False,
            eta_sufficient_lt_one=None,
            reference_values=reference,
            residual_values=residual,
        )

    eta_empirical = _empirical_sd(residual) / sigma_reference
    eta_sufficient = sufficient_taylor_eta(
        values,
        analysis,
        order,
        sigma_reference=sigma_reference,
        support=(support_lower, support_upper),
    )
    status: TaylorStatus = "computed" if eta_empirical < 1.0 else "eta_ge_one"
    return TaylorReferenceDiagnostics(
        transform_key=analysis.key,
        order=order,
        status=status,
        support_source=support_source,
        support_lower=support_lower,
        support_upper=support_upper,
        mu_y=mu_y,
        sigma_reference=sigma_reference,
        eta_empirical=float(eta_empirical),
        eta_sufficient=eta_sufficient,
        eta_empirical_lt_one=bool(eta_empirical < 1.0),
        eta_sufficient_lt_one=None if eta_sufficient is None else bool(eta_sufficient < 1.0),
        reference_values=reference,
        residual_values=residual,
    )


def sufficient_taylor_eta(
    y: np.ndarray,
    analysis: SmoothPointwiseTransformAnalysis,
    order: int,
    *,
    sigma_reference: float | None = None,
    support: tuple[float, float] | None = None,
) -> float | None:
    """Return the Taylor theorem sufficient ``eta`` bound when available."""
    values = _as_output_vector(y)
    order_value = _validate_order(order, max_order=analysis.max_taylor_order)
    sigma = sigma_reference
    if sigma is None:
        sigma = _empirical_sd(taylor_reference_values(values, analysis, order_value))
    if sigma <= 0.0:
        return None
    support_lower, support_upper, _ = _resolve_support(values, support)
    rho = analysis.derivative_supremum(order_value + 1, support_lower, support_upper)
    if rho is None:
        return None
    centered = values - float(values.mean())
    moment = float(np.mean(np.abs(centered) ** (2 * order_value + 2))) ** 0.5
    return float(rho * moment / (float(factorial(order_value + 1)) * sigma))


def local_affine_diagnostics(
    y: np.ndarray,
    analysis: SmoothPointwiseTransformAnalysis,
    *,
    support: tuple[float, float] | None = None,
    atol: float = 1e-12,
) -> LocalAffineDiagnostics:
    """Compute nonzero-slope local-affine diagnostics for one transform."""
    values = _as_output_vector(y)
    support_lower, support_upper, support_source = _resolve_support(values, support)
    mu_y = float(values.mean())
    centered = values - mu_y
    sigma_y = _empirical_sd(values)
    mu4 = float(np.mean(centered**4))
    phi_prime_mu = float(analysis.derivative(np.array([mu_y], dtype=float), 1)[0])

    base_kwargs: dict[str, Any] = {
        "transform_key": analysis.key,
        "support_source": support_source,
        "support_lower": support_lower,
        "support_upper": support_upper,
        "mu_y": mu_y,
        "sigma_y": sigma_y,
        "mu4": mu4,
        "phi_prime_mu": phi_prime_mu,
    }
    if sigma_y <= atol:
        return LocalAffineDiagnostics(
            status="zero_output_variance",
            rho2=None,
            kappa=None,
            moment_ratio=None,
            lambda_value=None,
            eta_upper=None,
            lambda_lt_two=None,
            **base_kwargs,
        )
    if abs(phi_prime_mu) <= atol:
        return LocalAffineDiagnostics(
            status="zero_slope",
            rho2=None,
            kappa=None,
            moment_ratio=None,
            lambda_value=None,
            eta_upper=None,
            lambda_lt_two=None,
            **base_kwargs,
        )
    rho2 = analysis.derivative_supremum(2, support_lower, support_upper)
    if rho2 is None:
        return LocalAffineDiagnostics(
            status="missing_second_derivative_sup",
            rho2=None,
            kappa=None,
            moment_ratio=None,
            lambda_value=None,
            eta_upper=None,
            lambda_lt_two=None,
            **base_kwargs,
        )

    kappa = float(rho2 * sigma_y / abs(phi_prime_mu))
    moment_ratio = float(np.sqrt(mu4) / sigma_y**2)
    lambda_value = float(moment_ratio * kappa)
    status: LocalAffineStatus = "computed" if lambda_value < 2.0 else "lambda_ge_two"
    return LocalAffineDiagnostics(
        status=status,
        rho2=float(rho2),
        kappa=kappa,
        moment_ratio=moment_ratio,
        lambda_value=lambda_value,
        eta_upper=lambda_value / 2.0,
        lambda_lt_two=bool(lambda_value < 2.0),
        **base_kwargs,
    )


def _as_output_vector(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if values.ndim != 1:
        raise ValueError("y must be a one-dimensional scalar-output sample")
    if values.size == 0:
        raise ValueError("y must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("y must contain only finite values")
    return values


def _validate_eta(value: float, *, upper: float, name: str) -> float:
    eta = float(value)
    if not np.isfinite(eta) or eta < 0.0 or eta >= upper:
        raise ValueError(f"{name} must satisfy 0 <= {name} < {upper:g}")
    return eta


def _validate_probability_like(p: float | np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return values


def _validate_order(order: int, *, max_order: int) -> int:
    if order < 1:
        raise ValueError("order must be at least 1")
    if order > max_order:
        raise ValueError(f"order {order} exceeds supported maximum order {max_order}")
    return order


def _scalar_or_array(values: np.ndarray, original: float | np.ndarray) -> float | np.ndarray:
    if np.asarray(original).ndim == 0:
        return float(values)
    return values


def _empirical_sd(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    return float(np.sqrt(np.mean(centered**2)))


def _resolve_support(
    values: np.ndarray,
    support: tuple[float, float] | None,
) -> tuple[float, float, SupportSource]:
    if support is None:
        return float(np.min(values)), float(np.max(values)), "sample_range"
    lower, upper = map(float, support)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("support must be finite and ordered as (lower, upper)")
    if np.min(values) < lower or np.max(values) > upper:
        raise ValueError("support must contain all observed y values")
    return lower, upper, "provided_support"


def _constant_derivative(value: float) -> Derivative:
    def derivative(y: np.ndarray, order: int) -> np.ndarray:
        if order == 1:
            return np.full_like(y, value, dtype=float)
        return np.zeros_like(y, dtype=float)

    return derivative


def _constant_derivative_sup(value: float) -> DerivativeSupremum:
    def derivative_supremum(order: int, lower: float, upper: float) -> float | None:
        _validate_support_bounds(lower, upper)
        if order == 1:
            return abs(value)
        return 0.0

    return derivative_supremum


def _polynomial_transform(coefficients: tuple[float, ...]) -> ArrayTransform:
    def transform(y: np.ndarray) -> np.ndarray:
        result = np.zeros_like(y, dtype=float)
        for power, coefficient in enumerate(coefficients):
            if coefficient != 0.0:
                result += coefficient * y**power
        return result

    return transform


def _polynomial_derivative(coefficients: tuple[float, ...]) -> Derivative:
    def derivative(y: np.ndarray, order: int) -> np.ndarray:
        result = np.zeros_like(y, dtype=float)
        for power, coefficient in enumerate(coefficients):
            if power < order or coefficient == 0.0:
                continue
            multiplier = 1.0
            for step in range(order):
                multiplier *= power - step
            result += coefficient * multiplier * y ** (power - order)
        return result

    return derivative


def _polynomial_derivative_sup(coefficients: tuple[float, ...]) -> DerivativeSupremum:
    derivative = _polynomial_derivative(coefficients)

    def derivative_supremum(order: int, lower: float, upper: float) -> float | None:
        _validate_support_bounds(lower, upper)
        if order >= len(coefficients):
            return 0.0
        points = np.linspace(lower, upper, 2049)
        return float(np.max(np.abs(derivative(points, order))))

    return derivative_supremum


def _exp_transform(scale: float) -> ArrayTransform:
    def transform(y: np.ndarray) -> np.ndarray:
        return np.exp(scale * y)

    return transform


def _exp_derivative(scale: float) -> Derivative:
    def derivative(y: np.ndarray, order: int) -> np.ndarray:
        if order < 1:
            raise ValueError("order must be at least 1")
        return scale**order * np.exp(scale * y)

    return derivative


def _exp_derivative_sup(scale: float) -> DerivativeSupremum:
    def derivative_supremum(order: int, lower: float, upper: float) -> float | None:
        _validate_support_bounds(lower, upper)
        endpoint = upper if scale >= 0.0 else lower
        return float(abs(scale) ** order * np.exp(scale * endpoint))

    return derivative_supremum


def _sin_transform(freq: float) -> ArrayTransform:
    def transform(y: np.ndarray) -> np.ndarray:
        return np.sin(freq * y)

    return transform


def _sin_derivative(freq: float) -> Derivative:
    def derivative(y: np.ndarray, order: int) -> np.ndarray:
        if order < 1:
            raise ValueError("order must be at least 1")
        phase = order * np.pi / 2.0
        return freq**order * np.sin(freq * y + phase)

    return derivative


def _cos_transform(freq: float) -> ArrayTransform:
    def transform(y: np.ndarray) -> np.ndarray:
        return np.cos(freq * y)

    return transform


def _cos_derivative(freq: float) -> Derivative:
    def derivative(y: np.ndarray, order: int) -> np.ndarray:
        if order < 1:
            raise ValueError("order must be at least 1")
        phase = order * np.pi / 2.0
        return freq**order * np.cos(freq * y + phase)

    return derivative


def _trig_derivative_sup(freq: float) -> DerivativeSupremum:
    def derivative_supremum(order: int, lower: float, upper: float) -> float | None:
        _validate_support_bounds(lower, upper)
        return float(abs(freq) ** order)

    return derivative_supremum


def _validate_support_bounds(lower: float, upper: float) -> None:
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("support bounds must be finite and ordered")


def _analysis(
    key: str,
    name: str,
    transform: ArrayTransform,
    derivative: Derivative,
    derivative_supremum: DerivativeSupremum,
    max_taylor_order: int = 3,
) -> SmoothPointwiseTransformAnalysis:
    return SmoothPointwiseTransformAnalysis(
        key=key,
        name=name,
        max_taylor_order=max_taylor_order,
        transform=transform,
        derivative=derivative,
        derivative_supremum=derivative_supremum,
    )


_SMOOTH_POINTWISE_ANALYSES: dict[str, SmoothPointwiseTransformAnalysis] = {
    "affine_a2_b1": _analysis(
        "affine_a2_b1",
        "Affine (a=2, b=1)",
        lambda y: 2.0 * y + 1.0,
        _constant_derivative(2.0),
        _constant_derivative_sup(2.0),
    ),
    "affine_a05_bm3": _analysis(
        "affine_a05_bm3",
        "Affine (a=0.5, b=-3)",
        lambda y: 0.5 * y - 3.0,
        _constant_derivative(0.5),
        _constant_derivative_sup(0.5),
    ),
    "square_pointwise": _analysis(
        "square_pointwise",
        "Square",
        _polynomial_transform((0.0, 0.0, 1.0)),
        _polynomial_derivative((0.0, 0.0, 1.0)),
        _polynomial_derivative_sup((0.0, 0.0, 1.0)),
    ),
    "neg_square": _analysis(
        "neg_square",
        "Negative square",
        _polynomial_transform((0.0, 0.0, -1.0)),
        _polynomial_derivative((0.0, 0.0, -1.0)),
        _polynomial_derivative_sup((0.0, 0.0, -1.0)),
    ),
    "cube_pointwise": _analysis(
        "cube_pointwise",
        "Cube",
        _polynomial_transform((0.0, 0.0, 0.0, 1.0)),
        _polynomial_derivative((0.0, 0.0, 0.0, 1.0)),
        _polynomial_derivative_sup((0.0, 0.0, 0.0, 1.0)),
    ),
    "exp_pointwise": _analysis(
        "exp_pointwise",
        "Exponential (scale=0.1)",
        _exp_transform(0.1),
        _exp_derivative(0.1),
        _exp_derivative_sup(0.1),
    ),
    "sin_pointwise": _analysis(
        "sin_pointwise",
        "Sine (freq=0.5)",
        _sin_transform(0.5),
        _sin_derivative(0.5),
        _trig_derivative_sup(0.5),
    ),
    "cos_pointwise": _analysis(
        "cos_pointwise",
        "Cosine (freq=0.5)",
        _cos_transform(0.5),
        _cos_derivative(0.5),
        _trig_derivative_sup(0.5),
    ),
}
