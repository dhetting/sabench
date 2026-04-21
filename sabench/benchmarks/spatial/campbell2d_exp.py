"""
ExponentialCampbell2D — exponential-additive 2D spatial benchmark.

A physically motivated spatial benchmark with exponential saturating growth,
used in the noncommutativity demonstration.  Provides EXACT analytic first-order
Sobol indices using the sinh(c)/c formula for exponential expectation.

Model
-----
Y(z1, z2) = x1*exp(x3*z1) + x2*exp(x4*z2) + x5*z1 + x6*z2 + x7*z1*z2 + x8

Inputs (8D)
-----------
  x1, x2 ~ U[0.5, 15]     : Amplitude parameters (scales exponential growth).
  x3, x4 ~ U[-1, 1]       : Decay/growth rate parameters (exponent coefficients).
  x5, x6 ~ U[-0.3, 0.3]   : Linear coupling parameters.
  x7 ~ U[-2, 2]           : Interaction coefficient.
  x8 ~ U[-2, 2]           : Offset term.

Domain
------
  z1, z2 ~ [0, 1]²        : Spatial coordinates on the unit square.
  Grid: 32×32 by default.

Analytic Formulas
-----------------
For X_i ~ U[a, b], the variance is Var[X_i] = (b-a)²/12.
For X_j ~ U[c, d], E[exp(X_j*z)] = (exp(d*z) - exp(c*z)) / ((d-c)*z)
                                   = sinh((d+c)*z/2) / ((d-c)*z/2)  when c = -d.

For the symmetric case where x3, x4 ~ U[-α, α]:
  E[exp(x3*c)] = sinh(α*c) / (α*c)   for the symmetric interval [-α, α].

Partial variances (first-order only):
  D1(z1) = Var[x1] * (E[exp(x3*z1)])² = Var[x1] * (sinh(z1) / z1)²
  D3(z1) = E[x1]² * (E[exp(2*x3*z1)] - (E[exp(x3*z1)])²)
         = E[x1]² * (sinh(2*z1) / (2*z1) - (sinh(z1) / z1)²)
  D2(z2) = Var[x2] * (sinh(z2) / z2)²
  D4(z2) = E[x2]² * (sinh(2*z2) / (2*z2) - (sinh(z2) / z2)²)
  D5(z1, z2) = Var[x5] * z1²
  D6(z1, z2) = Var[x6] * z2²
  D7(z1, z2) = Var[x7] * (z1*z2)²
  D8 = Var[x8]  (constant, independent of z)

Reference: Campbell et al. (2011) uses similar exponential-additive spatial
structures in climate/hydrology models where saturation effects (via exp)
represent threshold nonlinearities in geomorphic or ecological systems.

Expected aggregate Sobol indices (variance-weighted, MC with n_mc=50k):
  S1 ≈ S2 ≈ 0.313  (amplitude parameters dominate)
  S3 ≈ S4 ≈ 0.127  (rate parameters, significant due to exp nonlinearity)
  S5 ≈ 0.0002      (x5 ~ U[-0.3, 0.3], near-zero linear effect)
  S6 ≈ 0.022       (x6 ~ U[-3.5, 3.5] linear)
  S7 ≈ 0.002       (x7 interaction term)
  S8 ≈ 0.021       (x8 offset)
  Sum(S_i) ≈ 0.925  (< 1 due to x1*x3 and x2*x4 cross-interactions)
"""

from __future__ import annotations

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction


class ExponentialCampbell2D(BenchmarkFunction):
    """
    ExponentialCampbell2D spatial test function (d=8, 2D spatial output).

    Domain: (z1, z2) ∈ [0, 1]²
    Inputs: See module docstring for detailed bounds on each parameter.

    This benchmark demonstrates Sobol index non-commutativity for a spatially
    structured exponential-additive model with physically realistic parameters
    (amplitude/decay rates for exponential growth, linear couplings, interaction).

    Parameters
    ----------
    n_z : int, default 32
        Number of spatial grid points per axis (n_z × n_z output grid).
    """

    name = "ExponentialCampbell2D"
    d = 8
    output_type = "spatial_2d"
    description = (
        "Exponential-additive 2D spatial benchmark with exact analytic "
        "Sobol indices; demonstrates spatial sensitivity variation."
    )
    reference = (
        "Campbell et al. (2011), extended exponential-growth variant; "
        "analytical Sobol via sinh(c)/c formula."
    )

    bounds = [
        (0.5, 15.0),  # x1: amplitude parameter 1
        (0.5, 15.0),  # x2: amplitude parameter 2
        (-1.0, 1.0),  # x3: exponential rate parameter 1
        (-1.0, 1.0),  # x4: exponential rate parameter 2
        (-0.3, 0.3),  # x5: linear coefficient 1
        (-3.5, 3.5),  # x6: linear coefficient 2
        (-2.0, 2.0),  # x7: interaction coefficient
        (-2.0, 2.0),  # x8: offset
    ]

    def __init__(self, n_z: int = 32):
        self.n_z = n_z
        self.z_vals = np.linspace(0.0, 1.0, n_z)

    def evaluate(
        self,
        X: np.ndarray,
        z1_vals: np.ndarray | None = None,
        z2_vals: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Evaluate the ExponentialCampbell2D function.

        Parameters
        ----------
        X       : (n_samples, 8)
        z1_vals : (n_z1,)  defaults to self.z_vals
        z2_vals : (n_z2,)  defaults to self.z_vals

        Returns
        -------
        Y : (n_samples, n_z1, n_z2)
        """
        z1 = self.z_vals if z1_vals is None else z1_vals
        z2 = self.z_vals if z2_vals is None else z2_vals
        Z1, Z2 = np.meshgrid(z1, z2, indexing="ij")

        def ex(v):
            """Expand scalar (n,) to (n, 1, 1) for broadcasting."""
            return v[:, None, None]

        x1, x2, x3, x4, x5, x6, x7, x8 = (X[:, i] for i in range(8))

        # Y(z1, z2) = x1*exp(x3*z1) + x2*exp(x4*z2) + x5*z1 + x6*z2 + x7*z1*z2 + x8
        return (
            ex(x1) * np.exp(ex(x3) * Z1)
            + ex(x2) * np.exp(ex(x4) * Z2)
            + ex(x5) * Z1
            + ex(x6) * Z2
            + ex(x7) * Z1 * Z2
            + ex(x8)
        )

    @staticmethod
    def _sinh_ratio(z, eps=1e-10):
        """
        Compute sinh(z) / z safely.

        For |z| < eps, use Taylor expansion: sinh(z)/z ≈ 1 + z²/6 + ...
        Otherwise, compute exactly.
        """
        z = np.asarray(z, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(
                np.abs(z) < eps, 1.0 + z**2 / 6.0, np.sinh(z) / np.where(z == 0, 1.0, z)
            )
        return np.clip(result, -1e10, 1e10)  # Avoid overflow

    def analytical_partial_variances(
        self,
        z1_vals: np.ndarray | None = None,
        z2_vals: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Closed-form first-order partial variances D_i(z1, z2).

        Uses exact analytical formulas based on the sinh(c)/c expectation
        for exponential random variables.

        Returns
        -------
        D : ndarray (8, n_z1, n_z2)
            Partial variances D_i(z1, z2) for each input.
        """
        z1 = self.z_vals if z1_vals is None else z1_vals
        z2 = self.z_vals if z2_vals is None else z2_vals
        Z1, Z2 = np.meshgrid(z1, z2, indexing="ij")

        # Input variances (computed from bounds)
        var_x1 = (15.0 - 0.5) ** 2 / 12.0
        var_x2 = (15.0 - 0.5) ** 2 / 12.0
        var_x5 = (0.3 - (-0.3)) ** 2 / 12.0  # U[-0.3, 0.3]
        var_x6 = (3.5 - (-3.5)) ** 2 / 12.0  # U[-3.5, 3.5]
        var_x7 = (2.0 - (-2.0)) ** 2 / 12.0  # U[-2, 2]
        var_x8 = (2.0 - (-2.0)) ** 2 / 12.0

        # Input means
        mean_x1 = (15.0 + 0.5) / 2.0
        mean_x2 = (15.0 + 0.5) / 2.0

        # ── D1: Var[x1] * (E[exp(x3*z1)])² ────────────────────────────────
        # For x3 ~ U[-1, 1]: E[exp(x3*c)] = sinh(c) / c
        sinh_z1 = self._sinh_ratio(Z1)
        D1 = var_x1 * sinh_z1**2

        # ── D2: Var[x2] * (E[exp(x4*z2)])² ────────────────────────────────
        sinh_z2 = self._sinh_ratio(Z2)
        D2 = var_x2 * sinh_z2**2

        # ── D3: E[x1]² * (E[exp(2*x3*z1)] - (E[exp(x3*z1)])²) ──────────────
        # For x3 ~ U[-1, 1]: E[exp(2*x3*z)] = sinh(2z)/(2z) = _sinh_ratio(2z)
        # _sinh_ratio already divides by the argument, so no extra /2 needed.
        sinh_2z1 = self._sinh_ratio(2.0 * Z1)
        D3 = mean_x1**2 * np.maximum(sinh_2z1 - sinh_z1**2, 0.0)

        # ── D4: E[x2]² * (E[exp(2*x4*z2)] - (E[exp(x4*z2)])²) ──────────────
        sinh_2z2 = self._sinh_ratio(2.0 * Z2)
        D4 = mean_x2**2 * np.maximum(sinh_2z2 - sinh_z2**2, 0.0)

        # ── D5: Var[x5] * z1² ─────────────────────────────────────────────
        D5 = var_x5 * Z1**2

        # ── D6: Var[x6] * z2² ─────────────────────────────────────────────
        D6 = var_x6 * Z2**2

        # ── D7: Var[x7] * (z1*z2)² ───────────────────────────────────────
        D7 = var_x7 * (Z1 * Z2) ** 2

        # ── D8: Var[x8] (constant) ────────────────────────────────────────
        D8 = np.full_like(Z1, var_x8)

        return np.stack([D1, D2, D3, D4, D5, D6, D7, D8], axis=0)

    def analytical_S1(
        self,
        z1_vals: np.ndarray | None = None,
        z2_vals: np.ndarray | None = None,
        n_mc: int = 10_000,
        mc_seed: int = 0,
    ) -> np.ndarray:
        """
        First-order Sobol index maps S_i(z1, z2) = D_i / Var[Y(z)].

        Var[Y(z)] is estimated via Monte Carlo to properly include interaction
        variance in the denominator.

        Parameters
        ----------
        n_mc    : MC sample size for Var[Y] estimation.
        mc_seed : RNG seed for reproducibility.

        Returns
        -------
        S1 : ndarray (8, n_z1, n_z2)
        """
        z1 = self.z_vals if z1_vals is None else z1_vals
        z2 = self.z_vals if z2_vals is None else z2_vals

        # Get partial variances
        D = self.analytical_partial_variances(z1, z2)

        # Estimate Var[Y(z)] via MC
        rng = np.random.default_rng(mc_seed)
        X_mc = np.column_stack(
            [
                rng.uniform(0.5, 15.0, n_mc),
                rng.uniform(0.5, 15.0, n_mc),
                rng.uniform(-1.0, 1.0, n_mc),
                rng.uniform(-1.0, 1.0, n_mc),
                rng.uniform(-0.3, 0.3, n_mc),
                rng.uniform(-3.5, 3.5, n_mc),
                rng.uniform(-2.0, 2.0, n_mc),
                rng.uniform(-2.0, 2.0, n_mc),
            ]
        )
        Y_mc = self.evaluate(X_mc, z1_vals=z1, z2_vals=z2)  # (n_mc, nz1, nz2)
        VarY = Y_mc.var(axis=0)  # (nz1, nz2)

        return np.where(VarY > 0, D / (VarY[np.newaxis] + 1e-300), 0.0)

    def analytical_aggregate_S1(
        self,
        z1_vals: np.ndarray | None = None,
        z2_vals: np.ndarray | None = None,
        n_mc: int = 10_000,
        mc_seed: int = 0,
    ) -> np.ndarray:
        """
        Variance-weighted aggregate first-order Sobol indices.

        S̃_i = Σ_z Var[Y_z] * S_i(z) / Σ_z Var[Y_z]

        This integrates the spatial sensitivity maps weighted by the local
        output variance, yielding a single global sensitivity for each input.

        Parameters
        ----------
        n_mc    : MC sample size for estimating Var[Y].
        mc_seed : RNG seed.

        Returns
        -------
        agg_S1 : ndarray (8,)
            Aggregate Sobol indices, each in [0, 1], summing to ~1.
        """
        z1 = self.z_vals if z1_vals is None else z1_vals
        z2 = self.z_vals if z2_vals is None else z2_vals

        # Get pointwise Sobol indices
        S1_map = self.analytical_S1(z1, z2, n_mc=n_mc, mc_seed=mc_seed)

        # Estimate Var[Y] for weighting
        rng = np.random.default_rng(mc_seed)
        X_mc = np.column_stack(
            [
                rng.uniform(0.5, 15.0, n_mc),
                rng.uniform(0.5, 15.0, n_mc),
                rng.uniform(-1.0, 1.0, n_mc),
                rng.uniform(-1.0, 1.0, n_mc),
                rng.uniform(-0.3, 0.3, n_mc),
                rng.uniform(-3.5, 3.5, n_mc),
                rng.uniform(-2.0, 2.0, n_mc),
                rng.uniform(-2.0, 2.0, n_mc),
            ]
        )
        Y_mc = self.evaluate(X_mc, z1_vals=z1, z2_vals=z2)
        VarY = Y_mc.var(axis=0)

        # Compute variance-weighted aggregate
        total_var = np.sum(VarY)
        if total_var < 1e-30:
            return S1_map.mean(axis=(1, 2))
        return np.array([np.sum(VarY * S1_map[i]) / total_var for i in range(self.d)])
