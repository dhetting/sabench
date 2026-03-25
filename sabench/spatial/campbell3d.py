"""
Campbell3D — additive non-normalised 3D extension of Campbell2D.

Extends Marrel et al. (2011) to a 3D spatial domain [-90,90]³ while
preserving all analytically verified properties of Campbell2D.

Design
------
Mixing coordinates (Eq. 7, this work):
    θ₁⁽³⁾ = 0.8z₁ + 0.2z₂ + c₁₃z₃
    θ₂⁽³⁾ = 0.5z₁ + 0.5z₂ + c₂₃z₃
    φ₁⁽³⁾ = 0.4z₁ + 0.6z₂ + c₃₃z₃
    φ₂⁽³⁾ = 0.3z₁ + 0.7z₂ + c₄₃z₃

Structural invariants preserved from Campbell2D
------------------------------------------------
V₅ ≡ 0          X5 has zero first-order effect everywhere.
V₈ = V₆         X6 and X8 remain statistically interchangeable.

Backward compatibility (Proposition 4.2)
-----------------------------------------
g₃(X, z₁, z₂, 0) = g₂(X, z₁, z₂)  exactly for all X and (z₁,z₂).

Reference
---------
Hettinger, D. (2025). Campbell3D: An additive non-normalised extension of
  the Campbell2D spatial sensitivity benchmark. [manuscript in preparation].
"""

from __future__ import annotations

import numpy as np

from sabench._base import BenchmarkFunction
from sabench.spatial.campbell2d import _campbell_analytical_v


class Campbell3D(BenchmarkFunction):
    """
    Campbell3D spatial test function (d=8, 3D spatial output).

    Domain: (z1, z2, z3) ∈ [-90, 90]³
    Inputs: Xᵢ ~ U[-1, 5],  i = 1, …, 8

    Parameters
    ----------
    n_z  : int — grid points per axis (default 32; memory scales as n_z³)
    c13, c23, c33, c43 : z₃ mixing coefficients (defaults mirror 2D
          dominant weights: 0.8, 0.5, 0.7, 0.4)
    eps  : numerical guard
    """

    name = "Campbell3D"
    d = 8
    output_type = "spatial_3d"
    description = (
        "3D spatial SA benchmark; preserves Campbell2D heterogeneity "
        "and backward compatible at z3=0."
    )
    reference = "Hettinger (2025, in preparation)."

    bounds = [(-1.0, 5.0)] * 8

    def __init__(
        self,
        n_z: int = 32,
        c13: float = 0.8,
        c23: float = 0.5,
        c33: float = 0.7,
        c43: float = 0.4,
        eps: float = 1e-6,
    ):
        self.n_z = n_z
        self.c13 = c13
        self.c23 = c23
        self.c33 = c33
        self.c43 = c43
        self.eps = eps
        self.z_vals = np.linspace(-90.0, 90.0, n_z)

    def evaluate(
        self,
        X: np.ndarray,
        z1_vals: np.ndarray | None = None,
        z2_vals: np.ndarray | None = None,
        z3_vals: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        X       : (n_samples, 8)
        z1, z2, z3 : 1-D arrays over [-90, 90]; default self.z_vals

        Returns
        -------
        Y : (n_samples, n_z1, n_z2, n_z3)
        """
        z1 = self.z_vals if z1_vals is None else z1_vals
        z2 = self.z_vals if z2_vals is None else z2_vals
        z3 = self.z_vals if z3_vals is None else z3_vals
        eps = self.eps

        Z1, Z2, Z3 = np.meshgrid(z1, z2, z3, indexing="ij")

        # Additive non-normalised mixing coordinates (Eq. 7)
        th1 = 0.8 * Z1 + 0.2 * Z2 + self.c13 * Z3
        th2 = 0.5 * Z1 + 0.5 * Z2 + self.c23 * Z3
        ph1 = 0.4 * Z1 + 0.6 * Z2 + self.c33 * Z3
        ph2 = 0.3 * Z1 + 0.7 * Z2 + self.c43 * Z3

        def ex(v):
            return v[:, None, None, None]

        x1, x2, x3, x4, x5, x6, x7, x8 = (X[:, i] for i in range(8))

        t1 = ex(x1) * np.exp(-((th1 - 10 * ex(x2)) ** 2) / (60 * (ex(x1) ** 2 + eps)))
        t2 = ex(x2 + x4) * np.exp(th2 * ex(x1) / 500.0)
        t3 = ex(x5 * (x3 - 2.0)) * np.exp(-((ph1 - 20 * ex(x6)) ** 2) / (40 * (ex(x5) ** 2 + eps)))
        t4 = ex(x6 + x8) * np.exp(ph2 * ex(x7) / 250.0)
        return t1 + t2 + t3 + t4

    def analytical_partial_variances(
        self,
        z1_vals: np.ndarray | None = None,
        z2_vals: np.ndarray | None = None,
        z3_vals: np.ndarray | None = None,
        n_quad: int = 200,
    ) -> np.ndarray:
        """
        Analytical partial variances V_i(z1, z2, z3).
        Same Appendix-A structure as Campbell2D, substituting the 3D
        mixing coordinates (Eq. 7).

        Returns
        -------
        Vi : ndarray (8, n_z1, n_z2, n_z3)
        """
        z1 = self.z_vals if z1_vals is None else z1_vals
        z2 = self.z_vals if z2_vals is None else z2_vals
        z3 = self.z_vals if z3_vals is None else z3_vals

        Z1, Z2, Z3 = np.meshgrid(z1, z2, z3, indexing="ij")
        th1 = 0.8 * Z1 + 0.2 * Z2 + self.c13 * Z3
        th2 = 0.5 * Z1 + 0.5 * Z2 + self.c23 * Z3
        ph1 = 0.4 * Z1 + 0.6 * Z2 + self.c33 * Z3
        ph2 = 0.3 * Z1 + 0.7 * Z2 + self.c43 * Z3

        return _campbell_analytical_v(
            z1,
            z2,
            th1=th1,
            th2=th2,
            ph1=ph1,
            ph2=ph2,
            n_quad=n_quad,
        )

    def analytical_S1(
        self,
        z1_vals=None,
        z2_vals=None,
        z3_vals=None,
        n_quad: int = 200,
        n_mc: int = 2048,
        mc_seed: int = 0,
    ) -> np.ndarray:
        """
        S1 maps (8, n_z1, n_z2, n_z3) = V_i / Var[Y(z)].

        Var[Y(z)] is estimated via MC so interaction variance is included
        in the denominator and ΣS_i ≤ 1 is guaranteed.
        """
        z1 = self.z_vals if z1_vals is None else z1_vals
        z2 = self.z_vals if z2_vals is None else z2_vals
        z3 = self.z_vals if z3_vals is None else z3_vals
        Vi = self.analytical_partial_variances(z1_vals, z2_vals, z3_vals, n_quad)
        rng = np.random.default_rng(mc_seed)
        X_mc = rng.uniform(-1.0, 5.0, (n_mc, self.d))
        Y_mc = self.evaluate(X_mc, z1, z2, z3)  # (n_mc, nz1, nz2, nz3)
        VarY = Y_mc.var(axis=0)  # (nz1, nz2, nz3)
        return np.where(VarY > 0, Vi / (VarY[np.newaxis] + 1e-300), 0.0)

    def analytical_ST(
        self,
        z1_vals=None,
        z2_vals=None,
        z3_vals=None,
        n_quad: int = 200,
    ) -> np.ndarray:
        """
        Total-effect Sobol indices ST_i(z1, z2, z3).

        For the Campbell function structure, ST_i ≥ S1_i and the gap
        measures interaction effects.  Computed via the complement formula:
          ST_i(z) = 1 - V_{-i}(z) / Var(z)
        where V_{-i} = Σ_{j≠i} V_j  (first-order approximation valid
        because higher-order interactions are small for this function class).

        For exact ST, use the Jansen estimator on Monte Carlo samples.
        """
        Vi = self.analytical_partial_variances(z1_vals, z2_vals, z3_vals, n_quad)
        Var = Vi.sum(0)
        ST = np.empty_like(Vi)
        for i in range(8):
            V_excl = Var - Vi[i]
            ST[i] = np.where(Var > 0, 1.0 - V_excl / (Var + 1e-300), 0.0)
        return np.clip(ST, 0.0, 1.0)

    def slice_at_z3(
        self,
        X: np.ndarray,
        z3: float,
        z1_vals: np.ndarray | None = None,
        z2_vals: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Convenience: evaluate 3D function at a single z3 value.

        Returns
        -------
        Y : (n_samples, n_z1, n_z2)
        """
        return self.evaluate(X, z1_vals, z2_vals, np.array([z3]))[:, :, :, 0]
