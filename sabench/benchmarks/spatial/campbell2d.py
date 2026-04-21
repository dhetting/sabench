"""
Campbell2D function — exact implementation of Marrel et al. (2011) Eq. (6).

The canonical benchmark for 2D spatial sensitivity analysis.  Provides
analytical first-order partial variances V_i(z1, z2) via Appendix A,
Eqs. (16)-(23) of the reference.

Reference
---------
Marrel, A., Iooss, B., Laurent, B., & Roustant, O. (2011). Calculations
  of Sobol indices for the Gaussian process metamodel. Environmetrics,
  22(3), 383-397. https://doi.org/10.1002/env.1041
"""

from __future__ import annotations

import math

import numpy as np

from sabench.benchmarks.base import BenchmarkFunction


class Campbell2D(BenchmarkFunction):
    """
    Campbell2D spatial test function (d=8, 2D spatial output).

    Domain: (z1, z2) ∈ [-90, 90]²
    Inputs: Xᵢ ~ U[-1, 5],  i = 1, …, 8

    Structural invariants
    ----------------------
    V₅ ≡ 0    : X5 has zero first-order effect everywhere.
    V₈ = V₆   : X6 and X8 are statistically interchangeable.

    Parameters
    ----------
    n_z : int, default 64
        Number of spatial grid points per axis (n_z × n_z output grid).
    eps : float, default 1e-6
        Numerical guard for 1/X₁² and 1/X₅² denominators.
    """

    name = "Campbell2D"
    d = 8
    output_type = "spatial_2d"
    description = (
        "Spatial SA benchmark with analytical V_i(z1,z2); "
        "sharp Gaussian ridges and exponential growth terms."
    )
    reference = (
        "Marrel, Iooss, Laurent & Roustant (2011), Environmetrics "
        "22(3):383-397. https://doi.org/10.1002/env.1041"
    )

    bounds = [(-1.0, 5.0)] * 8

    def __init__(self, n_z: int = 64, eps: float = 1e-6):
        self.n_z = n_z
        self.eps = eps
        self.z_vals = np.linspace(-90.0, 90.0, n_z)

    def evaluate(
        self,
        X: np.ndarray,
        z1_vals: np.ndarray | None = None,
        z2_vals: np.ndarray | None = None,
    ) -> np.ndarray:
        """
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
        eps = self.eps

        th1 = 0.8 * Z1 + 0.2 * Z2  # θ₁ — dominant weight on z1
        th2 = 0.5 * Z1 + 0.5 * Z2  # θ₂ — symmetric
        ph1 = 0.4 * Z1 + 0.6 * Z2  # φ₁ — dominant weight on z2
        ph2 = 0.3 * Z1 + 0.7 * Z2  # φ₂ — dominant weight on z2

        def ex(v):
            return v[:, None, None]

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
        n_quad: int = 400,
    ) -> np.ndarray:
        """
        Closed-form partial variances V_i(z1, z2), Eqs. (16)-(23) of
        Marrel et al. (2011) Appendix A.  Evaluated via 1-D quadrature.

        Returns
        -------
        Vi : ndarray (8, n_z1, n_z2)
        """
        z1 = self.z_vals if z1_vals is None else z1_vals
        z2 = self.z_vals if z2_vals is None else z2_vals
        return _campbell_analytical_v(z1, z2, n_quad=n_quad)

    def analytical_S1(
        self,
        z1_vals: np.ndarray | None = None,
        z2_vals: np.ndarray | None = None,
        n_quad: int = 400,
        n_mc: int = 4096,
        mc_seed: int = 0,
    ) -> np.ndarray:
        """
        First-order Sobol index maps S_i(z1, z2) = V_i / Var[Y(z)].

        Var[Y(z)] is estimated via Monte Carlo integration (n_mc samples),
        so interaction variance is properly included in the denominator.
        For a purely additive model, Var[Y] = ΣV_i and the two approaches
        agree; for the Campbell2D function with cross-term interactions,
        they diverge, and this method gives the correct S_i ∈ [0,1] with
        ΣS_i ≤ 1.

        Parameters
        ----------
        n_mc     : MC sample size for Var[Y] estimation (default 4096).
        mc_seed  : RNG seed for reproducibility.

        Returns
        -------
        S1 : ndarray (8, n_z1, n_z2)
        """
        z1 = self.z_vals if z1_vals is None else z1_vals
        z2 = self.z_vals if z2_vals is None else z2_vals
        Vi = self.analytical_partial_variances(z1_vals, z2_vals, n_quad)
        # Estimate Var[Y(z)] via MC
        rng = np.random.default_rng(mc_seed)
        X_mc = rng.uniform(-1.0, 5.0, (n_mc, self.d))
        Y_mc = self.evaluate(X_mc, z1_vals=z1, z2_vals=z2)  # (n_mc, nz1, nz2)
        VarY = Y_mc.var(axis=0)  # (nz1, nz2)
        return np.where(VarY > 0, Vi / (VarY[np.newaxis] + 1e-300), 0.0)


# ── Shared analytical kernel (also used by Campbell3D) ───────────────────────


def _phi_cdf(x: np.ndarray) -> np.ndarray:
    """Vectorised standard-normal CDF via math.erf (no scipy)."""
    _sqrt2 = math.sqrt(2.0)
    vf = np.frompyfunc(lambda v: 0.5 * (1.0 + math.erf(v / _sqrt2)), 1, 1)
    return vf(np.asarray(x, dtype=float)).astype(float)


def _campbell_analytical_v(
    z1_vals: np.ndarray,
    z2_vals: np.ndarray,
    th1: np.ndarray | None = None,
    th2: np.ndarray | None = None,
    ph1: np.ndarray | None = None,
    ph2: np.ndarray | None = None,
    n_quad: int = 400,
) -> np.ndarray:
    """
    Compute partial variances V_i for any set of mixing coordinates.

    If th1/th2/ph1/ph2 are None they are computed from z1_vals, z2_vals
    using the standard 2D weights (Marrel et al. 2011, Eq. 14).

    Returns Vi : ndarray (8, *spatial_shape)
    """
    Z1, Z2 = np.meshgrid(z1_vals, z2_vals, indexing="ij")
    if th1 is None:
        th1_arr = 0.8 * Z1 + 0.2 * Z2
        th2_arr = 0.5 * Z1 + 0.5 * Z2
        ph1_arr = 0.4 * Z1 + 0.6 * Z2
        ph2_arr = 0.3 * Z1 + 0.7 * Z2
    else:
        assert th2 is not None
        assert ph1 is not None
        assert ph2 is not None
        th1_arr = th1
        th2_arr = th2
        ph1_arr = ph1
        ph2_arr = ph2

    xq = np.linspace(-1.0, 5.0, n_quad)

    # spatial shape may be (nz1, nz2) or (nz1, nz2, nz3)
    def ex(v):
        return v.reshape((-1,) + (1,) * th1_arr.ndim)

    def var1d(f):
        return np.var(f, axis=0)

    # ── V1 ──────────────────────────────────────────────────────────────────
    # E[Y|X1=x1] = sqrt(pi/60)*x1*|x1|*[Phi((th1+10)/s30) - Phi((th1-50)/s30)]
    #              + 4*exp(th2*x1/500)
    # where s30 = sqrt(30)*|x1|.  Derivation: substitution w=(th1-10x2)/sqrt(60x1²),
    # Gaussian integral with σ=sqrt(30)|x1|.  Prefactor x1*|x1| (not x1²) handles
    # sign correctly for negative x1.
    sx1 = np.where(xq == 0, 1e-10, np.abs(xq))  # |x1|
    s30 = np.sqrt(30.0) * ex(sx1)  # σ = sqrt(30)|x1|
    f1 = np.sqrt(np.pi / 60.0) * ex(xq * np.abs(xq)) * (
        _phi_cdf((th1_arr + 10.0) / s30) - _phi_cdf((th1_arr - 50.0) / s30)
    ) + 4.0 * np.exp(th2_arr * ex(xq) / 500.0)
    V1 = var1d(f1)

    # ── V2 ──────────────────────────────────────────────────────────────────
    # E_{X1}[exp(θ₂X₁/500)] over U[-1,5]: ∫₋₁⁵ exp(θ₂x/500)dx/6 = (500/(6θ₂))*(exp(θ₂/100)-exp(-θ₂/500))
    th2s = np.where(np.abs(th2_arr) > 1e-8, th2_arr, 1e-8)
    pA = (500.0 / (6.0 * th2s)) * (np.exp(th2s / 100.0) - np.exp(-th2s / 500.0))
    pA = np.where(np.abs(th2_arr) > 1e-8, pA, 1.0)
    sx1q = np.where(xq == 0, 1e-10, np.abs(xq))
    f2r = []
    for x2v in xq:
        gi = ex(xq / 6.0) * np.exp(
            -((th1_arr - 10.0 * x2v) ** 2) / (60.0 * (ex(sx1q) ** 2 + 1e-12))
        )
        f2r.append(x2v * pA + gi.mean(0) * 6.0)
    V2 = var1d(np.stack(f2r, axis=0))

    # ── V3 ──────────────────────────────────────────────────────────────────
    sx3 = np.where(xq == 0, 1e-10, xq)
    s20 = np.sqrt(20.0) * ex(sx3)
    i3 = ex(xq**2 / 6.0) * (_phi_cdf((100.0 - ph1_arr) / s20) - _phi_cdf((-20.0 - ph1_arr) / s20))
    V3 = (np.pi / 120.0) * (i3.mean(0) * 6.0) ** 2

    # ── V4 ──────────────────────────────────────────────────────────────────
    # E_{X1}[exp(θ₂X₁/500)] = (500/(6θ₂))*(exp(θ₂/100)-exp(-θ₂/500)); Var_{X4}[X4]=1/3 over U[-1,5]? No:
    # X4~U[-1,5], Var[X4+X2]=Var[X4]=(5-(-1))²/12=3; but V4=Var_{X4}[E_{~4}[Y|X4]].
    # E_{~4}[Y|X4] = X4 * E_{X1}[exp(θ₂X₁/500)] + const(X4). So V4 = b4² * Var[X4] = b4² * 3.
    # b4 = E_{X1}[exp(θ₂X₁/500)] = (500/(6θ₂))*(exp(θ₂/100)-exp(-θ₂/500))
    b4 = (500.0 / (6.0 * th2s)) * (np.exp(th2s / 100.0) - np.exp(-th2s / 500.0))
    b4 = np.where(np.abs(th2_arr) > 1e-8, b4, 1.0)
    V4 = 3.0 * b4**2

    # ── V5 = 0 ──────────────────────────────────────────────────────────────
    V5 = np.zeros_like(th1_arr)

    # ── V6 ──────────────────────────────────────────────────────────────────
    # t4 uses exp(φ₂X₇/250). E_{X7}[exp(φ₂X₇/250)] = (250/(6φ₂))*(exp(φ₂/50)-exp(-φ₂/250))
    # V6 = Var_{X6}[E_{~6}[Y|X6]] = E_{X7}[exp(φ₂X₇/250)]² * Var[X6] = b6² * 3
    ph2s = np.where(np.abs(ph2_arr) > 1e-8, ph2_arr, 1e-8)
    b6 = (250.0 / (6.0 * ph2s)) * (np.exp(ph2s / 50.0) - np.exp(-ph2s / 250.0))
    b6 = np.where(np.abs(ph2_arr) > 1e-8, b6, 1.0)
    V6 = 3.0 * b6**2

    # ── V7 ──────────────────────────────────────────────────────────────────
    V7 = var1d(4.0 * np.exp(ph2_arr * ex(xq) / 250.0))

    # ── V8 = V6 ─────────────────────────────────────────────────────────────
    V8 = V6.copy()

    return np.stack([V1, V2, V3, V4, V5, V6, V7, V8], axis=0)
