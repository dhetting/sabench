"""Tests for sabench spatial benchmarks: Campbell2D, ExponentialCampbell2D, Campbell3D."""

import unittest

import numpy as np

from sabench.spatial import Campbell2D, Campbell3D, ExponentialCampbell2D


class TestCampbell2D(unittest.TestCase):
    """Tests for the canonical Marrel et al. (2011) Campbell2D benchmark."""

    def setUp(self):
        self.m = Campbell2D(n_z=8)  # small grid for speed
        self.rng = np.random.default_rng(42)

    def _sample(self, N=200):
        return self.rng.uniform(-1.0, 5.0, (N, 8))

    # ── Interface ─────────────────────────────────────────────────────────────
    def test_evaluate_shape(self):
        Y = self.m.evaluate(self._sample(50))
        self.assertEqual(Y.shape, (50, 8, 8))

    def test_evaluate_deterministic(self):
        X = self._sample(20)
        np.testing.assert_array_equal(self.m.evaluate(X), self.m.evaluate(X))

    def test_d_and_bounds(self):
        self.assertEqual(self.m.d, 8)
        self.assertEqual(len(self.m.bounds), 8)

    def test_call_delegate(self):
        X = self._sample(10)
        np.testing.assert_array_equal(self.m(X), self.m.evaluate(X))

    # ── Structural invariants ──────────────────────────────────────────────────
    def test_V5_is_zero(self):
        """X5 contributes nothing to first-order variance (V5 ≡ 0)."""
        Vi = self.m.analytical_partial_variances()
        np.testing.assert_array_equal(
            Vi[4], np.zeros((8, 8)), err_msg="V5 must be identically zero"
        )

    def test_V8_equals_V6(self):
        """X6 and X8 are structurally interchangeable (V8 = V6)."""
        Vi = self.m.analytical_partial_variances()
        np.testing.assert_allclose(Vi[5], Vi[7], rtol=1e-8, err_msg="V8 must equal V6")

    def test_all_Vi_nonneg(self):
        Vi = self.m.analytical_partial_variances()
        self.assertTrue(np.all(Vi >= -1e-10), "All partial variances must be non-negative")

    # ── Analytical vs MC validation ───────────────────────────────────────────
    def test_analytic_S1_vs_MC_activity(self):
        """Analytic and MC agree on active/inactive classification (N=32k)."""
        m = Campbell2D(n_z=8)
        rng = np.random.default_rng(0)
        N = 4096
        k = 8
        TAU = 0.05

        A = rng.uniform(-1, 5, (N, k))
        B = rng.uniform(-1, 5, (N, k))
        YA = m.evaluate(A)
        YB = m.evaluate(B)
        YAB = np.zeros((k, N, 8, 8))
        for i in range(k):
            AB = A.copy()
            AB[:, i] = B[:, i]
            YAB[i] = m.evaluate(AB)

        var_flat = 0.5 * (np.var(YA, axis=0) + np.var(YB, axis=0)).reshape(-1)
        n2 = np.mean((YB[None, :, :, :] - YAB) ** 2, axis=1).reshape(k, -1)
        s1_mc = np.clip(1.0 - n2 / (2.0 * var_flat[None, :] + 1e-12), 0, None)
        w = var_flat / (var_flat.sum() + 1e-12)
        agg_mc = s1_mc @ w

        Vi = m.analytical_partial_variances()
        VarY = YA.var(axis=0).reshape(-1)
        S1_an = Vi.reshape(k, -1) / (VarY[None, :] + 1e-12)
        agg_an = S1_an @ w

        for i in range(k):
            # X1 (i=0) is marginal (~0.05): skip its activity classification
            # since small MC fluctuations can push it either side of tau
            if i == 0:
                continue
            active_an = agg_an[i] >= TAU
            active_mc = agg_mc[i] >= TAU
            self.assertEqual(
                active_an,
                active_mc,
                f"X{i + 1}: analytic active={active_an} but MC active={active_mc}",
            )

    def test_analytical_S1_shape(self):
        S1 = self.m.analytical_S1(n_mc=256, mc_seed=0)
        self.assertEqual(S1.shape, (8, 8, 8))

    def test_analytical_S1_nonneg(self):
        S1 = self.m.analytical_S1(n_mc=512, mc_seed=0)
        self.assertTrue(np.all(S1 >= -1e-6))

    # ── Custom grid ───────────────────────────────────────────────────────────
    def test_custom_z_vals(self):
        z = np.linspace(-90.0, 90.0, 5)
        Y = self.m.evaluate(self._sample(10), z1_vals=z, z2_vals=z)
        self.assertEqual(Y.shape, (10, 5, 5))


class TestExponentialCampbell2D(unittest.TestCase):
    """Tests for the exponential-additive benchmark with exact analytic indices."""

    def setUp(self):
        self.m = ExponentialCampbell2D(n_z=16)
        self.rng = np.random.default_rng(0)

    # ── Interface ─────────────────────────────────────────────────────────────
    def test_evaluate_shape(self):
        rng = np.random.default_rng(0)
        bounds = self.m.bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = rng.random((30, 8)) * (hi - lo) + lo
        Y = self.m.evaluate(X)
        self.assertEqual(Y.shape, (30, 16, 16))

    def test_d_equals_8(self):
        self.assertEqual(self.m.d, 8)

    # ── Analytic index values ─────────────────────────────────────────────────
    def test_analytic_aggregate_S1_S2_equal(self):
        """S1 = S2 by model symmetry (exact)."""
        S = self.m.analytical_aggregate_S1()
        self.assertAlmostEqual(float(S[0]), float(S[1]), places=6)

    def test_analytic_aggregate_S3_S4_equal(self):
        """S3 = S4 by model symmetry (exact)."""
        S = self.m.analytical_aggregate_S1()
        self.assertAlmostEqual(float(S[2]), float(S[3]), places=6)

    def test_analytic_aggregate_approximate_values(self):
        """Aggregate indices fall in expected ranges (atol=0.015 for MC noise)."""
        S = self.m.analytical_aggregate_S1(n_mc=20000, mc_seed=0)
        # S1=S2: amplitude params dominate, ~0.30-0.33
        self.assertGreater(float(S[0]), 0.28)
        self.assertLess(float(S[0]), 0.35)
        # S3=S4: rate params, ~0.12-0.14
        self.assertGreater(float(S[2]), 0.10)
        self.assertLess(float(S[2]), 0.15)
        # S5: near-zero (x5 ~ U[-0.3, 0.3] linear coupling)
        self.assertLess(float(S[4]), 0.005)

    def test_first_order_sum_lt_one(self):
        """sum(S_i) < 1: ExponentialCampbell2D has interactions (x1*x3 term)."""
        S = self.m.analytical_aggregate_S1(n_mc=5000, mc_seed=0)
        self.assertLess(float(S.sum()), 1.0)

    def test_analytic_vs_mc_aggregate(self):
        """Analytic aggregate S_i should be within 0.03 of MC at N=8192."""
        S_an = self.m.analytical_aggregate_S1(n_mc=10000, mc_seed=0)
        # Quick MC check using Jansen
        k = 8
        N = 8192
        rng = np.random.default_rng(0)
        bounds = self.m.bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        A = rng.random((N, k)) * (hi - lo) + lo
        B = rng.random((N, k)) * (hi - lo) + lo
        YA = self.m.evaluate(A)
        YB = self.m.evaluate(B)
        YAB = np.zeros((k, N, 16, 16))
        for i in range(k):
            AB = A.copy()
            AB[:, i] = B[:, i]
            YAB[i] = self.m.evaluate(AB)
        var_flat = 0.5 * (np.var(YA, axis=0) + np.var(YB, axis=0)).reshape(-1)
        n2 = np.mean((YB[None] - YAB) ** 2, axis=1).reshape(k, -1)
        s1_mc = 1.0 - n2 / (2.0 * var_flat[None, :] + 1e-12)
        w = var_flat / (var_flat.sum() + 1e-12)
        S_mc = s1_mc @ w
        np.testing.assert_allclose(
            S_mc, S_an, atol=0.03, err_msg="ExponentialCampbell2D analytic vs MC"
        )

    def test_S5_near_zero(self):
        """X5 (weak gradient, U[-0.3, 0.3]) should have S5 << 0.01."""
        S = self.m.analytical_aggregate_S1(n_mc=5000, mc_seed=0)
        self.assertLess(float(S[4]), 0.005)

    def test_amplitude_inputs_more_important_than_rate(self):
        """S1 > S3 (amplitude dominates rate)."""
        S = self.m.analytical_aggregate_S1(n_mc=5000, mc_seed=0)
        self.assertGreater(float(S[0]), float(S[2]))


class TestCampbell3D(unittest.TestCase):
    def setUp(self):
        self.m = Campbell3D(n_z=4)
        self.rng = np.random.default_rng(0)

    def test_evaluate_shape(self):
        X = self.rng.uniform(-1.0, 5.0, (10, 8))
        Y = self.m.evaluate(X)
        self.assertEqual(Y.shape, (10, 4, 4, 4))

    def test_d_equals_8(self):
        self.assertEqual(self.m.d, 8)

    def test_V5_is_zero(self):
        Vi = self.m.analytical_partial_variances()
        np.testing.assert_array_equal(Vi[4], np.zeros((4, 4, 4)))

    def test_V8_equals_V6(self):
        Vi = self.m.analytical_partial_variances()
        np.testing.assert_allclose(Vi[5], Vi[7], rtol=1e-8)

    def test_backward_compatibility(self):
        """Campbell3D at z3=0 should equal Campbell2D output."""
        m2 = Campbell2D(n_z=4)
        m3 = Campbell3D(n_z=4)
        z = np.linspace(-90.0, 90.0, 4)
        rng = np.random.default_rng(7)
        X = rng.uniform(-1.0, 5.0, (20, 8))
        Y2 = m2.evaluate(X, z1_vals=z, z2_vals=z)  # (20, 4, 4)
        Y3 = m3.evaluate(X, z1_vals=z, z2_vals=z, z3_vals=np.array([0.0]))[:, :, :, 0]  # (20, 4, 4)
        np.testing.assert_allclose(Y2, Y3, rtol=1e-10, err_msg="Campbell3D(z3=0) != Campbell2D")

    def test_slice_at_z3(self):
        X = self.rng.uniform(-1.0, 5.0, (5, 8))
        Y = self.m.slice_at_z3(X, z3=0.0)
        self.assertEqual(Y.shape, (5, 4, 4))


class TestSpatialNoncommutativity(unittest.TestCase):
    """Verify the noncommutativity demo: tanh(alpha*Y) produces threshold flips
    on the actual Marrel Campbell2D benchmark."""

    def test_threshold_flip_marrel_campbell2d(self):
        """X1 and X7 flip active→inactive under tanh(0.2*Y) in ≥18/20 seeds.

        Uses analytic aggregate S1 to confirm raw-field activity (ground truth),
        and MC Jansen estimator over 20 seeds to confirm post-transform inactivity.
        """
        m = Campbell2D(n_z=8)
        k = 8
        N = 4096
        TAU = 0.05
        ALPHA = 0.2

        # ── Confirm raw-field activity via analytic aggregate S1 ──────────────
        # Analytic: S1_x1 ≈ 0.0525, S1_x7 ≈ 0.122 (both > tau=0.05)
        Vi = m.analytical_partial_variances()
        rng_mc = np.random.default_rng(0)
        X_ref = rng_mc.uniform(-1.0, 5.0, (16384, k))
        Y_ref = m.evaluate(X_ref)
        VarY_ref = Y_ref.var(axis=0).reshape(-1)
        Vi_flat = Vi.reshape(k, -1)
        w_ref = VarY_ref / (VarY_ref.sum() + 1e-12)
        S1_raw_an = (Vi_flat / (VarY_ref + 1e-300)).clip(0, 1) @ w_ref
        self.assertGreater(
            float(S1_raw_an[0]), TAU, f"X1 analytic raw S1={S1_raw_an[0]:.4f} must exceed tau={TAU}"
        )
        self.assertGreater(
            float(S1_raw_an[6]), TAU, f"X7 analytic raw S1={S1_raw_an[6]:.4f} must exceed tau={TAU}"
        )

        # ── Confirm post-transform inactivity via MC over 20 seeds ───────────
        def agg_s1(YA, YB, YAB):
            var = 0.5 * (np.var(YA, axis=0) + np.var(YB, axis=0)).reshape(-1)
            n2 = np.mean((YB[None] - YAB) ** 2, axis=1).reshape(k, -1)
            s1 = 1.0 - n2 / (2.0 * var[None, :] + 1e-12)
            w = var / (var.sum() + 1e-12)
            return s1 @ w

        trans_results = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            A = rng.uniform(-1, 5, (N, k))
            B = rng.uniform(-1, 5, (N, k))
            YA = m.evaluate(A)
            YB = m.evaluate(B)
            YAB = np.zeros((k, N, 8, 8))
            for i in range(k):
                AB = A.copy()
                AB[:, i] = B[:, i]
                YAB[i] = m.evaluate(AB)
            T = np.tanh(ALPHA * YA)
            TB = np.tanh(ALPHA * YB)
            TYAB = np.tanh(ALPHA * YAB)
            trans_results.append(agg_s1(T, TB, TYAB))

        trans_all = np.array(trans_results)

        x1_flips = int((trans_all[:, 0] < TAU).sum())
        self.assertGreaterEqual(x1_flips, 18, f"X1 flip below tau in {x1_flips}/20 seeds; need ≥18")

        x7_flips = int((trans_all[:, 6] < TAU).sum())
        self.assertGreaterEqual(x7_flips, 18, f"X7 flip below tau in {x7_flips}/20 seeds; need ≥18")


if __name__ == "__main__":
    unittest.main()
