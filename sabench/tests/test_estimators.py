"""Tests for sabench.analysis: Jansen estimator."""

import unittest

import numpy as np

from sabench.analysis import first_order, jansen_s1_st, total_effect
from sabench.sampling import saltelli_sample
from sabench.scalar import Ishigami, LinearModel, SobolG


class TestJansenEstimator(unittest.TestCase):
    def _run_jansen(self, model, N=8192, seed=0):
        X = saltelli_sample(model.d, model.bounds, N=N, seed=seed)
        Y = model.evaluate(X)
        return jansen_s1_st(Y, N=N, d=model.d)

    # ── Output shape ──────────────────────────────────────────────────────────
    def test_scalar_output_shape(self):
        model = Ishigami()
        S1, ST = self._run_jansen(model, N=512)
        self.assertEqual(S1.shape, (3,))
        self.assertEqual(ST.shape, (3,))

    def test_multi_output_shape(self):
        """For 2-column output, shapes are (d, 2)."""
        N = 256
        d = 3
        rng = np.random.default_rng(0)
        Y = np.column_stack([rng.standard_normal(N * (d + 2))] * 2)
        S1, ST = jansen_s1_st(Y, N=N, d=d)
        self.assertEqual(S1.shape, (3, 2))
        self.assertEqual(ST.shape, (3, 2))

    # ── Clipping ─────────────────────────────────────────────────────────────
    def test_clipping_default(self):
        S1, ST = self._run_jansen(Ishigami(), N=512)
        self.assertTrue(np.all(S1 >= 0) and np.all(S1 <= 1))
        self.assertTrue(np.all(ST >= 0) and np.all(ST <= 1))

    def test_no_clipping(self):
        """With clip=False, noisy near-zero values may go negative."""
        # a=[2,1,0]: X3 has zero effect → noisy near-zero S1
        model = LinearModel(a=[2.0, 1.0, 0.0])
        X = saltelli_sample(model.d, model.bounds, N=256, seed=0)
        Y = model.evaluate(X)
        S1, ST = jansen_s1_st(Y, N=256, d=model.d, clip=False)
        # Just check shape and dtype
        self.assertEqual(S1.shape, (3,))

    # ── Convenience aliases ───────────────────────────────────────────────────
    def test_first_order_alias(self):
        model = Ishigami()
        X = saltelli_sample(model.d, model.bounds, N=1024, seed=0)
        Y = model.evaluate(X)
        S1_alias = first_order(Y, N=1024, d=model.d)
        S1, _ = jansen_s1_st(Y, N=1024, d=model.d)
        np.testing.assert_array_equal(S1_alias, S1)

    def test_total_effect_alias(self):
        model = Ishigami()
        X = saltelli_sample(model.d, model.bounds, N=1024, seed=0)
        Y = model.evaluate(X)
        ST_alias = total_effect(Y, N=1024, d=model.d)
        _, ST = jansen_s1_st(Y, N=1024, d=model.d)
        np.testing.assert_array_equal(ST_alias, ST)

    # ── Accuracy against known analytic values ────────────────────────────────
    def test_ishigami_S1_accuracy(self):
        """MC estimates should be within 0.02 of analytic for N=8192."""
        model = Ishigami()
        S1_an = model.analytical_S1()  # [0.314, 0.442, 0.0]
        S1_mc, _ = self._run_jansen(model, N=8192)
        np.testing.assert_allclose(S1_mc, S1_an, atol=0.02, err_msg="Ishigami S1 MC vs analytic")

    def test_ishigami_ST_accuracy(self):
        model = Ishigami()
        ST_an = model.analytical_ST()
        _, ST_mc = self._run_jansen(model, N=8192)
        np.testing.assert_allclose(ST_mc, ST_an, atol=0.03, err_msg="Ishigami ST MC vs analytic")

    def test_sobol_g_S1_accuracy(self):
        """SobolG d=4 with well-separated a values."""
        model = SobolG(a=[0.0, 1.0, 4.5, 9.0])
        S1_an = model.analytical_S1()
        S1_mc, _ = self._run_jansen(model, N=8192)
        np.testing.assert_allclose(S1_mc, S1_an, atol=0.02, err_msg="SobolG S1 MC vs analytic")

    def test_linear_model_S1(self):
        """Linear model: S_i = c_i^2 * Var_i / Var_Y."""
        model = LinearModel(a=[2.0, 1.0, 0.5])
        S1_an = model.analytical_S1()
        S1_mc, _ = self._run_jansen(model, N=8192)
        np.testing.assert_allclose(S1_mc, S1_an, atol=0.02, err_msg="LinearModel S1")

    def test_S1_sum_leq_1(self):
        """Sum of first-order indices ≤ 1 for any model."""
        for model in [Ishigami(), SobolG()]:
            S1, _ = self._run_jansen(model, N=2048)
            self.assertLessEqual(
                S1.sum(),
                1.05,  # small MC slack
                f"S1 sum > 1 for {model.name}",
            )

    def test_ST_geq_S1(self):
        """ST_i ≥ S1_i (total effect ≥ first-order)."""
        S1, ST = self._run_jansen(Ishigami(), N=8192)
        self.assertTrue(np.all(ST >= S1 - 0.02), "ST should be >= S1 for Ishigami")


if __name__ == "__main__":
    unittest.main()
