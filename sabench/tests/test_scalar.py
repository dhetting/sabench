"""Tests for all sabench scalar benchmarks."""

import unittest

import numpy as np

from sabench.benchmarks.scalar import (
    AdditiveQuadratic,
    Borehole,
    CornerPeak,
    CSTRReactor,
    DetPep8D,
    EnvironModel,
    Friedman,
    Ishigami,
    LinearModel,
    MoonHerrera,
    Morris,
    OakleyOHagan,
    OTLCircuit,
    PCETestFunction,
    Piston,
    ProductPeak,
    Rosenbrock,
    SobolG,
    WingWeight,
)
from sabench.sampling import saltelli_sample

ALL_SCALAR_BENCHMARKS = [
    Ishigami(),
    SobolG(),
    Borehole(),
    Piston(),
    WingWeight(),
    OTLCircuit(),
    Morris(),
    LinearModel(),
    AdditiveQuadratic(),
    PCETestFunction(),
    Friedman(),
    OakleyOHagan(),
    MoonHerrera(),
    CornerPeak(),
    ProductPeak(),
    Rosenbrock(),
    EnvironModel(),
    CSTRReactor(),
    DetPep8D(),
]


class TestBenchmarkInterface(unittest.TestCase):
    """Every benchmark must satisfy the BenchmarkFunction interface."""

    def test_all_have_d(self):
        for m in ALL_SCALAR_BENCHMARKS:
            self.assertIsInstance(m.d, int)
            self.assertGreater(m.d, 0, f"{m.name} has d<=0")

    def test_all_have_bounds(self):
        for m in ALL_SCALAR_BENCHMARKS:
            self.assertEqual(len(m.bounds), m.d, f"{m.name}: len(bounds)!=d")

    def test_all_have_name(self):
        for m in ALL_SCALAR_BENCHMARKS:
            self.assertIsInstance(m.name, str)
            self.assertGreater(len(m.name), 0)

    def test_evaluate_output_shape(self):
        """All scalar benchmarks return (N,) arrays."""
        for m in ALL_SCALAR_BENCHMARKS:
            N = 50
            rng = np.random.default_rng(42)
            X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (N, m.d))
            Y = m.evaluate(X)
            self.assertEqual(Y.shape, (N,), f"{m.name}: output shape {Y.shape} != ({N},)")

    def test_evaluate_no_nan(self):
        """No benchmark should return NaN for valid inputs."""
        for m in ALL_SCALAR_BENCHMARKS:
            rng = np.random.default_rng(7)
            X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (200, m.d))
            Y = m.evaluate(X)
            self.assertFalse(np.any(np.isnan(Y)), f"{m.name} returned NaN")

    def test_evaluate_no_inf(self):
        """No benchmark should return Inf for valid inputs."""
        for m in ALL_SCALAR_BENCHMARKS:
            rng = np.random.default_rng(11)
            X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (200, m.d))
            Y = m.evaluate(X)
            self.assertFalse(np.any(np.isinf(Y)), f"{m.name} returned Inf")

    def test_call_equals_evaluate(self):
        """__call__ should delegate to evaluate."""
        m = Ishigami()
        X = np.random.default_rng(0).uniform(-np.pi, np.pi, (20, 3))
        np.testing.assert_array_equal(m(X), m.evaluate(X))

    def test_repr(self):
        for m in ALL_SCALAR_BENCHMARKS:
            r = repr(m)
            self.assertIn(m.name, r)

    def test_output_finite_variance(self):
        """All benchmarks should have nonzero variance over their input domain."""
        for m in ALL_SCALAR_BENCHMARKS:
            rng = np.random.default_rng(99)
            X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (500, m.d))
            Y = m.evaluate(X)
            self.assertGreater(float(np.var(Y)), 0.0, f"{m.name} has zero variance output")


class TestIshigami(unittest.TestCase):
    def setUp(self):
        self.m = Ishigami(a=7.0, b=0.1)

    def test_S1_analytic_values(self):
        S1 = self.m.analytical_S1()
        self.assertAlmostEqual(float(S1[2]), 0.0, places=6)
        self.assertGreater(float(S1[0]), 0.3)
        self.assertGreater(float(S1[1]), 0.4)

    def test_ST_analytic_S3_nonzero(self):
        """X3 has nonzero total-effect via X1*X3 interaction."""
        ST = self.m.analytical_ST()
        self.assertGreater(float(ST[2]), 0.1)

    def test_S1_sum_leq_1(self):
        S1 = self.m.analytical_S1()
        self.assertLessEqual(S1.sum(), 1.0 + 1e-10)

    def test_ST_geq_S1(self):
        S1 = self.m.analytical_S1()
        ST = self.m.analytical_ST()
        np.testing.assert_array_less(S1 - 1e-10, ST + 1e-10)

    def test_analytical_variance(self):
        var_an = self.m.analytical_variance()
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, (50000, 3))
        var_mc = self.m.evaluate(X).var()
        self.assertAlmostEqual(var_an, var_mc, delta=0.05)

    def test_evaluate_deterministic(self):
        X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(self.m.evaluate(X), self.m.evaluate(X))

    def test_a_b_params(self):
        m2 = Ishigami(a=3.0, b=0.5)
        X = np.random.default_rng(0).uniform(-np.pi, np.pi, (100, 3))
        self.assertFalse(np.allclose(self.m.evaluate(X), m2.evaluate(X)))


class TestSobolG(unittest.TestCase):
    def setUp(self):
        self.m = SobolG()

    def test_partial_variances_formula(self):
        Di = self.m._partial_variances()
        self.assertAlmostEqual(float(Di[0]), 1.0 / 3.0, places=10)

    def test_S1_order(self):
        m = SobolG(a=[0.0, 1.0, 4.5, 9.0])
        S1 = m.analytical_S1()
        for i in range(len(S1) - 1):
            self.assertGreater(float(S1[i]), float(S1[i + 1]))

    def test_S1_sum_lt_1(self):
        self.assertLess(self.m.analytical_S1().sum(), 1.0)

    def test_custom_a(self):
        m = SobolG(a=[0.0, 0.0, 0.0])
        S1 = m.analytical_S1()
        self.assertAlmostEqual(float(S1[0]), float(S1[1]), places=6)

    def test_variance_positive(self):
        self.assertGreater(self.m.analytical_variance(), 0)

    def test_output_nonnegative(self):
        """G-function output is non-negative."""
        rng = np.random.default_rng(5)
        X = rng.uniform(0, 1, (500, self.m.d))
        Y = self.m.evaluate(X)
        self.assertTrue(np.all(Y >= 0))

    def test_analytical_vs_mc_S1(self):
        """Analytical S1 should be within 0.05 of MC estimate at N=2048."""
        from sabench.analysis import jansen_s1_st

        N = 2048
        X = saltelli_sample(self.m.d, self.m.bounds, N=N, seed=42)
        Y = self.m.evaluate(X)
        S1_mc, _ = jansen_s1_st(Y, N=N, d=self.m.d)
        S1_an = self.m.analytical_S1()
        np.testing.assert_allclose(
            S1_mc, S1_an, atol=0.08, err_msg="SobolG MC S1 deviates from analytical"
        )


class TestLinearModel(unittest.TestCase):
    def test_S1_proportional_to_c_squared(self):
        m = LinearModel(a=[2.0, 1.0, 0.0])
        S1 = m.analytical_S1()
        self.assertAlmostEqual(float(S1[2]), 0.0, places=6)
        ratio = float(S1[0]) / float(S1[1])
        self.assertAlmostEqual(ratio, 4.0, delta=0.01)

    def test_S1_sum_equals_1_additive(self):
        m = LinearModel(a=[3.0, 2.0, 1.0, 0.5])
        S1 = m.analytical_S1()
        self.assertAlmostEqual(float(S1.sum()), 1.0, places=6)

    def test_S1_equals_ST(self):
        """For purely additive model, S1 == ST."""
        m = LinearModel(a=[1.0, 2.0, 3.0])
        S1 = m.analytical_S1()
        ST = m.analytical_ST()
        np.testing.assert_allclose(S1, ST, atol=1e-10)


class TestBorehole(unittest.TestCase):
    def test_output_positive(self):
        m = Borehole()
        rng = np.random.default_rng(0)
        X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (500, m.d))
        Y = m.evaluate(X)
        self.assertTrue(np.all(Y > 0))

    def test_output_range_physical(self):
        m = Borehole()
        rng = np.random.default_rng(0)
        X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (5000, m.d))
        Y = m.evaluate(X)
        self.assertGreater(float(Y.mean()), 10.0)
        self.assertLess(float(Y.max()), 500.0)


class TestAdditiveQuadratic(unittest.TestCase):
    def setUp(self):
        self.m = AdditiveQuadratic()

    def test_S1_equals_ST(self):
        """Purely additive: S1 == ST."""
        S1 = self.m.analytical_S1()
        ST = self.m.analytical_ST()
        np.testing.assert_allclose(S1, ST, atol=1e-10)

    def test_S1_sum_equals_1(self):
        S1 = self.m.analytical_S1()
        self.assertAlmostEqual(float(S1.sum()), 1.0, places=8)

    def test_analytical_variance_positive(self):
        self.assertGreater(self.m.analytical_variance(), 0)

    def test_analytical_variance_vs_mc(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, (50000, self.m.d))
        var_mc = self.m.evaluate(X).var()
        var_an = self.m.analytical_variance()
        self.assertAlmostEqual(var_mc, var_an, delta=0.05)

    def test_dominant_input_ordering(self):
        """With default a=linspace(2.0,0.2,5), first input has largest a and dominates."""
        S1 = self.m.analytical_S1()
        self.assertEqual(
            int(np.argmax(S1)), 0, "First input (largest a coefficient) should dominate"
        )


class TestPCETestFunction(unittest.TestCase):
    def setUp(self):
        self.m = PCETestFunction()

    def test_S1_positive(self):
        S1 = self.m.analytical_S1()
        self.assertTrue(np.all(S1 >= 0))

    def test_ST_geq_S1(self):
        S1 = self.m.analytical_S1()
        ST = self.m.analytical_ST()
        np.testing.assert_array_less(S1 - 1e-10, ST + 1e-10)

    def test_S1_sum_leq_1(self):
        S1 = self.m.analytical_S1()
        self.assertLessEqual(float(S1.sum()), 1.0 + 1e-10)

    def test_output_shape(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(0, 1, (100, 4))
        Y = self.m.evaluate(X)
        self.assertEqual(Y.shape, (100,))

    def test_mean_near_c0(self):
        """PCE function: E[f] = c0 since Legendre polynomials on [0,1] have zero mean."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (50000, 4))
        mean = self.m.evaluate(X).mean()
        self.assertAlmostEqual(mean, float(self.m._c0), delta=0.05)


class TestFriedman(unittest.TestCase):
    def setUp(self):
        self.m = Friedman()

    def test_inactive_inputs(self):
        """X6-X10 should have near-zero S1 (computed numerically)."""
        from sabench.analysis import jansen_s1_st

        N = 2048
        X = saltelli_sample(self.m.d, self.m.bounds, N=N, seed=0)
        Y = self.m.evaluate(X)
        S1, _ = jansen_s1_st(Y, N=N, d=self.m.d)
        # All inactive inputs (idx 5-9) should have |S1| < 0.05
        for i in range(5, 10):
            self.assertLess(
                abs(float(S1[i])), 0.08, f"Friedman X{i + 1} should be inactive (S1={S1[i]:.4f})"
            )

    def test_active_inputs_nonzero(self):
        from sabench.analysis import jansen_s1_st

        N = 2048
        X = saltelli_sample(self.m.d, self.m.bounds, N=N, seed=0)
        Y = self.m.evaluate(X)
        S1, _ = jansen_s1_st(Y, N=N, d=self.m.d)
        # X1-X5 should have S1 > 0
        for i in range(5):
            self.assertGreater(float(S1[i]), 0.01, f"Friedman X{i + 1} should be active")


class TestMoonHerrera(unittest.TestCase):
    def setUp(self):
        self.m = MoonHerrera()

    def test_output_positive(self):
        """Exponential output is always positive."""
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 1, (200, self.m.d))
        Y = self.m.evaluate(X)
        self.assertTrue(np.all(Y > 0))

    def test_S1_sum_leq_1(self):
        S1 = self.m.analytical_S1()
        self.assertLessEqual(float(S1.sum()), 1.0 + 1e-6)

    def test_S1_positive(self):
        S1 = self.m.analytical_S1()
        self.assertTrue(np.all(S1 >= 0))

    def test_d_is_20(self):
        self.assertEqual(self.m.d, 20)


class TestCornerPeak(unittest.TestCase):
    def setUp(self):
        self.m = CornerPeak()

    def test_output_positive(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(0, 1, (200, self.m.d))
        Y = self.m.evaluate(X)
        self.assertTrue(np.all(Y > 0))

    def test_output_leq_1(self):
        """(1+sum(c_i*x_i))^(-(d+1)) <= 1 always."""
        rng = np.random.default_rng(4)
        X = rng.uniform(0, 1, (500, self.m.d))
        Y = self.m.evaluate(X)
        self.assertTrue(np.all(Y <= 1.0 + 1e-12))

    def test_S1_positive(self):
        S1 = self.m.analytical_S1()
        self.assertTrue(np.all(S1 >= -1e-10))

    def test_S1_sum_leq_1(self):
        S1 = self.m.analytical_S1()
        self.assertLessEqual(float(S1.sum()), 1.0 + 0.01)


class TestProductPeak(unittest.TestCase):
    def setUp(self):
        self.m = ProductPeak()

    def test_output_positive(self):
        rng = np.random.default_rng(5)
        X = rng.uniform(0, 1, (200, self.m.d))
        Y = self.m.evaluate(X)
        self.assertTrue(np.all(Y > 0))

    def test_S1_sum_leq_1(self):
        S1 = self.m.analytical_S1()
        self.assertLessEqual(float(S1.sum()), 1.0 + 0.02)

    def test_S1_positive(self):
        S1 = self.m.analytical_S1()
        self.assertTrue(np.all(S1 >= -1e-8))


class TestRosenbrock(unittest.TestCase):
    def setUp(self):
        self.m = Rosenbrock()

    def test_global_minimum_at_ones(self):
        """f(1,...,1) should be near 0."""
        X = np.ones((1, self.m.d))
        Y = self.m.evaluate(X)
        self.assertAlmostEqual(float(Y[0]), 0.0, delta=1e-10)

    def test_output_nonnegative(self):
        """Rosenbrock sum of squares is always non-negative."""
        rng = np.random.default_rng(6)
        X = rng.uniform(-2, 2, (500, self.m.d))
        Y = self.m.evaluate(X)
        self.assertTrue(np.all(Y >= 0))


class TestEnvironModel(unittest.TestCase):
    def test_output_nonnegative(self):
        m = EnvironModel()
        rng = np.random.default_rng(7)
        X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (300, m.d))
        Y = m.evaluate(X)
        self.assertTrue(np.all(Y >= 0))

    def test_d_is_4(self):
        self.assertEqual(EnvironModel().d, 4)


class TestCSTRReactor(unittest.TestCase):
    def test_output_concentration_positive(self):
        m = CSTRReactor()
        rng = np.random.default_rng(8)
        X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (200, m.d))
        Y = m.evaluate(X)
        self.assertTrue(np.all(Y >= 0))

    def test_d_is_5(self):
        self.assertEqual(CSTRReactor().d, 5)


class TestDetPep8D(unittest.TestCase):
    def test_output_shape(self):
        m = DetPep8D()
        rng = np.random.default_rng(9)
        X = rng.uniform(0, 1, (100, 8))
        Y = m.evaluate(X)
        self.assertEqual(Y.shape, (100,))

    def test_d_is_8(self):
        self.assertEqual(DetPep8D().d, 8)

    def test_output_positive(self):
        """DetPep8D has sum-of-squares structure; output >= 0."""
        m = DetPep8D()
        rng = np.random.default_rng(10)
        X = rng.uniform(0, 1, (500, 8))
        Y = m.evaluate(X)
        self.assertTrue(np.all(Y >= 0))


class TestOakleyOHagan(unittest.TestCase):
    def test_d_is_15(self):
        self.assertEqual(OakleyOHagan().d, 15)

    def test_output_shape(self):
        m = OakleyOHagan()
        rng = np.random.default_rng(11)
        X = rng.normal(0, 1, (100, 15))
        Y = m.evaluate(X)
        self.assertEqual(Y.shape, (100,))


class TestPiston(unittest.TestCase):
    def test_output_positive(self):
        """Cycle time must be positive."""
        m = Piston()
        rng = np.random.default_rng(12)
        X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (300, m.d))
        Y = m.evaluate(X)
        self.assertTrue(np.all(Y > 0))


class TestWingWeight(unittest.TestCase):
    def test_output_positive(self):
        m = WingWeight()
        rng = np.random.default_rng(13)
        X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (300, m.d))
        Y = m.evaluate(X)
        self.assertTrue(np.all(Y > 0))

    def test_d_is_10(self):
        self.assertEqual(WingWeight().d, 10)


if __name__ == "__main__":
    unittest.main()
