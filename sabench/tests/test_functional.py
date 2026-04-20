"""Tests for all sabench functional benchmarks."""

import unittest

import numpy as np

from sabench.benchmarks.functional import (
    BoussinesqRecession,
    DampedOscillator,
    EpidemicSIR,
    HeatDiffusion1D,
    Lorenz96,
    LotkaVolterra,
    TwoCompartmentPK,
)

ALL_FUNCTIONAL = [
    BoussinesqRecession(),
    DampedOscillator(),
    LotkaVolterra(),
    EpidemicSIR(),
    HeatDiffusion1D(),
    Lorenz96(),
    TwoCompartmentPK(),
]


class TestFunctionalInterface(unittest.TestCase):
    """All functional benchmarks must satisfy the BenchmarkFunction interface."""

    def test_all_have_d(self):
        for m in ALL_FUNCTIONAL:
            self.assertIsInstance(m.d, int)
            self.assertGreater(m.d, 0, f"{m.name} has d<=0")

    def test_all_have_bounds(self):
        for m in ALL_FUNCTIONAL:
            self.assertEqual(
                len(m.bounds), m.d, f"{m.name}: len(bounds)={len(m.bounds)} != d={m.d}"
            )

    def test_all_have_name(self):
        for m in ALL_FUNCTIONAL:
            self.assertIsInstance(m.name, str)
            self.assertGreater(len(m.name), 0)

    def test_evaluate_output_first_dim_equals_N(self):
        """All functional benchmarks return (N, n_t) or (N, n_x, n_t) arrays."""
        for m in ALL_FUNCTIONAL:
            N = 20
            rng = np.random.default_rng(42)
            X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (N, m.d))
            Y = m.evaluate(X)
            self.assertEqual(Y.shape[0], N, f"{m.name}: output first dim {Y.shape[0]} != {N}")

    def test_evaluate_no_nan(self):
        for m in ALL_FUNCTIONAL:
            rng = np.random.default_rng(7)
            X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (30, m.d))
            Y = m.evaluate(X)
            self.assertFalse(np.any(np.isnan(Y)), f"{m.name} returned NaN")

    def test_evaluate_no_inf(self):
        for m in ALL_FUNCTIONAL:
            rng = np.random.default_rng(11)
            X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (30, m.d))
            Y = m.evaluate(X)
            self.assertFalse(np.any(np.isinf(Y)), f"{m.name} returned Inf")


class TestBoussinesqRecession(unittest.TestCase):
    def setUp(self):
        self.m = BoussinesqRecession()

    def test_output_shape(self):
        N, n_t = 20, 100
        m = BoussinesqRecession(n_t=n_t)
        rng = np.random.default_rng(0)
        X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (N, m.d))
        Y = m.evaluate(X)
        self.assertEqual(Y.shape, (N, n_t))

    def test_output_nonnegative(self):
        """Flow rate must be non-negative."""
        rng = np.random.default_rng(1)
        X = rng.uniform(
            [b[0] for b in self.m.bounds], [b[1] for b in self.m.bounds], (100, self.m.d)
        )
        Y = self.m.evaluate(X)
        self.assertTrue(np.all(Y >= 0))

    def test_monotone_decreasing(self):
        """Recession: flow decreases monotonically over time."""
        rng = np.random.default_rng(2)
        X = rng.uniform(
            [b[0] for b in self.m.bounds], [b[1] for b in self.m.bounds], (50, self.m.d)
        )
        Y = self.m.evaluate(X)
        diffs = np.diff(Y, axis=1)
        self.assertTrue(
            np.all(diffs <= 1e-10), "Boussinesq recession should be monotone decreasing"
        )

    def test_d_is_3(self):
        self.assertEqual(self.m.d, 3)


class TestDampedOscillator(unittest.TestCase):
    def setUp(self):
        self.m = DampedOscillator()

    def test_output_shape(self):
        N, n_t = 15, 200
        m = DampedOscillator(n_t=n_t)
        rng = np.random.default_rng(0)
        X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (N, m.d))
        Y = m.evaluate(X)
        self.assertEqual(Y.shape, (N, n_t))

    def test_zero_initial_condition(self):
        """With x0=0 and no forcing, zero initial response."""
        # Boundary case: set x0=0 in input
        rng = np.random.default_rng(3)
        X = rng.uniform(
            [b[0] for b in self.m.bounds], [b[1] for b in self.m.bounds], (10, self.m.d)
        )
        X[:, -1] = 0.0  # x0 = 0
        Y = self.m.evaluate(X)
        # Output at t=0 should be 0 (initial condition)
        np.testing.assert_allclose(Y[:, 0], 0.0, atol=1e-10)

    def test_d_is_6(self):
        self.assertEqual(self.m.d, 6)


class TestLotkaVolterra(unittest.TestCase):
    def setUp(self):
        self.m = LotkaVolterra()

    def test_output_nonnegative(self):
        """Population must be non-negative."""
        rng = np.random.default_rng(4)
        X = rng.uniform(
            [b[0] for b in self.m.bounds], [b[1] for b in self.m.bounds], (50, self.m.d)
        )
        Y = self.m.evaluate(X)
        self.assertTrue(np.all(Y >= 0))

    def test_output_shape(self):
        N, n_t = 10, 200
        m = LotkaVolterra(n_t=n_t)
        rng = np.random.default_rng(0)
        X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (N, m.d))
        Y = m.evaluate(X)
        self.assertEqual(Y.shape, (N, n_t))

    def test_d_is_4(self):
        self.assertEqual(self.m.d, 4)

    def test_oscillatory(self):
        """LV prey should oscillate: not purely monotone."""
        rng = np.random.default_rng(5)
        X = rng.uniform(
            [b[0] for b in self.m.bounds], [b[1] for b in self.m.bounds], (10, self.m.d)
        )
        Y = self.m.evaluate(X)
        for i in range(5):
            diffs = np.diff(Y[i, :])
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            self.assertGreater(sign_changes, 2, f"LotkaVolterra row {i} not oscillatory")


class TestEpidemicSIR(unittest.TestCase):
    def setUp(self):
        self.m = EpidemicSIR()

    def test_output_bounded_01(self):
        """Infected fraction I(t) in [0,1]."""
        rng = np.random.default_rng(6)
        X = rng.uniform(
            [b[0] for b in self.m.bounds], [b[1] for b in self.m.bounds], (100, self.m.d)
        )
        Y = self.m.evaluate(X)
        self.assertTrue(np.all(Y >= 0))
        self.assertTrue(np.all(Y <= 1.0 + 1e-6))

    def test_initially_small(self):
        """I(0) should be near I0 (small)."""
        rng = np.random.default_rng(7)
        X = rng.uniform(
            [b[0] for b in self.m.bounds], [b[1] for b in self.m.bounds], (50, self.m.d)
        )
        Y = self.m.evaluate(X)
        # First time point should be small (< 0.1 for all samples)
        self.assertTrue(np.all(Y[:, 0] < 0.1))

    def test_eventually_decays(self):
        """Epidemic eventually dies out: I(T) < I(peak)."""
        rng = np.random.default_rng(8)
        X = rng.uniform(
            [b[0] for b in self.m.bounds], [b[1] for b in self.m.bounds], (20, self.m.d)
        )
        Y = self.m.evaluate(X)
        peak = Y.max(axis=1)
        final = Y[:, -1]
        self.assertTrue(np.all(final <= peak + 1e-6))

    def test_d_is_3(self):
        self.assertEqual(self.m.d, 3)


class TestHeatDiffusion1D(unittest.TestCase):
    def setUp(self):
        self.m = HeatDiffusion1D(n_x=20)

    def test_output_shape(self):
        N = 10
        rng = np.random.default_rng(9)
        X = rng.uniform([b[0] for b in self.m.bounds], [b[1] for b in self.m.bounds], (N, self.m.d))
        Y = self.m.evaluate(X)
        # Output shape: (N, n_x)
        self.assertEqual(Y.shape[0], N)
        self.assertEqual(Y.shape[1], self.m.n_x)

    def test_boundary_conditions_satisfied(self):
        """T(x=0) ~ T_left, T(x=L) ~ T_right at large t_obs (near steady state)."""
        m = HeatDiffusion1D(n_x=20, t_obs=1e6)
        X = np.array([[0.05, 25.0, 50.0, 100.0]])  # alpha, T0, T_left, T_right
        Y = m.evaluate(X)  # shape (1, n_x)
        self.assertAlmostEqual(float(Y[0, 0]), 50.0, delta=2.0)
        self.assertAlmostEqual(float(Y[0, -1]), 100.0, delta=2.0)

    def test_d_is_4(self):
        self.assertEqual(self.m.d, 4)


class TestLorenz96(unittest.TestCase):
    def setUp(self):
        self.m = Lorenz96(N=8, n_t=50)

    def test_output_shape(self):
        N = 10
        rng = np.random.default_rng(10)
        X = rng.uniform([b[0] for b in self.m.bounds], [b[1] for b in self.m.bounds], (N, self.m.d))
        Y = self.m.evaluate(X)
        self.assertEqual(Y.shape, (N, 50))

    def test_d_equals_N(self):
        for n in [4, 8, 16]:
            m = Lorenz96(N=n)
            self.assertEqual(m.d, n)

    def test_butterfly_effect(self):
        """Two nearby initial conditions should diverge."""
        m = Lorenz96(N=8, n_t=100, t_max=10.0)
        X1 = np.ones((1, 8)) * 8.0
        X2 = X1.copy()
        X2[0, 0] += 1e-4
        Y1 = m.evaluate(X1)
        Y2 = m.evaluate(X2)
        final_diff = abs(float(Y1[0, -1]) - float(Y2[0, -1]))
        # After 10 time units, the difference should have grown significantly
        self.assertGreater(final_diff, 1e-3)


class TestTwoCompartmentPK(unittest.TestCase):
    def setUp(self):
        self.m = TwoCompartmentPK()

    def test_output_shape(self):
        N, n_t = 20, 100
        m = TwoCompartmentPK(n_t=n_t)
        rng = np.random.default_rng(11)
        X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (N, m.d))
        Y = m.evaluate(X)
        self.assertEqual(Y.shape, (N, n_t))

    def test_concentration_nonnegative(self):
        rng = np.random.default_rng(12)
        X = rng.uniform(
            [b[0] for b in self.m.bounds], [b[1] for b in self.m.bounds], (100, self.m.d)
        )
        Y = self.m.evaluate(X)
        self.assertTrue(np.all(Y >= 0))

    def test_concentration_eventually_decays(self):
        """Drug concentration must eventually approach zero."""
        m = TwoCompartmentPK(n_t=200, t_max=48.0)
        rng = np.random.default_rng(13)
        X = rng.uniform([b[0] for b in m.bounds], [b[1] for b in m.bounds], (50, m.d))
        Y = m.evaluate(X)
        peak = Y.max(axis=1)
        final = Y[:, -1]
        # Final concentration should be less than 50% of peak
        self.assertTrue(np.all(final < 0.5 * peak + 1e-6))

    def test_dose_scaling(self):
        """Doubling dose D should approximately double peak concentration."""
        X1 = np.array([[100.0, 0.1, 0.05, 0.05, 10.0]])
        X2 = X1.copy()
        X2[0, 0] = 200.0
        Y1 = self.m.evaluate(X1)
        Y2 = self.m.evaluate(X2)
        ratio = Y2.max() / Y1.max()
        self.assertAlmostEqual(ratio, 2.0, delta=0.01)

    def test_d_is_5(self):
        self.assertEqual(self.m.d, 5)


if __name__ == "__main__":
    unittest.main()
