"""
test_transforms_expanded.py  --  Tests for the 87 new transforms added in the expansion.

Covers:
  - All new transforms run without error on scalar and 2D inputs
  - Mathematical property verification (convexity, monotonicity, smoothness)
  - Output range constraints (boundedness)
  - Symmetry properties (even/odd)
  - Property set membership consistency
"""

import unittest

import numpy as np

from sabench.transforms import (
    CONCAVE_TRANSFORMS,
    CONVEX_TRANSFORMS,
    MONOTONE_TRANSFORMS,
    NONMONOTONE_TRANSFORMS,
    NONSMOOTH_TRANSFORMS,
    POINTWISE_TRANSFORMS,
    SMOOTH_TRANSFORMS,
    TRANSFORMS,
    apply_transform,
)

RNG = np.random.default_rng(42)
Y_SCALAR = RNG.standard_normal(200)  # 1D, (200,)
Y_2D = RNG.standard_normal((50, 16))  # 2D, (50, 16)

# Keys added in the expansion
NEW_KEYS = {
    # Polynomial
    "poly4",
    "poly5",
    "poly6",
    "signed_power_p15",
    "signed_power_p05",
    "legendre_p3",
    "chebyshev_t4",
    "hermite_he2",
    "hermite_he3",
    "bernstein_b3",
    # Sigmoid/activation
    "atan2pi",
    "gompertz_cdf",
    "algebraic_sigmoid",
    "swish",
    "mish",
    "selu",
    "softsign",
    "bent_identity",
    "hard_sigmoid",
    "hard_tanh",
    # Oscillatory
    "sinc",
    "sin_squared",
    "cos_squared",
    "damped_sin",
    "sawtooth",
    "square_wave",
    "double_sin",
    "sin_cos_product",
    # Threshold/piecewise
    "soft_threshold",
    "hard_threshold",
    "ramp",
    "spike_gaussian",
    "breakpoint",
    "hockey_stick",
    "deadzone",
    "bimodal_flip",
    "donut",
    # Variance-stabilising
    "anscombe",
    "freeman_tukey",
    "asinh_vst",
    "modulus_lam05",
    "dual_power_lam03",
    "log2_shift",
    "log10_shift",
    # Curvature extremes
    "exp_neg_sq",
    "exp_pos_sq",
    "inverse_sq",
    "log_log",
    "power_exp",
    "gev_cdf",
    "pareto_tail",
    "log_logistic_cdf",
    # Financial
    "var_q95",
    "cvar_q95",
    "sharpe_proxy",
    "drawdown",
    "fold_change",
    "excess_return",
    # Ecological
    "hellinger",
    "chord_normalise",
    "relative_abundance",
    "log_ratio",
    # Climate
    "anomaly_pct",
    "bias_correction",
    "quantile_delta",
    "growing_degree_days",
    "std_precip_idx",
    # Hydrology
    "nash_sutcliffe",
    "pot_log",
    "log_flow",
    # Medical
    "hill_response",
    "log_auc",
    "emax_model",
    # Engineering
    "von_mises_stress",
    "safety_factor",
    "cumulative_damage",
    "stress_life",
    # Statistical summary
    "sample_variance",
    "sample_skewness",
    "sample_kurtosis",
    "percentile_q10",
    "percentile_q90",
    "iqr",
    # Information theory
    "negentropy_proxy",
    "wasserstein_proxy",
    "energy_distance",
    "renyi_entropy_a2",
}


class TestNewTransformsRegistry(unittest.TestCase):
    """All new keys are registered and have required fields."""

    def test_all_new_keys_in_registry(self):
        missing = NEW_KEYS - set(TRANSFORMS.keys())
        self.assertEqual(missing, set(), f"Missing new keys: {missing}")

    def test_new_keys_have_required_fields(self):
        for k in NEW_KEYS:
            t = TRANSFORMS[k]
            self.assertIn("fn", t, f"{k} missing 'fn'")
            self.assertIn("params", t, f"{k} missing 'params'")
            self.assertIn("category", t, f"{k} missing 'category'")
            self.assertIn("name", t, f"{k} missing 'name'")


class TestNewTransformsRun(unittest.TestCase):
    """All new transforms execute without error on 1D and 2D inputs."""

    def test_scalar_input_runs(self):
        for key in NEW_KEYS:
            with self.subTest(key=key):
                result = apply_transform(Y_SCALAR, key)
                self.assertEqual(
                    len(result),
                    len(Y_SCALAR),
                    f"{key}: output length mismatch ({len(result)} vs {len(Y_SCALAR)})",
                )

    def test_2d_input_runs(self):
        skip_2d = {"drawdown"}  # requires 2D but flattening behaviour differs
        for key in NEW_KEYS - skip_2d:
            with self.subTest(key=key):
                try:
                    result = apply_transform(Y_2D, key)
                    self.assertEqual(result.shape[0], Y_2D.shape[0])
                except Exception as e:
                    self.fail(f"{key} raised {e} on 2D input")

    def test_no_all_nan_output(self):
        """Transforms should not produce all-NaN output (may have some NaN/Inf)."""
        for key in NEW_KEYS:
            with self.subTest(key=key):
                result = apply_transform(Y_SCALAR, key)
                finite_frac = np.isfinite(result).mean()
                self.assertGreater(
                    finite_frac,
                    0.5,
                    f"{key}: more than half of outputs are non-finite (frac finite={finite_frac:.2f})",
                )


class TestBoundedTransforms(unittest.TestCase):
    """Bounded transforms stay within expected ranges."""

    def _check_bounded(self, key, lo, hi):
        result = apply_transform(Y_SCALAR, key)
        result = result[np.isfinite(result)]
        self.assertGreaterEqual(result.min(), lo - 1e-9, f"{key} below lower bound {lo}")
        self.assertLessEqual(result.max(), hi + 1e-9, f"{key} above upper bound {hi}")

    def test_atan2pi_bounded(self):
        self._check_bounded("atan2pi", -1, 1)

    def test_algebraic_sigmoid_bounded(self):
        self._check_bounded("algebraic_sigmoid", -1, 1)

    def test_softsign_bounded(self):
        self._check_bounded("softsign", -1, 1)

    def test_hard_tanh_bounded(self):
        self._check_bounded("hard_tanh", -1, 1)

    def test_hard_sigmoid_bounded(self):
        self._check_bounded("hard_sigmoid", 0, 1)

    def test_sin_squared_bounded(self):
        self._check_bounded("sin_squared", 0, 1)

    def test_cos_squared_bounded(self):
        self._check_bounded("cos_squared", 0, 1)

    def test_spike_gaussian_bounded(self):
        self._check_bounded("spike_gaussian", 0, 1)

    def test_exp_neg_sq_bounded(self):
        self._check_bounded("exp_neg_sq", 0, 1)

    def test_ramp_bounded(self):
        self._check_bounded("ramp", -1, 1)

    def test_gompertz_bounded(self):
        self._check_bounded("gompertz_cdf", 0, 1)

    def test_gev_cdf_bounded(self):
        result = apply_transform(Y_SCALAR, "gev_cdf")
        result = result[np.isfinite(result)]
        self.assertTrue(np.all((result >= 0) & (result <= 1)), "gev_cdf output out of [0,1]")

    def test_pareto_tail_bounded(self):
        result = apply_transform(Y_SCALAR, "pareto_tail")
        result = result[np.isfinite(result)]
        self.assertTrue(np.all((result >= 0) & (result <= 1)), "pareto_tail output out of [0,1]")

    def test_log_logistic_bounded(self):
        result = apply_transform(Y_SCALAR, "log_logistic_cdf")
        result = result[np.isfinite(result)]
        self.assertTrue(
            np.all((result >= 0) & (result <= 1)), "log_logistic_cdf output out of [0,1]"
        )

    def test_hill_response_bounded(self):
        result = apply_transform(Y_SCALAR, "hill_response")
        result = result[np.isfinite(result)]
        self.assertTrue(np.all((result >= 0) & (result <= 1)), "hill_response output out of [0,1]")

    def test_bernstein_b3_bounded(self):
        result = apply_transform(Y_SCALAR, "bernstein_b3")
        result = result[np.isfinite(result)]
        self.assertTrue(np.all(result >= -1e-9), f"bernstein_b3 negative: {result.min()}")

    def test_anscombe_nonnegative(self):
        result = apply_transform(Y_SCALAR, "anscombe")
        result = result[np.isfinite(result)]
        self.assertTrue(np.all(result >= -1e-9), "anscombe produced negative values")

    def test_freeman_tukey_nonnegative(self):
        result = apply_transform(Y_SCALAR, "freeman_tukey")
        result = result[np.isfinite(result)]
        self.assertTrue(np.all(result >= -1e-9), "freeman_tukey produced negative values")


class TestMonotoneTransforms(unittest.TestCase):
    """Monotone transforms preserve ordering."""

    def _check_monotone(self, key):
        Y_sorted = np.sort(Y_SCALAR)
        result = apply_transform(Y_sorted, key)
        result = result[np.isfinite(result)]
        diffs = np.diff(result)
        violations = np.sum(diffs < -1e-4)
        frac = violations / max(len(diffs), 1)
        self.assertLess(
            frac, 0.05, f"{key}: {violations}/{len(diffs)} monotone violations ({frac:.1%})"
        )

    def test_atan2pi_monotone(self):
        self._check_monotone("atan2pi")

    def test_algebraic_sigmoid_monotone(self):
        self._check_monotone("algebraic_sigmoid")

    def test_softsign_monotone(self):
        self._check_monotone("softsign")

    def test_bent_identity_monotone(self):
        self._check_monotone("bent_identity")

    def test_anscombe_monotone(self):
        self._check_monotone("anscombe")

    def test_freeman_tukey_monotone(self):
        self._check_monotone("freeman_tukey")

    def test_asinh_vst_monotone(self):
        self._check_monotone("asinh_vst")

    def test_modulus_lam05_monotone(self):
        self._check_monotone("modulus_lam05")

    def test_log2_shift_monotone(self):
        self._check_monotone("log2_shift")

    def test_log10_shift_monotone(self):
        self._check_monotone("log10_shift")

    def test_log_log_monotone(self):
        self._check_monotone("log_log")

    def test_signed_power_p15_monotone(self):
        self._check_monotone("signed_power_p15")

    def test_hockey_stick_nondecreasing(self):
        self._check_monotone("hockey_stick")

    def test_growing_degree_days_nondecreasing(self):
        self._check_monotone("growing_degree_days")

    def test_selu_monotone(self):
        self._check_monotone("selu")

    def test_gompertz_monotone(self):
        self._check_monotone("gompertz_cdf")


class TestSymmetryProperties(unittest.TestCase):
    """Even/odd symmetry checks."""

    def test_poly4_even(self):
        Y = np.array([1.0, 2.0, 3.0]) * 0.1
        pos = apply_transform(Y, "poly4")
        neg = apply_transform(-Y, "poly4")
        np.testing.assert_allclose(pos, neg, rtol=1e-6, err_msg="poly4 should be even")

    def test_poly5_odd(self):
        Y = np.array([1.0, 2.0, 3.0]) * 0.1
        pos = apply_transform(Y, "poly5")
        neg = apply_transform(-Y, "poly5")
        np.testing.assert_allclose(pos, -neg, rtol=1e-6, err_msg="poly5 should be odd")

    def test_atan2pi_odd(self):
        Y = np.array([0.5, 1.0, 2.0])
        pos = apply_transform(Y, "atan2pi")
        neg = apply_transform(-Y, "atan2pi")
        np.testing.assert_allclose(pos, -neg, rtol=1e-6, err_msg="atan2pi should be odd")

    def test_algebraic_sigmoid_odd(self):
        Y = np.array([0.5, 1.0, 2.0])
        pos = apply_transform(Y, "algebraic_sigmoid")
        neg = apply_transform(-Y, "algebraic_sigmoid")
        np.testing.assert_allclose(pos, -neg, rtol=1e-6, err_msg="algebraic_sigmoid should be odd")

    def test_asinh_vst_odd(self):
        Y = np.array([0.5, 1.0, 2.0])
        pos = apply_transform(Y, "asinh_vst")
        neg = apply_transform(-Y, "asinh_vst")
        np.testing.assert_allclose(pos, -neg, rtol=1e-6, err_msg="asinh_vst should be odd")

    def test_exp_neg_sq_even(self):
        Y = np.array([0.5, 1.0, 2.0])
        pos = apply_transform(Y, "exp_neg_sq")
        neg = apply_transform(-Y, "exp_neg_sq")
        np.testing.assert_allclose(pos, neg, rtol=1e-6, err_msg="exp_neg_sq should be even")

    def test_inverse_sq_even(self):
        Y = np.array([0.5, 1.0, 2.0])
        pos = apply_transform(Y, "inverse_sq")
        neg = apply_transform(-Y, "inverse_sq")
        np.testing.assert_allclose(pos, neg, rtol=1e-6, err_msg="inverse_sq should be even")


class TestConvexTransforms(unittest.TestCase):
    """Convexity via midpoint inequality: phi((a+b)/2) <= (phi(a)+phi(b))/2."""

    def _check_convex(self, key, Y=None):
        if Y is None:
            Y = np.linspace(-2, 2, 50)
        n = len(Y) - 1
        mid = 0.5 * (Y[:-1] + Y[1:])
        f_lo = apply_transform(Y[:-1], key)
        f_hi = apply_transform(Y[1:], key)
        f_mid = apply_transform(mid, key)
        violations = np.sum(f_mid > 0.5 * (f_lo + f_hi) + 1e-6)
        self.assertLess(violations / n, 0.1, f"{key}: {violations}/{n} convexity violations")

    def test_poly4_convex(self):
        self._check_convex("poly4")

    def test_poly6_convex(self):
        self._check_convex("poly6")

    def test_exp_pos_sq_convex(self):
        self._check_convex("exp_pos_sq", np.linspace(-1, 1, 50))

    def test_hockey_stick_convex(self):
        self._check_convex("hockey_stick")

    def test_inverse_sq_has_maximum_at_zero(self):
        """1/(y^2+eps) has a maximum at y=0 so is NOT globally convex (concave near 0)."""
        import numpy as np

        from sabench.transforms.mathematical import t_inverse_sq

        # Value at 0 should be largest for nearby points
        Y_zero = np.array([0.0])
        Y_away = np.array([1.0, 2.0])
        val0 = t_inverse_sq(Y_zero)[0]
        vals = t_inverse_sq(Y_away)
        for v in vals:
            self.assertLess(v, val0, msg="1/(y^2+eps) should peak at y=0")


class TestNonmonotonicTransforms(unittest.TestCase):
    """Non-monotone transforms should not be globally monotone."""

    def _check_nonmonotone(self, key):
        Y = np.linspace(-5, 5, 200)
        result = apply_transform(Y, key)
        diffs = np.diff(result)
        n_pos = np.sum(diffs > 1e-6)
        n_neg = np.sum(diffs < -1e-6)
        self.assertGreater(
            min(n_pos, n_neg),
            3,
            f"{key}: expected non-monotone but only {min(n_pos, n_neg)} direction changes",
        )

    def test_poly4_nonmonotone(self):
        self._check_nonmonotone("poly4")

    def test_hermite_he3_nonmonotone(self):
        self._check_nonmonotone("hermite_he3")

    def test_sin_squared_nonmonotone(self):
        self._check_nonmonotone("sin_squared")

    def test_cos_squared_nonmonotone(self):
        self._check_nonmonotone("cos_squared")

    def test_exp_neg_sq_nonmonotone(self):
        self._check_nonmonotone("exp_neg_sq")

    def test_swish_nonmonotone(self):
        self._check_nonmonotone("swish")


class TestSpecialOutputProperties(unittest.TestCase):
    """Check specific output properties of selected new transforms."""

    def test_sin_squared_plus_cos_squared_near_one(self):
        """sin^2 + cos^2 = 1 at matched frequency."""
        freq = 0.5
        Y = np.linspace(-5, 5, 100)
        from sabench.transforms.pointwise import t_cos_squared, t_sin_squared

        s2 = t_sin_squared(Y, freq=freq)
        c2 = t_cos_squared(Y, freq=freq)
        np.testing.assert_allclose(s2 + c2, 1.0, atol=1e-10, err_msg="sin^2 + cos^2 != 1")

    def test_soft_threshold_zero_in_band(self):
        """Soft threshold maps (-lam, lam) to 0."""
        Y = np.linspace(-0.5, 0.5, 50)
        from sabench.transforms.nonlinear import t_soft_threshold

        result = t_soft_threshold(Y, lam=1.0)
        np.testing.assert_allclose(
            result, 0.0, atol=1e-9, err_msg="soft_threshold should be 0 for |y|<lam"
        )

    def test_hockey_stick_zero_below_bp(self):
        """Hockey stick returns 0 below breakpoint."""
        Y = np.linspace(-5, 0, 50)
        from sabench.transforms.nonlinear import t_hockey_stick

        result = t_hockey_stick(Y, bp=0.0)
        np.testing.assert_allclose(
            result, 0.0, atol=1e-9, err_msg="hockey_stick should be 0 for y<bp"
        )

    def test_deadzone_zero_in_band(self):
        """Deadzone returns 0 in (-hw, hw)."""
        Y = np.linspace(-0.5, 0.5, 50)
        from sabench.transforms.nonlinear import t_deadzone

        result = t_deadzone(Y, half_width=1.0)
        np.testing.assert_allclose(
            result, 0.0, atol=1e-9, err_msg="deadzone should be 0 for |y|<half_width"
        )

    def test_hermite_he2_min_at_zero(self):
        """He2(u) = u^2 - 1 has minimum at u=0."""
        from sabench.transforms.mathematical import t_hermite_he2

        Y_near_zero = np.array([0.0])
        Y_away = np.array([0.5, 1.0])
        val_zero = t_hermite_he2(Y_near_zero, scale=1.0)[0]
        vals_away = t_hermite_he2(Y_away, scale=1.0)
        self.assertAlmostEqual(val_zero, -1.0, places=10, msg="He2(0) = 0^2 - 1 = -1")
        for v in vals_away:
            self.assertGreater(v, val_zero, msg="He2 should be > He2(0) for y!=0")

    def test_anscombe_variance_stabilises(self):
        """Anscombe approx. makes std ~ 1 for Poisson-like counts."""
        from sabench.transforms.statistical import t_anscombe

        counts = np.maximum(np.round(RNG.exponential(scale=10, size=500)), 0)
        result = t_anscombe(counts)
        # std of transformed should be close to 1 (roughly)
        std_after = np.std(result)
        self.assertLess(std_after, 5.0, msg=f"Anscombe output std too large: {std_after:.2f}")

    def test_excess_return_zero_mean(self):
        """Excess return has zero sample mean."""
        from sabench.transforms.financial import t_excess_return

        result = t_excess_return(Y_SCALAR)
        np.testing.assert_allclose(
            result.mean(), 0.0, atol=1e-9, err_msg="excess_return should have zero mean"
        )

    def test_relative_abundance_sums_to_one(self):
        """Relative abundance sums to 1 per sample."""
        from sabench.transforms.ecological import t_relative_abundance

        Y_pos = np.abs(Y_2D) + 0.1
        result = t_relative_abundance(Y_pos)
        row_sums = result.reshape(len(Y_pos), -1).sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-9, err_msg="relative_abundance should sum to 1 per sample"
        )

    def test_growing_degree_days_nonnegative(self):
        """GDD output is always >= 0."""
        from sabench.transforms.environmental import t_growing_degree_days

        result = t_growing_degree_days(Y_SCALAR, base=0.0)
        self.assertTrue(np.all(result >= -1e-12), "GDD should be non-negative")

    def test_drawdown_nonpositive(self):
        """Drawdown output is always <= 0."""
        from sabench.transforms.financial import t_drawdown

        result = t_drawdown(Y_2D)
        self.assertTrue(np.all(result <= 1e-12), "Drawdown should be <= 0")


class TestPropertySetsIncludeNewTransforms(unittest.TestCase):
    """Selected new transforms appear in the correct property sets."""

    def test_polynomial_convex_in_convex_set(self):
        for k in ["poly4", "poly6", "exp_pos_sq", "hockey_stick"]:
            self.assertIn(k, CONVEX_TRANSFORMS, f"{k} should be in CONVEX_TRANSFORMS")

    def test_variance_stabilising_in_concave_set(self):
        for k in ["anscombe", "freeman_tukey", "asinh_vst", "log2_shift", "log10_shift"]:
            self.assertIn(k, CONCAVE_TRANSFORMS, f"{k} should be in CONCAVE_TRANSFORMS")

    def test_monotone_new_in_monotone_set(self):
        for k in [
            "atan2pi",
            "algebraic_sigmoid",
            "softsign",
            "gompertz_cdf",
            "anscombe",
            "log_flow",
            "hill_response",
        ]:
            self.assertIn(k, MONOTONE_TRANSFORMS, f"{k} should be in MONOTONE_TRANSFORMS")

    def test_oscillatory_in_nonmonotone(self):
        for k in ["sin_squared", "cos_squared", "double_sin", "sawtooth"]:
            self.assertIn(k, NONMONOTONE_TRANSFORMS, f"{k} should be in NONMONOTONE_TRANSFORMS")

    def test_smooth_new_in_smooth_set(self):
        for k in [
            "poly4",
            "poly5",
            "poly6",
            "atan2pi",
            "algebraic_sigmoid",
            "sinc",
            "sin_squared",
            "anscombe",
            "asinh_vst",
        ]:
            self.assertIn(k, SMOOTH_TRANSFORMS, f"{k} should be in SMOOTH_TRANSFORMS")

    def test_discontinuous_in_nonsmooth_set(self):
        for k in ["square_wave", "hard_threshold", "sawtooth"]:
            self.assertIn(k, NONSMOOTH_TRANSFORMS, f"{k} should be in NONSMOOTH_TRANSFORMS")

    def test_pointwise_new_in_pointwise_set(self):
        for k in [
            "poly4",
            "poly5",
            "atan2pi",
            "swish",
            "mish",
            "sinc",
            "sin_squared",
            "soft_threshold",
            "spike_gaussian",
            "growing_degree_days",
        ]:
            self.assertIn(k, POINTWISE_TRANSFORMS, f"{k} should be in POINTWISE_TRANSFORMS")

    def test_financial_not_pointwise(self):
        for k in ["var_q95", "cvar_q95", "sharpe_proxy", "excess_return"]:
            self.assertNotIn(k, POINTWISE_TRANSFORMS, f"{k} should NOT be in POINTWISE_TRANSFORMS")

    def test_ecological_not_pointwise(self):
        for k in ["hellinger", "chord_normalise", "relative_abundance"]:
            self.assertNotIn(k, POINTWISE_TRANSFORMS, f"{k} should NOT be in POINTWISE_TRANSFORMS")


class TestTotalTransformCount(unittest.TestCase):
    def test_at_least_170_transforms(self):
        self.assertGreaterEqual(
            len(TRANSFORMS), 170, f"Expected >= 170 transforms, got {len(TRANSFORMS)}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
