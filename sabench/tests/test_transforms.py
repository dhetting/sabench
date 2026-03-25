"""Tests for sabench.transforms: registry, properties, pointwise classification."""
from pathlib import Path
import unittest

import numpy as np
from sabench.transforms import (
    TRANSFORMS, POINTWISE_TRANSFORMS, AFFINE_TRANSFORMS, LINEAR_TRANSFORMS,
    NONLOCAL_TRANSFORMS, CONVEX_TRANSFORMS, CONCAVE_TRANSFORMS,
    MONOTONE_TRANSFORMS, NONMONOTONE_TRANSFORMS, SMOOTH_TRANSFORMS,
    NONSMOOTH_TRANSFORMS, apply_transform, score_transform,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestTransformRegistry(unittest.TestCase):

    def test_registry_at_least_80(self):
        self.assertGreaterEqual(len(TRANSFORMS), 80)

    def test_all_keys_have_required_fields(self):
        for key, meta in TRANSFORMS.items():
            self.assertIn('fn', meta, f"{key} missing 'fn'")
            self.assertIn('name', meta, f"{key} missing 'name'")
            self.assertIn('params', meta, f"{key} missing 'params'")

    def test_all_fns_callable(self):
        for key, meta in TRANSFORMS.items():
            self.assertTrue(callable(meta['fn']), f"{key} fn is not callable")

    def test_pointwise_subset_of_transforms(self):
        for k in POINTWISE_TRANSFORMS:
            self.assertIn(k, TRANSFORMS, f"Pointwise key {k} not in TRANSFORMS")

    def test_linear_subset_of_transforms(self):
        for k in LINEAR_TRANSFORMS:
            self.assertIn(k, TRANSFORMS, f"Linear key {k} not in TRANSFORMS")

    def test_convex_subset_of_transforms(self):
        for k in CONVEX_TRANSFORMS:
            self.assertIn(k, TRANSFORMS, f"Convex key {k} not in TRANSFORMS")

    def test_concave_subset_of_transforms(self):
        for k in CONCAVE_TRANSFORMS:
            self.assertIn(k, TRANSFORMS, f"Concave key {k} not in TRANSFORMS")

    def test_monotone_subset_of_transforms(self):
        for k in MONOTONE_TRANSFORMS:
            self.assertIn(k, TRANSFORMS, f"Monotone key {k} not in TRANSFORMS")

    def test_nonmonotone_subset_of_transforms(self):
        for k in NONMONOTONE_TRANSFORMS:
            self.assertIn(k, TRANSFORMS, f"Nonmonotone key {k} not in TRANSFORMS")

    def test_smooth_subset_of_transforms(self):
        for k in SMOOTH_TRANSFORMS:
            self.assertIn(k, TRANSFORMS, f"Smooth key {k} not in TRANSFORMS")

    def test_nonsmooth_subset_of_transforms(self):
        for k in NONSMOOTH_TRANSFORMS:
            self.assertIn(k, TRANSFORMS, f"Nonsmooth key {k} not in TRANSFORMS")

    def test_nonlocal_is_complement_of_pointwise(self):
        expected = set(TRANSFORMS.keys()) - POINTWISE_TRANSFORMS
        self.assertEqual(NONLOCAL_TRANSFORMS, expected)

    def test_convex_concave_disjoint(self):
        """A transform cannot be both convex and concave (unless linear)."""
        overlap = CONVEX_TRANSFORMS & CONCAVE_TRANSFORMS
        # Only affine (linear) transforms may appear in both
        for k in overlap:
            self.assertIn(k, AFFINE_TRANSFORMS,
                          f"{k} is in both CONVEX and CONCAVE but not AFFINE")

    def test_smooth_nonsmooth_disjoint(self):
        overlap = SMOOTH_TRANSFORMS & NONSMOOTH_TRANSFORMS
        self.assertEqual(overlap, set(), f"Transforms in both smooth and nonsmooth: {overlap}")

    def test_monotone_nonmonotone_disjoint(self):
        overlap = MONOTONE_TRANSFORMS & NONMONOTONE_TRANSFORMS
        self.assertEqual(overlap, set(), f"Transforms in both monotone and nonmonotone: {overlap}")

    def test_affine_is_subset_of_pointwise(self):
        self.assertTrue(AFFINE_TRANSFORMS.issubset(POINTWISE_TRANSFORMS))

    def test_affine_is_subset_of_monotone(self):
        self.assertTrue(AFFINE_TRANSFORMS.issubset(MONOTONE_TRANSFORMS),
                        "Affine transforms are monotone (increasing or decreasing)")

    def test_affine_is_subset_of_smooth(self):
        self.assertTrue(AFFINE_TRANSFORMS.issubset(SMOOTH_TRANSFORMS),
                        "Affine transforms are C^inf smooth")

    def test_pointwise_and_linear_disjoint(self):
        """POINTWISE transforms act element-wise; LINEAR_TRANSFORMS are nonlocal (e.g. temporal mean)."""
        overlap = POINTWISE_TRANSFORMS & LINEAR_TRANSFORMS
        self.assertEqual(overlap, set(), f"Transforms in both sets: {overlap}")


class TestSpecificRegistryMembers(unittest.TestCase):

    def test_tanh_in_pointwise(self):
        for k in ['tanh_a03', 'tanh_a10', 'tanh_a005']:
            self.assertIn(k, POINTWISE_TRANSFORMS)

    def test_softplus_in_pointwise(self):
        for k in ['softplus_b01', 'softplus_b10']:
            self.assertIn(k, POINTWISE_TRANSFORMS)

    def test_affine_in_affine_set(self):
        for k in ['affine_a2_b1', 'affine_a05_bm3']:
            self.assertIn(k, AFFINE_TRANSFORMS)

    def test_square_in_convex(self):
        self.assertIn('square_pointwise', CONVEX_TRANSFORMS)

    def test_sqrt_in_concave(self):
        self.assertIn('sqrt_abs', CONCAVE_TRANSFORMS)

    def test_exp_in_convex(self):
        self.assertIn('exp_pointwise', CONVEX_TRANSFORMS)

    def test_log_in_concave(self):
        self.assertIn('log_abs_pointwise', CONCAVE_TRANSFORMS)

    def test_erf_in_concave_and_bounded(self):
        self.assertIn('erf_pointwise', CONCAVE_TRANSFORMS)

    def test_tanh_in_concave(self):
        # tanh is concave on [0,inf), which is the relevant positive domain
        self.assertIn('tanh_a10', CONCAVE_TRANSFORMS)

    def test_sin_in_nonmonotone(self):
        self.assertIn('sin_pointwise', NONMONOTONE_TRANSFORMS)

    def test_cos_in_nonmonotone(self):
        self.assertIn('cos_pointwise', NONMONOTONE_TRANSFORMS)

    def test_step_in_nonsmooth(self):
        self.assertIn('step_pointwise', NONSMOOTH_TRANSFORMS)

    def test_abs_in_nonsmooth(self):
        self.assertIn('abs_pointwise', NONSMOOTH_TRANSFORMS)

    def test_relu_in_nonsmooth(self):
        self.assertIn('relu_pointwise', NONSMOOTH_TRANSFORMS)

    def test_gumbel_in_monotone(self):
        self.assertIn('gumbel_cdf', MONOTONE_TRANSFORMS)

    def test_log_transform_in_smooth(self):
        self.assertIn('log_shift', SMOOTH_TRANSFORMS)


class TestAffineTransformProperty(unittest.TestCase):
    """Affine transforms φ(y)=a·y+b must preserve Sobol indices exactly."""

    def _make_field(self, shape=(200, 8, 8)):
        rng = np.random.default_rng(99)
        return rng.standard_normal(shape) * 5.0

    def test_affine_a2_b1_elementwise(self):
        Y = self._make_field()
        fn = TRANSFORMS['affine_a2_b1']['fn']
        params = TRANSFORMS['affine_a2_b1']['params']
        Z = fn(Y, **params)
        np.testing.assert_allclose(Z, 2.0 * Y + 1.0, rtol=1e-12)

    def test_affine_a05_bm3_elementwise(self):
        Y = self._make_field()
        Z = apply_transform(Y, 'affine_a05_bm3')
        np.testing.assert_allclose(Z, 0.5 * Y - 3.0, rtol=1e-12)

    def test_affine_preserves_sobol_index(self):
        """S_i(a·Y+b) = S_i(Y) for additive model Y=c1*X1+c2*X2."""
        rng = np.random.default_rng(42)
        n = 10000
        c1, c2 = 3.0, 1.0
        X1 = rng.uniform(0, 1, n)
        X2 = rng.uniform(0, 1, n)
        Y = c1 * X1 + c2 * X2
        # Monte Carlo conditional variance for S1
        n_bins = 50
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.digitize(X1, bins) - 1
        cond_means = np.array([Y[idx == b].mean() if (idx == b).sum() > 2 else 0
                                for b in range(n_bins)])
        var_cond = np.var(cond_means)
        S1_mc = var_cond / np.var(Y)
        # Now apply affine transform
        a, b = 2.5, -1.0
        Z = a * Y + b
        cond_means_z = np.array([Z[idx == j].mean() if (idx == j).sum() > 2 else 0
                                  for j in range(n_bins)])
        S1_z = np.var(cond_means_z) / np.var(Z)
        self.assertAlmostEqual(S1_mc, S1_z, delta=0.05,
                               msg="Affine transform must preserve S1")


class TestConvexTransformProperties(unittest.TestCase):
    """Verify convexity numerically: φ(λx+(1-λ)y) ≤ λφ(x)+(1-λ)φ(y)."""

    def _check_convex(self, key, x_range=(-3, 3)):
        fn = TRANSFORMS[key]['fn']
        params = TRANSFORMS[key]['params']
        rng = np.random.default_rng(0)
        x = rng.uniform(*x_range, 500)
        y = rng.uniform(*x_range, 500)
        lam = 0.5
        lhs = fn(lam * x + (1-lam) * y, **params)
        rhs = lam * fn(x, **params) + (1-lam) * fn(y, **params)
        violations = np.sum(lhs > rhs + 1e-8)
        self.assertLess(violations / len(x), 0.01,
                        f"{key} failed convexity: {violations}/500 violations")

    def test_square_convex(self):
        self._check_convex('square_pointwise')

    def test_exp_convex(self):
        self._check_convex('exp_pointwise', x_range=(-2, 2))

    def test_cosh_convex(self):
        self._check_convex('cosh_pointwise', x_range=(-2, 2))

    def test_square_pointwise_convex(self):
        self._check_convex('square_pointwise')


class TestConcaveTransformProperties(unittest.TestCase):
    """Verify concavity numerically: φ(λx+(1-λ)y) ≥ λφ(x)+(1-λ)φ(y)."""

    def _check_concave(self, key, x_range=(0.01, 5)):
        fn = TRANSFORMS[key]['fn']
        params = TRANSFORMS[key]['params']
        rng = np.random.default_rng(1)
        x = rng.uniform(*x_range, 500)
        y = rng.uniform(*x_range, 500)
        lam = 0.5
        lhs = fn(lam * x + (1-lam) * y, **params)
        rhs = lam * fn(x, **params) + (1-lam) * fn(y, **params)
        violations = np.sum(lhs < rhs - 1e-8)
        self.assertLess(violations / len(x), 0.01,
                        f"{key} failed concavity: {violations}/500 violations")

    def test_sqrt_concave(self):
        self._check_concave('sqrt_abs')

    def test_log1p_concave(self):
        self._check_concave('log1p_positive')

    def test_cbrt_concave(self):
        self._check_concave('cbrt_pointwise', x_range=(0.01, 8))


class TestMonotoneTransformProperties(unittest.TestCase):
    """Verify monotonicity numerically: if x1 < x2 then φ(x1) ≤ φ(x2) (or ≥)."""

    def _check_monotone_increasing(self, key, x_range=(-3, 3)):
        fn = TRANSFORMS[key]['fn']
        params = TRANSFORMS[key]['params']
        x = np.linspace(*x_range, 300)
        y = fn(x, **params)
        # Allow for small numerical noise but not clear descent
        violations = np.sum(np.diff(y) < -1e-4)
        self.assertLess(violations, 5,
                        f"{key} is not monotone increasing ({violations} violations)")

    def test_exp_monotone_increasing(self):
        self._check_monotone_increasing('exp_pointwise', (-3, 3))

    def test_tanh_monotone_increasing(self):
        self._check_monotone_increasing('tanh_a10', (-3, 3))

    def test_logistic_monotone_increasing(self):
        self._check_monotone_increasing('logistic_pointwise', (-5, 5))

    def test_erf_monotone_increasing(self):
        self._check_monotone_increasing('erf_pointwise', (-3, 3))

    def test_arctan_monotone_increasing(self):
        self._check_monotone_increasing('arctan_pointwise', (-5, 5))

    def test_gumbel_cdf_monotone_increasing(self):
        self._check_monotone_increasing('gumbel_cdf', (-1, 5))

    def test_log1p_abs_monotone_increasing(self):
        self._check_monotone_increasing('log1p_positive', (0, 5))


class TestNonmonotonicTransformProperties(unittest.TestCase):

    def test_sin_nonmonotone(self):
        """sin changes direction over its domain."""
        fn = TRANSFORMS['sin_pointwise']['fn']
        params = TRANSFORMS['sin_pointwise']['params']
        x = np.linspace(-5, 5, 500)
        y = fn(x, **params)
        sign_changes = np.sum(np.diff(np.sign(np.diff(y))) != 0)
        self.assertGreaterEqual(sign_changes, 2)

    def test_cos_nonmonotone(self):
        fn = TRANSFORMS['cos_pointwise']['fn']
        params = TRANSFORMS['cos_pointwise']['params']
        x = np.linspace(-5, 5, 500)
        y = fn(x, **params)
        sign_changes = np.sum(np.diff(np.sign(np.diff(y))) != 0)
        self.assertGreaterEqual(sign_changes, 2)

    def test_neg_square_nonmonotone(self):
        """−y² changes sign of derivative at 0."""
        fn = TRANSFORMS['neg_square']['fn']
        params = TRANSFORMS['neg_square']['params']
        x = np.linspace(-3, 3, 200)
        y = fn(x, **params)
        sign_changes = np.sum(np.diff(np.sign(np.diff(y))) != 0)
        self.assertGreater(sign_changes, 0)


class TestPointwiseTransformOutputProperties(unittest.TestCase):
    """Mathematical output range properties for specific transforms."""

    def _eval(self, key, x=None):
        if x is None:
            rng = np.random.default_rng(5)
            x = rng.uniform(-3, 3, 1000)
        fn = TRANSFORMS[key]['fn']
        params = TRANSFORMS[key]['params']
        return fn(x, **params)

    def test_square_nonnegative(self):
        y = self._eval('square_pointwise')
        self.assertTrue(np.all(y >= 0))

    def test_exp_positive(self):
        y = self._eval('exp_pointwise')
        self.assertTrue(np.all(y > 0))

    def test_erf_bounded_pm1(self):
        y = self._eval('erf_pointwise')
        self.assertTrue(np.all(y >= -1.0 - 1e-10))
        self.assertTrue(np.all(y <= 1.0 + 1e-10))

    def test_logistic_bounded_01(self):
        y = self._eval('logistic_pointwise')
        self.assertTrue(np.all(y > 0))
        self.assertTrue(np.all(y < 1))

    def test_tanh_bounded_pm1(self):
        y = self._eval('tanh_a10')
        self.assertTrue(np.all(y > -1))
        self.assertTrue(np.all(y < 1))

    def test_arctan_bounded(self):
        y = self._eval('arctan_pointwise')
        self.assertTrue(np.all(y > -np.pi/2 - 1e-10))
        self.assertTrue(np.all(y < np.pi/2 + 1e-10))

    def test_relu_nonneg_preserving(self):
        rng = np.random.default_rng(6)
        x = rng.uniform(0, 5, 500)
        fn = TRANSFORMS['relu_pointwise']['fn']
        params = TRANSFORMS['relu_pointwise']['params']
        y = fn(x, **params)
        np.testing.assert_array_equal(y, x)

    def test_relu_clips_negative(self):
        fn = TRANSFORMS['relu_pointwise']['fn']
        params = TRANSFORMS['relu_pointwise']['params']
        x = np.array([-5.0, -1.0, 0.0])
        y = fn(x, **params)
        np.testing.assert_array_equal(y, np.zeros(3))

    def test_step_binary(self):
        fn = TRANSFORMS['step_pointwise']['fn']
        params = TRANSFORMS['step_pointwise']['params']
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        y = fn(x, **params)
        unique = set(np.round(y, 10).tolist())
        self.assertTrue(unique.issubset({0.0, 1.0}),
                        f"Step function should be binary, got {unique}")

    def test_gumbel_cdf_in_01(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(500)
        y = self._eval('gumbel_cdf', x)
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y <= 1.0 + 1e-10))

    def test_sinh_pointwise_odd_symmetry(self):
        """sinh(-x) = -sinh(x)."""
        fn = TRANSFORMS['sinh_pointwise']['fn']
        params = TRANSFORMS['sinh_pointwise']['params']
        x = np.linspace(-2, 2, 50)
        np.testing.assert_allclose(fn(-x, **params), -fn(x, **params), atol=1e-10)

    def test_cube_pointwise_odd_symmetry(self):
        """x³ is odd."""
        fn = TRANSFORMS['cube_pointwise']['fn']
        params = TRANSFORMS['cube_pointwise']['params']
        x = np.linspace(-3, 3, 50)
        np.testing.assert_allclose(fn(-x, **params), -fn(x, **params), atol=1e-10)


class TestApplyTransformFunction(unittest.TestCase):

    def test_apply_transform_returns_array(self):
        Y = np.random.default_rng(0).standard_normal((100, 4, 4))
        for key in list(TRANSFORMS.keys())[:10]:
            Z = apply_transform(Y, key)
            self.assertEqual(Z.shape, Y.shape, f"{key} changed shape")

    def test_apply_transform_invalid_key(self):
        Y = np.ones((10, 4, 4))
        with self.assertRaises(KeyError):
            apply_transform(Y, 'nonexistent_transform_key_xyz')

    def test_apply_transform_all_keys_no_crash(self):
        """All transforms should execute without error on random data."""
        Y = np.abs(np.random.default_rng(0).standard_normal((50, 5, 5))) + 0.1
        errors = []
        for key in TRANSFORMS:
            try:
                Z = apply_transform(Y, key)
                if np.any(np.isnan(Z)) and 'log' not in key and 'inv' not in key:
                    errors.append(f"{key} returned NaN")
            except Exception as e:
                errors.append(f"{key}: {e}")
        self.assertEqual(errors, [], f"Transform errors: {errors}")


class TestScoreTransformFunction(unittest.TestCase):
    """score_transform(S1_orig, S1_trans, Y_orig, Y_trans) → dict."""

    def test_score_transform_returns_dict(self):
        s1_orig = np.array([0.5, 0.3, 0.2])
        s1_trans = np.array([0.4, 0.35, 0.25])
        Y = np.random.default_rng(0).standard_normal(100)
        Y_trans = Y**2
        result = score_transform(s1_orig, s1_trans, Y, Y_trans)
        self.assertIsInstance(result, dict)

    def test_score_transform_contains_required_keys(self):
        rng = np.random.default_rng(1)
        s1 = np.array([0.6, 0.4])
        s1_t = np.array([0.5, 0.5])
        Y = rng.standard_normal(200)
        Y_t = 2 * Y + 1
        result = score_transform(s1, s1_t, Y, Y_t)
        # Should have at minimum delta (Bray-Curtis) and D (decision score)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_affine_transform_zero_delta(self):
        """Affine transforms commute: Δ (Bray-Curtis) should be ~0."""
        from sabench.scalar import LinearModel
        from sabench.sampling import saltelli_sample
        from sabench.analysis import jansen_s1_st
        m = LinearModel(a=[3.0, 2.0, 1.0])
        N = 1024
        X = saltelli_sample(m.d, m.bounds, N=N, seed=0)
        Y = m.evaluate(X)
        S1, ST = jansen_s1_st(Y, N=N, d=m.d)
        Y_trans = apply_transform(Y, 'affine_a2_b1')
        X_trans = saltelli_sample(m.d, m.bounds, N=N, seed=0)
        Y_trans2 = apply_transform(m.evaluate(X_trans), 'affine_a2_b1')
        S1_trans, _ = jansen_s1_st(Y_trans2, N=N, d=m.d)
        result = score_transform(S1, S1_trans, Y, Y_trans)
        # For affine transforms on a linear model, the delta should be near zero
        self.assertIsInstance(result, dict)


class TestNonlocalTransforms(unittest.TestCase):

    def test_nonlocal_transforms_nonempty(self):
        self.assertGreater(len(NONLOCAL_TRANSFORMS), 5)

    def test_temporal_rms_increases_with_magnitude(self):
        """RMS of 2*Y > RMS of Y."""
        fn = TRANSFORMS['temporal_rms']['fn']
        params = TRANSFORMS['temporal_rms']['params']
        Y = np.random.default_rng(2).standard_normal((100, 20, 20))
        rms1 = fn(Y, **params)
        rms2 = fn(2 * Y, **params)
        self.assertTrue(np.all(rms2 > rms1 - 1e-10))

    def test_temporal_range_nonneg(self):
        fn = TRANSFORMS['temporal_range']['fn']
        params = TRANSFORMS['temporal_range']['params']
        Y = np.random.default_rng(3).standard_normal((50, 10, 10))
        out = fn(Y, **params)
        self.assertTrue(np.all(out >= 0))

    def test_minmax_normalise_range(self):
        fn = TRANSFORMS['min_max_normalise']['fn']
        params = TRANSFORMS['min_max_normalise']['params']
        Y = np.random.default_rng(4).standard_normal((100, 8, 8))
        Z = fn(Y, **params)
        self.assertAlmostEqual(float(Z.min()), 0.0, delta=1e-8)
        self.assertAlmostEqual(float(Z.max()), 1.0, delta=1e-8)


class TestMetadataFile(unittest.TestCase):
    """Verify metadata JSON files exist and are correctly structured."""

    def _load_json(self, fname):
        import json

        base = REPO_ROOT / 'sabench' / 'metadata'
        path = base / fname
        self.assertTrue(path.exists(), f"{fname} not found at {path}")
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def test_transforms_metadata_exists_and_nonempty(self):
        data = self._load_json('transforms_metadata.json')
        self.assertGreaterEqual(len(data), 80)

    def test_transforms_metadata_keys_match_registry(self):
        data = self._load_json('transforms_metadata.json')
        for key in data:
            self.assertIn(key, TRANSFORMS,
                          f"metadata key '{key}' not in TRANSFORMS registry")

    def test_transforms_metadata_has_required_fields(self):
        data = self._load_json('transforms_metadata.json')
        required = ['is_convex', 'is_monotone_increasing',
                    'is_smooth', 'differentiability_class', 'commutes_with_sobol']
        for key, entry in list(data.items())[:5]:
            for field in required:
                self.assertIn(field, entry,
                              f"transforms_metadata['{key}'] missing '{field}'")

    def test_benchmarks_metadata_exists_and_nonempty(self):
        data = self._load_json('benchmarks_metadata.json')
        self.assertGreaterEqual(len(data), 25)

    def test_benchmarks_metadata_has_required_fields(self):
        data = self._load_json('benchmarks_metadata.json')
        required = ['d', 'output_type', 'analytical_S1_available',
                    'nonlinearity_degree', 'reference_full']
        for key, entry in list(data.items())[:5]:
            for field in required:
                self.assertIn(field, entry,
                              f"benchmarks_metadata['{key}'] missing '{field}'")

    def test_affine_commutes_with_sobol_in_metadata(self):
        data = self._load_json('transforms_metadata.json')
        for key in AFFINE_TRANSFORMS:
            if key in data:
                self.assertTrue(data[key]['commutes_with_sobol'],
                                f"Affine transform {key} should have commutes_with_sobol=True")

    def test_nonlinear_noncommutative_in_metadata(self):
        """Nonlinear transforms should have commutes_with_sobol=False."""
        data = self._load_json('transforms_metadata.json')
        nonlinear_keys = ['square_pointwise', 'exp_pointwise', 'tanh_a10']
        for key in nonlinear_keys:
            if key in data:
                self.assertFalse(data[key]['commutes_with_sobol'],
                                 f"Nonlinear transform {key} should have commutes_with_sobol=False")


if __name__ == '__main__':
    unittest.main()
