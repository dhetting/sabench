"""Tests for sabench.sampling: Saltelli sample design."""

import unittest

import numpy as np

from sabench.sampling import saltelli_sample


class TestSaltelliSampler(unittest.TestCase):
    def test_output_shape(self):
        """Output shape is N*(d+2) x d."""
        X = saltelli_sample(3, [(0.0, 1.0)] * 3, N=100, seed=0)
        self.assertEqual(X.shape, (100 * 5, 3))

    def test_output_shape_d8(self):
        X = saltelli_sample(8, [(-1.0, 5.0)] * 8, N=64, seed=1)
        self.assertEqual(X.shape, (64 * 10, 8))

    def test_values_within_bounds(self):
        bounds = [(-2.0, 3.0), (0.0, 1.0), (-10.0, 10.0)]
        X = saltelli_sample(3, bounds, N=200, seed=42)
        for i, (lo, hi) in enumerate(bounds):
            self.assertTrue(np.all(X[:, i] >= lo), f"Col {i} has values below lower bound {lo}")
            self.assertTrue(np.all(X[:, i] <= hi), f"Col {i} has values above upper bound {hi}")

    def test_deterministic_with_seed(self):
        X1 = saltelli_sample(4, [(0.0, 1.0)] * 4, N=50, seed=7)
        X2 = saltelli_sample(4, [(0.0, 1.0)] * 4, N=50, seed=7)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds_differ(self):
        X1 = saltelli_sample(4, [(0.0, 1.0)] * 4, N=50, seed=0)
        X2 = saltelli_sample(4, [(0.0, 1.0)] * 4, N=50, seed=1)
        self.assertFalse(np.allclose(X1, X2))

    def test_ab_blocks_are_different(self):
        """A and B blocks should be independently drawn."""
        d = 3
        N = 200
        X = saltelli_sample(d, [(0.0, 1.0)] * d, N=N, seed=0)
        A = X[:N]
        B = X[N : 2 * N]
        self.assertFalse(np.allclose(A, B))

    def test_abi_has_correct_columns(self):
        """AB_i block should match A except column i which equals B[:,i]."""
        d = 3
        N = 100
        X = saltelli_sample(d, [(0.0, 1.0)] * d, N=N, seed=0)
        A = X[:N]
        B = X[N : 2 * N]
        for i in range(d):
            ABi = X[(2 + i) * N : (3 + i) * N]
            for j in range(d):
                if j == i:
                    np.testing.assert_array_equal(ABi[:, j], B[:, j])
                else:
                    np.testing.assert_array_equal(ABi[:, j], A[:, j])


if __name__ == "__main__":
    unittest.main()
