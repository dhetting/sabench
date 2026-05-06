import numpy as np
import pytest

from sabench.transforms import utilities as util


def test_safe_range_1d_returns_eps_for_single_value_samples():
    y = np.array([3.14, 2.71, 1.62])
    out = util._safe_range(y, eps=1e-8)
    assert out.shape == (3,)
    assert np.allclose(out, np.full(3, 1e-8))


def test_safe_range_2d():
    y = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 5.0]])
    out = util._safe_range(y)
    assert np.allclose(out, np.array([2.0, 3.0]))


def test_safe_range_empty():
    y = np.empty((0,))
    out = util._safe_range(y)
    assert out.size == 0


def test_bc_1d_vector():
    y = np.zeros((3, 4))
    v = np.array([1, 2, 3])
    out = util._bc(v, y)
    assert out.shape == (3, 1)
    assert np.all(out[:, 0] == v)


def test_bc_scalar():
    y = np.zeros((3, 4))
    v = 5
    out = util._bc(v, y)
    assert out.shape == (3, 1)
    assert np.all(out == 5)


def test_bc_length_mismatch_raises():
    y = np.zeros((3, 4))
    v = np.array([1, 2])
    with pytest.raises(ValueError):
        util._bc(v, y)
