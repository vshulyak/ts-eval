import numpy as np

from ts_eval.viz.stats import mw_is_equal


def test_mw_is_equal__equal_arrays():
    arr = np.ones((5, 5))
    assert mw_is_equal(arr, arr)


def test_mw_is_equal__different_arrays():
    arr = np.ones((5, 5))
    assert not mw_is_equal(arr, arr + 1e-5)
