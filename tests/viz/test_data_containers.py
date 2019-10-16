import numpy as np
import pytest

from ts_eval.viz.data_containers import xr_2d_factory, xr_3d_factory
from ts_eval.viz.utils import time_align


"""
xarray format checks
"""


def test_xr_2d_factory__xarray_fmt(dataset_2d):

    xarr = xr_2d_factory(dataset_2d)

    assert np.allclose(xarr.mean_.values, time_align(dataset_2d), equal_nan=True)
    assert xarr.sizes["dt"] == dataset_2d.shape[0] + dataset_2d.shape[1]
    assert xarr.sizes["h"] == dataset_2d.shape[1]


def test_xr_3d_factory__xarray_fmt(dataset_3d):

    xarr = xr_3d_factory(dataset_3d)

    assert np.allclose(
        xarr.upper.values, time_align(dataset_3d[:, :, 0]), equal_nan=True
    )
    assert np.allclose(
        xarr.mean_.values, time_align(dataset_3d[:, :, 1]), equal_nan=True
    )
    assert np.allclose(
        xarr.lower.values, time_align(dataset_3d[:, :, 2]), equal_nan=True
    )
    assert xarr.sizes["dt"] == dataset_3d.shape[0] + dataset_3d.shape[1]
    assert xarr.sizes["h"] == dataset_3d.shape[1]


"""
Input data checks
"""


def test_xr_2d_factory__nan(dataset_2d):
    dataset_2d = dataset_2d.copy()
    dataset_2d[:] = np.nan

    with pytest.raises(AssertionError):
        xr_2d_factory(dataset_2d)


def test_xr_2d_factory__shape(dataset_3d):

    # 3d array => 2d array
    with pytest.raises(AssertionError):
        xr_2d_factory(dataset_3d)


def test_xr_3d_factory__nan(dataset_3d):
    dataset_3d = dataset_3d.copy()
    dataset_3d[:] = np.nan

    with pytest.raises(AssertionError):
        xr_3d_factory(dataset_3d)


def test_xr_3d_factory__shape(dataset_2d):

    # 2d array => 3d array
    with pytest.raises(AssertionError):
        xr_3d_factory(dataset_2d)
