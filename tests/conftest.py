import numpy as np
import pandas as pd
import pytest

from ts_eval.viz.data_containers import xr_2d_factory, xr_3d_factory


def mk_1d():
    np.random.seed(0)
    return np.random.random((24 * 4 * 2,))


def mk_2d():
    np.random.seed(0)
    return np.random.random((24 * 4 * 2, 24))


def mk_3d():
    np.random.seed(0)
    return np.random.random((24 * 4 * 2, 24, 3))


@pytest.fixture(scope="module")
def dataset_1d():
    return mk_1d()


@pytest.fixture(scope="module")
def dataset_2d():
    return mk_2d()


@pytest.fixture(scope="module")
def dataset_3d():
    return mk_3d()


@pytest.fixture(scope="module")
def dataset_1d__pd_index_ordinal(dataset_1d):
    return pd.DataFrame(dataset_1d)


@pytest.fixture(scope="module")
def dataset_1d__pd_index_datetime(dataset_1d):
    return pd.DataFrame(
        dataset_1d,
        index=pd.date_range("2001-01-01", periods=dataset_1d.shape[0], freq="D"),
    )


@pytest.fixture(scope="module")
def xarray_2d__index_ordinal():
    return xr_2d_factory(mk_2d())


@pytest.fixture(scope="module")
def xarray_3d__index_ordinal():
    return xr_2d_factory(mk_2d())


@pytest.fixture(scope="module")
def xarray_2d__index_dt():
    return xr_2d_factory(mk_2d(), start_date="2001-12-01", freq="D")


@pytest.fixture(scope="module")
def xarray_3d__index_dt():
    return xr_3d_factory(mk_3d(), start_date="2001-12-01", freq="D")
