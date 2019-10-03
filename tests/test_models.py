import numpy as np
import pytest

from ts_eval.models.naive import naive_pi, snaive_pi


"""
Conditional tests depending on presence of R/rpy2 libs.
"""
try:
    import rpy2
    import rpy2.robjects as ro  # noqa  # isort:skip
    from rpy2.robjects import numpy2ri  # noqa  # isort:skip
    from rpy2.robjects.conversion import localconverter  # noqa  # isort:skip
    from rpy2.robjects.packages import importr  # noqa  # isort:skip
except ImportError:
    pytest.skip("R is not available, skipping", allow_module_level=True)

try:
    forecast = importr("forecast")
    stats = importr("stats")
except rpy2.rinterface_lib.embedded.RRuntimeError:
    pytest.skip("R-libraries not installed", allow_module_level=True)


@pytest.fixture
def endog(request):
    return request.getfixturevalue(request.param)


def nr(data):
    # pandas => numpy if needed
    np_arr = data.values.reshape(-1) if hasattr(data, "values") else data
    with localconverter(ro.default_converter + numpy2ri.converter):
        return ro.conversion.py2rpy(np_arr)


@pytest.mark.parametrize(
    "endog", ["dataset_1d", "dataset_1d__pd_index_ordinal"], indirect=["endog"]
)
def test_naive(endog):

    rmodel_res = forecast.naive(nr(endog), h=10, level=ro.IntVector((95,)))
    exp_mean = np.array(list(rmodel_res.rx2["mean"]))
    exp_upper = np.array(list(rmodel_res.rx2["upper"]))
    exp_lower = np.array(list(rmodel_res.rx2["lower"]))

    fc, ub, lb = naive_pi(endog, h=10, cl=95)

    assert np.allclose(fc, exp_mean)
    assert np.allclose(ub, exp_upper)
    assert np.allclose(lb, exp_lower)


@pytest.mark.parametrize(
    "endog", ["dataset_1d", "dataset_1d__pd_index_ordinal"], indirect=["endog"]
)
def test_snaive(endog):

    rmodel_res = forecast.snaive(
        stats.ts(nr(endog), freq=7), h=10, level=ro.IntVector((95,))
    )
    exp_mean = np.array(list(rmodel_res.rx2["mean"]))
    exp_upper = np.array(list(rmodel_res.rx2["upper"]))
    exp_lower = np.array(list(rmodel_res.rx2["lower"]))

    fc, ub, lb = snaive_pi(endog, freq=7, h=10, cl=95)

    assert np.allclose(fc, exp_mean)
    assert np.allclose(ub, exp_upper)
    assert np.allclose(lb, exp_lower)
