# TODO: comment out
import numpy as np

import rpy2.robjects as ro

from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from ts_eval.models.naive import naive_pi, snaive_pi


forecast = importr("forecast")
stats = importr("stats")


def nr(np_arr):
    with localconverter(ro.default_converter + numpy2ri.converter):
        return ro.conversion.py2rpy(np_arr)


def test_naive(dataset_1d):

    rmodel_res = forecast.naive(nr(dataset_1d), h=10, level=ro.IntVector((95,)))
    exp_mean = np.array(list(rmodel_res.rx2["mean"]))
    exp_upper = np.array(list(rmodel_res.rx2["upper"]))
    exp_lower = np.array(list(rmodel_res.rx2["lower"]))

    fc, ub, lb = naive_pi(dataset_1d, h=10, cl=95)

    assert np.allclose(fc, exp_mean)
    assert np.allclose(ub, exp_upper)
    assert np.allclose(lb, exp_lower)


def test_snaive(dataset_1d):

    rmodel_res = forecast.snaive(
        stats.ts(nr(dataset_1d), freq=7), h=10, level=ro.IntVector((95,))
    )
    exp_mean = np.array(list(rmodel_res.rx2["mean"]))
    exp_upper = np.array(list(rmodel_res.rx2["upper"]))
    exp_lower = np.array(list(rmodel_res.rx2["lower"]))

    fc, ub, lb = snaive_pi(dataset_1d, freq=7, h=10, cl=95)

    assert np.allclose(fc, exp_mean)
    assert np.allclose(ub, exp_upper)
    assert np.allclose(lb, exp_lower)
