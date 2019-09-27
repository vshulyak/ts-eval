import numpy as np

from ..data_containers import MetricRes
from ..stats import mw_is_equal
from ..utils import filter_nan


def se(arr1, arr2):
    """
    Computes SE – squared error
    """
    return (arr1 - arr2) ** 2


def ae(arr1, arr2):
    """
    Computes AE – absolute error
    """
    return np.abs(arr1 - arr2)


def rMIS(target, base, other, points=None, fv=False):
    """
    Computes rMIS

    TODO:
    - duplicate code with relative metric
    """
    points = points or list(range(target.h))

    m1 = compute_is(target.mean_.values, other.upper.values, other.lower.values)
    m2 = compute_is(target.mean_.values, base.upper.values, base.lower.values)

    overall = np.nanmean(m1) / np.nanmean(m2)
    steps = np.nanmean(m1[:, points], 0) / np.nanmean(m2[:, points], 0)

    if fv:
        overall = 1 - overall
        steps = 1 - steps

    return MetricRes(
        overall=overall,
        overall_is_same=mw_is_equal(
            filter_nan(np.nanmean(m1, 1)), filter_nan(np.nanmean(m2, 1))
        ),
        steps=steps,
        steps_is_same=[
            mw_is_equal(filter_nan(m1[:, p]), filter_nan(m2[:, p])) for p in points
        ],
    )


def compute_is(y, u, l, a=0.05):
    """
    Computes IS (Interval Score), used in M4 competition.
    """
    ml = u - l
    p1 = 2 / a * np.where(y < l, l - y, 0)
    p2 = 2 / a * np.where(y > u, y - u, 0)

    return ml + p1 + p2


def compute_mis(y, u, l, a=0.05):
    """
    Computes MIS (Mean Interval Score), used in M4 competition. Lower = better.

    Resources:
        - https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
        - https://stats.stackexchange.com/questions/194660/forecast-accuracy-metric-that-involves-prediction-intervals
    """
    return compute_is(y, u, l, a=a).mean()
