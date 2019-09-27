import numpy as np

from ..data_containers import MetricRes
from ..stats import mw_is_equal
from ..utils import filter_nan


def absolute_metric(target, base, other, metric_fn, points=None):
    """
    Computes absolute metric (single value)
    """
    m1 = metric_fn(target.mean_.values, other.mean_.values)

    return MetricRes(
        overall=np.nanmean(m1),
        overall_is_same=False,
        steps=np.nanmean(m1[:, points], 0),
        steps_is_same=[False for p in points],
    )


def relative_metric(target, base, other, metric_fn, points=None, fv=False):
    """
    Computes relative metric of two absolute metics (two metrics – reciprocal)
    """
    points = points or list(range(target.h))

    m1 = metric_fn(target.mean_.values, other.mean_.values)
    m2 = metric_fn(target.mean_.values, base.mean_.values)

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
