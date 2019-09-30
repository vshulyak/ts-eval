from typing import Callable, Iterable, Optional

import numpy as np
import xarray as xr

from ts_eval.utils import nanmeanw

from ..data_containers import MetricRes
from ..stats import mw_is_equal
from ..utils import filter_nan


def adaptor_point_metric(target: xr.Dataset, pred: xr.Dataset):
    """
    Adapts internal xarray data format to the format of arguments needed for metrics functions.
    Does so for point metrics.
    """
    return target.mean_.values, pred.mean_.values


def adaptor_interval_metric(target: xr.Dataset, pred: xr.Dataset):
    """
    Adapts internal xarray data format to the format of arguments needed for metrics functions.
    Does so for interval metrics.
    """
    assert hasattr(
        pred, "upper"
    ), "Upper bound PI prediction is missing in the data you supplied"
    assert hasattr(
        pred, "lower"
    ), "Lower bound PI predictio is missing in the data you supplied"
    return target.mean_.values, pred.upper.values, pred.lower.values


def absolute_metric(
    target: xr.Dataset,
    base_pred: xr.Dataset,
    other_pred: xr.Dataset,
    metric_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    points: Optional[Iterable[int]] = None,
    adaptor=adaptor_point_metric,
) -> MetricRes:
    """
    Computes absolute metric (single value)
    """
    m1 = metric_fn(*adaptor(target, other_pred))

    return MetricRes(
        overall=nanmeanw(m1),
        overall_is_same=False,
        steps=nanmeanw(m1[:, points], 0),
        steps_is_same=[False for p in points],
    )


def relative_metric(
    target: xr.Dataset,
    base_pred: xr.Dataset,
    other_pred: xr.Dataset,
    metric_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    points: Optional[Iterable[int]] = None,
    fv: bool = False,
    adaptor=adaptor_point_metric,
) -> MetricRes:
    """
    Computes relative metric of two absolute metics (two metrics – reciprocal)
    """
    assert (
        base_pred is not None
    ), "Relative metrics require a base/reference metric to be set"
    points = points or list(range(target.h))

    m1 = metric_fn(*adaptor(target, other_pred))
    m2 = metric_fn(*adaptor(target, base_pred))

    overall = nanmeanw(m1) / nanmeanw(m2)
    steps = nanmeanw(m1[:, points], 0) / nanmeanw(m2[:, points], 0)

    if fv:
        overall = 1 - overall
        steps = 1 - steps

    return MetricRes(
        overall=overall,
        overall_is_same=mw_is_equal(
            filter_nan(nanmeanw(m1, 1)), filter_nan(nanmeanw(m2, 1))
        ),
        steps=steps,
        steps_is_same=[
            mw_is_equal(filter_nan(m1[:, p]), filter_nan(m2[:, p])) for p in points
        ],
    )
