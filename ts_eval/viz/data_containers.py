from collections import namedtuple
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .utils import time_align


MetricRes = namedtuple(
    "MetricRes", ["overall", "overall_is_same", "steps", "steps_is_same"]
)


def xr_2d_factory(
    arr: np.ndarray, start_date: Optional[datetime] = None, freq: Optional[str] = None
) -> xr.Dataset:
    """
    Constructs a time aligned 2d-xarray
    """
    assert arr.ndim == 2, "Can work only with 2 dimentional arrays"
    assert not np.isnan(arr).any(), "The input data can't contain NaNs"

    n, h = arr.shape

    ts_index = _gen_dataset_index(arr_len=n + h, start_date=start_date, freq=freq)

    return xr.Dataset(
        {
            # indices
            "dt": ts_index,
            "h": list(range(h)),
            # series
            "mean_": (("dt", "h"), time_align(arr)),
        }
    )


def xr_3d_factory(
    arr: np.ndarray, start_date: Optional[datetime] = None, freq: Optional[str] = None
) -> xr.Dataset:
    """
    Constructs a time aligned 3d-xarray
    """
    assert arr.ndim == 3, "Can work only with 3 dimentional arrays"
    assert not np.isnan(arr).any(), "The input data can't contain NaNs"
    assert arr.shape[2] == 3, "The 2nd axis has to have dim=3 (upper, mean, lower)"

    n, h, _ = arr.shape

    ts_index = _gen_dataset_index(arr_len=n + h, start_date=start_date, freq=freq)

    return xr.Dataset(
        {
            # indices
            "dt": ts_index,
            "h": list(range(h)),
            # series
            "upper": (("dt", "h"), time_align(arr[:, :, 0])),
            "mean_": (("dt", "h"), time_align(arr[:, :, 1])),
            "lower": (("dt", "h"), time_align(arr[:, :, 2])),
        }
    )


def _gen_dataset_index(arr_len, start_date, freq):
    if start_date and freq:
        return pd.date_range(start_date, freq=freq, periods=arr_len)
    else:
        return list(range(arr_len))
