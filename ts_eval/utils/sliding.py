import numpy as np
import pandas as pd

from ts_eval.viz.data_containers import xr_2d_factory


def create_sliding_dataset(dataset, h=1):
    """
    Compute sliding window dataset (useful for target array creation)
    """
    return np.stack([dataset[i : i + h] for i in range(dataset.shape[0] - h)])


def create_sliding_dataset_xr(dataset, h=1):
    """
    Compute sliding window dataset as xarray, keeping date information if possible
    """
    is_dt_pandas = isinstance(dataset, pd.DataFrame) and isinstance(
        dataset.index, pd.DatetimeIndex
    )
    return xr_2d_factory(
        create_sliding_dataset(dataset, h=h).squeeze(),
        start_date=dataset.index[0] if is_dt_pandas else None,
        freq=dataset.index.freq if is_dt_pandas else None,
    )
