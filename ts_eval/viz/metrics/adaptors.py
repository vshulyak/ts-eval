import xarray as xr


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
