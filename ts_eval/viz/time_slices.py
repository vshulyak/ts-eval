import xarray as xr


def all(ds: xr.Dataset):
    return ds


def weekend(ds: xr.Dataset):
    return ds.where(ds["dt.dayofweek"].isin([5, 6]), drop=True)


def weekday(ds: xr.Dataset):
    return ds.where(ds["dt.dayofweek"].isin(list(range(0, 5))), drop=True)


all.name = "all"
weekend.name = "weekend"
weekday.name = "weekday"
