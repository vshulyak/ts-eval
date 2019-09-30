import xarray as xr


def all(ds: xr.Dataset):
    return ds


def weekend(ds: xr.Dataset):
    return ds.where(ds["dt.dayofweek"].isin([5, 6]), drop=True)


def weekday(ds: xr.Dataset):
    return ds.where(ds["dt.dayofweek"].isin(list(range(0, 5))), drop=True)


def mk_holiday(*holiday_args, **holiday_kwargs):
    import holidays

    country_holidays = holidays.CountryHoliday(*holiday_args, **holiday_kwargs)

    def _filter(ds: xr.Dataset):
        holiday_dates = country_holidays[
            ds.dt.to_index()[0].date() : ds.dt.to_index()[-1].date()
        ]
        return ds.sel(dt=holiday_dates)

    _filter.name = "holiday"
    return _filter


all.name = "all"
weekend.name = "weekend"
weekday.name = "weekday"
