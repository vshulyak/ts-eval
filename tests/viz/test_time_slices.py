import holidays

from ts_eval.viz.time_slices import mk_holiday


HOLIDAY_TEST_LOCATION = dict(country="DE", prov="BW")


def test_holiday_slice(xarray_2d__index_dt):

    country_holidays = holidays.CountryHoliday(**HOLIDAY_TEST_LOCATION)
    expected_holiday_dates = country_holidays[
        xarray_2d__index_dt.dt.to_index()[0]
        .date() : xarray_2d__index_dt.dt.to_index()[-1]
        .date()
    ]

    holiday = mk_holiday(**HOLIDAY_TEST_LOCATION)
    obtained_holiday_ds = holiday(xarray_2d__index_dt)

    assert len(expected_holiday_dates) > 0  # make sure we cover at least some holidays
    assert set(expected_holiday_dates) == set(
        [d.date() for d in obtained_holiday_ds.dt.to_index()]
    )
