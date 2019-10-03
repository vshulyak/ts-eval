from ts_eval.viz.data_containers import xr_3d_factory


class ForecastStrategy:
    def __init__(self):
        self._forecast_result = None
        self._first_forecast_dt = None
        self._freq = None

    def forecast(self):
        raise NotImplementedError

    def numpy(self):
        if self._forecast_result is None:
            raise Exception("Call 'forecast' method first")
        return self._forecast_result

    def xarray(self):
        if self._forecast_result is None:
            raise Exception("Call 'forecast' method first")
        return xr_3d_factory(
            self._forecast_result, start_date=self._first_forecast_dt, freq=self._freq
        )
