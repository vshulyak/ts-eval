import numpy as np
import pandas as pd

from ts_eval.models.naive import naive_pi, snaive_pi

from .base import ForecastStrategy


class BaseNaiveForecastStrategy(ForecastStrategy):
    naive_fn = None

    def __init__(self, train_endog, train_test_split_index, freq=7, cl=95):
        super().__init__()
        self.endog = train_endog
        self.train_test_split_index = train_test_split_index
        self.freq = freq
        self.cl = cl

        # TODO: ugly
        if isinstance(self.endog, pd.DataFrame) and isinstance(
            self.endog.index, pd.DatetimeIndex
        ):
            dt = self.endog[self.train_test_split_index :].index[0].to_pydatetime()
            self._first_forecast_dt = dt
            self._freq = self.endog.index.freq

    def forecast(self, h, omit_last_horizon=True):
        preds_batched = []

        ht = h if omit_last_horizon else 0

        for i in range(self.endog.shape[0] - self.train_test_split_index - ht):
            slice_ = self.endog[: self.train_test_split_index + i]
            fc, ub, lb = self.naive_fn(slice_, freq=self.freq, h=h, cl=self.cl)
            preds_batched += [np.stack([ub, fc, lb], 1)]

        self._forecast_result = np.stack(preds_batched, 0)
        return self


class NaiveForecastStrategy(BaseNaiveForecastStrategy):
    """
    Naive forecast sliding stragegy
    """

    naive_fn = staticmethod(naive_pi)


class SNaiveForecastStrategy(BaseNaiveForecastStrategy):
    """
    Seasonal Naive forecast sliding stragegy
    """

    naive_fn = staticmethod(snaive_pi)
