import copy

import numpy as np
import pandas as pd

from .base import ForecastStrategy


class SMForecastStrategy(ForecastStrategy):
    """
    Forecasts using statsmodels classes, updating params via kalman smoothing.
    """

    # TODO: make this stateless, so that model_fit is passed in the forecast loop
    def __init__(self, model_fit, model_kwargs):
        super().__init__()
        self.model_fit = copy.deepcopy(model_fit)
        self.model_kwargs = model_kwargs
        self.orig_params = self.model_fit.params
        self.model_class = model_fit.model.__class__

        # TODO: ugly
        if isinstance(self.model_fit.data.orig_endog, pd.DataFrame) and isinstance(
            self.model_fit.data.orig_endog.index, pd.DatetimeIndex
        ):
            dt = self.model_fit.data.orig_endog.index.shift(1)[-1].to_pydatetime()
            self._first_forecast_dt = dt
            self._freq = self.model_fit.data.orig_endog.index.freq

    def forecast(self, test_endog, test_exog, h=24, omit_last_horizon=True):
        assert test_endog.shape == test_exog.shape if test_exog is not None else True

        preds_batched = []

        ht = h if omit_last_horizon else 0

        for i in range(test_endog.shape[0] - ht):
            pred = self._predict(
                h,
                exog_for_horizon=test_exog[i : i + h]
                if test_exog is not None
                else None,
            )
            preds_batched += [pred]

            # array/dataframe have to be accessed via slice for interoperability
            self._update(
                test_endog[i : i + 1],
                test_exog[i : i + 1] if test_exog is not None else None,
            )

        self._forecast_result = np.stack(preds_batched, 0)
        return self

    def _predict(self, h, exog_for_horizon=None):
        assert h == exog_for_horizon.shape[0] if exog_for_horizon is not None else True

        # requirement by statsmodels to have 2-dim array
        exog_for_horizon = (
            exog_for_horizon.reshape(-1, 1)
            if exog_for_horizon is not None and exog_for_horizon.ndim == 1
            else exog_for_horizon
        )

        fc = self.model_fit.get_forecast(h, exog=exog_for_horizon)

        pi = fc.conf_int()
        pi = pi.values if hasattr(pi, "values") else pi
        mean = (
            fc.predicted_mean.values
            if hasattr(fc.predicted_mean, "values")
            else fc.predicted_mean
        )

        return np.stack([pi[:, 1], mean, pi[:, 0]], 1)

    def _update(self, updated_endog, updated_exog=None):
        endog = concat_safe(self.model_fit.data.orig_endog[-1:], updated_endog)
        exog = concat_safe(
            self.model_fit.data.orig_exog[-1:]
            if self.model_fit.data.orig_exog is not None
            else None,
            updated_exog,
        )

        new_model = self.model_class(endog=endog, exog=exog, **self.model_kwargs)
        new_model.initialize_known(
            self.model_fit.predicted_state[:, -2],
            self.model_fit.predicted_state_cov[:, :, -2],
        )
        new_model_fit = new_model.smooth(self.orig_params)

        self.model_fit = new_model_fit


def concat_safe(row_1, row_2):
    """
    Concats np or pd arrays
    """
    if row_1 is None or row_2 is None:
        return None
    if isinstance(row_1, pd.DataFrame):
        return pd.concat([row_1, row_2], axis=0)
    return np.vstack([row_1, row_2])
