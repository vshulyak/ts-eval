import pytest

from ts_eval.forecast_strategy.naive import (
    NaiveForecastStrategy,
    SNaiveForecastStrategy,
)


H = 24


@pytest.mark.parametrize(
    "endog",
    ["dataset_1d", "dataset_1d__pd_index_ordinal", "dataset_1d__pd_index_datetime"],
    indirect=["endog"],
)
def test_fc_strategy__naive(endog):
    """
    Tests interative prediction on different input data (numpy/pandas/None)
    """

    preds_3d = NaiveForecastStrategy(endog).forecast(h=H)

    assert preds_3d.shape[0] == endog.shape[0] - H
    assert preds_3d.shape[1] == H
    assert preds_3d.shape[2] == 3


@pytest.mark.parametrize(
    "endog",
    ["dataset_1d", "dataset_1d__pd_index_ordinal", "dataset_1d__pd_index_datetime"],
    indirect=["endog"],
)
def test_fc_strategy__snaive(endog):
    """
    Tests interative prediction on different input data (numpy/pandas/None)
    """

    preds_3d = SNaiveForecastStrategy(endog).forecast(h=H)

    assert preds_3d.shape[0] == endog.shape[0] - H
    assert preds_3d.shape[1] == H
    assert preds_3d.shape[2] == 3
