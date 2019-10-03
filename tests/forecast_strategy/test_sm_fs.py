import pytest
import statsmodels.api as sm

from ts_eval.forecast_strategy.sm import SMForecastStrategy


H = 24
SARIMAX_DEF = {"order": (2, 0, 0), "trend": "c"}
SARIMAX_FIT_KWARGS = {"maxiter": 100, "disp": -1}


@pytest.mark.parametrize(
    "endog, exog",
    [
        ("dataset_1d", "dataset_1d"),
        ("dataset_1d", None),
        ("dataset_1d__pd_index_ordinal", "dataset_1d__pd_index_ordinal"),
        ("dataset_1d__pd_index_ordinal", None),
        ("dataset_1d__pd_index_datetime", "dataset_1d__pd_index_datetime"),
        ("dataset_1d__pd_index_datetime", None),
    ],
    indirect=["endog", "exog"],
)
def test_fc_strategy__sm(endog, exog):
    """
    Tests interative prediction on different input data (numpy/pandas/None)
    """

    train_endog, test_endog = endog[:100], endog[100:200]
    train_exog, test_exog = (
        (endog[:100], endog[100:200]) if exog is not None else (None, None)
    )

    model = sm.tsa.SARIMAX(train_endog, **SARIMAX_DEF)
    model_fit = model.fit(**SARIMAX_FIT_KWARGS)

    preds_3d = (
        SMForecastStrategy(model_fit, SARIMAX_DEF)
        .forecast(test_endog, test_exog=None, h=H)
        .numpy()
    )

    assert preds_3d.shape[0] == test_endog.shape[0] - H
    assert preds_3d.shape[1] == H
    assert preds_3d.shape[2] == 3


@pytest.mark.parametrize(
    "endog, exog", [("dataset_1d__pd_index_datetime", None)], indirect=["endog", "exog"]
)
def test_fc_strategy__xarray_dt(endog, exog):
    """
    Tests interative prediction on different input data (numpy/pandas/None)
    """

    train_endog, test_endog = endog[:100], endog[100:200]
    train_exog, test_exog = (
        (endog[:100], endog[100:200]) if exog is not None else (None, None)
    )

    model = sm.tsa.SARIMAX(train_endog, **SARIMAX_DEF)
    model_fit = model.fit(**SARIMAX_FIT_KWARGS)

    preds_3d = (
        SMForecastStrategy(model_fit, SARIMAX_DEF)
        .forecast(test_endog, test_exog=None, h=H)
        .xarray()
    )

    assert preds_3d.dt.shape[0] == test_endog.shape[0]
    assert preds_3d.dt[0] == test_endog.index[0]
