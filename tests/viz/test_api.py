from ts_eval.viz import metrics, time_slices
from ts_eval.viz.api import ts_inspect_3d
from ts_eval.viz.data_containers import xr_3d_factory


def test_api__smoke(dataset_2d, dataset_3d):
    dataset_3d_naive = dataset_3d - 1
    dataset_3d_2 = dataset_3d + 1

    start_date = "2000-01-02"
    freq = "H"

    (
        ts_inspect_3d(
            dataset_2d, dataset_3d, dataset_3d_2, start_date=start_date, freq=freq
        )
        .use_reference(
            xr_3d_factory(dataset_3d_naive, start_date=start_date, freq=freq)
        )
        .for_horizons(0, 1, 5, 23)
        .for_time_slices(time_slices.all, time_slices.weekend)
        .with_decription()
        .with_metrics(metrics.FVrMSE, metrics.FVrMAE, metrics.FVrMIS)
        .with_predictions_plot()
        .show()
        ._repr_html_()
    )


def test_api__smoke__base_preds(dataset_2d, dataset_3d):
    dataset_3d_2 = dataset_3d + 1

    start_date = "2000-01-02"
    freq = "H"

    (
        ts_inspect_3d(
            dataset_2d, dataset_3d, dataset_3d_2, start_date=start_date, freq=freq
        )
        .with_metrics(metrics.FVrMSE, metrics.FVrMIS)
        .show()
        ._repr_html_()
    )
