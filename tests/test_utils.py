from ts_eval.utils import create_sliding_dataset, create_sliding_dataset_xr


H = 7


def test_sliding_dataset_numpy(dataset_1d):
    res = create_sliding_dataset(dataset_1d, h=H)

    assert res.ndim == 2
    assert res.shape[0] == dataset_1d.shape[0] - H
    assert res.shape[1] == H


def test_sliding_dataset_xarray(dataset_1d__pd_index_datetime):
    res = create_sliding_dataset_xr(dataset_1d__pd_index_datetime, h=H)

    assert res.dt.shape[0] == dataset_1d__pd_index_datetime.shape[0]
    assert res.dt[0] == dataset_1d__pd_index_datetime.index[0]
