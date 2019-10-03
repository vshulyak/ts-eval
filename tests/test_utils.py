from ts_eval.utils import create_sliding_dataset


H = 7


def test_sliding_dataset(dataset_1d):
    res = create_sliding_dataset(dataset_1d, h=H)

    assert res.ndim == 2
    assert res.shape[0] == dataset_1d.shape[0] - H
    assert res.shape[1] == H
