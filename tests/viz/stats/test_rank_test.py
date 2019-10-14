import numpy as np
import pytest

from ts_eval.viz.stats.rank_test import rank_test_2d, rank_test_3d


def test_rank__1d_fail(dataset_1d):
    with pytest.raises(AssertionError):
        rank_test_2d(dataset_1d)
    with pytest.raises(AssertionError):
        rank_test_3d(dataset_1d)


def test_rank__singlestep__2series_mw__identical_data(dataset_1d):

    arr = np.stack([dataset_1d + 10e-5, dataset_1d], 1)

    res = rank_test_2d(arr)

    assert res.ranks.shape[0] == 2
    assert res.mean_ranks.shape[0] == 2
    assert res.equality_bool_mask.shape[0] == 2

    assert np.array_equal(res.ranks, np.array([2, 1]))
    assert np.allclose(
        res.equality_bool_mask, np.full_like(res.equality_bool_mask, True)
    )


def test_rank__singlestep__2series_mw__different_data(dataset_1d):

    arr = np.stack([dataset_1d + 1, dataset_1d], 1)

    res = rank_test_2d(arr)

    assert res.ranks.shape[0] == 2
    assert res.mean_ranks.shape[0] == 2
    assert res.equality_bool_mask.shape[0] == 2

    assert np.array_equal(res.ranks, np.array([2, 1]))
    assert np.allclose(
        res.equality_bool_mask, np.full_like(res.equality_bool_mask, False)
    )


def test_rank__singlestep__3series_friedman__identical_data(dataset_1d):

    arr = np.stack([dataset_1d, dataset_1d, dataset_1d], 1)

    res = rank_test_2d(arr)

    assert res.ranks.shape[0] == 3
    assert res.mean_ranks.shape[0] == 3
    assert res.equality_bool_mask.shape[0] == 3

    assert np.array_equal(res.ranks, np.array([1, 2, 3]))
    assert np.allclose(
        res.equality_bool_mask, np.full_like(res.equality_bool_mask, True)
    )


def test_rank__singlestep__3series_friedman__different_data(dataset_1d):

    arr = np.stack([dataset_1d, dataset_1d, dataset_1d - 1], 1)

    res = rank_test_2d(arr)

    assert res.ranks.shape[0] == 3
    assert res.mean_ranks.shape[0] == 3
    assert res.equality_bool_mask.shape[0] == 3

    assert np.array_equal(res.ranks, np.array([3, 1, 2]))
    assert np.allclose(res.equality_bool_mask, np.array([True, True, False]))


def test_rank__multistep__2series_friedman__identical_data(dataset_1d):

    arr = np.stack(
        [
            np.stack([dataset_1d, dataset_1d, dataset_1d], 1),
            np.stack([dataset_1d, dataset_1d, dataset_1d], 1),
        ],
        1,
    )

    res = rank_test_3d(arr)

    assert res.ranks.shape == (2, 3)
    assert res.mean_ranks.shape == (2, 3)
    assert res.equality_bool_mask.shape == (2, 3)

    assert np.array_equal(res.ranks, np.array([[1, 2, 3], [1, 2, 3]]))
    assert np.allclose(
        res.equality_bool_mask, np.full_like(res.equality_bool_mask, True)
    )


def test_rank__multistep__2series_friedman__different_data(dataset_1d):

    arr = np.stack(
        [
            np.stack([dataset_1d, dataset_1d, dataset_1d - 5], 1),
            np.stack([dataset_1d - 10, dataset_1d, dataset_1d + 8], 1),
        ],
        1,
    )

    res = rank_test_3d(arr)

    assert res.ranks.shape == (2, 3)
    assert res.mean_ranks.shape == (2, 3)
    assert res.equality_bool_mask.shape == (2, 3)

    assert np.array_equal(res.ranks, np.array([[3, 1, 2], [1, 2, 3]]))
    assert np.allclose(
        res.equality_bool_mask, np.array([[True, True, False], [False, False, False]])
    )


def test_rank__multistep_1series(dataset_1d):

    arr = np.stack([dataset_1d, dataset_1d]).reshape(-1, 4, 1)

    res = rank_test_3d(arr)

    assert res.ranks.shape == (4, 1)
    assert res.mean_ranks.shape == (4, 1)
    assert res.equality_bool_mask.shape == (4, 1)

    assert np.array_equal(res.ranks, np.array([[1], [1], [1], [1]]))
    assert np.allclose(
        res.equality_bool_mask, np.array([[False], [False], [False], [False]])
    )
