import numpy as np

from ts_eval.viz.utils import time_align, time_unalign, nphash


def test_time_align(dataset_2d):
    aligned = time_align(dataset_2d)

    n, h = dataset_2d.shape

    # check NAs surrounding the original dataset
    for ih in range(h):
        # NaNs before the original data
        assert all(np.isnan(aligned[:ih, ih]))
        # elements of the original array
        assert np.allclose(aligned[ih:ih + n, ih], dataset_2d[:, ih])
        # trailing elements (NaNs)
        assert all(np.isnan(aligned[-h + ih:, ih]))


def test_time_align_unalign(dataset_2d):
    assert np.allclose(time_unalign(time_align(dataset_2d)), dataset_2d)


def test_nphash(dataset_2d):
    # idempotent
    assert nphash(dataset_2d) == nphash(dataset_2d)

    # small nudge
    assert nphash(dataset_2d) != nphash(dataset_2d + 1e-10)
