import numpy as np

from hypothesis import given
from hypothesis.extra.numpy import arrays

from ts_eval.viz.utils import nphash, time_align, time_unalign


def test_time_align(dataset_2d):
    aligned = time_align(dataset_2d)

    n, h = dataset_2d.shape

    # check NAs surrounding the original dataset
    for ih in range(h):
        # NaNs before the original data
        assert all(np.isnan(aligned[:ih, ih]))
        # elements of the original array
        assert np.allclose(aligned[ih : ih + n, ih], dataset_2d[:, ih])
        # trailing elements (NaNs)
        assert all(np.isnan(aligned[-h + ih :, ih]))


def test_time_align_unalign(dataset_2d):
    assert np.allclose(time_unalign(time_align(dataset_2d)), dataset_2d)


@given(arrays(np.float64, (1, 1)).filter(lambda x: x >= 0 and x < 1e6))
def test_nphash(arr):
    # idempotent
    assert nphash(arr) == nphash(arr)

    # small nudge
    assert nphash(arr) != nphash(arr + 0.001)
