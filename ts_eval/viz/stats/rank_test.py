from functools import partial

import numpy as np

from dataclasses import dataclass
from scipy.stats import friedmanchisquare, rankdata
from statsmodels.stats.libqsturng import qsturng

from ts_eval.utils import nanmeanw

from .mann_whitney_u import mw_is_equal


@dataclass
class FriedmanNemenyiResult:
    ranks: np.ndarray
    mean_ranks: np.ndarray
    equality_bool_mask: np.ndarray


def rank_test_2d(arr: np.ndarray, level=0.95) -> bool:
    assert arr.ndim == 2
    assert arr.shape[0] > arr.shape[1]

    r = rank_test_3d(np.expand_dims(arr, 1))

    return FriedmanNemenyiResult(
        ranks=r.ranks[0],
        mean_ranks=r.mean_ranks[0],
        equality_bool_mask=r.equality_bool_mask[0],
    )


def rank_test_3d(arr: np.ndarray, level=0.95) -> bool:
    """
    """
    assert arr.ndim == 3
    n, h, f = arr.shape
    assert arr.shape[0] > arr.shape[1]

    # when we have only one dataset to compare to itself, then there's no need to run any tests...
    # just return ranks
    if f == 1:
        ranks = np.ones((h, f), dtype=np.uint8)
        mean_ranks = nanmeanw(arr, 0)
        equality_bool_mask = np.full((h, f), False)

        return FriedmanNemenyiResult(
            ranks=ranks, mean_ranks=mean_ranks, equality_bool_mask=equality_bool_mask
        )

    pre_test = friedman if f >= 3 else mannwhitney

    ranks = np.empty((h, f), dtype=np.uint8)
    mean_ranks = np.empty((h, f))
    equality_bool_mask = np.empty((h, f), dtype=np.bool)

    for hi in range(arr.shape[1]):

        step_arr = arr[:, hi, :]

        if pre_test(step_arr):
            # H0: datasets not different, just return this information in an appropriate way
            ranks[hi, :] = nanmeanw(step_arr, 0).argsort() + 1
            mean_ranks[hi, :] = nanmeanw(step_arr, 0)
            equality_bool_mask[hi, :] = np.full((f,), True)

        else:
            # Ha: at least one is different, run nemenyi to determine ranks
            ranks_s, mean_ranks_s, equality_bool_mask_s = nemenyi(step_arr)

            ranks[hi, :] = ranks_s
            mean_ranks[hi, :] = mean_ranks_s
            equality_bool_mask[hi, :] = equality_bool_mask_s

    return FriedmanNemenyiResult(
        ranks=ranks, mean_ranks=mean_ranks, equality_bool_mask=equality_bool_mask
    )


def friedman(arr: np.ndarray, level=0.95) -> bool:
    """

    True = Ha
    """
    assert arr.ndim == 2

    # a shortcut to skip friedman if all arrays are equal.
    # Friedman would return NaN in this case, issuing a warning as well.
    if np.allclose(arr[:, 0], arr.sum(1) / arr.shape[1]):
        return True

    # H0: identical
    # Ha: at least one is different
    pvalue = friedmanchisquare(*arr.T).pvalue

    return np.isnan(pvalue) or pvalue >= 1 - level


def mannwhitney(arr: np.ndarray) -> bool:

    return mw_is_equal(*arr.T)


def nemenyi(arr: np.ndarray, level=0.95):
    """
    Compute 2-step rank test. Friedman and then Nemenyi posthoc.

    <observations> x <groups>
    """
    assert arr.ndim == 2
    assert arr.shape[0] > arr.shape[1]

    # get critical distance which depends on the size of the input matrix
    cd = get_critical_distance(arr)

    """
    Get ranks across rows like this:
    array([[2, 0, 1],
           [2, 0, 1],
           ...,
           [2, 1, 0],
           [2, 0, 1]])
    """

    # rank data by each row.
    # note that we can't use a simpler algorithm like "arr.argsort().argsort() + 1" as we need to handle ties.
    rankdatap = partial(rankdata, method="average")
    ranks = np.apply_along_axis(rankdatap, 1, arr)

    # find means per column
    mranks = ranks.mean(0)

    # store indexes to revert back to the original order later
    original_order_idxs = mranks.argsort()

    # sort the array
    sorted_mranks = mranks[original_order_idxs]

    # we have intervals (fence sections) between mean ranks (fence points),
    # but for each "fencepost" both "fence sections" on left and right are important to see if
    # this mean rank intersects with points on left and right. Hence, we pad this array on left and right
    # so that leftmost and rightmost mean ranks have enough elements to slide through the seq
    sorted_mranks_padded = np.pad(
        sorted_mranks, (1, 1), "constant", constant_values=np.inf
    )

    sorted_mranks_diff = np.abs(np.diff(sorted_mranks_padded))

    # find intervals between ranks which are bigger than critical distance.
    # note, that one critical distance can cover up many mean ranks, but
    # in the current use case we care only about at least one intersection
    # the returned mask has True value if critical distance is lower than the interval
    mask = np.array(
        [
            a >= cd and b >= cd
            for a, b in zip(sorted_mranks_diff[:-1], sorted_mranks_diff[1:])
        ]
    )

    ranks = mranks.argsort() + 1
    mean_ranks = mranks

    # invert the mask (shows non-equality) => to show equality
    equality_bool_mask = np.invert(mask[original_order_idxs[original_order_idxs]])

    return ranks, mean_ranks, equality_bool_mask


def get_critical_distance(x):

    n_rows, n_cols = x.shape

    a = qsturng(0.95, n_cols, np.inf)
    b = np.sqrt((n_cols * (n_cols + 1)) / (12 * n_rows))

    return a * b
