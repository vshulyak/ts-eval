import base64
import hashlib

import numpy as np


NPHASH_MAX_LEN = 10


def time_align(arr: np.ndarray):
    """
    Converts unaligned in time 2d array (timestamp,horizon) to aliged in time.
    """
    assert arr.ndim == 2, "I only know how to work on 2-dim arrays"

    n, h = arr.shape

    # Our new array will have all points aligned on one time axis,
    # but we'll get NaNs for shifted points.
    # As a result our new array will be longer by 2nd dim size (horizon)
    res = np.empty((n + h, h))
    res[:] = np.nan

    for i in range(h):
        res[i : i + n, i] = arr[:, i]

    return res


def time_unalign(arr: np.ndarray):
    """
    Converts aligned in time 2d array to not-aligned.
    """
    assert arr.ndim == 2, "I only know how to work on 2-dim arrays"

    n, h = arr.shape

    res = np.empty((n - h, h))
    res[:] = np.nan

    for i in range(h):
        res[:, i] = arr[i : i + n - h, i]

    return res


def filter_nan(arr: np.ndarray):
    """
    Filters NaNs, flattenning the resulting array
    """
    return arr[~np.isnan(arr)]


def nphash(arr: np.ndarray):
    """
    Hash from an array
    """
    return base64.b64encode(hashlib.sha256(arr).digest())[:NPHASH_MAX_LEN]
