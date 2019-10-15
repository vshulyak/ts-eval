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
    return base64.b64encode(hashlib.sha256(arr).digest())[:NPHASH_MAX_LEN].decode(
        "utf-8"
    )


def get_pretty_var_names(target_vars, local_vars, fallback_name_prefix):
    """
    Gets name of the variable passed into a function,
    then prettifies it by replacing "_" with spaces and capitalizes every token.
    """
    lookup = {id(var_val): var_name for var_name, var_val in local_vars}
    try:
        names = [lookup[id(pred)] for pred in target_vars]
    except KeyError:
        names_pretty = [f"{fallback_name_prefix} {i}" for i in range(len(target_vars))]
    else:
        names_pretty = list(
            map(lambda x: " ".join(t.title() for t in x.split("_")), names)
        )

    return names_pretty
