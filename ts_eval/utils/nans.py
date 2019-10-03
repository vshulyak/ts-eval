import warnings

import numpy as np


def nans_in_same_positions(*arrays):
    """
    Compares all provided arrays to see if they have NaNs in the same positions.
    """
    if len(arrays) == 0:
        return True
    for arr in arrays[1:]:
        if not (np.isnan(arrays[0]) == np.isnan(arr)).all():
            return False
    return True


def nanmeanw(arr, axis=None):
    """
    Computes nanmean without raising a warning in case of NaNs in the dataset
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=axis)
