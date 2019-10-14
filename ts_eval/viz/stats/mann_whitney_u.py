import numpy as np

from scipy.stats import mannwhitneyu


def mw_is_equal(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    # H0: distributions not equal
    try:
        return (
            mannwhitneyu(arr1, arr2, alternative="two-sided").pvalue > 0.05
        )  # TODO: pass in level
    except ValueError as e:
        # mannwhitneyu will raise a ValueError if numbers are equal
        if np.allclose(arr1, arr2):
            return True
        raise e
