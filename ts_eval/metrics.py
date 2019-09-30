import numpy as np


def se(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Computes SE – squared error
    """
    return (arr1 - arr2) ** 2


def ae(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Computes AE – absolute error
    """
    return np.abs(arr1 - arr2)


def is_(y: np.ndarray, u: np.ndarray, l: np.ndarray, a=0.05) -> np.ndarray:
    """
    Computes IS (Interval Score), used in M4 competition.
    """
    ml = u - l
    p1 = 2 / a * np.where(y < l, l - y, 0)
    p2 = 2 / a * np.where(y > u, y - u, 0)

    return ml + p1 + p2


def mse(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return se(arr1, arr2).mean()


def mae(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return ae(arr1, arr2).mean()


def mis(y: np.ndarray, u: np.ndarray, l: np.ndarray, a=0.05) -> np.ndarray:
    """
    Computes MIS (Mean Interval Score), used in M4 competition. Lower = better.

    Resources:
        - https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
        - https://stats.stackexchange.com/questions/194660/forecast-accuracy-metric-that-involves-prediction-intervals
    """
    return is_(y, u, l, a=a).mean()
