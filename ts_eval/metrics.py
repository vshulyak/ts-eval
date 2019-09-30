import numpy as np

from .utils import nans_in_same_positions


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

    :param y: predicted mean values
    :param u: prediction interval upper bound
    :param l: prediction interval lower bound
    :return: IS
    """
    assert nans_in_same_positions(y, u, l), "Arrays have NaNs in different positions"

    ml = u - l

    # NaNs can be in the dataset and this will cause lt/gt comparisons to raise a warning
    with np.errstate(invalid="ignore"):
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

    :param y: predicted mean values
    :param u: prediction interval upper bound
    :param l: prediction interval lower bound
    :return: IS
    """
    return is_(y, u, l, a=a).mean()


def smape(a, b):
    """
    From the M4 repo

    Calculates sMAPE
    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()


def mase(insample, y_test, y_hat_test, freq):
    """
    From the M4 repo

    Calculates MAsE
    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    return np.mean(abs(y_test - y_hat_test)) / masep
