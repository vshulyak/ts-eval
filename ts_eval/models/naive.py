import numpy as np

from scipy.stats import norm


def _naive_with_insample(insample: np.ndarray, freq: int = 1, h: int = 10):
    """
    Computes a naive forecast (with prediction interval).

    TODO:
     - add mean and drift forecasts https://otexts.com/fpp2/prediction-intervals.html
     - can be made more efficient
    """

    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])
    return y_hat_naive, np.tile(insample[-freq:], (h // freq) + 1)[:h]


def naive_pi(y: np.ndarray, freq: int = 1, h: int = 1, cl: int = 95):
    """
    Computes naive forecast (with prediction interval).
    """
    # pandas => numpy if needed
    y = y.values.reshape(-1) if hasattr(y, "values") else y

    insample, fc = _naive_with_insample(y, freq=freq, h=h)
    resid = y[freq:] - insample
    s = np.mean(resid ** 2)

    clp = norm.ppf(0.5 + cl / 200)

    ub, lb = [], []

    for h in range(1, h + 1):
        p = np.sqrt(s * h)
        ub += [fc[h - 1] + clp * p]
        lb += [fc[h - 1] - clp * p]

    return fc, np.array(ub), np.array(lb)


def snaive_pi(y: np.ndarray, freq: int = 1, h: int = 1, cl: int = 95):
    """
    Computes seasonal naive forecast (with prediction interval).
    """
    # pandas => numpy if needed
    y = y.values.reshape(-1) if hasattr(y, "values") else y

    insample, fc = _naive_with_insample(y, freq=freq, h=h)
    resid = y[freq:] - insample
    s = np.mean(resid ** 2)

    clp = norm.ppf(0.5 + cl / 200)

    ub, lb = [], []

    for h in range(1, h + 1):
        k = (h - 1) // freq
        p = np.sqrt(s * (k + 1))
        ub += [fc[h - 1] + clp * p]
        lb += [fc[h - 1] - clp * p]

    return fc, np.array(ub), np.array(lb)
