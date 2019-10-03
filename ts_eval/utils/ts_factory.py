import numpy as np


def seasonality(n, s_len=24):
    freq = n / s_len

    t = np.arange(n) / n
    c1 = 1.0 * np.sin(2 * np.pi * t * freq)
    c2 = 0.4 * np.sin(2 * np.pi * 15 * t)

    noise = np.random.rand(n)

    return c1 + c2 + noise


def trend(n, steepness=1.2):
    return np.arange(n) / (n ** steepness) + np.random.rand(n) * 0.1
