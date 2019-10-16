import numpy as np
import pandas as pd


FREQ_OFFSETS = {
    "day": pd.offsets.DateOffset(0),  # TODO: test
    "week": pd.offsets.Week(weekday=0),
    "month": pd.offsets.MonthBegin(),
    "quarter": pd.offsets.QuarterBegin(),  # TODO: test
    "year": pd.offsets.YearBegin(),  # TODO: test
}

FREQ_LEN = {
    "day": pd.offsets.Day(),
    "week": pd.offsets.Week(),
    "month": pd.offsets.MonthEnd(),  # TODO: test
    "quarter": pd.offsets.QuarterEnd(),  # TODO: test
    "year": pd.offsets.YearEnd(),
}


def fourier(k, n, freq):
    """
    Generates fourier series

    k – number of features; out=k*2
    n – length
    freq – selbstverstandlich
    """
    r = []
    for i in range(1, k + 1):
        r += [
            np.sin(2 * i * np.pi * np.arange(n) / freq),
            np.cos(2 * i * np.pi * np.arange(n) / freq),
        ]
    return np.vstack(r).T


def fourier_df(pd_index, freq, k, name="ffcomp"):
    """
    Generates fourier features
    """
    return pd.DataFrame(
        fourier(k=k, n=pd_index.shape[0], freq=freq),
        index=pd_index,
        columns=[f"{name}_{i}" for i in range(0, k * 2)],
    )


def fixed_start_fourier_df(pd_index, freq, k, name="ffcomp"):
    """
    Generates fourier features, with a fixed starting point
    """
    assert isinstance(
        pd_index, pd.Index
    ), "Provide pd.Index subclass as the first argument"
    assert pd_index.freq, "freq for input index is not defined"
    assert (
        freq in FREQ_OFFSETS.keys()
    ), f"Only freqs {FREQ_OFFSETS.keys()} are supported"
    assert k > 0, "k must be positive"

    # get the beginning of the interval, tied to calendar
    normalized_start = (pd_index[0] - FREQ_OFFSETS[freq]).normalize()
    normalized_index = pd.date_range(normalized_start, pd_index[-1], freq=pd_index.freq)

    # offset between the original input and the calendar
    offset = normalized_index.shape[0] - pd_index.shape[0]

    # compute needed frequency
    o_len = FREQ_LEN[freq]

    # note: -1 at the end as we don't need inclusive interval
    freq_p = (
        pd.date_range(
            normalized_start, normalized_start + o_len, freq=pd_index.freq
        ).shape[0]
        - 1
    )

    return pd.DataFrame(
        fourier(k=k, n=pd_index.shape[0] + offset, freq=freq_p)[offset:],
        index=pd_index,
        columns=[f"{name}_{i}" for i in range(0, k * 2)],
    )
