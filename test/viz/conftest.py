import pytest

import numpy as np


@pytest.fixture(scope="module")
def dataset_2d():
    np.random.seed(0)
    return np.random.random((24 * 4 * 2, 24))


@pytest.fixture(scope="module")
def dataset_3d():
    np.random.seed(0)
    return np.random.random((24 * 4 * 2, 24, 3))
