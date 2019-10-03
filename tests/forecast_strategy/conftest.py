import pytest


def _dataset_fixture(request):
    if request.param is None:
        return None
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="module")
def endog(request):
    return _dataset_fixture(request)


@pytest.fixture(scope="module")
def exog(request):
    return _dataset_fixture(request)
