import pytest


@pytest.mark.parametrize("sys", [205], indirect=True)
def test_predictor_mimo(sys, generate_ind):
    pass