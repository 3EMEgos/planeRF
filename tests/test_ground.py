import numpy as np
import pytest

from planeRF import compute_power_density


def test_compute_power_density():
    # test input values
    f = 1e9     # replace with your desired frequency
    theta = 45  # replace with your desired angle
    pol = 'TM'

    # perform the computation
    SH, SE = compute_power_density(f, theta, pol)

    # check the output types
    assert isinstance(SH, np.ndarray)
    assert isinstance(SE, np.ndarray)

    # do additional checks based on your knowledge of expected values
    # ...


if __name__ == '__main__':
    pytest.main()
