import numpy as np
from caliper.metrics import coverage_and_width

def test_coverage_and_width():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lo = np.array([-1.0, 0.0, 5.0, 2.0])
    hi = np.array([1.0, 2.0, 6.0, 4.0])
    cov, width = coverage_and_width(y, lo, hi)
    assert abs(cov - 0.75) < 1e-9
    assert abs(width - 1.75) < 1e-9