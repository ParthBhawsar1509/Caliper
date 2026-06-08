import numpy as np


def coverage_and_width(y, lower, upper):
    """Empirical coverage and mean interval width."""
    y = np.asarray(y)
    covered = (y >= lower) & (y <= upper)
    return float(covered.mean()), float(np.mean(np.asarray(upper) - np.asarray(lower)))