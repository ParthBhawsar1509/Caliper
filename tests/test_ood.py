import numpy as np
from caliper.ood import TrustScore

def test_flags_out_of_domain():
    rng = np.random.default_rng(0)
    X_in = rng.normal(0, 1, size=(500, 3))
    X_hold = rng.normal(0, 1, size=(300, 3))
    X_out = rng.normal(8, 1, size=(200, 3))
    ts = TrustScore(0.95).fit(X_in, X_holdout=X_hold)
    assert ts.in_domain(X_hold).mean() > 0.85
    assert (~ts.in_domain(X_out)).mean() > 0.95