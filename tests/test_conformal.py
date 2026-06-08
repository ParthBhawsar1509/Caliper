import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from caliper.conformal import SplitConformalRegressor, CQRRegressor

def _data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 6, 900).reshape(-1, 1)
    y = np.sin(X).ravel() + rng.normal(0, 0.1, 900)
    return X[:400], y[:400], X[400:600], y[400:600], X[600:], y[600:]

def test_split_conformal_coverage():
    Xtr, ytr, Xc, yc, Xte, yte = _data()
    cp = SplitConformalRegressor(GradientBoostingRegressor(random_state=0)).fit(Xtr, ytr, Xc, yc)
    _, lo, hi = cp.predict(Xte, alpha=0.1)
    assert ((yte >= lo) & (yte <= hi)).mean() >= 0.85

def test_cqr_coverage_and_ordering():
    Xtr, ytr, Xc, yc, Xte, yte = _data()
    cqr = CQRRegressor(alpha=0.1).fit(Xtr, ytr, Xc, yc)
    lo, hi = cqr.predict(Xte)
    assert (lo <= hi).all()
    assert ((yte >= lo) & (yte <= hi)).mean() >= 0.85