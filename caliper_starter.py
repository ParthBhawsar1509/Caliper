"""
caliper (starter)
=================
A tiny, dependency-light seed for the Caliper library. It shows the spine of the
whole project: a model-agnostic, calibrated-uncertainty core, plus an
out-of-domain detector that reuses the same idea of "distance from what we know".

Design principle: WRAP, don't replace. Caliper takes any already-trained
sklearn-style regressor (anything with .fit/.predict) and adds the trust layer.

Only needs: numpy, scikit-learn.

Run:  python caliper_starter.py
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import GradientBoostingRegressor


# ----------------------------------------------------------------------
# Pillar 1: calibrated uncertainty (the universal core)
# Split (inductive) conformal regression. Distribution-free, finite-sample
# coverage guarantee under exchangeability. Wraps ANY point predictor.
# ----------------------------------------------------------------------
class SplitConformalRegressor:
    def __init__(self, estimator):
        self.estimator = estimator  # any object with .fit(X, y) and .predict(X)

    def fit(self, X_train, y_train, X_cal, y_cal):
        # 1) fit the surrogate on the proper training split
        self.estimator.fit(X_train, y_train)
        # 2) measure how wrong it is on a held-out calibration split
        resid = np.abs(np.asarray(y_cal) - self.estimator.predict(X_cal))
        self.residuals_ = resid
        return self

    def _quantile(self, alpha):
        # finite-sample corrected quantile of the calibration residuals
        n = len(self.residuals_)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(self.residuals_, level, method="higher"))

    def predict(self, X, alpha=0.1):
        """Return (mean, lower, upper) for a (1 - alpha) interval."""
        mean = self.estimator.predict(X)
        q = self._quantile(alpha)
        return mean, mean - q, mean + q


def coverage_and_width(y, lower, upper):
    """How trustworthy were the intervals? This is how you PROVE the product works."""
    y = np.asarray(y)
    covered = (y >= lower) & (y <= upper)
    return float(covered.mean()), float(np.mean(np.asarray(upper) - np.asarray(lower)))


# ----------------------------------------------------------------------
# Pillar 2: out-of-domain detection ("is the model guessing here?")
# Mahalanobis distance from the training distribution, with a threshold
# calibrated on held-out in-domain data so the flag actually means something.
# ----------------------------------------------------------------------
class TrustScore:
    def __init__(self, target_in_domain=0.95):
        self.target = target_in_domain
        self.cov_ = LedoitWolf()  # shrinkage: robust when inputs are high-dim/correlated

    def fit(self, X_train, X_holdout=None):
        self.cov_.fit(np.asarray(X_train))
        ref = np.asarray(X_holdout if X_holdout is not None else X_train)
        d = self._score(ref)
        # set the boundary so ~target fraction of genuine in-domain points pass
        self.threshold_ = float(np.quantile(d, self.target))
        return self

    def _score(self, X):
        # sklearn returns squared Mahalanobis to the fitted mean; sqrt for a distance
        return np.sqrt(self.cov_.mahalanobis(np.asarray(X)))

    def score(self, X):
        """Higher = further from training data = trust less."""
        return self._score(X)

    def in_domain(self, X):
        """True = safe to trust, False = extrapolating."""
        return self._score(X) <= self.threshold_


# ----------------------------------------------------------------------
# Demo: the core pitch in ~20 lines
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def truth(x):
        return np.sin(x).ravel()

    # training/calibration live in [0, 6]; we will later probe far outside it
    X = rng.uniform(0, 6, size=400).reshape(-1, 1)
    y = truth(X) + rng.normal(0, 0.1, size=400)
    X_tr, X_cal = X[:280], X[280:]
    y_tr, y_cal = y[:280], y[280:]

    # in-domain test set
    X_in = rng.uniform(0, 6, size=300).reshape(-1, 1)
    y_in = truth(X_in) + rng.normal(0, 0.1, size=300)

    # out-of-domain probes in [8, 12], where the model never trained
    X_ood = rng.uniform(8, 12, size=300).reshape(-1, 1)

    # ---- Pillar 1: calibrated intervals on any model ----
    model = GradientBoostingRegressor(random_state=0)
    cp = SplitConformalRegressor(model).fit(X_tr, y_tr, X_cal, y_cal)
    _, lo, hi = cp.predict(X_in, alpha=0.1)          # ask for 90% coverage
    cov, width = coverage_and_width(y_in, lo, hi)
    print(f"[uncertainty]  target coverage = 0.90   "
          f"empirical = {cov:.3f}   mean width = {width:.3f}")

    # ---- Pillar 2: out-of-domain flag ----
    ts = TrustScore(target_in_domain=0.95).fit(X_tr, X_holdout=X_cal)
    print(f"[trust]        in-domain points flagged safe  = "
          f"{ts.in_domain(X_in).mean():.2%}")
    print(f"[trust]        out-of-domain points caught     = "
          f"{(~ts.in_domain(X_ood)).mean():.2%}")

    # ---- Pillar 3 hook ----
    # Active learning reuses the same uncertainty: query where intervals are widest
    # or (with an ensemble) where members disagree most. That comes next.
    print("\nThis is the spine. UQ first; OOD and active learning both reuse it.")
