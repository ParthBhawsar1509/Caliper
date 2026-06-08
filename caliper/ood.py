import numpy as np
from sklearn.covariance import LedoitWolf


class TrustScore:
    """Out-of-domain detector: Mahalanobis distance from training data,
    with a threshold calibrated on held-out in-domain points."""
    def __init__(self, target_in_domain=0.95):
        self.target = target_in_domain
        self.cov_ = LedoitWolf()

    def fit(self, X_train, X_holdout=None):
        self.cov_.fit(np.asarray(X_train))
        ref = np.asarray(X_holdout if X_holdout is not None else X_train)
        self.threshold_ = float(np.quantile(self._score(ref), self.target))
        return self

    def _score(self, X):
        return np.sqrt(self.cov_.mahalanobis(np.asarray(X)))

    def score(self, X):
        return self._score(X)

    def in_domain(self, X):
        return self._score(X) <= self.threshold_