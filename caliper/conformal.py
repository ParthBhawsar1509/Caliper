import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class SplitConformalRegressor:
    """Model-agnostic calibrated intervals via split conformal regression."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X_train, y_train, X_cal, y_cal):
        self.estimator.fit(X_train, y_train)
        self.residuals_ = np.abs(np.asarray(y_cal) - self.estimator.predict(X_cal))
        return self

    def _quantile(self, alpha):
        n = len(self.residuals_)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(self.residuals_, level, method="higher"))

    def predict(self, X, alpha=0.1):
        mean = self.estimator.predict(X)
        q = self._quantile(alpha)
        return mean, mean - q, mean + q


class CQRRegressor:
    """Conformalized Quantile Regression: adaptive intervals with a coverage guarantee."""
    def __init__(self, alpha=0.1, model_factory=None):
        self.alpha = alpha
        self.model_factory = model_factory or (
            lambda q: GradientBoostingRegressor(loss="quantile", alpha=q, random_state=0)
        )

    def fit(self, X_train, y_train, X_cal, y_cal):
        self.lo_ = self.model_factory(self.alpha / 2).fit(X_train, y_train)
        self.hi_ = self.model_factory(1 - self.alpha / 2).fit(X_train, y_train)
        y_cal = np.asarray(y_cal)
        scores = np.maximum(self.lo_.predict(X_cal) - y_cal, y_cal - self.hi_.predict(X_cal))
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.Q_ = float(np.quantile(scores, level, method="higher"))
        return self

    def predict(self, X):
        return self.lo_.predict(X) - self.Q_, self.hi_.predict(X) + self.Q_