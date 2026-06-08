"""
Caliper: CQR + 1D visual demo
=============================
CQRRegressor is the upgrade from the starter's symmetric split-conformal band:
it produces ADAPTIVE intervals (width changes with x) while keeping the same
finite-sample coverage guarantee. This is the method to make Caliper's headline.

The plot tells the whole story on one chart:
  - inside the training range, the band tracks the (growing) noise and covers ~90%
  - outside it, the surrogate is CONFIDENTLY WRONG and the band does NOT save you
    (conformal only guarantees coverage in-distribution) -> which is exactly why
    the Trust flag, shading the out-of-domain region, is a separate necessary piece.

Needs: numpy, scikit-learn, matplotlib.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.covariance import LedoitWolf



# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    LO, HI = 0.0, 6.0  # training range

    def truth(x):
        return np.sin(1.5 * x).ravel()

    def noisy(x):
        x = x.ravel()
        sigma = 0.05 + 0.15 * x          # heteroscedastic: noise grows with x
        return truth(x) + rng.normal(0, sigma)

    X = rng.uniform(LO, HI, 600).reshape(-1, 1)
    y = noisy(X)
    X_tr, y_tr, X_cal, y_cal = X[:420], y[:420], X[420:], y[420:]

    cqr = CQRRegressor(alpha=0.1).fit(X_tr, y_tr, X_cal, y_cal)
    ts = TrustScore(0.95).fit(X_tr, X_holdout=X_cal)

    # in-domain coverage check
    X_in = rng.uniform(LO, HI, 500).reshape(-1, 1)
    y_in = noisy(X_in)
    lo_in, hi_in = cqr.predict(X_in)
    cov = float(((y_in >= lo_in) & (y_in <= hi_in)).mean())

    # plot across a range that extends well outside training
    xs = np.linspace(-2, 10, 600).reshape(-1, 1)
    lo, hi = cqr.predict(xs)
    mid = 0.5 * (lo + hi)
    indom = ts.in_domain(xs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(X_tr, y_tr, s=8, alpha=0.15, color="#2E5A8C", label="training data")
    ax.plot(xs, truth(xs), color="#111111", lw=2, label="true function")
    ax.plot(xs, mid, color="#0F766E", lw=1.8, ls="--", label="surrogate estimate")
    ax.fill_between(xs.ravel(), lo, hi, color="#0F766E", alpha=0.18, label="90% CQR interval")

    # shade where the Trust detector says we are extrapolating
    ood = ~indom
    ax.fill_between(xs.ravel(), -3, 3, where=ood, color="#B91C1C", alpha=0.08,
                    label="flagged out-of-domain")
    ax.axvline(LO, color="#999", lw=0.8, ls=":")
    ax.axvline(HI, color="#999", lw=0.8, ls=":")

    ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Caliper on a 1D surrogate: in-domain coverage = {cov:.0%} (target 90%). "
                 f"Outside the training range the estimate is confidently wrong, and the\n"
                 f"interval does not catch it; the out-of-domain flag does.", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig("cqr_demo_1d.png", dpi=130)
    print(f"in-domain coverage = {cov:.3f}   conformal correction Q = {cqr.Q_:.3f}")
    print(f"out-of-domain points correctly flagged = {(~ts.in_domain(xs[(xs.ravel()<LO)|(xs.ravel()>HI)])).mean():.0%}")
    print("saved cqr_demo_1d.png")
