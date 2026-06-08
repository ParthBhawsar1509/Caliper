# Caliper

Trustworthy uncertainty for surrogate models. Caliper adds calibrated error bars, out-of-domain detection, and active learning to the fast ML models people use in place of slow, expensive simulations.

## What this is

Scientists and engineers replace slow simulations (fluid dynamics, materials, climate) with fast ML imitators called surrogates. The problem: a surrogate is only reliable near the data it was trained on. Outside that, it returns confident, precise-looking answers that are wrong, with no warning. Caliper makes the surrogate honest.

In one sentence: Caliper teaches the fast ML shortcuts that replace slow simulations to tell you when they actually know the answer and when they are just guessing.

## What it does

- **Calibrated uncertainty.** Wrap any model and get prediction intervals with a distribution-free coverage guarantee, via split conformal and conformalized quantile regression (CQR).
- **Out-of-domain detection.** A trust score flags inputs that fall outside the region the model was validated on, so you do not trust answers you should not.
- **Active learning.** Pick the most informative next simulation to run, so compute budget goes where it reduces error most.

[Caliper catching a surrogate that is confidently wrong outside its training range](assets/cqr_demo_1d.png)

## Quick start

```bash
pip install -e .
```

```python
from sklearn.ensemble import GradientBoostingRegressor
from caliper import SplitConformalRegressor, TrustScore

# calibrated 90% intervals around any regressor
cp = SplitConformalRegressor(GradientBoostingRegressor()).fit(X_train, y_train, X_cal, y_cal)
mean, lower, upper = cp.predict(X_new, alpha=0.1)

# flag where the model is extrapolating
ts = TrustScore().fit(X_train, X_holdout=X_cal)
trusted = ts.in_domain(X_new)   # False where it is guessing
```

See `examples/` for runnable 1D and Darcy-flow demos.

## For the technically minded

Surrogate models (fast ML stand-ins for expensive simulations like CFD or finite-element analysis) are unreliable outside their training domain and fail silently. Caliper wraps any surrogate and adds calibrated predictive uncertainty, out-of-domain detection, and active-learning acquisition. It is the trust and uncertainty layer that the large surrogate-training tools treat as an afterthought.

## Status

Early and experimental. The API may change while the core methods and the first real use cases are validated.

## License

MIT