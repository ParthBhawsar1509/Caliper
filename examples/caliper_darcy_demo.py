"""
Caliper: Darcy flow demo (the credibility demo for the SciML / neural-operator crowd)
=====================================================================================
Pipeline:
  1) generate random permeability fields a(x,y) (log-Gaussian random fields)
  2) solve the 2D Darcy equation  -div(a grad u) = f  for the pressure u
  3) learn a surrogate from the field to a scalar quantity of interest (mean pressure)
  4) Caliper wraps it: CQR intervals (coverage guarantee) + a Trust detector
  5) show it holds coverage in-domain, and that fields from a DIFFERENT distribution
     break the surrogate quietly while the Trust detector catches them.

Needs: numpy, scipy, scikit-learn, matplotlib.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.covariance import LedoitWolf

N = 14  # interior grid is N x N


# ---------- physics: random fields + Darcy solver ----------
def random_field(rng, length_scale, amp, scale=1.0):
    w = rng.standard_normal((N, N))
    f = gaussian_filter(w, sigma=length_scale, mode="reflect")
    f /= (f.std() + 1e-9)
    return scale * np.exp(amp * f)  # log-normal permeability, smooth and positive


def dataset(rng, m, length_scale, amp, scale=1.0):
    A, Q = [], []
    for _ in range(m):
        a = random_field(rng, length_scale, amp, scale)
        u = solve_darcy(a)
        A.append(a.ravel()); Q.append(u.mean())   # QoI = mean pressure
    return np.array(A), np.array(Q)


def solve_darcy(a, f=1.0):
    h = 1.0 / (N + 1)
    ap = np.pad(a, 1, mode="edge")
    idx = lambda i, j: i * N + j
    rows, cols, vals = [], [], []
    for i in range(N):
        for j in range(N):
            k = idx(i, j)
            ac = ap[i + 1, j + 1]
            aE, aW = 0.5 * (ac + ap[i + 2, j + 1]), 0.5 * (ac + ap[i, j + 1])
            aN, aS = 0.5 * (ac + ap[i + 1, j + 2]), 0.5 * (ac + ap[i + 1, j])
            rows.append(k); cols.append(k); vals.append(aE + aW + aN + aS)
            if i + 1 < N: rows.append(k); cols.append(idx(i + 1, j)); vals.append(-aE)
            if i - 1 >= 0: rows.append(k); cols.append(idx(i - 1, j)); vals.append(-aW)
            if j + 1 < N: rows.append(k); cols.append(idx(i, j + 1)); vals.append(-aN)
            if j - 1 >= 0: rows.append(k); cols.append(idx(i, j - 1)); vals.append(-aS)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N * N, N * N)).tocsr()
    u = spla.spsolve(A, np.full(N * N, f * h * h))
    return u.reshape(N, N)




def coverage(y, lo, hi):
    return float(((y >= lo) & (y <= hi)).mean())


# ---------- run ----------
if __name__ == "__main__":
    rng = np.random.default_rng(3)
    L_IN, AMP_IN = 2.0, 0.8                 # in-domain field statistics
    L_OOD, AMP_OOD, SCALE_OOD = 2.0, 2.2, 1.0  # OOD: same smoothness, much higher contrast

    A_tr, Q_tr = dataset(rng, 350, L_IN, AMP_IN)
    A_cal, Q_cal = dataset(rng, 150, L_IN, AMP_IN)
    A_te, Q_te = dataset(rng, 200, L_IN, AMP_IN)
    A_ood, Q_ood = dataset(rng, 200, L_OOD, AMP_OOD, SCALE_OOD)
    print(f"QoI range in-domain = [{Q_tr.min():.3f}, {Q_tr.max():.3f}]   "
          f"out-of-domain = [{Q_ood.min():.3f}, {Q_ood.max():.3f}]")

    pca = PCA(n_components=30).fit(A_tr)
    Xtr, Xcal, Xte, Xood = (pca.transform(A) for A in (A_tr, A_cal, A_te, A_ood))

    cqr = CQRRegressor(0.1).fit(Xtr, Q_tr, Xcal, Q_cal)
    ts = TrustScore(0.95).fit(Xtr, Xcal)

    lo_te, hi_te = cqr.predict(Xte)
    lo_oo, hi_oo = cqr.predict(Xood)
    cov_in = coverage(Q_te, lo_te, hi_te)
    cov_oo = coverage(Q_ood, lo_oo, hi_oo)
    flagged_in = 1 - ts.in_domain(Xte).mean()    # false-alarm rate (want low)
    flagged_oo = 1 - ts.in_domain(Xood).mean()   # detection rate (want high)

    print(f"in-domain coverage   = {cov_in:.3f}  (target 0.90)")
    print(f"out-of-domain coverage = {cov_oo:.3f}  (surrogate quietly unreliable)")
    print(f"OOD fields flagged    = {flagged_oo:.0%}   in-domain false alarms = {flagged_in:.0%}")

    # ---------- figure ----------
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.1))
    a_demo = random_field(rng, L_IN, AMP_IN)
    u_demo = solve_darcy(a_demo)
    im0 = ax[0].imshow(a_demo, cmap="viridis"); ax[0].set_title("permeability field a(x,y)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(u_demo, cmap="magma"); ax[1].set_title("solved pressure u (Darcy)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    for a in (ax[0], ax[1]):
        a.set_xticks([]); a.set_yticks([])

    bars = ax[2].bar(["coverage\nin-domain", "coverage\nout-of-domain", "OOD\ndetected"],
                     [cov_in, cov_oo, flagged_oo],
                     color=["#0F766E", "#B91C1C", "#1E3A5F"])
    ax[2].axhline(0.9, color="#999", ls="--", lw=1)
    ax[2].text(2.5, 0.91, "0.90 target", color="#666", fontsize=8, ha="right")
    ax[2].set_ylim(0, 1.05); ax[2].set_title("Caliper on the Darcy surrogate")
    for b, v in zip(bars, [cov_in, cov_oo, flagged_oo]):
        ax[2].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)

    fig.suptitle("Holds its promised coverage on in-distribution fields; "
                 "fields from a different distribution break the surrogate quietly, and the trust detector catches them.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("darcy_demo.png", dpi=130)
    print("saved darcy_demo.png")
