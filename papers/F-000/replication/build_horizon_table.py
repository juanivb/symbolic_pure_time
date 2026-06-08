"""F-000 v3 §5 — multi-horizon forecast quality across Cases 1 & 3.

Reports out-of-sample iterated h-step level forecast R^2 for:

  - plain OLS (AR(2) on Case 1, Engle-Granger on Case 3)
  - SPT_1: closed-form OLS with intercept on the kinematic jet
  - SPT_2: closed-form OLS with intercept on jet + Sym^2(jet)

Horizons: h = 1, 5, 10, 20.

Case 2 (fBm at H=0.7) is omitted: AR(1) and SPT yield indistinguishable
level forecasts at the persistent regime; the relevant contrast is the
recovery of H, treated in §5.2.

Output:
  results/case_horizon_table.csv
  results/case_horizon_table.tex
"""
from __future__ import annotations
import math
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = (1, 5, 10, 20)


# --------------------------------------------------------------
# Designs
# --------------------------------------------------------------

def jet(y):
    dz = np.diff(y, prepend=y[0])
    ddz = np.diff(dz, prepend=dz[0])
    return np.column_stack([y, dz, ddz])


def jet_quadratic(y):
    """SPT_2 univariate: jet + Sym^2(jet) = 9 features."""
    Q = jet(y)
    z, dz, ddz = Q[:, 0], Q[:, 1], Q[:, 2]
    return np.column_stack([
        Q,
        z * z, z * dz, z * ddz, dz * dz, dz * ddz, ddz * ddz,
    ])


def joint_jet(y1, y2):
    return np.concatenate([jet(y1), jet(y2)], axis=1)


def joint_jet_quadratic(y1, y2):
    """SPT_2 joint: 6D joint jet + Sym^2 (21 unique cross-products)."""
    J = joint_jet(y1, y2)
    n, d = J.shape
    cols = [J]
    for i in range(d):
        for j in range(i, d):
            cols.append((J[:, i] * J[:, j]).reshape(-1, 1))
    return np.concatenate(cols, axis=1)


# --------------------------------------------------------------
# OLS with intercept (multivariate)
# --------------------------------------------------------------

def fit_with_intercept(Q):
    """Fit Q[t+1] = M Q[t] + c. Returns Mhat (dxd), c (d,)."""
    n = Q.shape[0]
    X = np.column_stack([np.ones(n - 1), Q[:-1]])
    Y = Q[1:]
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    c = B[0, :]
    Mhat = B[1:, :].T
    return Mhat, c


def r2_score(yt, yp):
    yt = np.asarray(yt).ravel()
    yp = np.asarray(yp).ravel()
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# --------------------------------------------------------------
# Case 1 — logistic mu = 3.8
# --------------------------------------------------------------

def case1():
    mu = 3.8
    T = 4000
    y = np.empty(T); y[0] = 0.37
    for t in range(1, T):
        y[t] = mu * y[t - 1] * (1 - y[t - 1])

    n_train = int(0.8 * T)

    # ---- AR(2) plain OLS iterated h-step (level only) ----
    def ar2_iter(h):
        # In-sample fit on [1, y_{t-1}, y_{t-2}]
        X = np.column_stack([
            np.ones(n_train - 2),
            y[1: n_train - 1],
            y[0: n_train - 2],
        ])
        beta, *_ = np.linalg.lstsq(X, y[2: n_train], rcond=None)
        a, b1, b2 = beta
        preds = []
        # Test starts at n_train (so we have y[n_train-1], y[n_train-2])
        for t in range(n_train, T - h):
            yp = y[t - 1]; yp_prev = y[t - 2]
            for _ in range(h):
                yn = a + b1 * yp + b2 * yp_prev
                yp_prev = yp
                yp = yn
            preds.append(yp)
        target = y[n_train + h: T]
        return r2_score(target, preds)

    # ---- SPT_d iterated h-step (level component) ----
    def sptls_iter(h, builder):
        Q_tr = builder(y[:n_train])
        M, c = fit_with_intercept(Q_tr)
        Q_full = builder(y)
        preds = []
        for t in range(n_train, T - h):
            q = Q_full[t].copy()
            for _ in range(h):
                q = M @ q + c
                # Rebuild quadratic features from updated level for SPT_2
                if builder is jet_quadratic:
                    z, dz, ddz = q[0], q[1], q[2]
                    q[3] = z * z; q[4] = z * dz; q[5] = z * ddz
                    q[6] = dz * dz; q[7] = dz * ddz; q[8] = ddz * ddz
            preds.append(q[0])
        target = y[n_train + h: T]
        return r2_score(target, preds)

    return {
        "case": "1. Logistic $\\mu=3.8$",
        "rows": [
            ("AR(2) plain OLS",
             [ar2_iter(h) for h in HORIZONS]),
            ("$\\sPT_1$ (minimal jet)",
             [sptls_iter(h, jet) for h in HORIZONS]),
            ("$\\sPT_2$ (jet $+\\,\\mathrm{Sym}^{2}$)",
             [sptls_iter(h, jet_quadratic) for h in HORIZONS]),
        ],
    }


# --------------------------------------------------------------
# Case 3 — quadratic cointegration (multivariate)
# --------------------------------------------------------------

def case3():
    rng = np.random.default_rng(11)
    T = 2000
    eps2 = rng.standard_normal(T) * 0.6
    y2 = np.cumsum(eps2)
    eta = rng.standard_normal(T) * 1.5
    y1 = y2 ** 2 + eta
    n_train = int(0.8 * T)

    # Engle--Granger linear cointegration regression
    def eg_iter(h):
        Xtr = np.column_stack([np.ones(n_train), y2[:n_train]])
        beta, *_ = np.linalg.lstsq(Xtr, y1[:n_train], rcond=None)
        a, b = beta
        preds = a + b * y2[n_train - h: T - h]
        target = y1[n_train: T]
        return r2_score(target, preds)

    # SPT_d joint multivariate, iterated h-step (level y^(1))
    def sptls_joint_iter(h, builder):
        Q_tr = builder(y1[:n_train], y2[:n_train])
        M, c = fit_with_intercept(Q_tr)
        Q_full = builder(y1, y2)
        preds = []
        for t in range(n_train, T - h):
            q = Q_full[t].copy()
            for _ in range(h):
                q = M @ q + c
                # Rebuild Sym^2 features for SPT_2 joint
                if builder is joint_jet_quadratic:
                    base = q[:6]
                    idx = 6
                    for i in range(6):
                        for j in range(i, 6):
                            q[idx] = base[i] * base[j]
                            idx += 1
            preds.append(q[0])  # level of y^(1)
        target = y1[n_train + h: T]
        return r2_score(target, preds)

    return {
        "case": "3. Quadratic cointegration",
        "rows": [
            ("Engle--Granger linear OLS",
             [eg_iter(h) for h in HORIZONS]),
            ("$\\sPT_1$ joint (minimal joint jet)",
             [sptls_joint_iter(h, joint_jet) for h in HORIZONS]),
            ("$\\sPT_2$ joint (jet $+\\,\\mathrm{Sym}^{2}$)",
             [sptls_joint_iter(h, joint_jet_quadratic) for h in HORIZONS]),
        ],
    }


def main():
    blocks = [case1(), case3()]

    # CSV
    with open(OUT / "case_horizon_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "estimator"] + [f"h={h}" for h in HORIZONS])
        for b in blocks:
            for name, vals in b["rows"]:
                w.writerow([b["case"], name] + [f"{v:.3f}" for v in vals])
    print(f"wrote {OUT / 'case_horizon_table.csv'}")

    # Console
    print()
    print(f"{'case + estimator':55s}  " + "  ".join(f"h={h:<2d} R^2" for h in HORIZONS))
    print("-" * 100)
    for b in blocks:
        for name, vals in b["rows"]:
            label = f"{b['case']}  /  {name}"
            print(f"{label:55s}  " + "  ".join(f"{v:7.3f}" for v in vals))
        print()

    # LaTeX
    with open(OUT / "case_horizon_table.tex", "w") as f:
        f.write(r"""% generated by replication/build_horizon_table.py
\begin{tabular}{ll rrrr}
\toprule
Case & Estimator & $h{=}1$ & $h{=}5$ & $h{=}10$ & $h{=}20$ \\
\midrule
""")
        for i, b in enumerate(blocks):
            for j, (name, vals) in enumerate(b["rows"]):
                case_cell = b["case"] if j == 0 else ""
                f.write(
                    f"{case_cell} & {name} & "
                    + " & ".join(f"{v:+.3f}" for v in vals)
                    + " \\\\\n"
                )
            if i < len(blocks) - 1:
                f.write(r"\midrule" + "\n")
        f.write(r"""\bottomrule
\end{tabular}
""")
    print(f"wrote {OUT / 'case_horizon_table.tex'}")


if __name__ == "__main__":
    main()
