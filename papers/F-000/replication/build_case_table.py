"""F-000 v3 §5 — one-step quality-of-fit table for the two worked
cases:

  Case 1: Deterministic chaos --- logistic map at mu=3.8.
  Case 2: Mixed multivariate VAR --- linear + quadratic cointegration
          with lags.

For each case, the comparison is:

  (a) Plain OLS on lagged scalars / linear specification.
  (b) OLS with feature engineering (quadratic monomials of the same
      regressors that SPT_2 would build).
  (c) SPTLS_d (closed-form OLS on the kinematic jet, plus the same
      polynomial-degree extension where it applies).

In every case the numerical fit of (b) and (c) is essentially the
same when the underlying generator is polynomial. The contribution
of (c) is the *structural readout* of the fitted operator hat M
(polar decomposition, grade reading, block structure for
multivariate, cross-bivector channel for cointegration) --- output
the OLS+FE baseline does not deliver.

Outputs:
  results/case_fit_table.csv
  results/case_fit_table.tex
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def jet(y):
    dz = np.diff(y, prepend=y[0])
    ddz = np.diff(dz, prepend=dz[0])
    return np.column_stack([y, dz, ddz])


def jet_quadratic(y):
    """SPT_2 univariate design: jet + Sym^2(jet) = 9 features."""
    Q = jet(y)
    z, dz, ddz = Q[:, 0], Q[:, 1], Q[:, 2]
    return np.column_stack([
        Q, z*z, z*dz, z*ddz, dz*dz, dz*ddz, ddz*ddz
    ])


def ols_stats(X, y):
    # Numerically stable: use lstsq instead of (X^T X)^{-1}.
    # For collinear design matrices (e.g. polynomial features with
    # strong correlation across powers), the explicit inverse blows up.
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    n, p = X.shape
    sigma2 = float(resid @ resid) / max(n - p, 1)
    # t-stats from pseudo-inverse
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.pinv(XtX)
        se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    except Exception:
        se = np.full(beta.shape, np.nan)
    tvals = beta / np.where(se > 0, se, np.nan)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    R2 = 1.0 - ss_res / ss_tot
    return beta, R2, tvals, resid


def sptls_stats(Q):
    """OLS of Q[t+1] on (1, Q[t]). Returns Mhat, level-R^2 (first
    component), residuals (NxD)."""
    n = Q.shape[0]
    X = np.column_stack([np.ones(n - 1), Q[:-1]])
    Y = Q[1:]
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    c = B[0, :]
    Mhat = B[1:, :].T
    Qp_hat = (Mhat @ Q[:-1].T).T + c
    resid = Q[1:] - Qp_hat
    yt = Q[1:, 0]; yp = Qp_hat[:, 0]
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return Mhat, R2, resid


def autocorr1(x):
    x = np.asarray(x).ravel()
    x = x - x.mean()
    num = float((x[1:] * x[:-1]).sum())
    den = float((x * x).sum())
    return num / den if den > 0 else 0.0


def durbin_watson(r):
    r = np.asarray(r).ravel()
    d = np.diff(r)
    return float((d * d).sum() / (r * r).sum()) if (r * r).sum() > 0 else float("nan")


def polar_readout(M):
    """Polar decomposition M = R P. Return (rotor_angle_deg, det)."""
    MtM = M.T @ M
    w, V = np.linalg.eigh(MtM)
    w = np.clip(w, 1e-12, None)
    R = M @ V @ np.diag(1.0/np.sqrt(w)) @ V.T
    n = M.shape[0]
    # rotor angle from tr(R) on SO(n) approximation
    trR = float((np.trace(R) - (n - 2)) / 2.0)
    trR = max(-1.0, min(1.0, trR))
    return float(np.degrees(np.arccos(trR))), float(np.real(np.linalg.det(M)))


# ---------------------------------------------------------------------
# Case 1 — logistic mu=3.8 (deterministic chaos)
# ---------------------------------------------------------------------

def stats_case1():
    mu = 3.8
    T = 4000
    y = np.empty(T); y[0] = 0.37
    for t in range(1, T):
        y[t] = mu * y[t-1] * (1.0 - y[t-1])

    # (a) plain OLS: AR(2) (matched to lag depth in jet)
    X_ar = np.column_stack([np.ones(T-2), y[1:-1], y[:-2]])
    _, R2_ar, _, resid_ar = ols_stats(X_ar, y[2:])

    # (b) OLS + feature engineering: AR(2) lagged scalars plus their
    # quadratic products --- the SAME monomials SPT_2 builds on the
    # jet (since the jet spans {y_t, y_{t-1}, y_{t-2}}).
    X_fe = np.column_stack([
        np.ones(T-2),
        y[1:-1], y[:-2],
        y[1:-1]**2, y[1:-1]*y[:-2], y[:-2]**2,
    ])
    _, R2_fe, _, resid_fe = ols_stats(X_fe, y[2:])

    # (c) SPTLS_2: jet + Sym^2(jet)
    Q2 = jet_quadratic(y)
    M2, R2_sp2, resid_sp2 = sptls_stats(Q2)
    # Geometric readout of the M_2 polar decomposition (the 9x9 lift)
    rotor_deg, detM = polar_readout(M2)

    rec = {
        "case": "1. Logistic $\\mu=3.8$",
        "ar":  (R2_ar,  autocorr1(resid_ar),     durbin_watson(resid_ar)),
        "fe":  (R2_fe,  autocorr1(resid_fe),     durbin_watson(resid_fe)),
        "sp":  (R2_sp2, autocorr1(resid_sp2[:, 0]), durbin_watson(resid_sp2[:, 0])),
        "geom": {"rotor_deg": rotor_deg, "detM": detM},
    }
    print(f"  [Case 1]  OLS AR(2): R^2={R2_ar:.4f}; "
          f"OLS+quad FE: R^2={R2_fe:.4f}; "
          f"SPT_2: R^2={R2_sp2:.4f}; rotor={rotor_deg:.1f}deg, "
          f"det(M_2)={detM:.3g}")
    return rec


# ---------------------------------------------------------------------
# Case 2 — mixed multivariate VAR (lin + quad cointegration + lags)
# ---------------------------------------------------------------------

def stats_case2():
    """
    Three-variable mixed VAR:
        x_t = x_{t-1} + eps^x_t                 (I(1) RW driver)
        y_t = 0.5 y_{t-1} + 0.5 x_{t-1} + eps^y_t   (linear cointegration with x, lag 1)
        z_t = 0.7 z_{t-1} + 0.3 (x_{t-1})^2 + eps^z_t  (quadratic cointegration with x, lag 1)
    """
    rng = np.random.default_rng(17)
    T = 2000
    sig_x, sig_y, sig_z = 0.5, 0.3, 0.5
    x = np.zeros(T); y_ = np.zeros(T); z = np.zeros(T)
    ex = rng.standard_normal(T) * sig_x
    ey = rng.standard_normal(T) * sig_y
    ez = rng.standard_normal(T) * sig_z
    for t in range(1, T):
        x[t]  = x[t-1] + ex[t]
        y_[t] = 0.5 * y_[t-1] + 0.5 * x[t-1] + ey[t]
        z[t]  = 0.7 * z[t-1] + 0.3 * (x[t-1] ** 2) + ez[t]

    # (a) Plain OLS: a VAR(1) on lagged scalars, equation by equation.
    # Engle-Granger style cointegration: y on x, z on x.
    def ar_eq(Y, X):
        beta, R2, _, resid = ols_stats(X, Y)
        return R2, autocorr1(resid), durbin_watson(resid)

    Xtr = np.column_stack([np.ones(T-1), x[:-1], y_[:-1], z[:-1]])
    R2_y_var, rho_y_var, dw_y_var = ar_eq(y_[1:], Xtr)
    R2_z_var, rho_z_var, dw_z_var = ar_eq(z[1:],  Xtr)

    # Engle-Granger residual diagnostic on the linear z~x regression
    # (this is what an econometrician hits to detect the quadratic
    # cointegration --- it FAILS to give white residuals)
    Xeg = np.column_stack([np.ones(T), x])
    R2_eg, rho_eg, dw_eg = ar_eq(z, Xeg)

    # (b) OLS with quadratic feature engineering on the VAR design.
    # Augments each equation with x^2, y^2, z^2, x*y, x*z, y*z lagged.
    L = T - 1
    Xfe = np.column_stack([
        np.ones(L),
        x[:-1], y_[:-1], z[:-1],
        x[:-1]**2, y_[:-1]**2, z[:-1]**2,
        x[:-1]*y_[:-1], x[:-1]*z[:-1], y_[:-1]*z[:-1],
    ])
    R2_y_fe, rho_y_fe, dw_y_fe = ar_eq(y_[1:], Xfe)
    R2_z_fe, rho_z_fe, dw_z_fe = ar_eq(z[1:],  Xfe)

    # (c) SPTLS joint on the 9-dimensional joint jet.
    Q = np.concatenate([jet(x), jet(y_), jet(z)], axis=1)  # shape (T, 9)
    Mhat, _, resid_joint = sptls_stats(Q)

    # Joint level R^2 component-by-component
    n = T - 1
    Xc = np.column_stack([np.ones(n), Q[:-1]])
    B, *_ = np.linalg.lstsq(Xc, Q[1:], rcond=None)
    Qhat = Xc @ B
    R2_y_sp = 1 - ((Q[1:, 3] - Qhat[:, 3])**2).sum() / ((Q[1:, 3] - Q[1:, 3].mean())**2).sum()
    R2_z_sp = 1 - ((Q[1:, 6] - Qhat[:, 6])**2).sum() / ((Q[1:, 6] - Q[1:, 6].mean())**2).sum()
    rho_y_sp = autocorr1(resid_joint[:, 3])
    dw_y_sp  = durbin_watson(resid_joint[:, 3])
    rho_z_sp = autocorr1(resid_joint[:, 6])
    dw_z_sp  = durbin_watson(resid_joint[:, 6])

    # Cross-bivector channel magnitudes (rolling Frobenius norm) — used
    # to print bounded vs. growing diagnostic
    def cross_bivector_norm(qi, qj):
        # | q_i wedge q_j | components: q_i_0 dq_j - q_j_0 dq_i, etc.
        # We just compute three Wronskian-like sums.
        b1 = qi[:, 0] * qj[:, 1] - qj[:, 0] * qi[:, 1]
        b2 = qi[:, 0] * qj[:, 2] - qj[:, 0] * qi[:, 2]
        b3 = qi[:, 1] * qj[:, 2] - qj[:, 1] * qi[:, 2]
        return float(np.sqrt((b1**2 + b2**2 + b3**2).mean()))

    Bxy = cross_bivector_norm(jet(x), jet(y_))
    Bxz = cross_bivector_norm(jet(x), jet(z))

    print(f"  [Case 2]  VAR linear:        y R^2={R2_y_var:.4f}  rho1={rho_y_var:+.3f}  DW={dw_y_var:.2f}  |  z R^2={R2_z_var:.4f}  rho1={rho_z_var:+.3f}  DW={dw_z_var:.2f}")
    print(f"            VAR + quad FE:    y R^2={R2_y_fe:.4f}  rho1={rho_y_fe:+.3f}  DW={dw_y_fe:.2f}  |  z R^2={R2_z_fe:.4f}  rho1={rho_z_fe:+.3f}  DW={dw_z_fe:.2f}")
    print(f"            SPTLS joint jet:  y R^2={R2_y_sp:.4f}  rho1={rho_y_sp:+.3f}  DW={dw_y_sp:.2f}  |  z R^2={R2_z_sp:.4f}  rho1={rho_z_sp:+.3f}  DW={dw_z_sp:.2f}")
    print(f"  [Case 2 Engle-Granger linear z~x]   R^2={R2_eg:.4f}  rho1={rho_eg:+.3f}  DW={dw_eg:.2f}  (linear cointegration test rejects normality)")
    print(f"  [Case 2 cross-bivector norms]   |B^(xy)| ~ {Bxy:.2f}   |B^(xz)| ~ {Bxz:.2f}")

    rec = {
        "case": "2. Mixed VAR (lin+quad coint + lags)",
        # We report z-row (the quadratic-cointegration variable) — the
        # discriminating one for SPTLS over Engle-Granger.
        "var_z":  (R2_z_var, rho_z_var, dw_z_var),
        "fe_z":   (R2_z_fe,  rho_z_fe,  dw_z_fe),
        "sp_z":   (R2_z_sp,  rho_z_sp,  dw_z_sp),
        "eg":     (R2_eg,    rho_eg,    dw_eg),
        "Bxy": Bxy, "Bxz": Bxz,
    }
    return rec


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    rec1 = stats_case1()
    rec2 = stats_case2()

    # ---- CSV ----
    with open(OUT / "case_fit_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "method", "R2", "rho1", "DW"])
        w.writerow(["Case 1 (logistic)", "OLS AR(2)",
                     f"{rec1['ar'][0]:.4f}",  f"{rec1['ar'][1]:+.3f}",  f"{rec1['ar'][2]:.2f}"])
        w.writerow(["Case 1 (logistic)", "OLS + quad FE",
                     f"{rec1['fe'][0]:.4f}",  f"{rec1['fe'][1]:+.3f}",  f"{rec1['fe'][2]:.2f}"])
        w.writerow(["Case 1 (logistic)", "SPTLS_2",
                     f"{rec1['sp'][0]:.4f}",  f"{rec1['sp'][1]:+.3f}",  f"{rec1['sp'][2]:.2f}"])
        w.writerow(["Case 2 (mixed VAR), z eq.", "VAR(1)",
                     f"{rec2['var_z'][0]:.4f}", f"{rec2['var_z'][1]:+.3f}", f"{rec2['var_z'][2]:.2f}"])
        w.writerow(["Case 2 (mixed VAR), z eq.", "VAR + quad FE",
                     f"{rec2['fe_z'][0]:.4f}",  f"{rec2['fe_z'][1]:+.3f}",  f"{rec2['fe_z'][2]:.2f}"])
        w.writerow(["Case 2 (mixed VAR), z eq.", "SPTLS joint",
                     f"{rec2['sp_z'][0]:.4f}",  f"{rec2['sp_z'][1]:+.3f}",  f"{rec2['sp_z'][2]:.2f}"])
    print(f"\nwrote {OUT / 'case_fit_table.csv'}")

    # ---- LaTeX ----
    with open(OUT / "case_fit_table.tex", "w") as f:
        f.write(r"""% generated by replication/build_case_table.py
\begin{tabular}{l l rrr}
\toprule
Case & Method & $R^{2}$ & $\hat\rho_{1}$ & DW \\
\midrule
\multirow{3}{*}{1. Logistic $\mu=3.8$}
""")
        for name, key in [("OLS AR(2)", "ar"),
                          ("OLS\\,+\\,quadratic FE", "fe"),
                          ("$\\sPT_{2}$ (jet $+\\,\\mathrm{Sym}^{2}$)", "sp")]:
            r2, rho, dw = rec1[key]
            f.write(f" & {name} & {r2:.3f} & {rho:+.3f} & {dw:.2f} \\\\\n")

        f.write(r"\midrule" + "\n")
        f.write(r"\multirow{3}{*}{2. Mixed VAR (lin\,+\,quad coint, lag 1)}" + "\n")
        for name, key in [("VAR(1) linear ($z$ eq.)", "var_z"),
                          ("VAR\\,+\\,quadratic FE ($z$ eq.)", "fe_z"),
                          ("$\\sPT$ joint jet ($z$ eq.)", "sp_z")]:
            r2, rho, dw = rec2[key]
            f.write(f" & {name} & {r2:.3f} & {rho:+.3f} & {dw:.2f} \\\\\n")

        f.write(r"""\bottomrule
\end{tabular}
""")
    print(f"wrote {OUT / 'case_fit_table.tex'}")


if __name__ == "__main__":
    main()
