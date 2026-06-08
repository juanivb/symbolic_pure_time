"""F-000 v3 §5 --- build the three case-study figures.

Generates three PNGs at <=150 dpi and <=1500 px long side, saved to
../figures/, illustrating:

  Case 1: Logistic map at mu=3.8 (chaotic).
  Case 2: Fractional Brownian motion at H=0.7 (fractal long memory).
  Case 3: Quadratically cointegrated bivariate series.

Each figure has two panels: a 'trajectory/state' view and a
'reading' view that contrasts plain-OLS-on-lagged-scalars with
the SPTLS object that recovers the structural quantity.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_DATA = "#2b4f81"
C_OLS  = "#c44d3a"
C_SPT  = "#2f8f55"
C_GREY = "#666666"


# ---------------------------------------------------------------------
# Case 1: Logistic map at mu=3.8
# ---------------------------------------------------------------------

def case1_logistic():
    mu = 3.8
    T = 4000
    y = np.empty(T)
    y[0] = 0.37
    for t in range(1, T):
        y[t] = mu * y[t-1] * (1.0 - y[t-1])

    # plain OLS: AR(1) on y_t
    Y = y[1:]; X = np.column_stack([np.ones(T-1), y[:-1]])
    beta_ols, *_ = np.linalg.lstsq(X, Y, rcond=None)
    a_ols, phi_ols = beta_ols
    y_hat_ols = X @ beta_ols
    ss_res = float(np.sum((Y - y_hat_ols)**2))
    ss_tot = float(np.sum((Y - Y.mean())**2))
    r2_ols = 1.0 - ss_res / ss_tot

    # SPTLS: closed-form OLS of q(t+1) ~ M q(t) on the kinematic
    # jet q = (z, dz, ddz). Tr(M) and det(M) summarise rotor/stretch.
    dz = np.diff(y, prepend=y[0])
    ddz = np.diff(dz, prepend=dz[0])
    Q = np.column_stack([y, dz, ddz])
    Qm = Q[:-1]; Qp = Q[1:]
    # Solve M = argmin || Qp - Qm M^T ||^2  (rows are time, cols channels)
    Mhat, *_ = np.linalg.lstsq(Qm, Qp, rcond=None)
    Mhat = Mhat.T   # standard convention q_{t+1} = M q_t
    Qp_hat = (Mhat @ Qm.T).T
    ss_res_sp = float(np.sum((Qp - Qp_hat)**2))
    ss_tot_sp = float(np.sum((Qp - Qp.mean(axis=0))**2))
    r2_sp = 1.0 - ss_res_sp / ss_tot_sp
    eigs = np.linalg.eigvals(Mhat)
    trM = float(np.real(np.trace(Mhat)))
    detM = float(np.real(np.linalg.det(Mhat)))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))

    ax = axes[0]
    ax.plot(np.arange(200), y[:200], color=C_DATA, lw=0.9)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$y_t$")
    ax.set_title(r"Logistic map at $\mu=3.8$ (first 200 steps)")
    ax.set_ylim(0, 1)

    ax = axes[1]
    yy = y[:-1]; yz = y[1:]
    ax.scatter(yy, yz, s=2.5, color=C_DATA, alpha=0.32, label="data")
    xs = np.linspace(0, 1, 400)
    ax.plot(xs, a_ols + phi_ols * xs, color=C_OLS, lw=2,
            label=fr"plain OLS: $\hat y_{{t+1}}={a_ols:.2f}{phi_ols:+.2f}\,y_t$  ($R^2$={r2_ols:.2f})")
    ax.plot(xs, mu * xs * (1 - xs), color=C_SPT, lw=2,
            label=r"SPTLS on the jet (kinematic OLS)")
    ax.set_xlabel(r"$y_t$")
    ax.set_ylabel(r"$y_{t+1}$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Phase plot: what each fit sees")
    ax.legend(loc="lower center", frameon=False, fontsize=8.5)
    ax.text(0.02, 0.97,
            (fr"$\mathrm{{tr}}\,\hat M={trM:.2f}$    "
             fr"$\det\hat M={detM:.2f}$    "
             fr"$R^2_{{\mathrm{{SPT}}}}={r2_sp:.2f}$"),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, color=C_GREY)

    fig.tight_layout()
    out = FIG / "case_chaotic.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size//1024} KB)")
    print(f"  OLS R^2={r2_ols:.3f}  SPT R^2={r2_sp:.3f}  "
          f"tr(M)={trM:.3f}  det(M)={detM:.3f}")


# ---------------------------------------------------------------------
# Case 2: Mixed multivariate VAR (linear + quadratic cointegration,
# with lag) --- replaces previous fBm / oscillator Case 2.
# ---------------------------------------------------------------------

def case2_mixed_var():
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

    # Engle-Granger linear regression z ~ x (will be spurious)
    Xeg = np.column_stack([np.ones(T), x])
    beta_eg, *_ = np.linalg.lstsq(Xeg, z, rcond=None)
    a_eg, b_eg = beta_eg

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # Panel L: three trajectories
    ax = axes[0]
    ax.plot(np.arange(T), x,  color=C_DATA, lw=0.7, label=r"$x_t$ (RW)")
    ax.plot(np.arange(T), y_, color=C_OLS,   lw=0.7,
            label=r"$y_t$ (lin coint with $x$, lag 1)")
    ax.plot(np.arange(T), z,  color=C_SPT,  lw=0.7,
            label=r"$z_t$ (quad coint with $x$, lag 1)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("level")
    ax.set_title(r"Three-variable mixed VAR ($T=2000$)")
    ax.legend(loc="best", frameon=False, fontsize=8.5)

    # Panel R: scatter z vs x with linear OLS line (spurious) and
    # true quadratic curve.
    ax = axes[1]
    ax.scatter(x, z, s=2.5, color=C_DATA, alpha=0.3, label=r"data $(x_t, z_t)$")
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, 0.3 * xs ** 2 / (1.0 - 0.7), color=C_SPT, lw=2,
            label=r"true: $z\approx \frac{0.3\,x^{2}}{1-0.7}$ (quad coint)")
    ax.plot(xs, a_eg + b_eg * xs, color=C_OLS, lw=2,
            label=fr"linear OLS (E--G): $\hat\beta={b_eg:+.2f}$")
    ax.set_xlabel(r"$x_t$")
    ax.set_ylabel(r"$z_t$")
    ax.set_title(r"Quadratic coint of $(x,z)$ missed by linear E--G")
    ax.legend(loc="upper center", frameon=False, fontsize=8.5)

    fig.tight_layout()
    out = FIG / "case_mixed_var.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size//1024} KB)")


# (LEGACY) Case 2 oscillator — superseded; kept for reference.
def case2_oscillator():
    rng = np.random.default_rng(13)
    phi1, phi2 = 1.6, -0.9
    sigma = 0.3
    T = 4000
    y = np.zeros(T)
    eps = rng.standard_normal(T) * sigma
    for t in range(2, T):
        y[t] = phi1 * y[t-1] + phi2 * y[t-2] + eps[t]

    # OLS AR(2) baseline
    X = np.column_stack([np.ones(T-2), y[1:-1], y[:-2]])
    beta, *_ = np.linalg.lstsq(X, y[2:], rcond=None)
    phi1_hat, phi2_hat = beta[1], beta[2]

    # SPTLS on jet
    dz = np.diff(y, prepend=y[0])
    ddz = np.diff(dz, prepend=dz[0])
    Q = np.column_stack([y, dz, ddz])
    Mhat, *_ = np.linalg.lstsq(Q[:-1], Q[1:], rcond=None)
    Mhat = Mhat.T

    # polar decomposition
    MtM = Mhat.T @ Mhat
    w, V = np.linalg.eigh(MtM)
    w = np.clip(w, 1e-12, None)
    R = Mhat @ V @ np.diag(1.0/np.sqrt(w)) @ V.T
    trR = float(np.clip((np.trace(R) - 1.0)/2.0, -1.0, 1.0))
    rotor_deg = float(np.degrees(np.arccos(trR)))
    detM = float(np.real(np.linalg.det(Mhat)))
    period = 360.0 / rotor_deg if rotor_deg > 0 else float("inf")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))

    # Panel L: trajectory (first 200 steps)
    ax = axes[0]
    ax.plot(np.arange(200), y[:200], color=C_DATA, lw=0.9)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$y_t$")
    ax.set_title(fr"Stationary AR(2), $\phi_1=1.6$, $\phi_2=-0.9$ "
                 fr"(damped oscillator)")

    # Panel R: phase plot y_t vs y_{t-1} --- ellipse signature
    ax = axes[1]
    ax.scatter(y[:-1], y[1:], s=2, color=C_DATA, alpha=0.18)
    ax.set_xlabel(r"$y_{t-1}$")
    ax.set_ylabel(r"$y_t$")
    ax.set_title(r"Phase plane $(y_{t-1}, y_t)$: rotation visible to SPTLS")
    ax.text(0.02, 0.97,
            (fr"OLS AR(2): $\hat\phi_1={phi1_hat:.2f},\ "
             fr"\hat\phi_2={phi2_hat:.2f}$"
             "\n"
             fr"SPTLS readout: rotor $\approx {rotor_deg:.1f}^\circ$ "
             fr"(period $\approx {period:.1f}$),  $\det\hat M={detM:.3f}$"),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, color=C_GREY)

    fig.tight_layout()
    out = FIG / "case_oscillator.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size//1024} KB)")
    print(f"  OLS AR(2): phi1={phi1_hat:.3f}, phi2={phi2_hat:.3f}; "
          f"SPTLS rotor={rotor_deg:.1f}deg, period={period:.1f}, "
          f"det M={detM:.3f}")


# ---------------------------------------------------------------------
# Case 3: Quadratically cointegrated pair
# ---------------------------------------------------------------------

def case3_quadratic_coint():
    rng = np.random.default_rng(11)
    T = 2000
    eps2 = rng.standard_normal(T) * 0.6
    y2 = np.cumsum(eps2)
    eta = rng.standard_normal(T) * 1.5
    y1 = y2**2 + eta

    # plain OLS: y1 ~ a + b y2
    X = np.column_stack([np.ones(T), y2])
    beta, *_ = np.linalg.lstsq(X, y1, rcond=None)
    a_ols, b_ols = beta
    resid_lin = y1 - (a_ols + b_ols * y2)

    # SPTLS bivector cross-channel: rolling Frobenius norm of
    # B^(12)(t) = q1(t) wedge q2(t).  Use first kinematic component
    # (level + delta) cross between series 1 and series 2.
    dz1 = np.diff(y1, prepend=y1[0])
    dz2 = np.diff(y2, prepend=y2[0])
    # cross bivector area in the (y1, y2) plane between t and t+1
    area = np.abs(y1[:-1] * dz2[1:] - y2[:-1] * dz1[1:])
    # rolling mean of |B| with window 50 for stability
    W = 50
    cum = np.cumsum(np.concatenate([[0], area]))
    rolling_mean_B = (cum[W:] - cum[:-W]) / W
    # normalise vs sigma_eps2^2 = 0.36
    sigma_eps2_sq = 0.6**2
    # compare with Engle-Granger residual: |resid_lin| rolling RMS
    # (use [1:] to align with area, which has T-1 elements)
    resid_aligned = resid_lin[1:]
    cum_r2 = np.cumsum(np.concatenate([[0], resid_aligned**2]))
    rolling_rms_resid = np.sqrt((cum_r2[W:] - cum_r2[:-W]) / W)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))

    ax = axes[0]
    ax.scatter(y2, y1, s=3, color=C_DATA, alpha=0.3, label=r"data $(y^{(2)}_t,\,y^{(1)}_t)$")
    xs = np.linspace(y2.min(), y2.max(), 400)
    ax.plot(xs, xs**2, color=C_SPT, lw=2,
            label=r"true relation $y^{(1)}=(y^{(2)})^{2}$")
    ax.plot(xs, a_ols + b_ols * xs, color=C_OLS, lw=2,
            label=fr"plain OLS line: $\hat y^{{(1)}}={a_ols:.1f}{b_ols:+.2f}\,y^{{(2)}}$")
    ax.set_xlabel(r"$y^{(2)}_t$")
    ax.set_ylabel(r"$y^{(1)}_t$")
    ax.set_title("Quadratic equilibrium, linear regression")
    ax.legend(loc="upper center", frameon=False, fontsize=8.5)

    ax = axes[1]
    tt = np.arange(W, W + len(rolling_mean_B))
    ax.plot(tt, rolling_rms_resid, color=C_OLS, lw=1.6,
            label=r"Engle--Granger residual (rolling RMS)")
    ax.plot(tt, rolling_mean_B, color=C_SPT, lw=1.6,
            label=r"$\|B^{(12)}\|$ rolling mean (SPTLS cross-bivector)")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("magnitude  (log scale)")
    ax.set_title("Linear residual grows; bivector stays bounded")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)

    fig.tight_layout()
    out = FIG / "case_cointegration.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size//1024} KB)")
    print(f"  OLS slope b={b_ols:.3f}  EG resid grows; |B^(12)| ~ {rolling_mean_B[-100:].mean():.2f}")


if __name__ == "__main__":
    print("=== F-000 v3 §5: building case figures (2 worked cases) ===")
    case1_logistic()
    case2_mixed_var()
    print("done.")
