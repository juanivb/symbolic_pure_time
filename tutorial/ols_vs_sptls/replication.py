"""
Hand-check companion to F-000: OLS vs SPTLS on a phase-space rotation.

Reproduces every number and both figures in the companion note
`ols_vs_sptls_handcheck.tex`. Deterministic (seed 11, 2-decimal series).

    python3 replication.py        # prints numbers, writes figures/

Dependencies: numpy, matplotlib.
"""
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------
# 1. DGP: a damped rotation in phase space, observed through one coordinate.
#    state s_{t+1} = rho R(w) s_t + noise ; z_t = (s_t)_1
#    This is a clean oscillation: a single-lag scalar model cannot rotate.
# ---------------------------------------------------------------------
rng = np.random.default_rng(11)
T = 20
rho, w = 0.97, 0.62
R2 = rho * np.array([[np.cos(w), -np.sin(w)],
                     [np.sin(w),  np.cos(w)]])
s = np.array([1.0, 0.0])
z = []
for _ in range(T):
    z.append(s[0])
    s = R2 @ s + 0.02 * rng.standard_normal(2)
z = np.round(np.array(z), 3)
print("z_1..z_20:", " ".join(f"{v:.3f}" for v in z))


def metrics(pred, targ):
    rss = np.sum((targ - pred) ** 2)
    tss = np.sum((targ - targ.mean()) ** 2)
    return float(np.sqrt(np.mean((targ - pred) ** 2))), float(1 - rss / tss)


# ---------------------------------------------------------------------
# 2. OLS AR(1): z_t = c + phi z_{t-1}   (the "review" baseline)
#    Design X = [1, z_{t-1}]; Gram X^T X is the (Toeplitz) sample
#    autocovariance matrix; solve by SVD/normal equations.
# ---------------------------------------------------------------------
y = z[1:]
X1 = np.column_stack([np.ones(T - 1), z[:-1]])
G1 = X1.T @ X1                      # Gram = autocovariance block (Toeplitz)
b1 = np.linalg.solve(G1, X1.T @ y)
pred1 = X1 @ b1
rmse1, r21 = metrics(pred1, y)
print("\n[OLS AR(1)]")
print("  X^T X (sample autocovariance block) =\n", np.round(G1, 3))
print(f"  coef [c, phi] = {np.round(b1,3)}   RMSE={rmse1:.4f}  R2={r21:.3f}")

# OLS AR(2) for the honest note
y2 = z[2:]
X2 = np.column_stack([np.ones(T - 2), z[1:-1], z[:-2]])
b2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
rmse2, r22 = metrics(X2 @ b2, y2)
print(f"[OLS AR(2)] coef={np.round(b2,3)}  RMSE={rmse2:.4f}  R2={r22:.3f}")

# ---------------------------------------------------------------------
# 3. SPTLS: one-step map of the kinematic embedding q(t)=(z,Dz,D2z)
#    M-hat = (sum q(t+1) q(t)^T)(sum q(t) q(t)^T)^{-1} ; polar M=RP.
# ---------------------------------------------------------------------
Q = np.vstack([z[2:], z[2:] - z[1:-1], z[2:] - 2 * z[1:-1] + z[:-2]]).T
q0, q1 = Q[:-1], Q[1:]
C1 = q1.T @ q0          # cross-moment sum
C0 = q0.T @ q0          # Gram of the embedding
M = C1 @ np.linalg.inv(C0)
pred_q = (M @ q0.T).T
rmseS, r2S = metrics(pred_q[:, 0], q1[:, 0])   # z-component one-step
U, S, Vt = np.linalg.svd(M)
Rm = U @ Vt
if np.linalg.det(Rm) < 0:
    U = U.copy(); U[:, -1] *= -1; Rm = U @ Vt
P = Vt.T @ np.diag(S) @ Vt
ang = float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))
antisym = float(np.linalg.norm(M - M.T) / np.linalg.norm(M))
print("\n[SPTLS geometric]")
print("  sum q(t)q(t)^T =\n", np.round(C0, 3))
print("  M-hat =\n", np.round(M, 3))
print("  R (rotor) =\n", np.round(Rm, 3), f"\n  rotation = {ang:.1f} deg")
print("  P (stretch) =\n", np.round(P, 3))
print(f"  ||M-M^T||/||M|| = {antisym:.3f}   RMSE={rmseS:.4f}  R2={r2S:.3f}")

# ---------------------------------------------------------------------
# 4. Figures (<=1500 px, 150 dpi).  Not shown inline; opened separately.
# ---------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fig 1: one-step predictions, OLS AR(1) vs SPTLS
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
tt = np.arange(2, T + 1)
ax[0].plot(np.arange(1, T + 1), z, "o-", color="0.4", ms=3, lw=1, label="series $z_t$")
ax[0].plot(np.arange(2, T + 1), pred1, "s--", color="C3", ms=3, label="OLS AR(1) $\\hat z_t$")
ax[0].set_title(f"OLS AR(1):  $R^2={r21:.2f}$"); ax[0].set_xlabel("$t$"); ax[0].legend(fontsize=7)
ax[1].plot(np.arange(1, T + 1), z, "o-", color="0.4", ms=3, lw=1, label="series $z_t$")
ax[1].plot(np.arange(4, T + 1), pred_q[:, 0], "^--", color="C0", ms=3, label="SPTLS $\\hat z_t$")
ax[1].set_title(f"SPTLS $\\hat M$:  $R^2={r2S:.3f}$"); ax[1].set_xlabel("$t$"); ax[1].legend(fontsize=7)
for a in ax: a.axhline(0, color="0.85", lw=0.6, zorder=0)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fits.png")); plt.close(fig)

# Fig 2: phase space (z, Dz) — the rotation a scalar lag cannot represent
fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=150)
dz = np.diff(z)
ax.plot(z[:-1], dz, "o-", color="C0", ms=3, lw=1)
for i in range(0, len(dz) - 1, 1):
    ax.annotate("", xy=(z[i + 1], dz[i + 1]), xytext=(z[i], dz[i]),
                arrowprops=dict(arrowstyle="->", color="C0", lw=0.6, alpha=0.6))
ax.axhline(0, color="0.85", lw=0.6); ax.axvline(0, color="0.85", lw=0.6)
ax.set_xlabel("$z_t$"); ax.set_ylabel("$\\Delta z_t$")
ax.set_title("Phase space: the trajectory rotates")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "phase.png")); plt.close(fig)
print("\nfigures written to", OUT)
