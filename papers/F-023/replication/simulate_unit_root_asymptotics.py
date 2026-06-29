"""
Unit root asymptotics — SPTLS vs classical OLS.

Demonstrates that SPTLS on the kinematic jet q(t) = (y_t, Δy_t, Δ²y_t)
yields √T-normal characteristic polynomial coefficients even under I(1),
while classical OLS φ̂ follows the Phillips (1987) non-standard distribution.

DGPs: I(0) AR(1), I(1) random walk, I(2).
T = 500, N = 5000 replications.
"""
import numpy as np

np.random.seed(42)
N = 5000
T = 500


def build_jet(y):
    return np.column_stack([y[2:], y[2:]-y[1:-1], y[2:]-2*y[1:-1]+y[:-2]])


def sptls(jet):
    Q = jet[:-1]; Qp = jet[1:]
    M, _, _, _ = np.linalg.lstsq(Q, Qp, rcond=None)
    return M.T


def charpoly(M):
    c1 = np.trace(M)
    c2 = (M[0,0]*M[1,1] - M[0,1]*M[1,0] +
          M[0,0]*M[2,2] - M[0,2]*M[2,0] +
          M[1,1]*M[2,2] - M[1,2]*M[2,1])
    c3 = np.linalg.det(M)
    return np.array([c1, c2, c3])


def gen_ar1(T, phi, burn=100):
    y = np.zeros(T + burn)
    eps = np.random.randn(T + burn)
    for t in range(1, T+burn):
        y[t] = phi*y[t-1] + eps[t]
    return y[burn:]


# ──────────────────────────────────────────────────────────────
# Part 1: SPTLS characteristic polynomial — all DGPs
# ──────────────────────────────────────────────────────────────
for d, phi, label in [(0, 0.95, "I(0) AR(1) phi=0.95"),
                       (1, None, "I(1) RW"),
                       (2, None, "I(2)")]:
    c_all = np.zeros((N, 3))
    for i in range(N):
        if d == 0:
            y = gen_ar1(T+3, phi)
        elif d == 1:
            y = np.cumsum(np.random.randn(T+3))
        else:
            y = np.cumsum(np.cumsum(np.random.randn(T+3)))
        M = sptls(build_jet(y))
        c_all[i] = charpoly(M)

    print(f"\n{'='*60}")
    print(f"DGP: {label}, T={T}, N={N}")
    print(f"{'='*60}")
    print(f"  {'Coeff':<8} {'Mean':>10} {'√T·Std':>10} {'Skew':>8} {'Kurt':>8}")
    for j, name in enumerate(["c1=tr", "c2", "c3=det"]):
        v = c_all[:, j]
        m, s = np.mean(v), np.std(v)
        sk = np.mean(((v-m)/s)**3) if s > 1e-10 else 0
        ku = np.mean(((v-m)/s)**4) - 3 if s > 1e-10 else 0
        print(f"  {name:<8} {m:10.5f} {np.sqrt(T)*s:10.4f} {sk:8.3f} {ku:8.3f}")

# ──────────────────────────────────────────────────────────────
# Part 2: Classical OLS φ̂ under I(1) — Phillips distribution
# ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Comparison: classical OLS phi_hat under I(1)")
print(f"{'='*60}")
phi_hats = np.zeros(N)
for i in range(N):
    y = np.cumsum(np.random.randn(T+3))
    y = y[3:]
    X = y[:-1].reshape(-1, 1)
    phi_hats[i] = np.linalg.lstsq(X, y[1:], rcond=None)[0][0]

centered = T * (phi_hats - 1)
m, s = np.mean(centered), np.std(centered)
sk = np.mean(((centered-m)/s)**3)
ku = np.mean(((centered-m)/s)**4) - 3
print(f"  T*(phi_hat - 1): mean={m:.3f}, std={s:.3f}, skew={sk:.3f}, kurt={ku:.3f}")
print(f"  Non-normal (Phillips contamination): skew and kurtosis ≠ 0")
