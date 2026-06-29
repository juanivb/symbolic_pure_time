"""
Power analysis — W1* test vs ADF under near-unit-root alternatives.

Compares rejection rates of the SPTLS-based W1* test (chi-squared critical
values) against ADF with BIC lag selection under:
  - H0: I(1) random walk (size)
  - H1: AR(1) with phi = 0.99, 0.95, 0.90, 0.80, 0.50 (power)
  - I(2) process (W1* should not reject; ADF should not reject)

T = 500, N = 2000 replications, alpha = 5%.
"""
import numpy as np

np.random.seed(2026)
N = 2000
T = 500
ALPHA = 0.05

# Calibrated asymptotic variances (from simulate_tables1_2.py)
V1, V2, V3 = 0.984**2, 1.409**2, 1.021**2

# Critical values
CHI2_CV_3 = 7.815   # chi2(3) at 5%
ADF_CV = -1.941     # ADF 5% critical value (no constant, no trend)


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
    return c1, c2, c3


def sigma_res(M, jet):
    Q = jet[:-1]; Qp = jet[1:]
    return max(np.mean((Qp - Q @ M.T)**2), 1e-12)


def wald_calibrated(M, jet, c0=(1., 0., 0.)):
    sig2 = sigma_res(M, jet)
    c1, c2, c3 = charpoly(M)
    return T / sig2 * ((c1-c0[0])**2/V1 + (c2-c0[1])**2/V2 + (c3-c0[2])**2/V3)


def adf_bic(y, max_lags=8):
    dy = np.diff(y)
    best_bic = np.inf
    best_t = np.nan
    for lag in range(max_lags + 1):
        t0 = lag + 1
        if t0 >= len(dy) - 5:
            break
        Y = dy[t0:]
        X = y[t0:-1].reshape(-1, 1)
        if lag > 0:
            X = np.hstack([X, np.column_stack([dy[t0-k-1:len(dy)-k-1] for k in range(lag)])])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            s2 = np.sum((Y - X @ beta)**2) / (len(Y) - X.shape[1])
            bic = len(Y) * np.log(s2) + X.shape[1] * np.log(len(Y))
            if bic < best_bic:
                best_bic = bic
                se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
                best_t = beta[0] / se
        except Exception:
            pass
    return best_t


def gen_ar1(n, phi, burn=100):
    y = np.zeros(n + burn)
    eps = np.random.randn(n + burn)
    for t in range(1, n + burn):
        y[t] = phi * y[t-1] + eps[t]
    return y[burn:]


# ──────────────────────────────────────────────────────────────
# Size and power
# ──────────────────────────────────────────────────────────────
dgps = [
    ("I(1) RW",    lambda: np.cumsum(np.random.randn(T+3)),               True),
    ("AR phi=0.99", lambda: gen_ar1(T+3, 0.99),                           False),
    ("AR phi=0.95", lambda: gen_ar1(T+3, 0.95),                           False),
    ("AR phi=0.90", lambda: gen_ar1(T+3, 0.90),                           False),
    ("AR phi=0.80", lambda: gen_ar1(T+3, 0.80),                           False),
    ("AR phi=0.50", lambda: gen_ar1(T+3, 0.50),                           False),
    ("I(2)",        lambda: np.cumsum(np.cumsum(np.random.randn(T+3))),   False),
]

print(f"Power analysis: N={N}, T={T}, alpha={ALPHA}")
print(f"{'DGP':<16} {'W1*':>8} {'ADF':>8}  {'note'}")
print("-" * 45)

for name, gen, is_size in dgps:
    w_rej = adf_rej = 0
    for _ in range(N):
        y = gen()
        jet = build_jet(y)
        M = sptls(jet)
        W = wald_calibrated(M, jet)
        if W > CHI2_CV_3:
            w_rej += 1
        t = adf_bic(y[:T])
        if not np.isnan(t) and t < ADF_CV:
            adf_rej += 1
    note = "[size]" if is_size else "[power]"
    print(f"{name:<16} {w_rej/N:8.3f} {adf_rej/N:8.3f}  {note}")
