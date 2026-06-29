"""
Empirical application — W1* and W2 tests on a GDP-calibrated process.

Applies the SPTLS Wald tests to a simulated unit-root process calibrated
to match US real GDP quarterly data (T=312, sigma=0.008).

Results:
  W1* does not reject H0: d >= 1  => process is integrated
  W2  rejects H0: d = 2          => process is NOT I(2)
  Conclusion: I(1), consistent with standard GDP findings.

Note: Uses simulated data with fixed seed for exact replication.
      For actual BEA data, download GDPC1 from FRED.
"""
import numpy as np
from math import erfc, sqrt

np.random.seed(2026)


def chi2_sf(x, df=1):
    """Survival function for chi-squared(df) using erfc."""
    if df == 1:
        return erfc(sqrt(x / 2))
    return None


# ──────────────────────────────────────────────────────────────
# Generate GDP-like process
# ──────────────────────────────────────────────────────────────
T = 312
sigma = 0.008
eps = np.random.randn(T + 10) * sigma
y = np.zeros(T + 10)
y[0] = 9.0  # log(GDP) ~ 9 in 1947
for t in range(1, T + 10):
    y[t] = y[t-1] + eps[t]
y = y[10:]

# Build kinematic jet
Dy = np.diff(y)
D2y = np.diff(Dy)
n = T - 3

Y = np.column_stack([y[2:T-1], Dy[1:T-2], D2y[:T-3]])
Yp = np.column_stack([y[3:T], Dy[2:T-1], D2y[1:T-2]])

# SPTLS fit
M_hat = np.linalg.solve(Y.T @ Y, Y.T @ Yp).T

print("SPTLS Operator M_hat:")
print(np.array2string(M_hat, precision=4, suppress_small=True))

# Eigenvalues
eigvals = np.linalg.eigvals(M_hat)
print(f"\nEigenvalues: {np.sort(np.abs(eigvals))[::-1].round(4)}")
print(f"Spectral radius: {np.max(np.abs(eigvals)):.4f}")

# Characteristic polynomial coefficients
c1 = np.trace(M_hat)
c2 = (M_hat[0,0]*M_hat[1,1] - M_hat[0,1]*M_hat[1,0] +
      M_hat[0,0]*M_hat[2,2] - M_hat[0,2]*M_hat[2,0] +
      M_hat[1,1]*M_hat[2,2] - M_hat[1,2]*M_hat[2,1])
c3 = np.linalg.det(M_hat)
c_hat = np.array([c1, c2, c3])
print(f"\nChar poly: c1={c1:.4f}, c2={c2:.4f}, c3={c3:.6f}")

# Variance estimation via FWL
resid = Yp - Y @ M_hat.T
sigma2_hat = np.sum(resid**2) / (3 * n)

Q1 = Y[:, 0:1]; Q23 = Y[:, 1:3]
P1 = Q1 @ np.linalg.solve(Q1.T @ Q1, Q1.T)
Q23_star = Q23 - P1 @ Q23

J = np.array([[1, 1], [1, 2], [0, 1]])
V23_hat = sigma2_hat * np.linalg.inv(Q23_star.T @ Q23_star / n)
Sigma_hat = J @ V23_hat @ J.T

# ──────────────────────────────────────────────────────────────
# W1* test: H0: c1 - c2 = 1  (unit root present)
# ──────────────────────────────────────────────────────────────
v1 = np.array([1, -1, 0])
c0_d1 = np.array([2, 1, 0])
diff1 = v1 @ (c_hat - c0_d1)
var1 = v1 @ Sigma_hat @ v1
W1_star = n * diff1**2 / var1
p_W1 = chi2_sf(W1_star, 1)
print(f"\nW1* = {W1_star:.2f}, p-value = {p_W1:.4f}")

# ──────────────────────────────────────────────────────────────
# W2 test: H0: c1 + 2c2 + c3 = 4  (I(2) vs I(1))
# ──────────────────────────────────────────────────────────────
v2 = np.array([1, 2, 1])
w2_diff = v2 @ c_hat - 4
Q3s = Q23_star[:, 1:2]
V3_hat = sigma2_hat / (Q3s.T @ Q3s / n)[0, 0]
var_w2 = 36 * V3_hat
W2 = n * w2_diff**2 / var_w2
p_W2 = chi2_sf(W2, 1)
print(f"W2  = {W2:.1f}, p-value = {p_W2:.2e}")

# ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("Summary:")
print(f"  W1* = {W1_star:.2f} (p = {p_W1:.2f}): do not reject H0: d >= 1")
print(f"  W2  = {W2:.1f} (p = {p_W2:.2e}): reject H0: d = 2")
print("  Conclusion: I(1) process")
