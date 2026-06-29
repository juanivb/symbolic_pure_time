"""
GARCH(1,1) robustness — W1* and W2 size under conditional heteroskedasticity.

Verifies that the Newey-West HAC-based Wald tests maintain correct size
when innovations follow a GARCH(1,1) process with alpha=0.1, beta=0.85.

DGPs: I(0) AR(1) phi=0.5, I(1) random walk, I(2).
Both GARCH and IID innovations are run for comparison.
T = 500, N = 5000 replications.
"""
import numpy as np
from math import erfc, sqrt


def chi2_pvalue(x, k=1):
    """P-value for chi-squared(k) distribution using erfc."""
    if k == 1:
        return erfc(sqrt(x / 2))
    return None


def garch_innovations(T, alpha=0.1, beta=0.85, seed=None):
    omega = 1 - alpha - beta
    rng = np.random.RandomState(seed)
    z = rng.randn(T + 100)
    sigma2 = np.ones(T + 100)
    eps = np.zeros(T + 100)
    for t in range(1, T + 100):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
    return eps[100:]


def generate_process(eps, d, phi=0.5):
    if d == 0:
        y = np.zeros(len(eps))
        for t in range(1, len(eps)):
            y[t] = phi * y[t-1] + eps[t]
        return y
    y = eps.copy()
    for _ in range(d):
        y = np.cumsum(y)
    return y


def sptls_tests(y, T):
    Dy = np.diff(y); D2y = np.diff(Dy)
    n = T - 3
    Y = np.column_stack([y[2:T-1], Dy[1:T-2], D2y[:T-3]])
    Yp = np.column_stack([y[3:T], Dy[2:T-1], D2y[1:T-2]])

    try:
        M_hat = np.linalg.solve(Y.T @ Y, Y.T @ Yp).T
    except np.linalg.LinAlgError:
        return None, None

    c1 = np.trace(M_hat)
    c2 = (M_hat[0,0]*M_hat[1,1] - M_hat[0,1]*M_hat[1,0] +
          M_hat[0,0]*M_hat[2,2] - M_hat[0,2]*M_hat[2,0] +
          M_hat[1,1]*M_hat[2,2] - M_hat[1,2]*M_hat[2,1])
    c3 = np.linalg.det(M_hat)
    c_hat = np.array([c1, c2, c3])

    J = np.array([[1, 1], [1, 2], [0, 1]])
    resid = Yp - Y @ M_hat.T
    sigma2 = np.sum(resid**2) / (3 * n)

    Q1 = Y[:, 0:1]; Q23 = Y[:, 1:3]
    P1 = Q1 @ np.linalg.solve(Q1.T @ Q1, Q1.T)
    Q23s = Q23 - P1 @ Q23

    # Newey-West HAC
    u = resid[:, 1:3]
    S = Q23s * u
    bw = int(np.floor(4 * (n / 100)**(2/9)))
    Omega = S.T @ S / n
    for j in range(1, bw + 1):
        w = 1 - j / (bw + 1)
        Gamma_j = S[j:].T @ S[:-j] / n
        Omega += w * (Gamma_j + Gamma_j.T)
    V23_hac = np.linalg.inv(Q23s.T @ Q23s / n) @ Omega @ np.linalg.inv(Q23s.T @ Q23s / n)
    Sigma_hac = J @ V23_hac @ J.T

    # W1*: v1=(1,-1,0), H0: c1 - c2 = 1
    v1 = np.array([1, -1, 0])
    w1_diff = v1 @ c_hat - 1
    var_w1 = v1 @ Sigma_hac @ v1
    if var_w1 <= 0:
        return None, None
    W1 = n * w1_diff**2 / var_w1

    # W2: v2=(1,2,1), H0: c1 + 2c2 + c3 = 4
    v2 = np.array([1, 2, 1])
    w2_diff = v2 @ c_hat - 4
    Q3s = Q23s[:, 1:2]
    u3 = resid[:, 2:3]
    S3 = Q3s * u3
    Om3 = (S3.T @ S3)[0, 0] / n
    for j in range(1, bw + 1):
        w = 1 - j / (bw + 1)
        G3j = (S3[j:].T @ S3[:-j])[0, 0] / n
        Om3 += 2 * w * G3j
    A33 = (Q3s.T @ Q3s)[0, 0] / n
    V3_hac = Om3 / A33**2
    var_w2 = 36 * V3_hac
    if var_w2 <= 0:
        return None, None
    W2 = n * w2_diff**2 / var_w2

    return W1, W2


# ──────────────────────────────────────────────────────────────
T = 500
N = 5000
cv = 3.8415  # chi2(1) 5% critical value

print(f"GARCH(1,1) Robustness: alpha=0.1, beta=0.85, T={T}, N={N}")
print(f"{'DGP':<25} {'Innov':<8} {'W1* rej':>8} {'W2 rej':>8}")
print("-" * 55)

for innov_type in ["IID", "GARCH"]:
    for d, phi, label in [(0, 0.5, "I(0) AR(1) phi=0.5"),
                           (1, None, "I(1) RW"),
                           (2, None, "I(2)")]:
        rej_W1 = rej_W2 = valid = 0
        for i in range(N):
            if innov_type == "GARCH":
                eps = garch_innovations(T + 5, seed=i*7 + d*1000)
            else:
                rng = np.random.RandomState(i*7 + d*1000 + 99999)
                eps = rng.randn(T + 5)

            y = generate_process(eps[:T+5], d, phi=phi if phi else 0)
            y = y[:T]

            W1, W2 = sptls_tests(y, T)
            if W1 is None:
                continue
            valid += 1
            if W1 > cv:
                rej_W1 += 1
            if W2 > cv:
                rej_W2 += 1

        print(f"{label:<25} {innov_type:<8} {rej_W1/valid:8.3f} {rej_W2/valid:8.3f}")
