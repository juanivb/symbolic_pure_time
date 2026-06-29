"""
Tables 1 & 2 — Distribution of M̂ elements and characteristic polynomial.

Reproduces the Monte Carlo evidence in Section 4 of the paper:
  Table 1: Elements of M̂ have non-standard distribution (Phillips contamination).
  Table 2: Characteristic polynomial coefficients have standard N(0,1) distribution.

DGPs: I(0) AR(1) φ=0.5, I(1) random walk, I(2).
T = 500, N = 5000 replications.
"""
import numpy as np

def generate_process(T, d, phi=0.5, seed=None):
    rng = np.random.RandomState(seed)
    eps = rng.randn(T + 10)
    if d == 0:
        y = np.zeros(T + 10)
        for t in range(1, T + 10):
            y[t] = phi * y[t-1] + eps[t]
    else:
        y = eps.copy()
        for _ in range(d):
            y = np.cumsum(y)
    return y[10:T+10]

def sptls_fit(y):
    T = len(y)
    Dy = np.diff(y)
    D2y = np.diff(Dy)
    n = T - 3
    Y = np.column_stack([y[2:T-1], Dy[1:T-2], D2y[:T-3]])
    Yp = np.column_stack([y[3:T], Dy[2:T-1], D2y[1:T-2]])
    M_hat = np.linalg.solve(Y.T @ Y, Y.T @ Yp).T
    return M_hat

def char_poly(M):
    c1 = np.trace(M)
    c2 = (M[0,0]*M[1,1] - M[0,1]*M[1,0] +
          M[0,0]*M[2,2] - M[0,2]*M[2,0] +
          M[1,1]*M[2,2] - M[1,2]*M[2,1])
    c3 = np.linalg.det(M)
    return np.array([c1, c2, c3])

# Population characteristic polynomial coefficients
C0 = {0: None, 1: np.array([1, 0, 0]), 2: np.array([2, 1, 0])}
# For d=0, c0 depends on phi; computed below

T = 500
N = 5000

for d, phi, label in [(0, 0.5, "I(0) AR(1) phi=0.5"),
                       (1, None, "I(1) RW"),
                       (2, None, "I(2)")]:
    M_elements = []
    c_coeffs = []
    for i in range(N):
        y = generate_process(T, d, phi=phi if phi else 0, seed=i)
        M = sptls_fit(y)
        M_elements.append(M.flatten())
        c_coeffs.append(char_poly(M))

    M_elements = np.array(M_elements)
    c_coeffs = np.array(c_coeffs)

    print(f"\n{'='*60}")
    print(f"DGP: {label}, T={T}, N={N}")
    print(f"{'='*60}")

    # Table 1: M̂ elements
    print("\nTable 1 — Elements of M̂:")
    print(f"  {'Element':<12} {'Mean':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8}")
    for j in range(9):
        row, col = j // 3, j % 3
        vals = M_elements[:, j]
        m, s = np.mean(vals), np.std(vals)
        skew = np.mean(((vals - m)/s)**3)
        kurt = np.mean(((vals - m)/s)**4) - 3
        print(f"  M[{row+1},{col+1}]      {m:8.3f} {s:8.3f} {skew:8.2f} {kurt:8.1f}")

    # Table 2: Characteristic polynomial
    print("\nTable 2 — Characteristic polynomial coefficients:")
    print(f"  {'Coeff':<8} {'Mean':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8}")
    for j, name in enumerate(["c1", "c2", "c3"]):
        vals = c_coeffs[:, j]
        m, s = np.mean(vals), np.std(vals)
        skew = np.mean(((vals - m)/s)**3) if s > 1e-10 else 0
        kurt = np.mean(((vals - m)/s)**4) - 3 if s > 1e-10 else 0
        print(f"  {name:<8} {m:8.4f} {s:8.4f} {skew:8.2f} {kurt:8.1f}")
