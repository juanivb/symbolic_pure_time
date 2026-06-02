"""F-004 E23: trajectory entropy H_tray vs Shannon H of the marginal.

The classical Shannon entropy of a random variable X with distribution
p(x) is H_Shannon = -E[log p(X)]. It is a property of the DISTRIBUTION.
For an i.i.d. trajectory of X-valued steps the same number is the
entropy rate.

If we treat probability as a property of TRAJECTORIES rather than
distributions, the natural quantity is the conditional surprise of
the next step given history, averaged along the trajectory:

   H_tray  =  -E_t[ log p( X_{t+1} | history_t ) ]
           =  -E_t[ log p( eps_t ) ]

where eps_t = X_{t+1} - M̂(history_t) is the residual of the
best one-step algebraic predictor (the SPTLS operator). H_tray is
the entropy of the SUBSTRATE'S RESIDUE — what the substrate cannot
predict.

Two limit cases let us see the meaning:

(A) Fair coin: M̂ ≡ 0 (the past does not predict the next toss).
    The residual eps_t equals X_{t+1} itself. So H_tray = H_Shannon.
    The substrate gives up; Shannon's measure is the right one.

(B) AR(1)  X_{t+1} = phi · X_t + sigma · z_t with z ~ N(0,1):
    The marginal of X is N(0, sigma^2/(1-phi^2)). Its Shannon entropy
    is high (broad distribution). But the SUBSTRATE's M̂ on the
    kinematic jet reads phi exactly; the residual eps_t = sigma · z_t
    has variance sigma^2 ≪ sigma^2/(1-phi^2). So
    H_tray ≈ (1/2) log(2 pi e sigma^2)
          ≪ (1/2) log(2 pi e sigma^2/(1-phi^2)) ≈ H_Shannon.

The numerical demo here:
  - coin: 10000 tosses, verify H_tray = H_Shannon = log 2.
  - AR(1) with phi ∈ {0.0, 0.3, 0.6, 0.9, 0.99}, sigma = 1:
    show H_tray stays ≈ (1/2) log(2 pi e),
    H_Shannon = (1/2) log(2 pi e / (1 - phi^2)) diverges as phi -> 1,
    and the gap H_Shannon - H_tray = -(1/2) log(1 - phi^2)
    is exactly the operational information the substrate reads from
    the past.

Output: a CSV with (phi, n, H_tray_est, H_Shannon_est, gap_est) and
a short text summary printed to stdout.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "empirical" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# ============================== fair coin ================================
def fair_coin_demo(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=n).astype(float)  # 0/1
    # marginal Shannon entropy
    p = np.array([(X == 0).mean(), (X == 1).mean()])
    p = np.clip(p, 1e-12, 1.0)
    H_shannon = -np.sum(p * np.log(p))

    # trajectory entropy: predictor M̂ = best one-step linear predictor on
    # the kinematic jet (X_t, ΔX_t, ΔΔX_t). For an i.i.d. fair coin this
    # collapses to predicting X_{t+1} from past history -- which carries
    # no information. We compute eps_t = X_{t+1} - mean(X), apply the
    # plug-in Bernoulli density on the prediction.
    # Equivalent formulation: empirical conditional distribution of
    # X_{t+1} given history of length 1: should be 50/50 regardless of X_t.
    # So eps_t cloud = X cloud, and H_tray = H_shannon.
    p_cond = []
    for prev in [0.0, 1.0]:
        mask = (X[:-1] == prev)
        if mask.sum() == 0: continue
        future = X[1:][mask]
        p1 = future.mean()
        p_cond.append([1 - p1, p1])
    p_cond = np.array(p_cond)
    p_cond = np.clip(p_cond, 1e-12, 1.0)
    H_tray = -np.mean([np.sum(p * np.log(p)) for p in p_cond])
    return {
        "case": "fair_coin",
        "phi": 0.0, "sigma": float("nan"), "n": n,
        "H_shannon": round(H_shannon, 4),
        "H_tray": round(H_tray, 4),
        "gap": round(H_shannon - H_tray, 4),
        "log_2": round(math.log(2), 4),
    }


# ============================== AR(1) ===================================
def ar1_demo(phi, n=20000, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n) * sigma
    X = np.zeros(n)
    for t in range(1, n):
        X[t] = phi * X[t-1] + z[t]

    # Marginal Shannon entropy: empirical estimate via Gaussian formula
    # applied to the observed variance (the marginal is Gaussian).
    var_X = X.var()
    H_shannon = 0.5 * math.log(2 * math.pi * math.e * var_X)

    # Trajectory entropy: fit M̂ on the kinematic jet (X_t, ΔX_t, ΔΔX_t)
    # but for AR(1) it's enough to fit phi_hat from X_{t+1} = phi · X_t + eps.
    # eps_t = X_{t+1} - phi_hat · X_t, then H_tray is the Gaussian entropy
    # of eps_t.
    A = X[:-1].reshape(-1, 1)
    b = X[1:]
    phi_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    eps = b - A @ phi_hat
    var_eps = eps.var()
    H_tray = 0.5 * math.log(2 * math.pi * math.e * var_eps)

    # Closed-form check: H_shannon - H_tray = -(1/2) log(1 - phi^2)
    gap_th = -0.5 * math.log(1.0 - phi**2) if phi**2 < 1 else float("nan")

    return {
        "case": f"ar1_phi_{phi}",
        "phi": phi, "sigma": sigma, "n": n,
        "H_shannon": round(H_shannon, 4),
        "H_tray": round(H_tray, 4),
        "gap": round(H_shannon - H_tray, 4),
        "gap_theory": round(gap_th, 4) if not math.isnan(gap_th) else "inf",
    }


# ============================== driver ===================================
def main():
    rows = []
    print(f"\n{'case':<22s} | {'phi':<5s} | {'H_shannon':<10s} | {'H_tray':<10s} | "
          f"{'gap':<8s} | check", flush=True)
    print("-" * 86, flush=True)

    r = fair_coin_demo()
    rows.append(r)
    print(f"{r['case']:<22s} | --    | {r['H_shannon']:<10.4f} | {r['H_tray']:<10.4f} | "
          f"{r['gap']:<8.4f} | log 2 = {r['log_2']:.4f}", flush=True)

    for phi in [0.0, 0.3, 0.6, 0.9, 0.99]:
        r = ar1_demo(phi)
        rows.append(r)
        gth = r['gap_theory']
        print(f"{r['case']:<22s} | {phi:<5.2f} | {r['H_shannon']:<10.4f} | "
              f"{r['H_tray']:<10.4f} | {r['gap']:<8.4f} | -1/2·log(1-phi²) = {gth}",
              flush=True)

    out = RESULTS / "E23_trajectory_entropy.csv"
    fields = sorted({k for r in rows for k in r.keys()})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nwrote {out}", flush=True)

    # Interpretation
    print("\n--- Interpretation ---", flush=True)
    print("Fair coin: H_tray = H_Shannon = log 2 ≈ 0.6931.", flush=True)
    print("           The substrate's M̂ has nothing to read; Shannon is the right measure.", flush=True)
    print("AR(1):     as phi increases, H_Shannon grows (marginal variance inflates),", flush=True)
    print("           but H_tray stays near (1/2) log(2 pi e) ≈ 1.4189.", flush=True)
    print("           The gap H_Shannon - H_tray = -(1/2) log(1 - phi^2) is the", flush=True)
    print("           OPERATIONAL INFORMATION the substrate reads from the past.", flush=True)


if __name__ == "__main__":
    main()
