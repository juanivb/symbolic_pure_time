"""F-000 v3 appendix --- SPTLS dictionary GEE ergodicity test.

This is the WORKING test: the Geometric Ergodicity Exponent (GEE) on
the SPTLS dictionary (windowed moment-tensor dispersion), as
developed in the E002 paper. Decision via two-layer inspection:
pointwise signature + sub-window stability.

We compare two canonical systems where the answer is known:

  1. AR(1) stationary (phi = 0.7)  --- ergodic by construction.
  2. Fredrickson-Andersen 1D KCM at T_kcm = 0.3  --- non-ergodic
     by kinetic constraint arrest, a textbook glass-physics example.

The test should classify (1) as ergodic and (2) as non-ergodic
without false positives across multiple seeds. Three seeds each;
result tabulated for the appendix.
"""
from __future__ import annotations
import math
import sys
import types
from pathlib import Path
import numpy as np

# Load gsd.ergodicity from the legacy backup (production module)
HERE = Path(__file__).resolve().parent
LIB_ROOT = Path("/sessions/eager-brave-goldberg/mnt/research_hub/research/"
                "_legacy_backups/gsd_legacy_2026-05-29")
# Stub scipy submodules used by some sibling imports
for name in ("scipy", "scipy.special", "scipy.stats", "scipy.linalg",
              "scipy.optimize", "scipy.signal"):
    sys.modules.setdefault(name, types.ModuleType(name))

# Build a minimal `gsd` package whose submodules are loaded from the
# legacy backup
import importlib.util

def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, str(LIB_ROOT / rel))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

if "gsd" not in sys.modules:
    pkg = types.ModuleType("gsd")
    pkg.__path__ = [str(LIB_ROOT)]
    sys.modules["gsd"] = pkg

embedding = _load("gsd.embedding", "embedding.py")
symbolic = _load("gsd.symbolic", "symbolic.py")
erg = _load("gsd.ergodicity", "ergodicity.py")

ergodicity_test_sptls = erg.ergodicity_test_sptls


# ---------------------------------------------------------------------
# DGPs (same as E002 phase3 ladder)
# ---------------------------------------------------------------------

def dgp_ar1(T, rng, phi=0.7):
    """Stationary AR(1) — ergodic by construction."""
    eps = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t-1] + eps[t]
    return y


def dgp_fa_kcm(T, rng, T_kcm=0.3, N=120):
    """Fredrickson-Andersen 1D Kinetically Constrained Model.
    At low T_kcm the kinetic constraint produces ergodicity breaking
    (glass-like arrest). Observable: density of active spins.
    """
    c_eq = 1.0 / (1.0 + math.exp(1.0 / max(T_kcm, 1e-3)))
    spins = (rng.random(N) < c_eq).astype(np.int8)
    n_steps = 2 * T
    rho = np.empty(T)
    dt = 0.5
    j = 0
    for step in range(n_steps):
        left = np.roll(spins, 1)
        right = np.roll(spins, -1)
        constr = (left | right).astype(bool)
        u = rng.random(N)
        flip01 = constr & (spins == 0) & (u < c_eq * dt)
        flip10 = constr & (spins == 1) & (u < (1 - c_eq) * dt)
        spins = np.where(flip01, 1, spins)
        spins = np.where(flip10, 0, spins)
        if step % 2 == 0 and j < T:
            rho[j] = float(spins.sum()) / N
            j += 1
    return rho


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=== F-000 v3 appendix: SPTLS-GEE ergodicity test ===\n", flush=True)
    print("Two systems:", flush=True)
    print("  (1) AR(1) stationary phi=0.7 — ergodic by construction.",
          flush=True)
    print("  (2) Fredrickson-Andersen KCM at T=0.3 — non-ergodic by ",
          "kinetic arrest.\n", flush=True)

    T = 2500
    n_seeds = 5
    rows = []

    print(f"{'system':<25s} {'seed':>4s} {'a':>8s} {'b':>8s} {'Π':>8s} "
          f"{'R²':>7s} {'a_std':>7s} {'b_std':>7s} {'label':<28s} {'ergodic'}",
          flush=True)
    print("-" * 117, flush=True)

    for system, dgp_fn in [
        ("AR(1) phi=0.7 (ergodic)",     dgp_ar1),
        ("FA-KCM T=0.3 (non-erg)",     dgp_fa_kcm),
    ]:
        for seed in range(n_seeds):
            rng = np.random.default_rng(8000 + seed * 31)
            y = dgp_fn(T, rng)
            res = ergodicity_test_sptls(y, n_subwindows=5)
            print(f"{system:<25s} {seed:>4d} {res.signature.a:>8.4f} "
                  f"{res.signature.b:>8.4f} {res.signature.Pi:>8.4f} "
                  f"{res.signature.r_squared:>7.4f} "
                  f"{res.a_subwindow_std:>7.4f} {res.b_subwindow_std:>7.4f} "
                  f"{res.label:<28s} {'YES' if res.ergodic else 'NO'}",
                  flush=True)
            rows.append({"system": system, "seed": seed,
                         "a": float(res.signature.a),
                         "b": float(res.signature.b),
                         "Pi": float(res.signature.Pi),
                         "R2": float(res.signature.r_squared),
                         "a_std": float(res.a_subwindow_std),
                         "b_std": float(res.b_subwindow_std),
                         "label": res.label,
                         "ergodic": bool(res.ergodic)})

    # Aggregate
    print("\n=== AGGREGATE ===\n", flush=True)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[r["system"]].append(r["ergodic"])
    correct = 0; total = 0
    for system in [r["system"] for r in rows[:1]] + [
            "FA-KCM T=0.3 (non-erg)"]:
        items = agg[system]
        rate = sum(items) / len(items)
        if "ergodic)" in system:
            want = "ergodic"; ok = (rate >= 0.8)
        else:
            want = "non-ergodic"; ok = (rate <= 0.2)
        if ok: correct += 1
        total += 1
        print(f"  {system:<25s}  ergodic_frac={rate:.2f}   "
              f"truth={want:<12s}  {'✓' if ok else '✗'}", flush=True)
    print(f"\n  Classification accuracy: {correct}/{total} = "
          f"{100*correct/total:.0f}%", flush=True)

    # CSV
    import csv
    out = HERE.parent / "results" / "appendix_ergodicity_test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
