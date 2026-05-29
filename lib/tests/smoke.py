"""Smoke test for the numpy-only gsd package. Run:  python3 tests/smoke.py
Exercises every module without matplotlib (plotting tested separately)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import gsd

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

print("gsd", gsd.__version__)

# ---- univariate: rotation -------------------------------------------------
print("\n[univariate rotation]")
y = gsd.datasets.rotation(T=120)
ar2 = gsd.OLS(2).fit(y)
sp = gsd.SPTLS().fit(y)
check("AR(2) R2 high on linear oscillation", ar2.rsquared > 0.9)
check("SPTLS z-R2 high", sp.rsquared_mean > 0.9)
rep = sp.report()
s0 = rep.series[0]
check("rotor angle > 5 deg (oscillatory)", s0["rotor_angle_deg"] > 5)
check("verdict mentions oscillatory", "oscillatory" in s0["verdict"])
check("grade energies sum to 1", abs(sum(s0["grade_energy"].values()) - 1) < 1e-9)
lam, g0, g2 = rep.pgcf_curves(0)
check("PGCF grade-2 non-zero", float(np.max(g2)) > 0)
print(ar2.summary())
print(rep.summary())

# ---- univariate: logistic (OLS fails, grade-2 needed) ---------------------
print("\n[univariate logistic]")
z = gsd.datasets.logistic(T=120)
check("AR(2) R2 low on chaos", gsd.OLS(2).fit(z).rsquared < 0.4)
rep2 = gsd.SPTLS().fit(z).report()
print("  logistic verdict:", rep2.series[0]["verdict"])

# ---- random walk: unit root ----------------------------------------------
print("\n[random walk]")
rw = gsd.datasets.random_walk(T=300, seed=1)
rwrep = gsd.SPTLS().fit(rw).report()
check("random walk near unit root", abs(rwrep.series[0]["spectral_radius"] - 1) < 0.15)
print("  rw verdict:", rwrep.series[0]["verdict"])

# ---- multivariate: cointegrated VAR --------------------------------------
print("\n[multivariate cointegrated]")
Y = gsd.datasets.cointegrated_var(T=400)
var = gsd.OLS(1).fit(Y)
check("VAR returns per-equation R2 dict", isinstance(var.rsquared, dict) and len(var.rsquared) == 2)
mrep = gsd.SPTLS().fit(Y).report()
check("influence matrix is 2x2", mrep.influence.shape == (2, 2))
check("cointegration rank >= 1", mrep.coint_rank >= 1)
check("cointegrating vector length 2", len(mrep.cointegrating_vector) == 2)
print(mrep.summary())

# ---- validation -----------------------------------------------------------
print("\n[walk-forward CV]")
cmp = gsd.validation.compare({"AR(2)": gsd.OLS(2), "SPTLS": gsd.SPTLS()},
                             y, start=60, scheme="expanding")
print(cmp["_table"])
check("CV produced metrics for both", "AR(2)" in cmp and "SPTLS" in cmp)

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
