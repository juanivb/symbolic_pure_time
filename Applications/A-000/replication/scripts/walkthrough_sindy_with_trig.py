"""
Walkthrough — SINDy with system-aware trigonometric / gravitational
features. The honest comparator for PTLS path (ii): if SINDy is given
the *same* atomic dictionary as PTLS, does it also reach machine
precision? If yes, the PTLS contribution is not raw RMSE — it is the
pure-time parametrisation and the framing.
"""
from __future__ import annotations

import json
import os

import numpy as np


G_GRAVITY = 9.81


# ---------------------------------------------------------------------
# Pendulum
# ---------------------------------------------------------------------

def pendulum_rhs(z, g=G_GRAVITY):
    th1, th2, w1, w2 = z
    phi = th1 - th2; s_phi, c_phi = np.sin(phi), np.cos(phi)
    D = 3.0 - np.cos(2 * phi)
    n1 = -3*g*np.sin(th1) - g*np.sin(th1-2*th2) - 2*s_phi*(w2*w2 + w1*w1*c_phi)
    n2 =  2*s_phi*(2*w1*w1 + 2*g*np.cos(th1) + w2*w2*c_phi)
    return np.array([w1, w2, n1/D, n2/D])


def integrate_pendulum(z0, dt, T):
    Z = np.empty((T, 4)); Z[0] = z0
    for t in range(1, T):
        Z[t] = Z[t-1] + dt * pendulum_rhs(Z[t-1])
    return Z


def pendulum_atomic_features(Z):
    """The same EOM-matched atoms used in PTLS path (ii), divided by D
    where appropriate. SINDy has access to the *same* dictionary."""
    th1, th2, w1, w2 = Z.T
    phi = th1 - th2
    s_phi, c_phi = np.sin(phi), np.cos(phi)
    invD = 1.0 / (3.0 - np.cos(2 * phi))
    return np.column_stack([
        np.sin(th1) * invD,
        np.sin(th1 - 2 * th2) * invD,
        s_phi * w2 * w2 * invD,
        s_phi * c_phi * w1 * w1 * invD,
        s_phi * w1 * w1 * invD,
        s_phi * np.cos(th1) * invD,
        s_phi * c_phi * w2 * w2 * invD,
    ])


def sindy_with_trig_pendulum(Z, dt):
    """Predict ż from z_t with the EOM-atomic dictionary."""
    z_t = Z[:-1]; dz = (Z[1:] - Z[:-1]) / dt
    feats = np.hstack([z_t, pendulum_atomic_features(z_t)])
    coef, *_ = np.linalg.lstsq(feats, dz, rcond=None)
    res = feats @ coef - dz
    return float(np.sqrt(np.mean(np.sum(res * res, axis=1)))), feats.shape[1]


# ---------------------------------------------------------------------
# Three-body (figure-eight)
# ---------------------------------------------------------------------

def three_body_rhs(z, masses, G=1.0):
    r = z[:6].reshape(3, 2); v = z[6:].reshape(3, 2); a = np.zeros_like(r)
    for i in range(3):
        for j in range(3):
            if i == j: continue
            d = r[j] - r[i]; r3 = (d @ d) ** 1.5
            a[i] += G * masses[j] * d / r3
    return np.concatenate([v.ravel(), a.ravel()])


def integrate_threebody(z0, masses, G, dt, T):
    Z = np.empty((T, 12)); Z[0] = z0
    for t in range(1, T):
        Z[t] = Z[t-1] + dt * three_body_rhs(Z[t-1], masses, G)
    return Z


def figure_eight_ic():
    p1 = np.array([0.97000436, -0.24308753])
    p3 = -p1; p2 = np.zeros(2)
    v3 = np.array([-0.93240737, -0.86473146])
    v1 = -v3 / 2.0; v2 = -v3 / 2.0
    return np.concatenate([p1, p2, p3, v1, v2, v3])


def gravitational_features(Z):
    T = Z.shape[0]
    R = Z[:, :6].reshape(T, 3, 2)
    feats = []
    for i in range(3):
        for j in range(3):
            if i == j: continue
            d = R[:, j] - R[:, i]
            r3 = (np.sum(d * d, axis=1) ** 1.5)[:, None]
            feats.append(d / r3)
    return np.hstack(feats)


def sindy_with_grav_threebody(Z, dt):
    z_t = Z[:-1]; dz = (Z[1:] - Z[:-1]) / dt
    feats = np.hstack([z_t, gravitational_features(z_t)])
    coef, *_ = np.linalg.lstsq(feats, dz, rcond=None)
    res = feats @ coef - dz
    return float(np.sqrt(np.mean(np.sum(res * res, axis=1)))), feats.shape[1]


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.normpath(os.path.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "walkthrough_sindy_with_trig.json")

    print("=" * 72)
    print(" Walkthrough — SINDy with system-aware features (the honest comparator)")
    print("=" * 72)

    # Pendulum
    z0 = np.array([np.pi - 0.1, np.pi - 0.1, 0.0, 0.0])
    Z = integrate_pendulum(z0, 1e-3, 5000)
    rmse, nfeat = sindy_with_trig_pendulum(Z, dt=1e-3)
    print(f"\nPendulum SINDy + trig atoms (dt=1e-3, T=5000):")
    print(f"  features = {nfeat}, RMSE = {rmse:.4e}")

    # Three-body
    masses = np.array([1.0, 1.0, 1.0])
    z0_3b = figure_eight_ic()
    Z3 = integrate_threebody(z0_3b, masses, 1.0, 1e-4, 5000)
    rmse3, nfeat3 = sindy_with_grav_threebody(Z3, dt=1e-4)
    print(f"\nThree-body SINDy + gravitational atoms (dt=1e-4, T=5000):")
    print(f"  features = {nfeat3}, RMSE = {rmse3:.4e}")

    out = {
        "pendulum_sindy_with_trig":   {"rmse": rmse,  "n_features": nfeat},
        "three_body_sindy_with_grav": {"rmse": rmse3, "n_features": nfeat3},
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
