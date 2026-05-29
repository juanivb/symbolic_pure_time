"""
gsd.algebra — minimal Cl(3,0) / gl(3) primitives for the white-box reading.

The SPTLS one-step operator is a real 3x3 matrix acting on the kinematic
jet (z, Dz, D2z). Everything a practitioner reads off it is a decomposition
of that matrix:

    polar       M = R P          rotor (orientation) x stretch (gain)
    grades      M = s I + S0 + A scalar / traceless-symmetric / antisymmetric

These map onto the Clifford grade picture: the antisymmetric part is the
grade-2 (bivector) content — the genuinely non-abelian, rotational channel
that a scalar-lag (OLS) model cannot name; the symmetric part is the abelian
"coaxial shadow" (the gains OLS does see).

numpy only.
"""
from __future__ import annotations
import numpy as np

__all__ = [
    "polar_decomposition", "rotor_angle", "rotor_axis", "antisymmetry_index",
    "grade_decomposition", "grade_energy", "spectrum",
]


def polar_decomposition(M):
    """Right polar decomposition M = R @ P with R a proper rotation (det=+1)
    and P symmetric positive semidefinite. Read from the SVD; the rotor R is
    the orientation part, P the stretch part."""
    M = np.asarray(M, float)
    U, s, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:                # enforce a proper rotation
        U = U.copy(); U[:, -1] *= -1
        s = s.copy(); s[-1] *= -1
        R = U @ Vt
    P = Vt.T @ np.diag(s) @ Vt
    P = 0.5 * (P + P.T)                      # symmetrise away round-off
    return R, P


def rotor_angle(R, degrees=True):
    """Rotation angle of a 3x3 proper-rotation matrix, via trace(R)=1+2cos(theta)."""
    R = np.asarray(R, float)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.arccos(c))
    return np.degrees(ang) if degrees else ang


def rotor_axis(R):
    """Unit rotation axis of a 3x3 proper rotation (eigenvector with eigenvalue 1)."""
    R = np.asarray(R, float)
    w, V = np.linalg.eig(R)
    k = int(np.argmin(np.abs(w - 1.0)))
    axis = np.real(V[:, k])
    n = np.linalg.norm(axis)
    return axis / n if n > 0 else axis


def antisymmetry_index(M):
    """||M - M^T|| / ||M||  — the share of the operator that is rotational
    (grade-2). 0 = purely coaxial (OLS-like); ~1.4 = strongly rotating."""
    M = np.asarray(M, float)
    den = np.linalg.norm(M)
    return float(np.linalg.norm(M - M.T) / den) if den > 0 else 0.0


def grade_decomposition(M):
    """Split a real square matrix into the three gl(n) channels that mirror
    the Clifford grades:
        scalar     s I            (isotropic gain, grade-0)
        stretch    S0             (traceless symmetric, the abelian shadow)
        rotation   A              (antisymmetric, the grade-2 bivector)
    Returns (scalar_coeff, S0, A) with M = scalar*I + S0 + A."""
    M = np.asarray(M, float)
    n = M.shape[0]
    sym = 0.5 * (M + M.T)
    A = 0.5 * (M - M.T)
    s = np.trace(sym) / n
    S0 = sym - s * np.eye(n)
    return float(s), S0, A


def grade_energy(M, normalize=True):
    """Frobenius energy in each channel returned by grade_decomposition.
    Returns dict {scalar, stretch, rotation}. If normalize, they sum to 1."""
    s, S0, A = grade_decomposition(M)
    n = M.shape[0]
    e = {
        "scalar": float((s ** 2) * n),
        "stretch": float(np.sum(S0 ** 2)),
        "rotation": float(np.sum(A ** 2)),
    }
    if normalize:
        tot = sum(e.values()) or 1.0
        e = {k: v / tot for k, v in e.items()}
    return e


def spectrum(M):
    """Eigenvalues of the one-step operator, sorted by magnitude (descending).
    The SPTLS analogue of AR roots: a complex pair's modulus is the per-step
    damping/growth, its argument the rotation frequency (rad/step)."""
    M = np.asarray(M, float)
    w = np.linalg.eigvals(M)
    return w[np.argsort(-np.abs(w))]
