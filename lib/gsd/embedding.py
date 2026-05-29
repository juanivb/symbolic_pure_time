"""
gsd.embedding — the kinematic embedding (the lingua franca of SPTLS).

A scalar series z_t is lifted to its instantaneous jet

    q(t) = (z_t, Dz_t, D2z_t)          (value, velocity, acceleration)

with finite differences Dz_t = (z_t - z_{t-1})/h and D2z_t the second
difference. This is the forward, kinematic representation SPTLS fits a
one-step map on — as opposed to the backward, lagged-value representation
OLS uses.

For d series the per-series jets are stacked into q(t) in R^{3d}; the SPTLS
operator is then a 3d x 3d matrix whose off-diagonal 3x3 blocks carry the
cross-series coupling.

numpy only.
"""
from __future__ import annotations
import numpy as np

__all__ = ["kinematic_embedding", "stack_embedding"]


def kinematic_embedding(y, h=1.0):
    """Return the (T-2, 3) jet array [z, Dz, D2z] for a 1-D series y.

    Uses backward finite differences so q(t) depends only on present and
    past samples (causal). Rows are aligned to t = 2 .. T-1 (0-based 2..).
    """
    y = np.asarray(y, float).ravel()
    z = y[2:]
    dz = (y[2:] - y[1:-1]) / h
    d2z = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (h * h)
    return np.column_stack([z, dz, d2z])


def stack_embedding(Y, h=1.0):
    """Multivariate embedding. Y is (T, d) (columns = series). Returns the
    (T-2, 3d) stacked jet array with blocks [z_i, Dz_i, D2z_i] per series,
    column order  series-major: [z0,Dz0,D2z0, z1,Dz1,D2z1, ...]."""
    Y = np.atleast_2d(np.asarray(Y, float))
    if Y.shape[0] < Y.shape[1] and Y.shape[0] <= 3:
        # tolerate a (d, T) input only when it is unambiguous
        pass
    blocks = [kinematic_embedding(Y[:, j], h) for j in range(Y.shape[1])]
    return np.hstack(blocks)
