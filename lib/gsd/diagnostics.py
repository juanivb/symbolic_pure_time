"""
gsd.diagnostics — the white-box reading of a fitted SPTLS operator.

In the spirit of what ACF/PACF and the unit-circle root plot are to
Box-Jenkins: a set of *descriptive* readings that help characterise a process
rather than forecast it. Everything here is read off the one-step operator
M-hat and the embedding Gram C0 — no extra fitting.

These are **descriptive readings of the fitted operator, not hypothesis
tests.** Formal tests built on these readings (for stationarity, shared
directions, directional coupling, etc.) are under active development and will
arrive with their null distributions; until then, read the numbers as
indicative geometry, not as test decisions.

Univariate readings (phase space)
    polar readout      rotor angle / axis, stretch gains, antisymmetry index
    operator spectrum  eigenvalues of M-hat (the SPTLS "roots"): modulus =
                       per-step damping/growth, argument = rotation frequency
    grade energy       scalar / stretch / rotation shares of the dynamics
    regime reading     a descriptive label from the spectral radius + rotor
                       angle (stable / near-unit / explosive; oscillatory?)
    PGCF grades        grade-0 (power-spectrum shadow) vs grade-2 magnitude

Multivariate readings (cross structure)
    process nature     per-series own-block spectrum, read descriptively as
                       stable / near-unit / explosive, oscillatory or not
    cross-effects      directed influence matrix from the off-diagonal blocks,
                       split into coaxial (level/gain) and rotational (phase)
                       channels, and resolved by source channel (does y_j, its
                       velocity, or its acceleration move y_i?)
    shared directions  near-null directions of the joint embedding Gram, read
                       as candidate near-stationary combinations of the levels
                       (a descriptive companion to a cointegration test, not a
                       test in itself)

A directed causal *graph* over the influence matrix is a natural next
iteration; for now the influence matrix carries the same information as a
table.

numpy only for the readings; matplotlib only inside .plot().
"""
from __future__ import annotations
import numpy as np
from . import algebra as _alg

__all__ = ["SPTLSReport"]


def _block(M, i, j):
    return M[3 * i:3 * i + 3, 3 * j:3 * j + 3]


class SPTLSReport:
    def __init__(self, result):
        self.res = result
        self.M = result.M
        self.M_white = result.M_white
        self.C0 = result.C0
        self.d = result.d
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        d = self.d
        # ---- per-series own-block reading (diagonal 3x3 blocks) --------
        self.series = []
        for i in range(d):
            Bi = _block(self.M, i, i)
            BiW = _block(self.M_white, i, i)
            R, P = _alg.polar_decomposition(BiW)
            eig = _alg.spectrum(Bi)
            rho = float(np.max(np.abs(eig)))
            ge = _alg.grade_energy(Bi)
            ang = _alg.rotor_angle(R)
            self.series.append({
                "rotor_angle_deg": ang,
                "rotor_axis": _alg.rotor_axis(R),
                "stretch_gains": np.sort(np.linalg.eigvalsh(P))[::-1],
                "antisymmetry": _alg.antisymmetry_index(Bi),
                "spectral_radius": rho,
                "eigenvalues": eig,
                "grade_energy": ge,
                "verdict": _verdict(rho, ang),
            })
        # ---- joint stability + cross influence (multivariate) ----------
        self.spectral_radius = float(np.max(np.abs(_alg.spectrum(self.M))))
        if d > 1:
            G = np.zeros((d, d)); Gstr = np.zeros((d, d)); Grot = np.zeros((d, d))
            for i in range(d):
                for j in range(d):
                    if i == j:
                        continue
                    B = _block(self.M, i, j)
                    G[i, j] = np.linalg.norm(B)
                    _, S0, A = _alg.grade_decomposition(B)
                    Gstr[i, j] = np.linalg.norm(S0) + abs(np.trace(B)) / 3.0 * np.sqrt(3)
                    Grot[i, j] = np.linalg.norm(A)
            self.influence = G
            self.influence_coaxial = Gstr
            self.influence_rotational = Grot
            # source-channel resolution: z-row of each block tells which of
            # (level, velocity, accel) of j drives the level of i
            self.channel_effect = np.zeros((d, d, 3))
            for i in range(d):
                for j in range(d):
                    if i == j:
                        continue
                    self.channel_effect[i, j] = self.M[3 * i, 3 * j:3 * j + 3]
            # ---- shared-direction reading -----------------------------------
            self._cointegration()

    def _cointegration(self):
        d = self.d
        # normalise the joint Gram to a correlation-like matrix so the scale
        # of each series does not dominate the eigenvalues
        D = np.sqrt(np.clip(np.diag(self.C0), 1e-15, None))
        Cn = self.C0 / np.outer(D, D)
        w, V = np.linalg.eigh(0.5 * (Cn + Cn.T))
        order = np.argsort(w)              # smallest first = most stationary
        self.coint_eigenvalues = w[order]
        self.coint_vectors = V[:, order]
        # restrict the most-stationary direction to the z-components (the
        # levels) -> the cointegrating-vector analogue across the d series
        v0 = V[:, order[0]]
        zload = np.array([v0[3 * i] for i in range(d)])
        n = np.linalg.norm(zload)
        self.cointegrating_vector = zload / n if n > 0 else zload
        # rank analogue: number of eigenvalues below a small threshold
        thr = 0.05 * float(np.mean(w))
        self.coint_rank = int(np.sum(w < max(thr, 1e-9)))

    # ------------------------------------------------------------------
    def pgcf_curves(self, series=0, lam=None, n=60):
        """Empirical PGCF along xi = lambda e1 for one series' jet:
        grade-0 magnitude (the classical-CF / power-spectrum shadow) and
        grade-2 magnitude (the non-abelian coordinate beyond it)."""
        if lam is None:
            lam = np.linspace(0.0, 3.0, n)
        # reconstruct the series jet from the stored embedding
        zc = 3 * series
        q = np.vstack([self.res._q0[:, zc:zc + 3], self.res._q1[-1:, zc:zc + 3]])
        g0 = []; g2 = []
        for L in lam:
            rho = L * np.hypot(q[:, 1], q[:, 2]) + 1e-12
            c = np.cos(rho); sinc = np.sin(rho) / rho
            g0.append(np.mean(c))
            g12 = np.mean(sinc * L * q[:, 1]); g13 = np.mean(sinc * L * q[:, 2])
            g2.append(np.hypot(g12, g13))
        return lam, np.abs(np.array(g0)), np.array(g2)

    # ------------------------------------------------------------------
    def summary(self):
        L = []
        L.append("=" * 70)
        L.append(" SPTLS diagnostic report" + (f"  ({self.d} series)" if self.d > 1 else ""))
        L.append(" (descriptive readings of the fitted operator, not tests)")
        L.append("=" * 70)
        for i, s in enumerate(self.series):
            tag = f"series y{i}" if self.d > 1 else "process"
            L.append(f"\n[{tag}]  regime (descriptive): {s['verdict']}")
            L.append(f"  spectral radius   = {s['spectral_radius']:.4f}"
                     f"   (>1 explosive, ~1 unit-root, <1 stable)")
            L.append(f"  rotor angle       = {s['rotor_angle_deg']:.2f} deg/step"
                     f"   (0 = no rotation)")
            L.append(f"  antisymmetry idx  = {s['antisymmetry']:.3f}"
                     f"   (0 coaxial / OLS-like, ->1.41 strongly rotating)")
            gs = s["stretch_gains"]
            L.append(f"  stretch gains     = [{gs[0]:.3f}, {gs[1]:.3f}, {gs[2]:.3f}]")
            ge = s["grade_energy"]
            L.append(f"  grade energy      = scalar {ge['scalar']:.2f} | "
                     f"stretch {ge['stretch']:.2f} | rotation {ge['rotation']:.2f}")
            ev = s["eigenvalues"]
            evs = ", ".join(f"{e.real:+.3f}{e.imag:+.3f}i" for e in ev)
            L.append(f"  operator spectrum = {evs}")
        if self.d > 1:
            L.append("\n" + "-" * 70)
            L.append(f" joint spectral radius = {self.spectral_radius:.4f}")
            L.append("\n directed influence  ||block(i<-j)||   (row i driven by column j)")
            L.append(self._mat_str(self.influence))
            L.append("\n   ... rotational (phase-coupled) part:")
            L.append(self._mat_str(self.influence_rotational))
            L.append("\n source-channel effect on each level y_i  (level / vel / accel of y_j):")
            for i in range(self.d):
                for j in range(self.d):
                    if i == j:
                        continue
                    c = self.channel_effect[i, j]
                    L.append(f"   y{j} -> y{i}:  z {c[0]:+.3f}  Dz {c[1]:+.3f}  D2z {c[2]:+.3f}")
            L.append("\n shared-direction reading (joint embedding Gram, descriptive):")
            L.append(f"   smallest correlation-eigenvalues = "
                     + ", ".join(f"{v:.4f}" for v in self.coint_eigenvalues[:min(self.d, 4)]))
            L.append(f"   near-stationary directions (count) = {self.coint_rank}")
            cv = self.cointegrating_vector
            L.append("   most-stationary level combination = "
                     + " + ".join(f"({cv[i]:+.3f})·y{i}" for i in range(self.d)))
        L.append("=" * 70)
        return "\n".join(L)

    @staticmethod
    def _mat_str(M):
        return "\n".join("   " + "  ".join(f"{v:7.3f}" for v in row) for row in M)

    def __repr__(self):
        return self.summary()

    # ------------------------------------------------------------------
    def plot(self, series=0, show=True):
        """Visual companion to .summary(). Returns the matplotlib Figure.
        (Pedagogical/expository — practitioners usually read the table.)"""
        import matplotlib.pyplot as plt
        if self.d == 1:
            fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
            self._plot_spectrum(ax[0], 0)
            self._plot_grades(ax[1], 0)
            self._plot_pgcf(ax[2], 0)
        else:
            fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
            self._plot_spectrum(ax[0], series)
            self._plot_influence(ax[1])
            self._plot_coint(ax[2])
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def _plot_spectrum(self, ax, i):
        ev = self.series[i]["eigenvalues"]
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color="0.8", lw=1)
        ax.scatter(ev.real, ev.imag, c="C0", zorder=3)
        ax.axhline(0, color="0.9", lw=.6); ax.axvline(0, color="0.9", lw=.6)
        ax.set_aspect("equal"); ax.set_title(f"operator spectrum (y{i})" if self.d > 1 else "operator spectrum")
        ax.set_xlabel("Re"); ax.set_ylabel("Im")

    def _plot_grades(self, ax, i):
        ge = self.series[i]["grade_energy"]
        ax.bar(list(ge.keys()), list(ge.values()), color=["0.5", "C2", "C0"])
        ax.set_ylim(0, 1); ax.set_title("grade energy")

    def _plot_pgcf(self, ax, i):
        lam, g0, g2 = self.pgcf_curves(i)
        ax.plot(lam, g0, label="grade-0 (spectrum shadow)")
        ax.plot(lam, g2, label="grade-2 (beyond)")
        ax.set_xlabel(r"$\lambda$"); ax.set_title("PGCF grades"); ax.legend(fontsize=8)

    def _plot_influence(self, ax):
        im = ax.imshow(self.influence, cmap="viridis")
        ax.set_title("directed influence  ||i<-j||")
        ax.set_xlabel("source j"); ax.set_ylabel("target i")
        ax.figure.colorbar(im, ax=ax, fraction=0.046)

    def _plot_coint(self, ax):
        w = self.coint_eigenvalues
        ax.bar(range(len(w)), w, color="C3")
        ax.set_title("joint-Gram eigenvalues\n(small = stationary combos)")
        ax.set_xlabel("direction")


def _verdict(rho, angle, tol=0.04):
    # descriptive regime label from the operator's spectral radius — NOT a
    # stationarity test (a formal test with a null distribution is in
    # development). "near-unit" flags rho ~ 1, not an I(1) test decision.
    if rho > 1 + tol:
        base = "explosive"
    elif rho > 1 - tol:
        base = "near-unit"
    else:
        base = "stable"
    if angle > 5.0:
        base += ", oscillatory"
    return base
