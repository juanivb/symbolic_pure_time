"""
symbolic_generate_figures.py — Genera las figuras del paper Symbolic-PTLS.

Figuras generadas en figures/symbolic_*.png:
  fig_lattice         — esquema del retículo (i, j)
  fig_loglog_rates    — pendientes asintóticas Park-Phillips
  fig_bias_variance   — heatmap (p, q) × σ
  fig_irrationals     — convergencia de los cuatro irracionales canónicos
  fig_transcendentals — γ vs algebraico-en-operadores
  fig_causality       — descomposición causal por canal (i, j)

Tamaño objetivo: ≤ 1500 px largo, ≤ 150 dpi.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Fig 1: el retículo (i, j) — coordenadas operatoriales
# ---------------------------------------------------------------------

def fig_lattice():
    fig, ax = plt.subplots(figsize=(7, 5))
    P, Q = 4, 3
    for i in range(P + 1):
        for j in range(Q + 1):
            color = "#dddddd"
            edge = "#999999"
            lw = 1
            if (i, j) == (0, 1):
                color, edge, lw = "#ffd699", "#cc7a00", 2
            elif i == 0 and j >= 1:
                color = "#ffe8cc"
            elif j == 0 and i >= 1:
                color = "#cfe2f3"
            ax.add_patch(Rectangle((i - 0.4, j - 0.4), 0.8, 0.8,
                                     facecolor=color, edgecolor=edge,
                                     linewidth=lw))
            ax.text(i, j, f"$t^{{{i}}}\\,(hD)^{{{j}}}$", ha="center", va="center",
                    fontsize=10)
    ax.set_xlim(-0.7, P + 0.7)
    ax.set_ylim(-0.7, Q + 0.7)
    ax.set_xlabel("$i$ (temporal index)", fontsize=12)
    ax.set_ylabel("$j$ (derivative order)", fontsize=12)
    ax.set_title("Symbolic-PTLS basis: the $(i, j)$ lattice of operator monomials\n"
                  "Classic PTLS = $(0, 1)$ (orange); $j$-only column = velocity/curvature; "
                  "$i$-only column = explicit time-dependence",
                  fontsize=10)
    ax.set_xticks(range(P + 1))
    ax.set_yticks(range(Q + 1))
    ax.grid(False)
    ax.set_aspect("equal")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "symbolic_fig_lattice.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------
# Fig 2: pendientes asintóticas log-log
# ---------------------------------------------------------------------

def fig_loglog_rates():
    # Use Paso 3b data (var(α̂_i) vs T) for AR(1) and TVP-AR(1)
    paso3b_path = os.path.join(RESULTS_DIR, "hamilton_paso3b.json")
    with open(paso3b_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(
        axes,
        ["DGP_A", "DGP_B"],
        ["I(0): AR(1) stationary, ρ = 0.5",
         "I(0) + trend: TVP-AR(1), ρ = 0.3, γ = $10^{-5}$"],
    ):
        T_list = data[key]["T_list"]
        for i in range(3):
            vars_i = [data[key]["summaries"][str(T)]["var"][i]
                      for T in T_list]
            slope = data[key]["slopes"][i]
            expected = data[key]["expected_slopes"][i]
            ax.loglog(T_list, vars_i, "o-",
                       label=f"$\\hat\\alpha_{{{i},0}}$: slope = {slope:+.2f} (exp. {expected})")
        ax.set_xlabel("$T$", fontsize=11)
        ax.set_ylabel("$\\mathrm{Var}(\\hat\\alpha_{i,0})$", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "symbolic_fig_loglog_rates.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------
# Fig 3: heatmap bias-variance (p, q) × σ
# ---------------------------------------------------------------------

def fig_bias_variance():
    path = os.path.join(RESULTS_DIR, "hamilton_paso3c_noise.json")
    with open(path) as f:
        data = json.load(f)
    sigmas = data["config"]["sigmas"]
    pq_grid = data["config"]["pq_grid"]
    ps = sorted({pq[0] for pq in pq_grid})
    qs = sorted({pq[1] for pq in pq_grid})

    # Build matrix log(MSE) for each (p, q) at each sigma
    fig, axes = plt.subplots(1, len(sigmas), figsize=(2.5 * len(sigmas), 3.2))
    for ax, sigma in zip(axes, sigmas):
        M = np.full((len(ps), len(qs)), np.nan)
        for (p_idx, p) in enumerate(ps):
            for (q_idx, q) in enumerate(qs):
                key = f"sigma={sigma},p={p},q={q}"
                rec = data["grid"].get(key)
                if rec is not None and not np.isnan(rec["mean_test_mse"]):
                    M[p_idx, q_idx] = np.log10(max(rec["mean_test_mse"], 1e-300))
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis_r",
                        extent=[-0.5, len(qs) - 0.5, -0.5, len(ps) - 0.5])
        ax.set_xticks(range(len(qs))); ax.set_xticklabels(qs)
        ax.set_yticks(range(len(ps))); ax.set_yticklabels(ps)
        ax.set_xlabel("$q$"); ax.set_ylabel("$p$")
        ax.set_title(f"$\\sigma = {sigma:.0e}$", fontsize=10)
        # Mark the optimum
        best_idx = np.unravel_index(np.nanargmin(M), M.shape)
        ax.scatter([best_idx[1]], [best_idx[0]], marker="*",
                    s=180, color="white", edgecolor="red", linewidth=1.5,
                    label="$(p^*, q^*)$")
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="upper right")
        plt.colorbar(im, ax=ax, label="$\\log_{10}$ rel test MSE",
                      fraction=0.04)
    plt.suptitle("Bias-variance landscape: $\\log_{10}$ relative test MSE over $(p, q)$",
                  fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIGURES_DIR, "symbolic_fig_bias_variance.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------
# Fig 4: convergencia (p, q) en DGP exacto, para irracionales algebraicos
# ---------------------------------------------------------------------

def fig_irrationals():
    path = os.path.join(RESULTS_DIR, "hamilton_paso2b.json")
    with open(path) as f:
        data = json.load(f)
    B = data["experiment_B"]
    table = B["table"]
    ps = sorted({r["p"] for r in table})
    qs = sorted({r["q"] for r in table})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in qs:
        ys = []
        for p in ps:
            rec = next((r for r in table
                        if r["p"] == p and r["q"] == q), None)
            ys.append(rec["rms"] if rec else np.nan)
        ax.semilogy(ps, ys, "o-", label=f"$q = {q}$", linewidth=2,
                     markersize=8)
    ax.set_xlabel("$p$ (temporal index order)", fontsize=11)
    ax.set_ylabel("RMS fit residual", fontsize=11)
    ax.set_title("Symbolic-PTLS convergence on exact closed-form DGP $y(t) = e^{\\int P_2(s)\\,ds}$",
                  fontsize=10)
    ax.legend(title="derivative order", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(1e-15, color="gray", linestyle=":", linewidth=1)
    ax.text(0.05, 1.5e-15, "machine precision floor",
            transform=ax.get_yaxis_transform(), fontsize=8, color="gray")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "symbolic_fig_irrationals.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------
# Fig 5: γ vs algebraicos: convergencia de fracciones continuas
# ---------------------------------------------------------------------

def fig_transcendentals():
    path = os.path.join(RESULTS_DIR, "anharmonic_oq2_transcendentals.json")
    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    # γ via harmonic series
    ns = data["harmonic_minus_log"]["ns_sampled"]
    errs = data["harmonic_minus_log"]["errors"]
    ax.loglog(ns, errs, "s-", label="$\\gamma$ via $H_n - \\ln(n)$  "
              "(slope $\\approx -1$, algebraic)",
              linewidth=2, markersize=9, color="firebrick")

    # Algebraic-in-operators: CF convergents
    colors = {"φ": "steelblue", "√2": "darkgreen", "e": "darkorange", "π": "purple"}
    for name, info in data["convergents"].items():
        if name == "γ":
            continue
        errs = np.array(info["errors"])
        ks = np.arange(1, len(errs) + 1)
        # Filter out exact zeros (saturated to floor)
        nonzero = errs > 1e-17
        ax.loglog(ks[nonzero], errs[nonzero], "o-",
                   label=f"${name}$ via CF convergents (geometric)",
                   linewidth=2, markersize=6, color=colors[name])

    ax.set_xlabel("convergent index $k$  /  partial sum $n$", fontsize=11)
    ax.set_ylabel("absolute error from truth", fontsize=11)
    ax.set_title("Convergence of canonical irrationals:\n"
                  "algebraic-in-operators $\\{e, \\pi, \\phi, \\sqrt{2}\\}$ "
                  "(geometric)  vs  $\\gamma$ (algebraic $1/n$)",
                  fontsize=10)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(1e-15, color="gray", linestyle=":", linewidth=1)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "symbolic_fig_transcendentals.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------
# Fig 6: descomposición causal por canal (i, j)
# ---------------------------------------------------------------------

def fig_causality():
    path = os.path.join(RESULTS_DIR, "symbolic_causality_example.json")
    with open(path) as f:
        data = json.load(f)

    # Two-panel bar chart: scenario A (level coupling) and scenario B (velocity coupling)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    scenarios = [("scenario_A", "fit_Y", "expected_Y",
                   "Escenario A:  $Y_{n+1} = \\rho_Y\\,Y_n + c\\,X_n + \\varepsilon$\n"
                   "(level coupling $X \\to Y$)"),
                  ("scenario_B", "fit_Y", "expected_Y",
                   "Escenario B:  $Y_{n+1} = \\rho_Y\\,Y_n + c\\,\\Delta X_n + \\varepsilon$\n"
                   "(velocity coupling $X \\to Y$)")]
    for ax, (sckey, fit_key, exp_key, title) in zip(axes, scenarios):
        sc = data[sckey]
        labels = list(sc[fit_key].keys())
        est = [sc[fit_key][k] for k in labels]
        exp = [sc[exp_key][k] for k in labels]
        x = np.arange(len(labels))
        width = 0.4
        ax.bar(x - width / 2, est, width, label="estimated", color="steelblue")
        ax.bar(x + width / 2, exp, width, label="truth", color="darkorange",
                alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace(".t^0.", " ") for l in labels],
                            rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("coefficient value", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
    plt.suptitle("Graded causal reading: coupling channel is identifiable from $(i, j)$",
                  fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIGURES_DIR, "symbolic_fig_causality.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("Generating figures for Symbolic-PTLS paper...")
    fig_lattice()
    fig_loglog_rates()
    fig_bias_variance()
    fig_irrationals()
    fig_transcendentals()
    fig_causality()
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()
