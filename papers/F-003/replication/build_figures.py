"""
build_figures.py — generate the 10 figures of F-003.

Outputs PNGs at 150 dpi, max 1500 px on the long side, in ../figures/.

Sections:
  fig01_vector_add.png             §2.1
  fig02_wedge.png                  §2.2
  fig03_geometric_product.png      §2.3
  fig04_trivector.png              §2.4
  fig05_rotor.png                  §2.5
  fig06_dual.png                   §2.6
  fig07_translation.png            §3.1
  fig08_mult_by_t.png              §3.2
  fig09_clifford_circle.png        §5.1
  fig10_weyl_helix.png             §5.2
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "axes.linewidth":   0.9,
    "figure.dpi":       100,
})

# ---------------------------------------------------------------- helpers ----

def arrow2(ax, p, q, **kw):
    a = FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=15,
                         lw=1.6, **kw)
    ax.add_patch(a)


def save(fig, name):
    f = OUT / name
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {f}")


# ------------------------------------------- fig 1 — vector addition ------

def fig01_vector_add():
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.array([2.0, 1.0])
    y = np.array([-1.0, 3.0])
    z = x + y
    o = np.zeros(2)
    arrow2(ax, o, x, color="C0")
    arrow2(ax, o, y, color="C1")
    arrow2(ax, o, z, color="black")
    # parallelogram
    pgon = Polygon([o, x, z, y], closed=True, alpha=0.12, color="gray")
    ax.add_patch(pgon)
    # dashed continuations
    ax.plot([x[0], z[0]], [x[1], z[1]], "k--", lw=0.7)
    ax.plot([y[0], z[0]], [y[1], z[1]], "k--", lw=0.7)
    # labels
    ax.text(1.1, 0.3, r"$\mathbf{x} = 2\mathbf{e}_1 + \mathbf{e}_2$",
            color="C0", fontsize=11)
    ax.text(-1.4, 1.4, r"$\mathbf{y} = -\mathbf{e}_1 + 3\mathbf{e}_2$",
            color="C1", fontsize=11)
    ax.text(0.6, 2.5, r"$\mathbf{x}+\mathbf{y} = \mathbf{e}_1 + 4\mathbf{e}_2$",
            color="black", fontsize=11)
    ax.set_xlim(-2.5, 3.5); ax.set_ylim(-0.5, 4.7)
    ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel(r"$\mathbf{e}_1$"); ax.set_ylabel(r"$\mathbf{e}_2$")
    ax.set_title("Vector addition: the diagonal of the parallelogram")
    save(fig, "fig01_vector_add.png")


# ------------------------------------------- fig 2 — wedge --------------

def fig02_wedge():
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.array([2.0, 0.0])
    y = np.array([0.0, 1.0])
    o = np.zeros(2)
    # oriented parallelogram with arrow on the boundary
    pgon = Polygon([o, x, x+y, y], closed=True, alpha=0.22, color="C2")
    ax.add_patch(pgon)
    arrow2(ax, o, x, color="C0")
    arrow2(ax, x, x+y, color="C1")
    arrow2(ax, x+y, y, color="C0", linestyle="--")
    arrow2(ax, y, o, color="C1", linestyle="--")
    ax.annotate("",
                xy=(1.0, 0.7), xytext=(0.5, 0.3),
                arrowprops=dict(arrowstyle="->", color="C2", lw=2,
                                connectionstyle="arc3,rad=-0.4"))
    ax.text(0.65, 0.6, r"orientation", color="C2",
            fontsize=10, fontstyle="italic")
    ax.text(0.85, -0.18, r"$\mathbf{x}$", color="C0", fontsize=12)
    ax.text(-0.27, 0.45, r"$\mathbf{y}$", color="C1", fontsize=12)
    ax.text(0.9, 1.15, r"$\mathbf{x}\wedge\mathbf{y} = 2\,\mathbf{e}_{12}$",
            color="C2", fontsize=12)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.5, 1.6)
    ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel(r"$\mathbf{e}_1$"); ax.set_ylabel(r"$\mathbf{e}_2$")
    ax.set_title(r"Wedge product as oriented parallelogram in $\mathrm{Cl}^2(3,0)$")
    save(fig, "fig02_wedge.png")


# ------------------------------------- fig 3 — geometric product --------

def fig03_geometric_product():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.array([2.0, 1.0]); y = np.array([1.0, 1.0]); o = np.zeros(2)

    # left: the two vectors
    ax = axes[0]
    arrow2(ax, o, x, color="C0"); arrow2(ax, o, y, color="C1")
    ax.text(1.05, 0.4, r"$\mathbf{x}$", color="C0", fontsize=12)
    ax.text(0.45, 0.85, r"$\mathbf{y}$", color="C1", fontsize=12)
    ax.set_title("Inputs")
    for a in axes:
        a.set_xlim(-0.3, 2.6); a.set_ylim(-0.3, 1.8)
        a.axhline(0, color='k', lw=0.4); a.axvline(0, color='k', lw=0.4)
        a.set_aspect("equal"); a.grid(alpha=0.25)
        a.set_xlabel(r"$\mathbf{e}_1$"); a.set_ylabel(r"$\mathbf{e}_2$")

    # middle: dot product as projection
    ax = axes[1]
    arrow2(ax, o, x, color="C0", alpha=0.3); arrow2(ax, o, y, color="C1")
    # projection of x onto y_hat
    yhat = y / np.linalg.norm(y)
    proj = (x @ yhat) * yhat
    arrow2(ax, o, proj, color="C3")
    ax.plot([x[0], proj[0]], [x[1], proj[1]], "k--", lw=0.7)
    ax.text(0.4, 0.65, r"$\mathbf{x}\cdot\mathbf{y}=3$", color="C3", fontsize=12)
    ax.set_title("Grade-0 part: scalar")

    # right: wedge as oriented parallelogram
    ax = axes[2]
    pgon = Polygon([o, x, x+y, y], closed=True, alpha=0.22, color="C2")
    ax.add_patch(pgon)
    arrow2(ax, o, x, color="C0", alpha=0.3)
    arrow2(ax, o, y, color="C1", alpha=0.3)
    ax.text(1.5, 1.1, r"$\mathbf{x}\wedge\mathbf{y}=\mathbf{e}_{12}$",
            color="C2", fontsize=11)
    ax.set_title("Grade-2 part: bivector")

    fig.suptitle(r"$\mathbf{x}\mathbf{y}=\mathbf{x}\cdot\mathbf{y}+\mathbf{x}\wedge\mathbf{y}=3+\mathbf{e}_{12}$",
                 y=1.02, fontsize=13)
    save(fig, "fig03_geometric_product.png")


# ----------------------------------- fig 4 — trivector ------------------

def fig04_trivector():
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    o = np.zeros(3)
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    verts = np.array([
        o, x, x+y, y,                  # bottom face
        z, x+z, x+y+z, y+z,            # top face
    ])
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],   # bottom
        [verts[4], verts[5], verts[6], verts[7]],   # top
        [verts[0], verts[1], verts[5], verts[4]],   # front
        [verts[3], verts[2], verts[6], verts[7]],   # back
        [verts[0], verts[3], verts[7], verts[4]],   # left
        [verts[1], verts[2], verts[6], verts[5]],   # right
    ]
    box = Poly3DCollection(faces, alpha=0.2, facecolor="C2", edgecolor="k", lw=0.6)
    ax.add_collection3d(box)
    # axes
    ax.quiver(*o, *x, color="C0", arrow_length_ratio=0.1, lw=2)
    ax.quiver(*o, *y, color="C1", arrow_length_ratio=0.1, lw=2)
    ax.quiver(*o, *z, color="C3", arrow_length_ratio=0.1, lw=2)
    ax.text(1.05, 0, 0, r"$\mathbf{e}_1$", color="C0", fontsize=12)
    ax.text(0, 1.05, 0, r"$\mathbf{e}_2$", color="C1", fontsize=12)
    ax.text(0, 0, 1.05, r"$\mathbf{e}_3$", color="C3", fontsize=12)
    ax.text(0.5, 0.5, 1.15,
            r"$\mathbf{e}_1\wedge\mathbf{e}_2\wedge\mathbf{e}_3 = \mathbf{I}$",
            color="C2", fontsize=12, ha="center")
    ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.2); ax.set_zlim(0, 1.2)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.set_title("Trivector as oriented volume")
    save(fig, "fig04_trivector.png")


# ----------------------------------- fig 5 — rotor ----------------------

def fig05_rotor():
    fig, ax = plt.subplots(figsize=(6, 5))
    theta = np.pi / 4
    v0 = np.array([1.0, 0.0])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    v1 = R @ v0
    o = np.zeros(2)
    # circle
    t = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), "k", lw=0.6, alpha=0.4)
    # arc of rotation
    t_arc = np.linspace(0, theta, 80)
    ax.plot(0.55*np.cos(t_arc), 0.55*np.sin(t_arc), "C2", lw=2)
    ax.annotate("",
                xy=(0.55*np.cos(theta), 0.55*np.sin(theta)),
                xytext=(0.55*np.cos(theta-0.05), 0.55*np.sin(theta-0.05)),
                arrowprops=dict(arrowstyle="->", color="C2", lw=2))
    arrow2(ax, o, v0, color="C0")
    arrow2(ax, o, v1, color="C1")
    ax.text(0.7, 0.1, r"$\mathbf{v}$", color="C0", fontsize=12)
    ax.text(0.55, 0.78, r"$R\mathbf{v}R^{-1}$", color="C1", fontsize=12)
    ax.text(0.25, 0.35, r"$\theta=\pi/4$", color="C2", fontsize=11)
    ax.text(-0.95, 0.95,
            r"$R = \exp(-\theta\,\mathbf{e}_{12}/2)$",
            color="black", fontsize=11)
    ax.set_xlim(-1.2, 1.4); ax.set_ylim(-1.2, 1.3)
    ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel(r"$\mathbf{e}_1$"); ax.set_ylabel(r"$\mathbf{e}_2$")
    ax.set_title("Rotor sandwich rotates a vector in the bivector's plane")
    save(fig, "fig05_rotor.png")


# ---------------------------------- fig 6 — Hodge dual ------------------

def fig06_dual():
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    o = np.zeros(3)
    # bivector e_12 as the e1-e2 plane (shaded square)
    plane = np.array([
        [-0.7, -0.7, 0], [0.7, -0.7, 0],
        [0.7, 0.7, 0],   [-0.7, 0.7, 0]
    ])
    bivec = Poly3DCollection([plane], alpha=0.3, facecolor="C2", edgecolor="C2")
    ax.add_collection3d(bivec)
    # dual vector e_3
    ax.quiver(*o, 0, 0, 1.0, color="C3", arrow_length_ratio=0.13, lw=2.4)
    # circular arrow inside the plane showing orientation
    t_arc = np.linspace(0, 2*np.pi*0.85, 60)
    rad = 0.45
    ax.plot(rad*np.cos(t_arc), rad*np.sin(t_arc), 0*t_arc, "C2", lw=2)
    ax.text(0.7, 0.1, 0, r"$\mathbf{B}=\mathbf{e}_{12}$", color="C2", fontsize=12)
    ax.text(0.05, 0.05, 1.08,
            r"$\star\mathbf{B}=\mathbf{B}\,\mathbf{I}=\mathbf{e}_3$",
            color="C3", fontsize=12)
    # axes
    for vec, lab, color in [
            ((1.0, 0, 0), r"$\mathbf{e}_1$", "k"),
            ((0, 1.0, 0), r"$\mathbf{e}_2$", "k"),
            ((0, 0, 1.0), r"$\mathbf{e}_3$", "k"),
    ]:
        ax.quiver(*o, *vec, color=color, arrow_length_ratio=0.1, lw=0.9, alpha=0.5)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1.3)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([0, 1])
    ax.set_title("Hodge dual: bivector plane ↔ orthogonal vector")
    save(fig, "fig06_dual.png")


# ----------------------------- fig 7 — translation T_h ------------------

def fig07_translation():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    s = np.linspace(-1, 5, 300)
    f = lambda x: x**2
    ax.plot(s, f(s), "C0", lw=2, label=r"$f(s)=s^2$")
    h = 1.5
    ax.plot(s, f(s+h), "C1", lw=2, label=rf"$(T_h f)(s)=f(s+h),\ h={h}$")
    # mark a sample point
    s0 = 1.0
    ax.scatter([s0], [f(s0)], color="C0", s=60, zorder=5)
    ax.scatter([s0], [f(s0+h)], color="C1", s=60, zorder=5)
    ax.annotate("", xy=(s0+0.02, f(s0+h)-0.5), xytext=(s0+0.02, f(s0)+0.5),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.4))
    ax.text(s0+0.15, (f(s0)+f(s0+h))/2,
            r"$T_h=\exp(h\mathrm{D})$", color="gray", fontsize=11)
    ax.set_xlim(-1, 5); ax.set_ylim(-1, 25)
    ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$s$"); ax.set_ylabel("value")
    ax.legend(loc="upper left")
    ax.set_title(r"Translation operator $T_h$: slide the argument by $h$")
    save(fig, "fig07_translation.png")


# ----------------------------- fig 8 — multiplication by t --------------

def fig08_mult_by_t():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    h = 0.15
    n = np.arange(60)
    t = n * h
    y = np.cos(t)
    yt = t * y
    ax.plot(t, y, "C0", lw=2, label=r"$y_n=\cos(t_n)$")
    ax.plot(t, yt, "C1", lw=2, label=r"$(t\cdot y)_n = t_n\,y_n$")
    ax.plot(t, t, "k--", lw=0.7, alpha=0.4)
    ax.plot(t, -t, "k--", lw=0.7, alpha=0.4)
    ax.set_xlim(0, t[-1]); ax.set_ylim(-12, 12)
    ax.axhline(0, color='k', lw=0.4)
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$t_n = n h$"); ax.set_ylabel("value")
    ax.legend(loc="upper left")
    ax.set_title(r"Multiplication by $t$: linear amplitude envelope")
    save(fig, "fig08_mult_by_t.png")


# ----------------------------- fig 9 — Clifford circle (Stage 1) -------

def fig09_clifford_circle():
    fig, ax = plt.subplots(figsize=(6, 6))
    N = 8
    thetas = 2 * np.pi * np.arange(N) / N
    xs = np.cos(thetas)
    ys = np.sin(thetas)
    # unit circle
    t = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), "k--", lw=0.6, alpha=0.4)
    # connect consecutive vertices
    for k in range(N):
        ax.annotate("",
                    xy=(xs[(k+1) % N], ys[(k+1) % N]),
                    xytext=(xs[k], ys[k]),
                    arrowprops=dict(arrowstyle="->", color="C2", lw=1.5))
    # vertices and labels
    ax.scatter(xs, ys, color="C0", s=70, zorder=5)
    for k in range(N):
        off = 0.13
        ax.text(xs[k]*(1+off), ys[k]*(1+off),
                rf"$\mathbf{{q}}_{k}$", fontsize=11, ha="center", va="center")
    # initial vector arrow
    o = np.zeros(2)
    arrow2(ax, o, np.array([xs[0], ys[0]]), color="C0")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel(r"$\mathbf{e}_1$"); ax.set_ylabel(r"$\mathbf{e}_2$")
    ax.set_title("Stage 1: Clifford octagon via rotor sandwich (no time)")
    save(fig, "fig09_clifford_circle.png")


# ----------------------- fig 10 — Weyl helix (Stage 2) -----------------

def fig10_weyl_helix():
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    N = 8
    h = 0.5
    thetas = 2 * np.pi * np.arange(N) / N
    xs = np.cos(thetas)
    ys = np.sin(thetas)
    zs = np.arange(N) * h
    # continuous helix (visual aid)
    tt = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(tt), np.sin(tt), tt / (2*np.pi) * (N-1) * h,
            color="k", lw=0.6, alpha=0.3, linestyle="--")
    # discrete vertices
    ax.scatter(xs, ys, zs, color="C0", s=70, zorder=5)
    # connect consecutive
    for k in range(N-1):
        ax.plot([xs[k], xs[k+1]],
                [ys[k], ys[k+1]],
                [zs[k], zs[k+1]], color="C2", lw=1.5)
    # labels
    for k in range(N):
        ax.text(xs[k]*1.18, ys[k]*1.18, zs[k]+0.05,
                rf"$\mathbf{{q}}_{k}^{{(2)}}$", fontsize=10, ha="center")
    # projection to the e1-e2 plane (Stage 1)
    ax.plot(np.cos(tt), np.sin(tt), 0*tt, color="gray", lw=0.7, alpha=0.5)
    ax.scatter(xs, ys, 0*zs, color="C0", s=25, alpha=0.4)
    # axes labels
    ax.text(1.35, 0, 0, r"$\mathbf{e}_1$", fontsize=12)
    ax.text(0, 1.35, 0, r"$\mathbf{e}_2$", fontsize=12)
    ax.text(0, 0, zs[-1] + 0.4,
            r"$T_h^n$ direction", fontsize=11, color="C3")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(0, zs[-1] + 0.5)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_title("Stage 2: Clifford octagon + Weyl translation = helix")
    save(fig, "fig10_weyl_helix.png")


# ===================== fig 11 — biquaternion → substrate ==================

def fig11_biquaternion_to_substrate():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    def draw_box(ax, x0, y0, w, h, label, color, alpha=0.18):
        from matplotlib.patches import FancyBboxPatch
        p = FancyBboxPatch((x0, y0), w, h,
                            boxstyle="round,pad=0.08",
                            linewidth=1.4, edgecolor=color,
                            facecolor=color, alpha=alpha)
        ax.add_patch(p)
        ax.text(x0 + w/2, y0 + h/2, label,
                ha="center", va="center", fontsize=11)

    # --- LEFT: Hamilton 1853, C ⊗ H ---
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5); ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Hamilton 1853:  $\\mathbb{C}\\otimes\\mathbb{H}$  (biquaternions)",
                 fontsize=12)
    draw_box(ax, 1.0, 5.0, 3.0, 1.6,
             r"$\mathbb{H}$  spatial rotations",
             "C0")
    draw_box(ax, 6.0, 5.0, 3.0, 1.6,
             r"$\mathbb{C}$  temporal label",
             "C1")
    # tensor connector
    ax.plot([4.0, 6.0], [5.8, 5.8], "k", lw=1)
    ax.text(5.0, 6.1, r"$\otimes$", ha="center", fontsize=14)
    # external operator
    draw_box(ax, 4.0, 1.5, 3.0, 1.6,
             r"$\Delta$  external  difference",
             "gray", alpha=0.25)
    # arrow from C to external Delta
    ax.annotate("",
                xy=(5.5, 3.1), xytext=(7.0, 5.0),
                arrowprops=dict(arrowstyle="->", color="C3",
                                lw=1.6, linestyle="--"))
    ax.text(6.6, 4.05, "rate of change\n(external)",
            color="C3", fontsize=10, style="italic")
    # caption note
    ax.text(5.0, 0.5,
            r"$i \in \mathbb{C}$: inert scalar, $i^2=-1$, commutes with $\mathbb{H}$",
            ha="center", fontsize=9.5, style="italic")

    # --- RIGHT: substrate ---
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5); ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Substrate:  $\\mathcal{A}_1\\otimes\\mathrm{Cl}(3,0)$",
                 fontsize=12)
    draw_box(ax, 1.0, 5.0, 3.0, 1.6,
             r"$\mathrm{Cl}(3,0)$ spatial",
             "C0")
    draw_box(ax, 6.0, 5.0, 3.0, 1.6,
             r"$\mathcal{A}_1=\langle t, \mathrm{D}\rangle$",
             "C2")
    ax.plot([4.0, 6.0], [5.8, 5.8], "k", lw=1)
    ax.text(5.0, 6.1, r"$\otimes$", ha="center", fontsize=14)
    # primitive identity
    draw_box(ax, 3.5, 1.5, 4.0, 1.6,
             r"$\mathrm{D}\,t - t\,\mathrm{D} = 1$"
             "\nprimitive identity",
             "C2", alpha=0.25)
    ax.annotate("",
                xy=(5.5, 3.1), xytext=(7.5, 5.0),
                arrowprops=dict(arrowstyle="->", color="C2",
                                lw=1.6))
    ax.text(7.4, 4.05, "rate of change\n(algebraic)",
            color="C2", fontsize=10, style="italic")
    ax.text(5.0, 0.5,
            r"$\mathrm{D}\in\mathcal{A}_1$: operator generator; non-commutative with $t$",
            ha="center", fontsize=9.5, style="italic")

    save(fig, "fig11_biquaternion_to_substrate.png")


# ===================== fig 12 — comparative addition =====================

def fig12_compare_add():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    # Descartes: lengths along a reference line
    ax = axes[0]
    ax.hlines(0.5, 0, 5, color="k", lw=0.8)
    ax.hlines(1.2, 0, 2, color="C0", lw=4)
    ax.hlines(1.2, 2.05, 5, color="C1", lw=4)
    ax.hlines(2.0, 0, 5, color="black", lw=4)
    ax.text(1.0, 1.5, r"$a$", color="C0", fontsize=12)
    ax.text(3.5, 1.5, r"$b$", color="C1", fontsize=12)
    ax.text(2.5, 2.3, r"$a+b$", color="black", fontsize=12, ha="center")
    ax.text(2.5, 0.3, "reference line",
            color="gray", fontsize=9, ha="center", style="italic")
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("Descartes (1637)\nlengths along a reference line", fontsize=11)

    # Hamilton: quaternions add component-wise
    ax = axes[1]
    o = np.zeros(2)
    q1 = np.array([2.0, 1.0]); q2 = np.array([-0.5, 2.0])
    qsum = q1 + q2
    arrow2(ax, o, q1, color="C0")
    arrow2(ax, o, q2, color="C1")
    arrow2(ax, o, qsum, color="black")
    ax.plot([q1[0], qsum[0]], [q1[1], qsum[1]], "k--", lw=0.5)
    ax.plot([q2[0], qsum[0]], [q2[1], qsum[1]], "k--", lw=0.5)
    ax.text(1.2, 0.3, r"$q_1$", color="C0", fontsize=12)
    ax.text(-0.8, 1.0, r"$q_2$", color="C1", fontsize=12)
    ax.text(0.7, 3.1, r"$q_1+q_2$", color="black", fontsize=12)
    ax.set_xlim(-1.5, 2.6); ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Hamilton (1846)\ncomponent-wise in $\\mathbb{H}$", fontsize=11)

    # Substrate: vector + operator-valued vector
    ax = axes[2]
    o = np.zeros(2)
    v = np.array([1.6, 0.4])
    arrow2(ax, o, v, color="C0")
    # operator-valued: visualize as a "fuzzy" arrow (multiple translucent copies)
    for k, alpha in enumerate([0.18, 0.32, 0.55, 0.85]):
        dv = np.array([0.4 + 0.15*k, 1.6 + 0.1*k])
        arrow2(ax, o, dv, color="C2", alpha=alpha)
    # sum
    arrow2(ax, o, v + np.array([0.55, 1.7]), color="black")
    ax.text(1.5, 0.15, r"$\mathbf{v}$", color="C0", fontsize=12)
    ax.text(0.0, 1.4, r"$\mathrm{D}\otimes\mathbf{w}$", color="C2", fontsize=12)
    ax.text(1.7, 2.2, r"$\mathbf{v} + \mathrm{D}\otimes\mathbf{w}$",
            color="black", fontsize=12)
    ax.set_xlim(-0.5, 3.0); ax.set_ylim(-0.5, 2.8)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("$\\mathcal{A}_1\\otimes\\mathrm{Cl}(3,0)$\n" +
                 r"spatial + operator-valued", fontsize=11)

    fig.suptitle("Addition: what each picture lets you add", fontsize=12.5, y=1.02)
    save(fig, "fig12_compare_add.png")


# ===================== fig 13 — comparative multiplication ===============

def fig13_compare_mul():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    # Descartes: similar triangles for the product of lengths
    ax = axes[0]
    # unit segment on the y-axis, length a on the x-axis; line through (1, 0) and (1, a)
    # similar triangle yields b * a on the y-axis along the line through (b, 0) parallel to (1,a)
    a = 2.0; b = 1.5
    ax.plot([0, 1], [0, 0], color="C0", lw=3)
    ax.text(0.5, -0.25, "unit", color="C0", fontsize=10, ha="center")
    ax.plot([0, 0], [0, a], color="C1", lw=3)
    ax.text(-0.4, a/2, r"$a$", color="C1", fontsize=11)
    ax.plot([0, b], [0, 0], color="C2", lw=2, alpha=0.4)
    ax.text(b - 0.1, -0.25, r"$b$", color="C2", fontsize=11, ha="center")
    # construction line: from (1, a) to (b, a*b) along the ray from origin... actually
    # similar triangles: (1, a) on the line; (b, ?) gives ? = a*b.
    ax.plot([0, b], [0, a*b], "k--", lw=0.8)
    ax.plot([b, b], [0, a*b], color="black", lw=3)
    ax.scatter([1, b], [a, a*b], color="black", zorder=5, s=30)
    ax.text(b + 0.1, a*b - 0.15, r"$a\,b$", color="black", fontsize=12)
    ax.set_xlim(-0.7, 2.5); ax.set_ylim(-0.6, 3.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Descartes (1637)\nsimilar triangles\non scalar lengths", fontsize=11)

    # Hamilton: quaternion product as rotation composition
    ax = axes[1]
    t = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), "k", lw=0.5, alpha=0.4)
    v0 = np.array([1.0, 0.0])
    # rotate by theta1 then theta2
    theta1 = np.pi/6
    theta2 = np.pi/4
    R1 = np.array([[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1),  np.cos(theta1)]])
    R2 = np.array([[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2),  np.cos(theta2)]])
    v1 = R1 @ v0
    v2 = R2 @ v1
    o = np.zeros(2)
    arrow2(ax, o, v0, color="C0")
    arrow2(ax, o, v1, color="C1")
    arrow2(ax, o, v2, color="black")
    ax.text(0.8, -0.18, r"$\mathbf{v}$", color="C0", fontsize=11)
    ax.text(0.65, 0.5, r"$R_1\mathbf{v}$", color="C1", fontsize=11)
    ax.text(0.05, 0.9, r"$R_2 R_1\mathbf{v}$", color="black", fontsize=11)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.0, 1.3)
    ax.set_aspect("equal")
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Hamilton (1846)\nquaternion product\n= rotation composition", fontsize=11)

    # Substrate: rotation + time advance
    ax = axes[2]
    # show a vector being rotated AND advanced in time (visualize time as vertical)
    o = np.zeros(2)
    v0 = np.array([0.9, 0.0])
    v1 = np.array([0.9*np.cos(np.pi/4), 0.9*np.sin(np.pi/4)])
    arrow2(ax, o, v0, color="C0")
    arrow2(ax, np.array([0, 0.9]), v1 + np.array([0, 0.9]), color="C1")
    # connector showing time advance
    ax.annotate("", xy=(0.45, 0.9), xytext=(0.45, 0.05),
                arrowprops=dict(arrowstyle="->", color="C2",
                                lw=1.6, linestyle="--"))
    ax.text(0.55, 0.45, r"$T_h = \exp(h\mathrm{D})$",
            color="C2", fontsize=10)
    ax.text(0.95, -0.18, r"$\mathbf{v}$", color="C0", fontsize=11)
    ax.text(0.4, 1.65, r"$T_h \cdot (R\mathbf{v})$", color="C1", fontsize=11)
    # arc for the rotation
    arc = np.linspace(0, np.pi/4, 30)
    ax.plot(0.45*np.cos(arc), 0.9 + 0.45*np.sin(arc), "C1", lw=1.4)
    ax.text(-0.4, 1.7, "rotate", color="C1", fontsize=10)
    ax.text(-0.4, 0.1, "time =0", color="gray", fontsize=9)
    ax.text(-0.4, 0.95, "time =h", color="gray", fontsize=9)
    ax.set_xlim(-0.7, 1.5); ax.set_ylim(-0.4, 2.0)
    ax.set_aspect("equal")
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"$\mathcal{A}_1\otimes\mathrm{Cl}(3,0)$" +
                 "\nrotation AND time advance\nin one algebra", fontsize=11)

    fig.suptitle("Multiplication: what each picture composes", fontsize=12.5, y=1.02)
    save(fig, "fig13_compare_mul.png")


# ===================== fig 14 — comparative exponentiation ==============

def fig14_compare_exp():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    # Descartes: scalar exp on R
    ax = axes[0]
    x = np.linspace(-2, 2, 200)
    ax.plot(x, np.exp(x), "C0", lw=2)
    ax.scatter([0], [1], color="C0", zorder=5)
    ax.text(1.0, 5.0, r"$\exp(x)=\sum x^k/k!$", color="C0", fontsize=11)
    ax.set_xlim(-2, 2); ax.set_ylim(-0.5, 8)
    ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)
    ax.grid(alpha=0.2)
    ax.set_xlabel(r"$x \in \mathbb{R}$")
    ax.set_title("Descartes (1637)\nscalar exp; no geometric meaning", fontsize=11)

    # Hamilton: rotor exp generates Spin(3) one-parameter family
    ax = axes[1]
    thetas = np.linspace(0, 2*np.pi, 9)
    for k, theta in enumerate(thetas):
        v = np.array([np.cos(theta), np.sin(theta)])
        arrow2(ax, np.zeros(2), v, color=plt.cm.viridis(k/len(thetas)), alpha=0.7)
    # unit circle
    t = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), "k", lw=0.5, alpha=0.3)
    ax.text(0.0, 1.25,
            r"$R(\theta)=\exp(-\frac{\theta}{2}\mathbf{B})$",
            color="black", fontsize=11, ha="center")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Hamilton (1846)\nrotor exp; spatial Spin(3)", fontsize=11)

    # Substrate: joint rotation × time-advance
    ax = axes[2]
    # show a discrete helix to suggest joint exponential
    N = 12
    thetas = 2*np.pi * np.arange(N) / N
    xs = 0.8 * np.cos(thetas)
    ys = 0.8 * np.sin(thetas)
    zs = np.arange(N) * 0.18
    # project to 2D by collapsing y onto z + offset x
    ax.scatter(xs, zs + 0.2*ys, c=np.arange(N), cmap="viridis", s=80, zorder=5)
    for k in range(N-1):
        ax.plot([xs[k], xs[k+1]],
                [zs[k] + 0.2*ys[k], zs[k+1] + 0.2*ys[k+1]],
                color="gray", lw=1, alpha=0.6)
    ax.text(0.0, zs[-1] + 0.5,
            r"$\exp(\alpha\mathrm{D} + \beta\mathbf{B})$",
            color="black", fontsize=11, ha="center")
    ax.text(0.0, zs[-1] + 0.2,
            r"(BCH expansion via $[\mathrm{D},t]=1$)",
            color="gray", fontsize=9, ha="center", style="italic")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.2, zs[-1] + 0.8)
    ax.set_aspect("equal")
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"$\mathcal{A}_1\otimes\mathrm{Cl}(3,0)$" +
                 "\njoint rotation × time-translation", fontsize=11)

    fig.suptitle("Exponentiation: one-parameter families each picture supports",
                 fontsize=12.5, y=1.02)
    save(fig, "fig14_compare_exp.png")


# ===================== fig 15 — OLS vs SPTLS ===========================

def fig15_ols_vs_sptls():
    np.random.seed(42)
    T = 10
    phi_true = 0.7
    y = np.zeros(T)
    e = np.random.randn(T) * 0.5
    for n in range(1, T):
        y[n] = phi_true * y[n-1] + e[n]
    y = np.round(y, 4)

    # OLS Cartesian
    X_C = y[:-1].reshape(-1, 1)
    target_C = y[1:]
    phi_hat = float((X_C.T @ target_C).item() / (X_C.T @ X_C).item())
    yhat_C = X_C.flatten() * phi_hat

    # SPTLS backward diff
    rows, target_S = [], []
    for n in range(1, T-1):
        rows.append([y[n], y[n]-y[n-1]])
        target_S.append(y[n+1])
    X_S = np.array(rows); target_S = np.array(target_S)
    beta_S, *_ = np.linalg.lstsq(X_S, target_S, rcond=None)
    yhat_S = X_S @ beta_S

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    n_full = np.arange(T)
    # LEFT — OLS Cartesian
    ax = axes[0]
    ax.plot(n_full, y, "o-", color="C0", lw=1.5, ms=7, label=r"$y_n$ (data)")
    n_C = np.arange(1, T)
    ax.plot(n_C, yhat_C, "x", color="C3", ms=10, mew=2,
            label=r"$\hat y_n = \hat\varphi\, y_{n-1}$")
    ax.set_xlabel("sample index $n$"); ax.set_ylabel("value")
    ax.set_title(f"OLS Cartesian: $\\hat\\varphi = {phi_hat:.4f}$")
    ax.grid(alpha=0.25); ax.legend(loc="upper left", fontsize=9)
    rmse_C = np.sqrt(np.mean((target_C - yhat_C)**2))
    ax.text(0.55, 0.05, f"RMSE = {rmse_C:.4f}\ncond = 1",
            transform=ax.transAxes, fontsize=9.5,
            bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.3"))

    # RIGHT — SPTLS
    ax = axes[1]
    ax.plot(n_full, y, "o-", color="C0", lw=1.5, ms=7, label=r"$y_n$ (data)")
    n_S = np.arange(2, T)
    ax.plot(n_S, yhat_S, "s", color="C2", ms=8,
            label=r"$\hat y_{n+1}=\hat\alpha_{00}y_n + \hat\alpha_{01}\mathrm{D}y_n$")
    ax.set_xlabel("sample index $n$"); ax.set_ylabel("value")
    ax.set_title(f"SPTLS $(0,1)$: "
                 f"$\\hat\\alpha_{{00}}={beta_S[0]:.4f}$, "
                 f"$\\hat\\alpha_{{01}}={beta_S[1]:.4f}$")
    ax.grid(alpha=0.25); ax.legend(loc="upper left", fontsize=9)
    rmse_S = np.sqrt(np.mean((target_S - yhat_S)**2))
    cond_S = float(np.linalg.cond(X_S.T @ X_S))
    ax.text(0.55, 0.05, f"RMSE = {rmse_S:.4f}\ncond = {cond_S:.2f}",
            transform=ax.transAxes, fontsize=9.5,
            bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.3"))

    fig.suptitle("Same ten-observation series, two fits", fontsize=12.5, y=1.02)
    save(fig, "fig15_ols_vs_sptls.png")


# ===================== fig 16 — Gibbs vs Clifford on two snapshots ========

def fig16_gibbs_vs_clifford():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Two snapshots
    q1 = np.array([1.0, 0.0])
    q2 = np.array([np.cos(np.pi/4), np.sin(np.pi/4)])
    o = np.zeros(2)

    # --- LEFT: Gibbs picture ---
    ax = axes[0]
    arrow2(ax, o, q1, color="C0")
    arrow2(ax, o, q2, color="C1")
    # difference arrow q2 - q1
    arrow2(ax, q1, q2, color="C3", linestyle="--")
    ax.text(1.05, -0.1, r"$\mathbf{q}_1$", color="C0", fontsize=12)
    ax.text(0.65, 0.78, r"$\mathbf{q}_2$", color="C1", fontsize=12)
    ax.text(0.92, 0.45, r"$\mathbf{q}_2 - \mathbf{q}_1$",
            color="C3", fontsize=11)
    # legend of Gibbs operations
    ax.text(-1.05, 1.30,
            "Three independent operations:\n"
            r"  $\mathbf{q}_2 - \mathbf{q}_1$   (vector)" + "\n"
            r"  $\mathbf{q}_2 \cdot \mathbf{q}_1 \approx 0.71$   (scalar)" + "\n"
            r"  $\mathbf{q}_2 \times \mathbf{q}_1 \approx -0.71\,\mathbf{e}_3$" + "\n"
            "No single algebraic object\nencodes the relation.",
            fontsize=10, va="top",
            bbox=dict(facecolor="white", edgecolor="C0",
                      boxstyle="round,pad=0.4"))
    ax.set_xlim(-1.1, 1.6); ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.grid(alpha=0.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Gibbs (1881):\nthree independent ops on two snapshots",
                 fontsize=11)

    # --- RIGHT: Clifford picture ---
    ax = axes[1]
    arrow2(ax, o, q1, color="C0")
    arrow2(ax, o, q2, color="C1")
    # bivector parallelogram
    from matplotlib.patches import Polygon
    pgon = Polygon([o, q1, q1 + q2 - q1, q2], closed=True,
                   alpha=0.2, color="C2")
    ax.add_patch(pgon)
    # curved orientation arrow inside the parallelogram
    arc = np.linspace(0.15*np.pi, 0.85*np.pi, 30)
    rad = 0.22
    cx, cy = 0.4, 0.25
    ax.plot(cx + rad*np.cos(arc), cy + rad*np.sin(arc), "C2", lw=2)
    ax.annotate("",
                xy=(cx + rad*np.cos(arc[-1]), cy + rad*np.sin(arc[-1])),
                xytext=(cx + rad*np.cos(arc[-3]), cy + rad*np.sin(arc[-3])),
                arrowprops=dict(arrowstyle="->", color="C2", lw=2))
    ax.text(1.05, -0.1, r"$\mathbf{q}_1$", color="C0", fontsize=12)
    ax.text(0.65, 0.78, r"$\mathbf{q}_2$", color="C1", fontsize=12)
    # geometric product expression
    ax.text(-1.05, 1.30,
            r"One geometric product:" + "\n"
            r"$\mathbf{q}_2 \mathbf{q}_1 = "
            r"\mathbf{q}_2\!\cdot\!\mathbf{q}_1 + "
            r"\mathbf{q}_2\!\wedge\!\mathbf{q}_1$" + "\n"
            r"$ \quad \approx 0.71 - 0.71\,\mathbf{e}_{12}$" + "\n"
            "Reversing order:\n"
            r"$\mathbf{q}_1 \mathbf{q}_2 \approx 0.71 + 0.71\,\mathbf{e}_{12}$" + "\n"
            "(bivector flips sign)",
            fontsize=10, va="top",
            bbox=dict(facecolor="white", edgecolor="C2",
                      boxstyle="round,pad=0.4"))
    ax.set_xlim(-1.1, 1.6); ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.grid(alpha=0.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"$\mathrm{Cl}(3,0)$: " + "one multivector\nscalar + bivector,"
                 + " order matters",
                 fontsize=11)

    save(fig, "fig16_gibbs_vs_clifford.png")


# ===================== fig 17 — t external vs Weyl algebra ===============

def fig17_t_external_vs_weyl():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    from matplotlib.patches import FancyBboxPatch
    def draw_box(ax, x0, y0, w, h, label, color, alpha=0.18, fontsize=10):
        p = FancyBboxPatch((x0, y0), w, h,
                            boxstyle="round,pad=0.10",
                            linewidth=1.4, edgecolor=color,
                            facecolor=color, alpha=alpha)
        ax.add_patch(p)
        ax.text(x0 + w/2, y0 + h/2, label,
                ha="center", va="center", fontsize=fontsize)

    # --- LEFT: Newton / Cartesian ---
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Newton / Cartesian:\n$t$ external, $d/dt$ external",
                 fontsize=11)

    # Configuration algebra box
    draw_box(ax, 0.6, 4.5, 4.0, 2.5,
             "Configuration algebra\n(vectors, scalars, dot, cross)",
             "C0")
    # time axis
    ax.plot([5.6, 9.4], [3.0, 3.0], "k", lw=1.4)
    ax.plot([9.4, 9.4], [2.9, 3.1], "k", lw=1.4)
    ax.plot([5.6, 5.6], [2.9, 3.1], "k", lw=1.4)
    ax.text(7.5, 2.5, r"$t \in \mathbb{R}$", color="gray",
            fontsize=11, ha="center")
    # function f(t) curve
    x = np.linspace(5.6, 9.4, 60)
    fcurve = 1.2 + 0.4 * np.sin(2.0 * (x - 5.6))
    ax.plot(x, 1.3 + fcurve - 1.3, color="C1", lw=1.6)
    ax.text(7.5, 0.8, r"$f(t)$", color="C1", fontsize=11, ha="center")
    # d/dt arrow from outside
    ax.annotate("", xy=(2.6, 5.5), xytext=(6.5, 7.6),
                arrowprops=dict(arrowstyle="->", color="C3",
                                lw=1.6, linestyle="--"))
    draw_box(ax, 6.0, 7.4, 3.5, 1.0,
             r"$d/dt$  (external operator)",
             "C3", alpha=0.22, fontsize=10)
    # legend
    ax.text(5.0, 0.1,
            r"Three pieces, three sides of a divide.",
            ha="center", fontsize=10, style="italic", color="gray")

    # --- RIGHT: Weyl algebra A_1 ---
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"Weyl algebra $\mathcal{A}_1$:" + "\n"
                 r"$t$ and $\mathrm{D}$ inside one algebra",
                 fontsize=11)

    # One big algebra box containing t and D
    draw_box(ax, 0.6, 3.0, 8.8, 5.0,
             "",
             "C2", alpha=0.10)
    ax.text(5.0, 7.5, r"$\mathcal{A}_1 = \langle t, \mathrm{D}\rangle$",
            ha="center", fontsize=14)

    draw_box(ax, 1.5, 5.0, 2.8, 1.4,
             r"$t$  (time symbol)",
             "C0", fontsize=11)
    draw_box(ax, 5.7, 5.0, 2.8, 1.4,
             r"$\mathrm{D}$  (diff operator)",
             "C3", fontsize=11)
    # primitive identity box
    draw_box(ax, 2.5, 3.3, 5.0, 1.2,
             r"$\mathrm{D}\,t - t\,\mathrm{D} = 1$" + "\n"
             "primitive algebraic identity",
             "C2", alpha=0.28, fontsize=10)
    # legend
    ax.text(5.0, 1.7,
            "Rate of change is multiplication by $\\mathrm{D}$.\n"
            "Time advance is the element $\\exp(h\\mathrm{D})$.\n"
            "Non-commutativity = temporal order.",
            ha="center", fontsize=9.5, color="gray")

    save(fig, "fig17_t_external_vs_weyl.png")


# ===================== fig 18 — building blocks =========================

def fig18_building_blocks():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.5); ax.set_aspect("equal")
    ax.axis("off")

    from matplotlib.patches import FancyBboxPatch

    def box(x0, y0, w, h, label, color, alpha=0.18, lw=1.5,
            linestyle="solid", fontsize=11, italic=False):
        p = FancyBboxPatch((x0, y0), w, h,
                            boxstyle="round,pad=0.12",
                            linewidth=lw, edgecolor=color,
                            facecolor=color, alpha=alpha,
                            linestyle=linestyle)
        ax.add_patch(p)
        if italic:
            ax.text(x0 + w/2, y0 + h/2, label,
                    ha="center", va="center", fontsize=fontsize,
                    style="italic")
        else:
            ax.text(x0 + w/2, y0 + h/2, label,
                    ha="center", va="center", fontsize=fontsize)

    # --- Top label: core substrate ---
    ax.text(7.0, 8.2, "Substrate as a building block",
            ha="center", fontsize=13, weight="bold")

    # CORE — A_1 ⊗ Cl(3,0)
    # Outer frame for the core
    box(0.5, 4.4, 8.0, 2.6, "", "C0", alpha=0.05, lw=2.6)
    ax.text(4.5, 6.6, "CORE  (programme proposal, solid)",
            ha="center", fontsize=10.5, style="italic", color="C0")
    box(1.0, 4.7, 3.0, 1.4,
        r"$\mathcal{A}_1=\langle t,\mathrm{D}\rangle$" + "\n"
        r"$[\mathrm{D},t]=1$",
        "C2", fontsize=10)
    box(5.0, 4.7, 3.0, 1.4,
        r"$\mathrm{Cl}(3,0)$" + "\n"
        r"Pauli algebra / spatial",
        "C0", fontsize=10)
    ax.text(4.5, 5.4, r"$\otimes$", ha="center", fontsize=18)

    # EXTENSION — Cl(0,1) graded
    box(9.0, 4.4, 4.5, 2.6, "", "C3", alpha=0.05, lw=2.0,
        linestyle="dashed")
    ax.text(11.25, 6.6, "EXTENSION  (exploratory)",
            ha="center", fontsize=10.5, style="italic", color="C3")
    box(9.4, 4.7, 3.7, 1.4,
        r"$\mathrm{Cl}(0,1)$" + "\n"
        r"$\gamma_0^{\,2}=-1$,  relativistic",
        "C3", fontsize=10, alpha=0.22)
    # Graded tensor connector
    ax.plot([8.5, 9.0], [5.4, 5.4], "k", lw=1.2)
    ax.text(8.75, 5.55, r"$\hat\otimes$", ha="center", fontsize=15)

    # --- Bottom labels: regimes covered ---
    ax.annotate("",
                xy=(4.5, 4.3), xytext=(4.5, 3.5),
                arrowprops=dict(arrowstyle="<-", color="C0", lw=1.4))
    ax.text(4.5, 3.2,
            "Classical mechanics  ·  non-rel. QM  ·  GSD time series\n"
            "F-000, F-001, F-002, A-001, A-002, A-005, A-007",
            ha="center", fontsize=9.5, color="C0")

    ax.annotate("",
                xy=(11.25, 4.3), xytext=(11.25, 3.5),
                arrowprops=dict(arrowstyle="<-", color="C3", lw=1.4,
                                linestyle="dashed"))
    ax.text(11.25, 3.2,
            "Relativistic (post-Newtonian, QFT)\n"
            "A-004, A-006",
            ha="center", fontsize=9.5, color="C3")

    # --- Further candidates (greyed out) ---
    box(0.5, 0.6, 13.0, 1.7, "", "gray", alpha=0.04, lw=1.0,
        linestyle="dotted")
    ax.text(7.0, 1.95, "Further candidate building blocks (open, future work)",
            ha="center", fontsize=10, style="italic", color="dimgray")
    ax.text(7.0, 1.05,
            r"$\otimes\;\mathfrak{g}_{\mathrm{int}}$  (internal symmetries, gauge)"
            "       "
            r"$\otimes\;$bundle  (curvature)"
            "       "
            r"$\otimes\;\mathbb{V}_{\mathrm{flav}}$  (flavour)",
            ha="center", fontsize=10, color="dimgray")
    # admission criterion footer
    ax.text(7.0, 0.15,
            "Admission criterion: interpretability of generators preserved.",
            ha="center", fontsize=9, color="dimgray", style="italic")

    save(fig, "fig18_building_blocks.png")


# ==============================================================================

if __name__ == "__main__":
    fig01_vector_add()
    fig02_wedge()
    fig03_geometric_product()
    fig04_trivector()
    fig05_rotor()
    fig06_dual()
    fig07_translation()
    fig08_mult_by_t()
    fig09_clifford_circle()
    fig10_weyl_helix()
    fig11_biquaternion_to_substrate()
    fig12_compare_add()
    fig13_compare_mul()
    fig14_compare_exp()
    fig15_ols_vs_sptls()
    fig16_gibbs_vs_clifford()
    fig17_t_external_vs_weyl()
    fig18_building_blocks()
    print(f"All 18 figures generated in {OUT}")
