"""Build the README figures and animations (GIF) with compatpower itself.

Run from the repository root:  python docs/make_figures.py
Requires matplotlib and pillow (not package dependencies).
"""
from __future__ import annotations

import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from compatpower import AssayModel, allocate, design_table, power, simulate_counts

OUT = pathlib.Path(__file__).parent / "img"
OUT.mkdir(exist_ok=True)
M = AssayModel()  # defaults: mu0=25, k=4.3, sd_run=0.48, tau=0.15
BLUE, RED, GREEN, GREY = "#1f4e79", "#c0392b", "#2e8b57", "#7f7f7f"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def dish(ax, x, y, n, rng, r=0.42, colour=BLUE):
    ax.add_patch(Circle((x, y), r, fc="#fbf7ea", ec="#555", lw=1))
    ang = rng.uniform(0, 2 * np.pi, n); rad = r * 0.88 * np.sqrt(rng.uniform(0, 1, n))
    ax.scatter(x + rad * np.cos(ang), y + rad * np.sin(ang), s=4, c=colour, lw=0)


# 1. The assay model: each frame is one run (day) with 5 control and 5 treatment plates ------------------
def anim_model():
    rng = np.random.default_rng(3)
    runs = 8
    yc, yt = simulate_counts(M, runs, 5, 0.3, 1, rng)
    fig, ax = plt.subplots(figsize=(6.8, 3.3))

    def frame(i):
        ax.clear(); ax.set_xlim(-0.6, 5.6); ax.set_ylim(-0.75, 1.65); ax.set_aspect("equal"); ax.axis("off")
        prng = np.random.default_rng(100 + i)
        for j in range(5):
            dish(ax, j, 1.1, yc[0, i, j], prng, colour=BLUE)
            dish(ax, j, 0.0, yt[0, i, j], prng, colour=RED)
            ax.text(j, 0.6, f"{yc[0, i, j]}", ha="center", fontsize=8, color=BLUE)
            ax.text(j, -0.5, f"{yt[0, i, j]}", ha="center", fontsize=8, color=RED)
        mc, mt = yc[0, i].mean(), yt[0, i].mean()
        ax.text(4.75, 1.1, f"control\nmean {mc:.0f}", va="center", fontsize=9, color=BLUE)
        ax.text(4.75, 0.0, f"treatment\nmean {mt:.0f}", va="center", fontsize=9, color=RED)
        ax.set_title(f"Run (day) {i + 1} of {runs}: true change −30%, observed {mt / mc - 1:+.0%}", fontsize=10)

    FuncAnimation(fig, frame, frames=runs, interval=1300).save(OUT / "assay_model.gif", writer=PillowWriter(fps=0.8))
    plt.close(fig)


# 2. Pseudoreplication: more plates make plate-level tests worse when the effect varies between runs ------
def anim_pseudoreplication():
    model = AssayModel(tau=0.30)
    ns = [3, 5, 10, 15, 20, 30]
    pooled = [power(5, n, 0.0, model, "pooled", n_sim=2500, seed=11) for n in ns]
    run = [power(5, n, 0.0, model, "run", n_sim=2500, seed=11) for n in ns]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    def frame(i):
        ax.clear()
        ax.plot(ns[: i + 1], pooled[: i + 1], "o-", c=RED, label="plates as replicates")
        ax.plot(ns[: i + 1], run[: i + 1], "o-", c=GREEN, label="run as the unit")
        ax.axhline(0.05, c="k", ls=":", lw=1); ax.text(30.5, 0.05, "5%", va="center", fontsize=8)
        ax.set_xlim(0, 33); ax.set_ylim(0, 0.32); ax.set_xlabel("plates per group per run (5 runs)")
        ax.set_ylabel("false-positive rate"); ax.legend(frameon=False, loc="upper left")
        ax.set_title("No true effect, but the product effect varies between days (τ = 0.30)", fontsize=10)
        fig.tight_layout()

    FuncAnimation(fig, frame, frames=len(ns) + 3, interval=900).save(
        OUT / "pseudoreplication.gif", writer=PillowWriter(fps=1.1))
    plt.close(fig)


# 3. Allocation: the same 60 plates split into more runs ------------------------------------------------
def anim_allocation():
    al = allocate(60, 0.3, M, n_sim=2500, seed=5)
    al = al[al.runs.isin([2, 3, 4, 5, 6, 10, 12, 20])].reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})

    def frame(i):
        i = min(i, len(al) - 1)
        r, n, p = int(al.runs[i]), int(al.plates[i]), al.power[i]
        a1.clear(); a1.axis("off"); a1.set_xlim(-1, 31); a1.set_ylim(-21, 2)
        for d in range(r):
            a1.scatter(np.arange(n), np.full(n, -d), s=22, c=BLUE, ec="k", lw=0.3)
        a1.set_title(f"{r} runs × {n} plates (60 per group)", fontsize=10)
        a2.clear(); a2.bar([0], [p], color=BLUE, width=0.5); a2.set_ylim(0, 1); a2.set_xlim(-0.6, 0.6)
        a2.set_xticks([]); a2.axhline(0.8, c="k", ls=":", lw=1)
        a2.text(0, p + 0.03, f"{p:.0%}", ha="center", fontsize=14, fontweight="bold")
        a2.set_title("power to detect\na 30% reduction", fontsize=10)

    FuncAnimation(fig, frame, frames=len(al) + 3, interval=1000).save(
        OUT / "allocation.gif", writer=PillowWriter(fps=1))
    plt.close(fig)


# 4. Equivalence (TOST) explained, static -------------------------------------------------------------
def fig_tost():
    lo, hi = 0.75, 1 / 0.75
    cases = [("Equivalent: CI inside the margins", 0.98, 0.86, 1.12, GREEN),
             ("Not shown: CI too wide (few runs)", 1.00, 0.62, 1.60, GREY),
             ("Not equivalent: real reduction", 0.68, 0.58, 0.80, RED),
             ("'Not significant' ≠ compatible", 0.82, 0.60, 1.12, GREY)]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.axvspan(lo, hi, color=GREEN, alpha=0.12); ax.axvline(1, c="k", lw=0.8)
    for x in (lo, hi): ax.axvline(x, c=GREEN, ls="--", lw=1)
    for y, (lab, m, a, b, c) in enumerate(cases[::-1]):
        ax.plot([a, b], [y, y], c=c, lw=3); ax.plot(m, y, "o", c=c, ms=7); ax.text(1.68, y, lab, va="center", fontsize=9)
    ax.set_xscale("log"); ax.set_xticks([0.5, 0.75, 1, 1.33, 2], ["0.5", "0.75", "1", "1.33", "2"]); ax.minorticks_off()
    ax.set_yticks([]); ax.set_xlim(0.5, 3.6); ax.set_xlabel("treatment / control ratio (90% CI)")
    ax.spines["left"].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "tost.png", dpi=150); plt.close(fig)


# 5. Design table heatmap, static ------------------------------------------------------------------------
def fig_table():
    t = design_table(runs=(3, 4, 5, 6, 8, 10), plates=(3, 5, 10, 20), reductions=(0.3,), model=M, n_sim=1500)
    m = t.pivot(index="runs", columns="plates", values="power_30pct").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.imshow(m.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(m.shape[1]), m.columns); ax.set_yticks(range(m.shape[0]), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v > 0.6 else "black",
                    fontweight="bold" if v >= 0.8 else "normal")
    ax.set_xlabel("plates per group per run"); ax.set_ylabel("independent runs")
    ax.set_title("Power, 30% reduction (default model)", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT / "design_table.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    anim_model(); anim_pseudoreplication(); anim_allocation(); fig_tost(); fig_table()
    for f in sorted(OUT.iterdir()):
        print(f.name, f"{f.stat().st_size / 1024:.0f} KB")
