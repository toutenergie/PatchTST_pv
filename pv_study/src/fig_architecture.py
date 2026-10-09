# -*- coding: utf-8 -*-
"""fig_architecture.py — diagram of the proposed PatchTST-CA-X architecture (paper figure)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import figure_style as fs


def box(ax, x, y, w, h, text, fc="#eaf2fb", ec="#5d6d7e", bold=False, size=7.5):
    """Rounded box with centred text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal")


def arrow(ax, a, b, color="#5d6d7e", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, color=color, lw=1, ls=ls))


def draw():
    fs.apply()
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ax.set_xlim(0, 22); ax.set_ylim(0, 10.4); ax.axis("off")
    R = fs.COLORS["model"]
    # --- inputs
    box(ax, 0.2, 7.6, 3.4, 2.2, "Past (L hours)\n\nPV_total (target)\nThermal, Wind,\nManantali import\n+ 8 deterministic\ncovariates", fc="#fdf2e9")
    box(ax, 0.2, 0.4, 3.4, 2.0, "Known future (H hours)\n\nsin/cos: hour,\nday of year,\nday of week\ncos θz · clear-sky GHI", fc="#fdf2e9")
    # --- selective RevIN
    box(ax, 4.3, 7.9, 3.0, 1.6, "Selective RevIN\n(target + grid);\ndeterministic:\nglobal norm.")
    arrow(ax, (3.6, 8.7), (4.3, 8.7))
    # --- patches
    box(ax, 8.0, 7.9, 3.0, 1.6, "Patches of length P,\nstride P/2 → N tokens;\nprojection P→D\n+ position + channel")
    arrow(ax, (7.3, 8.7), (8.0, 8.7))
    # --- shared temporal encoder
    box(ax, 11.7, 7.9, 3.3, 1.6, "Transformer encoder\nshared across channels\n(temporal attention,\n2 layers, D = 64)", fc="#eaf2fb", bold=True)
    arrow(ax, (11.0, 8.7), (11.7, 8.7))
    # --- CI branch
    box(ax, 15.8, 8.3, 2.7, 1.2, "CI: target channel only\n(exogenous ignored)", fc="#f4f6f6", size=7)
    arrow(ax, (15.0, 9.0), (15.8, 8.9))
    # --- CA branch
    box(ax, 15.8, 5.6, 2.7, 1.9, "CA: CROSS-CHANNEL\nattention\nper patch position\nquery = target", fc="#fdecea", ec=R, bold=True)
    arrow(ax, (15.0, 8.2), (15.8, 6.8), color=R)
    # --- head
    box(ax, 19.1, 6.9, 2.7, 1.6, "Linear head\nN·D → H\n(direct forecast)")
    arrow(ax, (18.5, 8.9), (19.1, 8.0)); arrow(ax, (18.5, 6.6), (19.1, 7.3), color=R)
    # --- known-future conditioning X
    box(ax, 4.6, 0.5, 9.0, 1.8, "Known-future conditioning (X): for each step h,\nMLP[context(z); future covariates(h); ŷ_base(h)]\n→ additive correction Δŷ(h)", fc="#fdecea", ec=R, bold=True)
    arrow(ax, (3.6, 1.4), (4.6, 1.4), color=R)
    arrow(ax, (20.4, 6.9), (13.6, 2.0), color="#5d6d7e", ls="--")
    ax.text(17.4, 4.1, "context z, ŷ_base", fontsize=7, color="#5d6d7e", rotation=26)
    # --- output
    box(ax, 15.2, 0.5, 3.1, 1.8, "RevIN de-normalisation\n(scale of the target\nchannel's window)")
    arrow(ax, (13.6, 1.4), (15.2, 1.4), color=R)
    box(ax, 19.0, 0.5, 2.8, 1.8, "Physical gate (g)\nŷ = 0 if cos θz = 0\n→ ŷ(t+1 … t+H) MW", fc="#f4f6f6")
    arrow(ax, (18.3, 1.4), (19.0, 1.4))
    # --- legend
    ax.text(0.2, 4.9, "Variants studied:", fontsize=8, fontweight="bold")
    ax.text(0.2, 3.3, "CI     : standard PatchTST (channel independence)\n"
                      "CA     : + cross-channel attention\n"
                      "X      : + known-future conditioning\n"
                      "g      : + physical night gate\n"
                      "in red : contributions of this study", fontsize=7.4, family="monospace")
    ax.set_title("PatchTST-CA-X architecture for total PV generation forecasting", fontsize=10, fontweight="bold")
    fs.save(fig, "fig_architecture")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    draw()
