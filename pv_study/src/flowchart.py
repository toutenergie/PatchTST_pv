# -*- coding: utf-8 -*-
"""flowchart.py — flowchart of the procedure for each configuration.

The factor studied by the configuration is highlighted (red box).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import figure_style as fs

STEPS = [
    ("data", "SENELEC hourly log\nenergie.xlsx (2019–2023)"),
    ("target", "Target PV_total = Σ 11 plants\n(clipping, causal imputation)"),
    ("cov", "Covariates: grid (3) · cyclic (6)\n· solar geometry (2)"),
    ("split", "Strict temporal split\nTrain 2019–21 · Val 2022 · Test 10/22–05/23"),
    ("norm", "Normalisation fitted on Train\n+ per-window RevIN"),
    ("window", "Windowing: past L h → future H h"),
    ("patch", "Patching\nlength P, stride P/2"),
    ("channels", "Shared Transformer encoder\n+ channel handling (CI / CA)"),
    ("head", "Direct forecasting head (H steps)\n± known-future conditioning X"),
    ("train", "Training × 5 seeds\nAdam · early stopping on Val"),
    ("eval", "Test evaluation: accuracy · 5 % noise\n· 30-day rolling (drift) · efficiency"),
    ("score", "Decision score\n0.4/nRMSE + 0.3/σ + 0.3/drift + bonus"),
    ("choice", "Selected configuration →\nnext step"),
]

FACTOR = {1: ("patch", "P ∈ {12, 24, 48}"), 2: ("window", "L ∈ {96, 168, 336}"),
          3: ("channels", "CI · CI-X · CA · CA-X · CA-Xg"), 4: ("head", "H ∈ {1, 6, 12, 24, 168}")}


def draw(config_no: int, title: str, file_name: str):
    key, values = FACTOR[config_no]
    fig, ax = plt.subplots(figsize=(6.2, 9.2))
    ax.set_xlim(-1.0, 10); ax.set_ylim(0, len(STEPS) * 1.0 + 0.5); ax.axis("off")
    y0 = len(STEPS) * 1.0
    for i, (k, text) in enumerate(STEPS):
        y = y0 - i * 1.0
        highlighted = k == key
        face = "#fdecea" if highlighted else ("#eaf2fb" if i < 5 else "#f4f6f6")
        edge = fs.COLORS["model"] if highlighted else "#5d6d7e"
        ax.add_patch(FancyBboxPatch((1.2, y - 0.36), 6.2, 0.72, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=face, ec=edge, lw=1.8 if highlighted else 0.9))
        ax.text(4.3, y, text, ha="center", va="center", fontsize=7.6,
                fontweight="bold" if highlighted else "normal")
        if highlighted:
            ax.text(7.6, y, "studied factor:\n" + values, ha="left", va="center", fontsize=7.4,
                    color=fs.COLORS["model"], fontweight="bold")
        if i < len(STEPS) - 1:
            ax.add_patch(FancyArrowPatch((4.3, y - 0.37), (4.3, y - 0.63), arrowstyle="-|>",
                                         mutation_scale=9, color="#5d6d7e", lw=0.9))
    # selection loop: score → back to the factor
    y_key = y0 - [k for k, _ in STEPS].index(key) * 1.0
    y_score = y0 - [k for k, _ in STEPS].index("score") * 1.0
    ax.add_patch(FancyArrowPatch((1.2, y_score), (1.2, y_key), connectionstyle="arc3,rad=-0.45",
                                 arrowstyle="-|>", mutation_scale=9, color=fs.COLORS["model"],
                                 lw=1.0, ls="--"))
    ax.text(-0.95, (y_score + y_key) / 2, "one\nvalue\nper\nrun", fontsize=7, color=fs.COLORS["model"],
            va="center")
    ax.set_title(title, fontsize=10, fontweight="bold")
    fs.save(fig, file_name)
