"""Figure 5 - Cross-scenario transfer (Phase 5e).

(a) Transfer matrix: 6 trainers (rows) x 6 evals (cols), episode-mean return
    annotated per cell.
(b) Transfer gap: cell = R(train, eval) - R(eval, eval). Positive means the
    non-native trainer beats the eval-native policy. Diagonal is 0.
(c) Generalization index: horizontal bars per trainer showing mean off-
    diagonal return (transferred performance) with mean-including-diagonal
    as an overlay marker.
(d) Diagonal (self) vs off-diagonal (transferred) return per trainer,
    connected by a slope line. Shows specialization vs generalization.

All values loaded from disk. No hardcoded numbers.
"""

from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.style import (
    apply_style, panel_label, save_figure, PALETTE, PALETTE_LIST, BAR_WIDTH,
)

REPO = Path(__file__).resolve().parents[2]
PHASE5E = REPO / "results/phase5e"

ATTACKERS = ["reactive", "fresh_wallet", "static_signature",
             "low_and_slow", "perturbation", "mimicry"]
ATTACKER_LABEL = {
    "reactive":         "reactive",
    "fresh_wallet":     "fresh\nwallet",
    "static_signature": "static\nsignature",
    "low_and_slow":     "low &\nslow",
    "perturbation":     "perturb.",
    "mimicry":          "mimicry",
}


TICK_LABELS = ("Reactive", "Fresh", "Static", "Slow", "Perturb.", "Mimicry")


def main() -> None:
    apply_style()
    stats = json.loads((Path(os.environ["HONEYCHAIN_ATTACKER_AUDIT"]) / "phase5e/stats.json").read_text())

    n = len(ATTACKERS)
    R = np.zeros((n, n))
    for i, tr in enumerate(ATTACKERS):
        for j, ev in enumerate(ATTACKERS):
            R[i, j] = stats[f"{tr}::{ev}"]["episode_mean"]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.6))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # ---- (a) transfer matrix -------------------------------------------
    vabs = max(abs(R.min()), abs(R.max()))
    im_a = ax_a.imshow(R, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
    ax_a.set_xticks(np.arange(n))
    ax_a.set_xticklabels(TICK_LABELS, fontsize=10, rotation=32, ha="right")
    ax_a.set_yticks(np.arange(n))
    ax_a.set_yticklabels([ATTACKER_LABEL[a] for a in ATTACKERS], fontsize=13)
    ax_a.set_xlabel("Evaluation attacker")
    ax_a.set_ylabel("Training attacker")
    for i in range(n):
        for j in range(n):
            v = R[i, j]
            color = "white" if abs(v) > vabs * 0.55 else "black"
            weight = "bold" if i == j else "normal"
            ax_a.text(j, i, f"{v:.1f}", ha="center", va="center",
                      fontsize=11, color=color, fontweight=weight)
    cb_a = plt.colorbar(im_a, ax=ax_a, fraction=0.042, pad=0.02, shrink=0.9)
    cb_a.set_label("Episode-mean return")
    ax_a.set_title("Transfer matrix")
    panel_label(ax_a, "a")

    # ---- (b) transfer gap ---------------------------------------------
    diag = np.diag(R)                # eval-native returns
    G = R - diag[np.newaxis, :]      # gap vs eval-native
    gabs = max(abs(G.min()), abs(G.max()))
    im_b = ax_b.imshow(G, cmap="RdBu_r", vmin=-gabs, vmax=gabs, aspect="auto")
    ax_b.set_xticks(np.arange(n))
    ax_b.set_xticklabels(TICK_LABELS, fontsize=10, rotation=32, ha="right")
    ax_b.set_yticks(np.arange(n))
    ax_b.set_yticklabels([ATTACKER_LABEL[a] for a in ATTACKERS], fontsize=13)
    ax_b.set_xlabel("Evaluation attacker")
    ax_b.set_ylabel("Training attacker")
    for i in range(n):
        for j in range(n):
            v = G[i, j]
            if i == j:
                txt = "0"
            else:
                txt = f"{v:.1f}"
            color = "white" if abs(v) > gabs * 0.55 else "black"
            ax_b.text(j, i, txt, ha="center", va="center",
                      fontsize=11, color=color)
    cb_b = plt.colorbar(im_b, ax=ax_b, fraction=0.042, pad=0.02, shrink=0.9)
    cb_b.set_label("Transfer gap (train - eval-native)")
    ax_b.set_title("Transfer gap")
    panel_label(ax_b, "b")

    # ---- (c) generalization index ------------------------------------
    # Off-diagonal mean per row = mean transferred return.
    off_mean = np.array([
        R[i, [j for j in range(n) if j != i]].mean() for i in range(n)
    ])
    all_mean = R.mean(axis=1)  # includes diagonal

    order = np.argsort(off_mean)[::-1]   # best generalizer at top
    ypos = np.arange(n)

    bars = ax_c.barh(ypos, off_mean[order], height=0.35,
                     color=PALETTE_LIST[0], edgecolor="black", linewidth=0.7,
                     label="mean off-diagonal (transferred)")
    # Overlay all-eval mean as an unfilled marker.
    ax_c.plot(all_mean[order], ypos, "D",
              markerfacecolor="none", markeredgecolor="black",
              markersize=7, mew=1.2,
              label="mean incl. self (diagonal)")

    ax_c.axvline(0, color="grey", lw=0.6, ls=":")
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels([ATTACKER_LABEL[ATTACKERS[i]].replace("\n", " ") for i in order], fontsize=13)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Return (points)")
    ax_c.set_title("Generalization index")
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19),
                fontsize=11, frameon=False)
    # Extend xlim so annotations fit.
    xmax = max(off_mean.max(), all_mean.max())
    xmin = min(off_mean.min(), all_mean.min())
    pad = 0.15 * (xmax - xmin + 1)
    ax_c.set_xlim(xmin - pad, xmax + pad)
    panel_label(ax_c, "c")

    # (d) Plot both return measures on separate rows for each trainer.
    ax_d.hlines(ypos, np.minimum(diag[order], off_mean[order]),
                np.maximum(diag[order], off_mean[order]), color="0.65")
    ax_d.scatter(diag[order], ypos, s=55, color=PALETTE_LIST[2], label="Self-evaluation")
    ax_d.scatter(off_mean[order], ypos, s=55, marker="D", facecolors="white",
                 edgecolors="black", label="Off-diagonal mean")
    ax_d.set_yticks(ypos, [ATTACKER_LABEL[ATTACKERS[i]].replace("\n", " ") for i in order])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Return (points)")
    ax_d.set_title("Self-evaluation vs. transfer")
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19),
                fontsize=11, frameon=False)
    panel_label(ax_d, "d")

    fig.tight_layout()
    save_figure(fig, "fig5_cross_scenario_transfer_phase5e")


if __name__ == "__main__":
    main()
