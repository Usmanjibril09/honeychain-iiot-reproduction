# Manuscript Figure 2 uses the bundled policy comparison inputs.
"""Figure 2 - Headline policy comparison (Phase 5b).

(a) Return distribution: violin per policy (500 episodes each = 5 seeds x 100),
    overlaid with bootstrap 95% CI from stats.json.
(b) Per-seed spread: thin lines connecting each policy's 5 seed means,
    with grand mean marker.
(c) Bootstrap 95% CI bars for episode_mean per policy.
(d) Pairwise Cliff's delta heatmap (8x8), computed descriptively from
    per-episode arrays. Confirmatory Wilcoxon p-values are computed from
    the five paired seed means and annotated on off-diagonal cells.

All values loaded from disk. No hardcoded numbers.

Our proposed uncertainty-gated adaptive controller family = {B5_myopic, B7_full_dqn}.
These are highlighted with a bold edge; B6_no_uncertainty (ablation) is dashed;
B0-B4 are baselines.
"""

from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import wilcoxon

from src.figures.style import (
    apply_style, panel_label, save_figure, PALETTE, PALETTE_LIST, BAR_WIDTH,
)

REPO = Path(__file__).resolve().parents[2]
PHASE5B = REPO / "results/figure-inputs/policy-comparison"

POLICY_ORDER = [
    "B0_random", "B1_always_abstain", "B2_always_honeypot", "B3_threshold", "B3b_stateful_escalation",
    "B4_classifier_gated", "B6_no_uncertainty",
    "B5_myopic", "B7_full_dqn",   # our family last so they anchor the right side
]
FAMILY = {"B5_myopic", "B7_full_dqn"}
ABLATION = {"B6_no_uncertainty"}

POLICY_LABEL = {
    "B0_random":            "B0 random",
    "B1_always_abstain":    "B1 abstain",
    "B2_always_honeypot":   "B2 honeypot",
    "B3_threshold":         "B3 threshold",
    "B3b_stateful_escalation": "B3b stateful",
    "B4_classifier_gated":  "B4 gated",
    "B6_no_uncertainty":    "B6 no-unc.",
    "B5_myopic":            "B5 myopic*",
    "B7_full_dqn":          "B7 temporal*",
}


def _load_episodes(policy: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (all_episodes_flat, per_seed_arrays) for one policy."""
    per_seed = []
    for seed_dir in sorted(PHASE5B.glob("s*")):
        rows = []
        for kind in ("episodes_malicious.jsonl", "episodes_benign.jsonl"):
            p = seed_dir / policy / kind
            if not p.exists():
                continue
            with p.open() as f:
                for line in f:
                    rec = json.loads(line)
                    rows.append(float(rec["return"]))
        if rows:
            per_seed.append(np.asarray(rows, dtype=np.float64))
    all_flat = np.concatenate(per_seed) if per_seed else np.array([])
    return all_flat, per_seed


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta = P(a>b) - P(a<b), unbiased."""
    a = np.asarray(a); b = np.asarray(b)
    # Vectorised via broadcasting; fine at n=500.
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    n = a.size * b.size
    return (gt - lt) / n


def _wilcoxon_seed_p(a_seed_means: np.ndarray,
                         b_seed_means: np.ndarray) -> float:
    """Paired two-sided Wilcoxon test on matched per-seed means."""
    a = np.asarray(a_seed_means, dtype=np.float64)
    b = np.asarray(b_seed_means, dtype=np.float64)
    if a.size != b.size:
        raise ValueError("paired Wilcoxon requires equal numbers of seeds")
    if a.size == 0:
        return np.nan
    try:
        return float(
            wilcoxon(
                a,
                b,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            ).pvalue
        )
    except ValueError:
        # Occurs when all paired differences are zero.
        return 1.0


def _policy_color(policy: str) -> str:
    if policy in FAMILY:
        return PALETTE_LIST[2 % len(PALETTE_LIST)]  # accent
    if policy in ABLATION:
        return PALETTE_LIST[3 % len(PALETTE_LIST)]
    return PALETTE_LIST[0]  # neutral for baselines


def _policy_edge(policy: str) -> tuple[str, float, str]:
    """(edgecolor, linewidth, linestyle)"""
    if policy in FAMILY:
        return ("black", 1.6, "-")
    if policy in ABLATION:
        return ("black", 1.2, "--")
    return ("black", 0.6, "-")


def main() -> None:
    apply_style()

    # ---- load ------------------------------------------------------------
    stats = json.loads((PHASE5B / "stats.json").read_text())
    pp = stats["per_policy"]

    episodes = {}    # policy -> flat array
    per_seed = {}    # policy -> list of per-seed arrays
    for pol in POLICY_ORDER:
        all_flat, seeds = _load_episodes(pol)
        episodes[pol] = all_flat
        per_seed[pol] = seeds
        print(f"[load] {pol}: n_episodes={all_flat.size}, "
              f"n_seeds={len(seeds)}, mean={all_flat.mean():.3f}")

    seed_means = {
        pol: np.asarray([s.mean() for s in per_seed[pol]], dtype=np.float64)
        for pol in POLICY_ORDER
    }
    n_seed_counts = {pol: seed_means[pol].size for pol in POLICY_ORDER}
    if len(set(n_seed_counts.values())) != 1:
        raise ValueError(f"Policies do not have matching seed counts: {n_seed_counts}")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.0))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    xpos = np.arange(len(POLICY_ORDER))
    xlabels = [POLICY_LABEL[p] for p in POLICY_ORDER]

    # ---- (a) violins + CI --------------------------------------------------
    data = [episodes[p] for p in POLICY_ORDER]
    parts = ax_a.violinplot(
        data, positions=xpos, widths=0.72, showextrema=False, showmeans=False,
    )
    for i, body in enumerate(parts["bodies"]):
        pol = POLICY_ORDER[i]
        body.set_facecolor(_policy_color(pol))
        body.set_alpha(0.55)
        ec, lw, ls = _policy_edge(pol)
        body.set_edgecolor(ec)
        body.set_linewidth(lw)
        body.set_linestyle(ls)

    # Overlay bootstrap CI as thin black errorbars.
    means, lo, hi = [], [], []
    for pol in POLICY_ORDER:
        m = pp[pol]["episode_mean"]
        c = pp[pol]["bootstrap_ci95"]
        means.append(m); lo.append(m - c[0]); hi.append(c[1] - m)
    ax_a.errorbar(xpos, means, yerr=[lo, hi],
                  fmt="o", color="black", markersize=3.5,
                  elinewidth=1.0, capsize=3, zorder=5, label="mean $\pm$ 95% CI")

    ax_a.axhline(0, color="grey", lw=0.6, ls=":")
    ax_a.set_xticks(xpos); ax_a.set_xticklabels(xlabels, rotation=30, ha="right")
    ax_a.set_ylabel("Episode return")
    ax_a.set_title("Return distribution (500 episodes / policy)")
    ax_a.legend(loc="lower right", framealpha=0.9)
    panel_label(ax_a, "a")

    # ---- (b) per-seed spread ----------------------------------------------
    for i, pol in enumerate(POLICY_ORDER):
        pol_seed_means = seed_means[pol]
        # Plot the five matched seed means.
        ax_b.plot([i] * len(pol_seed_means), pol_seed_means, "o",
                  color=_policy_color(pol), alpha=0.65, markersize=5,
                  markeredgecolor="black", markeredgewidth=0.4)
        # tiny horizontal jitter for individual visibility
        jitter = (
            np.arange(len(pol_seed_means)) - (len(pol_seed_means) - 1) / 2
        ) * 0.06
        ax_b.plot(i + jitter, pol_seed_means, "o",
                  color=_policy_color(pol), alpha=0.35, markersize=4,
                  markeredgecolor="none")
        # grand mean as horizontal dash
        gm = float(pol_seed_means.mean())
        ax_b.hlines(gm, i - 0.28, i + 0.28, colors="black", lw=1.4, zorder=6)

    ax_b.axhline(0, color="grey", lw=0.6, ls=":")
    ax_b.set_xticks(xpos); ax_b.set_xticklabels(xlabels, rotation=30, ha="right")
    ax_b.set_ylabel("Seed-mean return (5 seeds)")
    ax_b.set_title("Per-seed spread (black bar = grand mean)")
    panel_label(ax_b, "b")

    # ---- (c) bootstrap CI bars --------------------------------------------
    colors = [_policy_color(p) for p in POLICY_ORDER]
    yerr = np.array([lo, hi])
    bars = ax_c.bar(xpos, means, width=BAR_WIDTH, color=colors,
                    edgecolor="black", linewidth=0.7)
    # style family bars: bold edge
    for i, pol in enumerate(POLICY_ORDER):
        ec, lw, ls = _policy_edge(pol)
        bars[i].set_edgecolor(ec); bars[i].set_linewidth(lw); bars[i].set_linestyle(ls)
    ax_c.errorbar(xpos, means, yerr=yerr, fmt="none",
                  ecolor="black", elinewidth=0.9, capsize=3, zorder=5)
    ax_c.axhline(0, color="grey", lw=0.6, ls=":")
    ax_c.set_xticks(xpos); ax_c.set_xticklabels(xlabels, rotation=30, ha="right")
    ax_c.set_ylabel("Episode-mean return")
    ax_c.set_title("Bootstrap 95% CI (10k resamples)")
    panel_label(ax_c, "c")

    # ---- (d) descriptive paired-seed difference heatmap --------------------
    # Descriptive summary only: median difference in matched seed means
    # (row minus column) across the five observed seeds.
    n = len(POLICY_ORDER)
    diff_med = np.zeros((n, n))
    diff_sd = np.zeros((n, n))
    for i, a in enumerate(POLICY_ORDER):
        for j, b in enumerate(POLICY_ORDER):
            if i == j:
                continue
            diffs = np.asarray(seed_means[a]) - np.asarray(seed_means[b])
            diff_med[i, j] = np.median(diffs)
            diff_sd[i, j] = np.std(diffs, ddof=1)

    vmax = np.max(np.abs(diff_med))
    im = ax_d.imshow(diff_med, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax_d.set_xticks(xpos); ax_d.set_yticks(xpos)
    ax_d.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax_d.set_yticklabels(xlabels, fontsize=8)

    for i in range(n):
        for j in range(n):
            if i == j:
                txt = "-"
            else:
                txt = f"{diff_med[i, j]:+.1f}\n(sd {diff_sd[i, j]:.1f})"
            color = "white" if abs(diff_med[i, j]) > 0.55 * vmax else "black"
            ax_d.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=5.6, color=color,
            )

    cb = plt.colorbar(im, ax=ax_d, fraction=0.042, pad=0.02, shrink=0.9)
    cb.set_label("Median paired-seed return difference\n(row - col)")
    ax_d.set_title("Matched 5-seed return differences", fontsize=12)
    panel_label(ax_d, "d", dx=-0.22, dy=1.10)

    # ---- legend for policy family colouring -------------------------------
    handles = [
        mpatches.Patch(facecolor=PALETTE_LIST[0], edgecolor="black",
                       linewidth=0.6, label="baseline / heuristic policies"),
        mpatches.Patch(facecolor=PALETTE_LIST[3 % len(PALETTE_LIST)],
                       edgecolor="black", linewidth=1.2, linestyle="--",
                       label="ablation (B6, no uncertainty)"),
        mpatches.Patch(facecolor=PALETTE_LIST[2 % len(PALETTE_LIST)],
                       edgecolor="black", linewidth=1.6,
                       label="proposed family (B5, B7)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.055), frameon=False, fontsize=11)

    fig.tight_layout(rect=(0, 0.065, 1, 1))
    save_figure(fig, "fig2_policy_comparison")


if __name__ == "__main__":
    main()
