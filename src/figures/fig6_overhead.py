"""Figure 6 - Overhead breakdown (Phase 5f).

(a) Stacked bar per policy: mean latency decomposed into RL, classifier
    components (preprocess/ood/lgbm), bridge-overhead, EVM, and residual
    step overhead. All values in milliseconds.
(b) Grouped bars for step_total_ms percentiles (mean, p50, p95, p99) per
    policy. Thin bars.
(c) Component share: horizontal 100%-stacked bar per policy. Shows relative
    contribution regardless of absolute latency.
(d) Raw step_total_ms samples per policy plotted as horizontal strip plots
    (jittered dots), with mean line overlaid.

All values loaded from disk (stats.json + per-policy timings.jsonl).
No hardcoded numbers.
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

from src.figures.style import (
    apply_style, panel_label, save_figure, PALETTE, PALETTE_LIST, BAR_WIDTH,
)

REPO = Path(__file__).resolve().parents[2]
PHASE5F = REPO / "results/phase5f"

# Policy order matches Fig 2 for cross-figure consistency.
POLICY_ORDER = [
    "B0_random", "B1_always_abstain", "B2_always_honeypot", "B3_threshold",
    "B4_classifier_gated", "B6_no_uncertainty",
    "B7_ablation_full", "B7_ablation_posterior_only",
    "B5_myopic", "B7_full_dqn",
]
POLICY_LABEL = {
    "B0_random":                  "B0 random",
    "B1_always_abstain":          "B1 abstain",
    "B2_always_honeypot":         "B2 honeypot",
    "B3_threshold":               "B3 threshold",
    "B4_classifier_gated":        "B4 gated",
    "B6_no_uncertainty":          "B6 no-unc.",
    "B7_ablation_full":           "B7-abl full",
    "B7_ablation_posterior_only": "B7-abl post",
    "B5_myopic":                  "B5 myopic*",
    "B7_full_dqn":                "B7 full-DQN*",
}

# Components stacked in (a), colours in (a)/(c) legend.
COMP_KEYS   = ["rl_inference_ms", "preprocess_ms", "ood_ms", "lgbm_ms",
               "bridge_overhead_ms", "evm_ms", "step_residual_ms"]
COMP_LABELS = ["RL inference", "preprocess", "OOD (Mahalanobis)",
               "LightGBM", "bridge overhead", "EVM call", "step residual"]


def _components(pstats: dict) -> dict:
    """Return dict of mean ms for each of COMP_KEYS."""
    rl   = pstats["rl_inference_ms"]["mean"]
    pre  = pstats["preprocess_ms"]["mean"]
    ood  = pstats["ood_ms"]["mean"]
    lgb  = pstats["lgbm_ms"]["mean"]
    clf  = pstats["classifier_total_ms"]["mean"]
    br   = pstats["bridge_total_ms"]["mean"]
    evm  = pstats["evm_ms"]["mean"]
    tot  = pstats["step_total_ms"]["mean"]
    br_ov = max(br - clf, 0.0)                        # bridge - classifier
    step_res = max(tot - (br + evm + rl), 0.0)         # residual env overhead
    return {
        "rl_inference_ms":    rl,
        "preprocess_ms":      pre,
        "ood_ms":             ood,
        "lgbm_ms":            lgb,
        "bridge_overhead_ms": br_ov,
        "evm_ms":             evm,
        "step_residual_ms":   step_res,
    }


def _load_raw_totals(policy: str) -> np.ndarray:
    p = PHASE5F / policy / "timings.jsonl"
    rows = []
    with p.open() as f:
        for line in f:
            rows.append(float(json.loads(line)["step_total_ms"]))
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    apply_style()
    stats = json.loads((PHASE5F / "stats.json").read_text())

    # Build the component matrix (rows=policies, cols=COMP_KEYS).
    M = np.zeros((len(POLICY_ORDER), len(COMP_KEYS)))
    for i, pol in enumerate(POLICY_ORDER):
        comps = _components(stats[pol])
        for j, k in enumerate(COMP_KEYS):
            M[i, j] = comps[k]
        print(f"[load] {pol}: total={M[i].sum():.2f} ms  "
              f"(rl={comps['rl_inference_ms']:.3f}, "
              f"clf_all={comps['preprocess_ms']+comps['ood_ms']+comps['lgbm_ms']:.3f}, "
              f"br_ov={comps['bridge_overhead_ms']:.3f}, "
              f"evm={comps['evm_ms']:.2f})")

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.6))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    xpos = np.arange(len(POLICY_ORDER))
    xlabels = [POLICY_LABEL[p] for p in POLICY_ORDER]
    comp_colors = [PALETTE_LIST[k % len(PALETTE_LIST)]
                   for k in range(len(COMP_KEYS))]

    # ---- (a) stacked-bar decomposition (ms) ---------------------------
    bottom = np.zeros(len(POLICY_ORDER))
    for j, k in enumerate(COMP_KEYS):
        ax_a.bar(xpos, M[:, j], width=BAR_WIDTH,
                 bottom=bottom, color=comp_colors[j],
                 edgecolor="black", linewidth=0.4,
                 label=COMP_LABELS[j])
        bottom = bottom + M[:, j]

    # Overlay total value above each stack.
    totals = M.sum(axis=1)
    for i, t in enumerate(totals):
        ax_a.text(i, t + max(totals) * 0.015, f"{t:.1f}",
                  ha="center", va="bottom", fontsize=13)

    ax_a.set_xticks(xpos)
    ax_a.set_xticklabels(xlabels, rotation=30, ha="right")
    ax_a.set_ylabel("Mean latency per step (ms)")
    ax_a.set_title("Latency decomposition (mean)")
    ax_a.set_ylim(0, max(totals) * 1.15)
    panel_label(ax_a, "a")

    # ---- (b) grouped bars: percentiles of step_total ------------------
    pct_keys = ["mean", "p50", "p95", "p99"]
    pct_colors = [PALETTE_LIST[0], PALETTE_LIST[2 % len(PALETTE_LIST)],
                  PALETTE_LIST[3 % len(PALETTE_LIST)],
                  PALETTE_LIST[4 % len(PALETTE_LIST)]]
    bw = 0.18
    offsets = (np.arange(len(pct_keys)) - (len(pct_keys) - 1) / 2) * bw
    for j, k in enumerate(pct_keys):
        vals = [stats[pol]["step_total_ms"][k] for pol in POLICY_ORDER]
        ax_b.bar(xpos + offsets[j], vals, width=bw,
                 color=pct_colors[j], edgecolor="black", linewidth=0.5,
                 label=k)
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels(xlabels, rotation=30, ha="right")
    ax_b.set_ylabel("Total step latency (ms)")
    ax_b.set_title("Total step latency: mean / p50 / p95 / p99")
    ax_b.legend(loc="upper right", fontsize=13, framealpha=0.9, ncol=2)
    panel_label(ax_b, "b")

    # ---- (c) 100%-stacked share ---------------------------------------
    row_sums = M.sum(axis=1, keepdims=True)
    P = M / np.where(row_sums > 0, row_sums, 1.0) * 100.0

    ypos = np.arange(len(POLICY_ORDER))
    left = np.zeros(len(POLICY_ORDER))
    for j, k in enumerate(COMP_KEYS):
        ax_c.barh(ypos, P[:, j], height=0.55,
                  left=left, color=comp_colors[j],
                  edgecolor="black", linewidth=0.4)
        # Annotate segment if wide enough (>7%).
        for i in range(len(POLICY_ORDER)):
            if P[i, j] >= 7.0:
                ax_c.text(left[i] + P[i, j] / 2, i, f"{P[i, j]:.0f}%",
                          ha="center", va="center", fontsize=13, color="black")
        left = left + P[:, j]

    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels(xlabels, fontsize=13)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 100)
    ax_c.set_xlabel("Share of total step latency (%)")
    ax_c.set_title("Component share per policy")
    panel_label(ax_c, "c")

    # ---- (d) raw sample strip plot ------------------------------------
    rng = np.random.default_rng(0)
    for i, pol in enumerate(POLICY_ORDER):
        samples = _load_raw_totals(pol)
        y_jitter = i + (rng.random(samples.size) - 0.5) * 0.55
        ax_d.plot(samples, y_jitter, "o",
                  color=PALETTE_LIST[0], markersize=2.2,
                  alpha=0.35, markeredgewidth=0)
        # Mean marker.
        mean_v = samples.mean()
        ax_d.plot([mean_v], [i], "|", color="black",
                  markersize=14, mew=1.6, zorder=6)

    ax_d.set_yticks(np.arange(len(POLICY_ORDER)))
    ax_d.set_yticklabels(xlabels, fontsize=13)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Total step latency (ms, raw samples, N=200/policy)")
    ax_d.set_title("Per-step latency distribution (| = mean)")
    ax_d.grid(True, axis="x", ls=":", lw=0.5)
    panel_label(ax_d, "d")

    # ---- component legend at figure bottom ----------------------------
    handles = [
        mpatches.Patch(facecolor=comp_colors[j],
                       edgecolor="black", linewidth=0.4,
                       label=COMP_LABELS[j])
        for j in range(len(COMP_KEYS))
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(COMP_KEYS),
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=13)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, "fig6_overhead_phase5f")


if __name__ == "__main__":
    main()
