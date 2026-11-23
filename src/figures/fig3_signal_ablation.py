"""Figure 3 - Signal ablation (Phase 5c).

(a) Return distribution: violins per mask (300 episodes each = 3 seeds x 100),
    overlaid with bootstrap 95% CI from stats.json. Reference mask 'full' is
    highlighted.
(b) Attack vs benign exposure rate scatter (AER on x, BER on y), bubble size
    proportional to exposure, one point per mask.
(c) Bootstrap 95% CI bars for episode-mean return per mask, retention % annotated (all using episode_mean for internal consistency).
(d) Descriptive matched-seed pooled-return differences for each ablation
    relative to the 'full' reference; no inferential tests.

All values loaded from disk. No hardcoded numbers.
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
PHASE5C = REPO / "results/figure-inputs/signal-ablation"

# Order: reference first, then ablations from most→least informative signal set.
MASK_ORDER = [
    "full",
    "posterior_uncertainty_ood",
    "posterior_uncertainty",
    "posterior_only",
    "confidence_only",
    "uncertainty_only",
]
MASK_LABEL = {
    "full":                      "full",
    "posterior_uncertainty_ood": "post+unc+ood",
    "posterior_uncertainty":     "post+unc",
    "posterior_only":            "post",
    "confidence_only":           "conf",
    "uncertainty_only":          "unc",
}
REFERENCE = "full"


def _load_episodes(mask: str) -> np.ndarray:
    """Return flat array of all episode returns for one mask."""
    rows = []
    for seed_dir in sorted((PHASE5C / mask).glob("s*")):
        for kind in ("episodes_malicious.jsonl", "episodes_benign.jsonl"):
            p = seed_dir / kind
            if not p.exists():
                continue
            with p.open() as f:
                for line in f:
                    rows.append(float(json.loads(line)["return"]))
    return np.asarray(rows, dtype=np.float64)



def _load_seed_means(mask: str) -> dict[str, float]:
    """Return pooled malicious+benign episode-return mean per seed."""
    result = {}
    for seed_dir in sorted((PHASE5C / mask).glob("s*")):
        values = []
        for kind in ("episodes_malicious.jsonl", "episodes_benign.jsonl"):
            path = seed_dir / kind
            if path.exists():
                with path.open() as f:
                    for line in f:
                        values.append(float(json.loads(line)["return"]))
        if values:
            result[seed_dir.name] = float(np.mean(values))
    return result


def _mask_color(mask: str) -> str:
    if mask == REFERENCE:
        return PALETTE_LIST[2 % len(PALETTE_LIST)]  # accent for reference
    return PALETTE_LIST[0]


def _mask_edge(mask: str) -> tuple[str, float, str]:
    if mask == REFERENCE:
        return ("black", 1.6, "-")
    return ("black", 0.6, "-")


def main() -> None:
    apply_style()

    stats = json.loads((PHASE5C / "stats.json").read_text())
    per_mask = {m["mask"]: m for m in stats["per_mask"]}
    comps    = {c["mask"]: c for c in stats["comparisons"]}

    episodes = {m: _load_episodes(m) for m in MASK_ORDER}
    for m in MASK_ORDER:
        print(f"[load] {m}: n_episodes={episodes[m].size}, "
              f"mean={episodes[m].mean():.3f}")

    # Matched seed-level pooled means for descriptive comparisons only.
    seed_means = {m: _load_seed_means(m) for m in MASK_ORDER}
    reference_seeds = set(seed_means[REFERENCE])
    if not reference_seeds:
        raise ValueError("No seed-level returns found for the full reference.")

    for m in MASK_ORDER:
        if set(seed_means[m]) != reference_seeds:
            raise ValueError(
                f"Seed mismatch for {m}: "
                f"{sorted(seed_means[m])} vs {sorted(reference_seeds)}"
            )

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.0))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    xpos = np.arange(len(MASK_ORDER))
    xlabels = [MASK_LABEL[m] for m in MASK_ORDER]

    # ---- (a) violins + CI ------------------------------------------------
    data = [episodes[m] for m in MASK_ORDER]
    parts = ax_a.violinplot(data, positions=xpos, widths=0.72,
                            showextrema=False, showmeans=False)
    for i, body in enumerate(parts["bodies"]):
        m = MASK_ORDER[i]
        body.set_facecolor(_mask_color(m))
        body.set_alpha(0.55)
        ec, lw, ls = _mask_edge(m)
        body.set_edgecolor(ec); body.set_linewidth(lw); body.set_linestyle(ls)

    means, lo, hi = [], [], []
    for m in MASK_ORDER:
        pm = per_mask[m]
        mn = pm["episode_mean"]  # M11: bars, CIs, and retention % all use episode_mean for internal consistency
        means.append(mn)
        lo.append(mn - pm["ci95_lo"])
        hi.append(pm["ci95_hi"] - mn)
    ax_a.errorbar(xpos, means, yerr=[lo, hi], fmt="o",
                  color="black", markersize=3.5, elinewidth=1.0,
                  capsize=3, zorder=5, label="mean $\\pm$ 95% CI")

    ax_a.axhline(0, color="grey", lw=0.6, ls=":")
    ax_a.set_xticks(xpos); ax_a.set_xticklabels(xlabels, rotation=30, ha="right")
    ax_a.set_ylabel("Episode return")
    ax_a.set_title("Return distribution (300 episodes / mask)")
    ax_a.legend(loc="lower left", framealpha=0.9)
    panel_label(ax_a, "a")

    # ---- (b) AER vs BER scatter ------------------------------------------
    # Some masks share (AER, BER, exposure) to plotting precision -- that is
    # a real finding (removing OOD / uncertainty on top of posterior does not
    # change exposure behaviour). Draw one marker per unique coordinate and
    # stack the grouped mask names on separate lines above the marker.
    from collections import defaultdict
    coord_groups = defaultdict(list)
    for m in MASK_ORDER:
        pm = per_mask[m]
        coord_groups[(round(pm["aer"], 3), round(pm["ber"], 3))].append(m)

    GROUP_OFFSETS = {
        "full":            ( 10,  -4),
        "unc":             (-14,   4),
        "conf":            ( 12,   4),
        "post":            ( 12,   6),
    }
    for (aer, ber), masks in coord_groups.items():
        exp = max(per_mask[m]["exposure"] for m in masks)
        size = 60 + exp * 400
        if REFERENCE in masks:
            fc = _mask_color(REFERENCE); ec, lw, _ = _mask_edge(REFERENCE)
        else:
            fc = _mask_color(masks[0]); ec, lw, _ = _mask_edge(masks[0])
        ax_b.scatter(aer, ber, s=size, color=fc,
                     edgecolor=ec, linewidth=lw, alpha=0.75, zorder=5)
        # Stack labels on separate lines.
        label = chr(10).join(MASK_LABEL[m] for m in masks)
        first = MASK_LABEL[masks[0]]
        dxp, dyp = GROUP_OFFSETS.get(first, (10, 6))
        ax_b.annotate(label, xy=(aer, ber),
                      xytext=(dxp, dyp), textcoords="offset points",
                      fontsize=9,
                      linespacing=1.1,
                      ha="left" if dxp >= 0 else "right",
                      va="bottom" if dyp >= 0 else "top")

    ax_b.set_xlabel("Attack Exposure Rate (AER)")
    ax_b.set_ylabel("Benign Exposure Rate (BER)")

    # --- M-extra: mean rotations per mask (bubble size for panel b) ---
    import json as _json
    _PHASE5C = PHASE5C
    def _mean_rotations(mask_name):
        rots = []
        mdir = _PHASE5C / mask_name
        if not mdir.exists():
            return 0.0
        for seed_dir in sorted(mdir.glob("s*")):
            for jf in ("episodes_malicious.jsonl", "episodes_benign.jsonl"):
                fp = seed_dir / jf
                if not fp.exists():
                    continue
                for line in fp.read_text().splitlines():
                    if line.strip():
                        rots.append(_json.loads(line).get("rotations", 0))
        return float(sum(rots)) / len(rots) if rots else 0.0
    _mean_rot = {m: _mean_rotations(m) for m in MASK_ORDER}
    ax_b.set_title("Signal fidelity (bubble size = exposure)")
    ax_b.grid(True, ls=":", lw=0.5)
    ax_b.margins(x=0.18, y=0.15)
    panel_label(ax_b, "b")

    # ---- (c) CI bars + retention annotation ------------------------------
    colors = [_mask_color(m) for m in MASK_ORDER]
    bars = ax_c.bar(xpos, means, width=BAR_WIDTH, color=colors,
                    edgecolor="black", linewidth=0.7)
    for i, m in enumerate(MASK_ORDER):
        ec, lw, ls = _mask_edge(m)
        bars[i].set_edgecolor(ec); bars[i].set_linewidth(lw); bars[i].set_linestyle(ls)
    ax_c.errorbar(xpos, means, yerr=[lo, hi], fmt="none",
                  ecolor="black", elinewidth=0.9, capsize=3, zorder=5)

    # Return-retention labels relative to the full-mask seed mean.
    full_episode_mean = per_mask[REFERENCE]["episode_mean"]  # M11: retention uses episode_mean, consistent with bars

    y_top = max(means) + max(hi) + 3
    for i, m in enumerate(MASK_ORDER):
        ret = 100.0 * per_mask[m]["episode_mean"] / full_episode_mean
        ax_c.text(i, means[i] + hi[i] + 0.8, f"{ret:.0f}%",
                ha="center", va="bottom", fontsize=9)

    ax_c.axhline(0, color="grey", lw=0.6, ls=":")
    ax_c.set_xticks(xpos); ax_c.set_xticklabels(xlabels, rotation=30, ha="right")
    ax_c.set_ylabel("Episode-mean return")
    ax_c.set_title("Episode-mean return per mask (retention % annotated)")
    ax_c.set_ylim(top=y_top + 2)
    panel_label(ax_c, "c")

    # ---- (d) descriptive seed-level differences vs full -------------------
    # Each point is one matched seed's pooled-return difference:
    # ablated mask minus full reference. No inferential test is performed.
    ablations = [m for m in MASK_ORDER if m != REFERENCE]
    ypos = np.arange(len(ablations))
    ref_seed = seed_means[REFERENCE]

    for i, m in enumerate(ablations):
        common = sorted(set(seed_means[m]) & set(ref_seed))
        diffs = np.asarray([
            seed_means[m][seed] - ref_seed[seed]
            for seed in common
        ], dtype=float)

        jitter = np.linspace(-0.10, 0.10, len(diffs))
        ax_d.scatter(
            diffs, np.full(len(diffs), ypos[i]) + jitter,
            s=42, color=PALETTE_LIST[1 % len(PALETTE_LIST)],
            edgecolor="black", linewidth=0.5, zorder=3,
            label="Observed seed" if i == 0 else None
        )

        mean_diff = float(np.mean(diffs))
        ax_d.scatter(
            mean_diff, ypos[i], marker="D", s=48,
            color=PALETTE_LIST[4 % len(PALETTE_LIST)],
            edgecolor="black", linewidth=0.5, zorder=4,
            label="Mean difference" if i == 0 else None
        )

    ax_d.axvline(0, color="black", lw=0.8)
    ax_d.set_yticks(ypos)
    ax_d.set_yticklabels([MASK_LABEL[m] for m in ablations])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Pooled-return difference (ablation − full)")
    ax_d.set_title("Observed matched-seed differences")
    ax_d.grid(True, axis="x", ls=":", lw=0.5)
    ax_d.legend(loc="best", framealpha=0.9, fontsize=8)
    panel_label(ax_d, "d")

    # ---- global legend for reference highlighting ------------------------
    handles = [
        mpatches.Patch(facecolor=PALETTE_LIST[0], edgecolor="black",
                       linewidth=0.6, label="ablated mask"),
        mpatches.Patch(facecolor=PALETTE_LIST[2 % len(PALETTE_LIST)],
                       edgecolor="black", linewidth=1.6,
                       label="reference (full, all 13 features)"),
    ]
    fig.legend(handles=handles,
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=11)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, "fig3_signal_ablation")


if __name__ == "__main__":
    main()
