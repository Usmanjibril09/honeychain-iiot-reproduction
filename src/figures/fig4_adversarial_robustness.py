"""Figure 4 - Adversarial robustness (Phase 5d).

(a) Return heatmap: 4 core policies (rows) x 5 attackers (cols), coloured by
    episode_mean, annotated with values.
(b) Retention % heatmap vs reactive baseline: same axes, diverging cmap
    centred at 100%.
(c) Grouped bars per attacker: mean return with 95% CI for the 4 core policies.
(d) Mimicry failure mode: violins per policy under the mimicry attacker for
    all 10 policies (4 core + 6 B7 ablations), showing that no signal mask
    fixes mimicry.

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
PHASE5D = REPO / "results/phase5d"

ATTACKERS = ["fresh_wallet", "static_signature", "low_and_slow",
             "perturbation", "mimicry"]
ATTACKER_LABEL = {
    "fresh_wallet":     "fresh\nwallet",
    "static_signature": "static\nsignature",
    "low_and_slow":     "low &\nslow",
    "perturbation":     "perturb.",
    "mimicry":          "mimicry",
}
# T1 = signature/identity evasion; T2 = behavioural evasion.
T1_ATTACKERS = {"fresh_wallet", "static_signature", "perturbation"}

CORE_POLICIES = ["B4_classifier_gated", "B6_no_uncertainty",
                 "B5_myopic", "B7_full_dqn"]
CORE_LABEL = {
    "B4_classifier_gated": "B4 gated",
    "B6_no_uncertainty":   "B6 no-unc.",
    "B5_myopic":           "B5 myopic*",
    "B7_full_dqn":         "B7 full-DQN*",
}
FAMILY = {"B5_myopic", "B7_full_dqn"}
ABLATION = {"B6_no_uncertainty"}

# All policies present in 5d (10 total): core + B7 ablations.
ALL_POLICIES = CORE_POLICIES + [
    "B7_ablation_full",
    "B7_ablation_posterior_uncertainty_ood",
    "B7_ablation_posterior_uncertainty",
    "B7_ablation_posterior_only",
    "B7_ablation_confidence_only",
    "B7_ablation_uncertainty_only",
]
ALL_LABEL = {**CORE_LABEL, **{
    "B7_ablation_full":                      "B7-abl full",
    "B7_ablation_posterior_uncertainty_ood": "B7-abl p+u+ood",
    "B7_ablation_posterior_uncertainty":     "B7-abl p+u",
    "B7_ablation_posterior_only":            "B7-abl post",
    "B7_ablation_confidence_only":           "B7-abl conf",
    "B7_ablation_uncertainty_only":          "B7-abl unc",
}}


def _load_episodes(attacker: str, policy: str) -> np.ndarray:
    rows = []
    base = PHASE5D / attacker / policy
    for seed_dir in sorted(base.glob("s*")):
        for kind in ("episodes_malicious.jsonl", "episodes_benign.jsonl"):
            p = seed_dir / kind
            if not p.exists(): continue
            with p.open() as f:
                for line in f:
                    rows.append(float(json.loads(line)["return"]))
    return np.asarray(rows, dtype=np.float64)


def _policy_color(pol: str) -> str:
    if pol in FAMILY:   return PALETTE_LIST[2 % len(PALETTE_LIST)]
    if pol in ABLATION: return PALETTE_LIST[3 % len(PALETTE_LIST)]
    if pol.startswith("B7_ablation"): return PALETTE_LIST[1 % len(PALETTE_LIST)]
    return PALETTE_LIST[0]


def _policy_edge(pol: str) -> tuple[str, float, str]:
    if pol in FAMILY:   return ("black", 1.6, "-")
    if pol in ABLATION: return ("black", 1.2, "--")
    return ("black", 0.6, "-")


def main() -> None:
    apply_style()
    stats = json.loads((Path(os.environ["HONEYCHAIN_ATTACKER_AUDIT"]) / "phase5d/stats.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.4))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # ---- (a) return heatmap ---------------------------------------------
    Ra = np.zeros((len(CORE_POLICIES), len(ATTACKERS)))
    for i, pol in enumerate(CORE_POLICIES):
        for j, atk in enumerate(ATTACKERS):
            v = stats[f"{atk}::{pol}"]
            Ra[i, j] = v["episode_mean"]
    vabs = max(abs(Ra.min()), abs(Ra.max()))
    im_a = ax_a.imshow(Ra, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
    ax_a.set_xticks(np.arange(len(ATTACKERS)))
    ax_a.set_xticklabels(["Fresh", "Static", "Slow", "Perturb.", "Mimicry"], fontsize=12, rotation=25, ha="right")
    ax_a.set_yticks(np.arange(len(CORE_POLICIES)))
    ax_a.set_yticklabels([CORE_LABEL[p] for p in CORE_POLICIES], fontsize=13)
    for i in range(Ra.shape[0]):
        for j in range(Ra.shape[1]):
            v = Ra[i, j]
            color = "white" if abs(v) > vabs * 0.55 else "black"
            ax_a.text(j, i, f"{v:+.1f}", ha="center", va="center",
                      fontsize=13, color=color)
    cb_a = plt.colorbar(im_a, ax=ax_a, fraction=0.042, pad=0.02, shrink=0.9)
    cb_a.set_label("Episode-mean return")
    ax_a.set_title("Return by attacker")
    panel_label(ax_a, "a")

    # ---- (b) retention % heatmap ----------------------------------------
    Rb = np.zeros((len(CORE_POLICIES), len(ATTACKERS)))
    for i, pol in enumerate(CORE_POLICIES):
        for j, atk in enumerate(ATTACKERS):
            v = stats[f"{atk}::{pol}"]
            rp = v.get("retention_pct")
            Rb[i, j] = np.nan if rp is None else float(rp)
    # Centre diverging colormap at 100 (perfect retention).
    lo, hi = np.nanmin(Rb), np.nanmax(Rb)
    span = max(abs(100 - lo), abs(100 - hi))
    im_b = ax_b.imshow(Rb, cmap="RdBu_r",
                       vmin=100 - span, vmax=100 + span, aspect="auto")
    ax_b.set_xticks(np.arange(len(ATTACKERS)))
    ax_b.set_xticklabels(["Fresh", "Static", "Slow", "Perturb.", "Mimicry"], fontsize=12, rotation=25, ha="right")
    ax_b.set_yticks(np.arange(len(CORE_POLICIES)))
    ax_b.set_yticklabels([CORE_LABEL[p] for p in CORE_POLICIES], fontsize=13)
    for i in range(Rb.shape[0]):
        for j in range(Rb.shape[1]):
            v = Rb[i, j]
            if np.isnan(v): txt = "n/a"
            else:           txt = f"{v:+.0f}%"
            color = "white" if not np.isnan(v) and abs(v - 100) > span * 0.55 else "black"
            ax_b.text(j, i, txt, ha="center", va="center",
                      fontsize=13, color=color)
    cb_b = plt.colorbar(im_b, ax=ax_b, fraction=0.042, pad=0.02, shrink=0.9)
    cb_b.set_label("Retention % (100 = reactive baseline)")
    ax_b.set_title("Retention vs reactive baseline")
    panel_label(ax_b, "b")

    # ---- (c) grouped bars per attacker ----------------------------------
    x = np.arange(len(ATTACKERS))
    n_pol = len(CORE_POLICIES)
    bw = 0.18
    offsets = (np.arange(n_pol) - (n_pol - 1) / 2) * bw
    for i, pol in enumerate(CORE_POLICIES):
        means, lo_err, hi_err = [], [], []
        for atk in ATTACKERS:
            v = stats[f"{atk}::{pol}"]
            m = v["episode_mean"]; c = v["ci95"]
            means.append(m); lo_err.append(m - c[0]); hi_err.append(c[1] - m)
        ec, lw, ls = _policy_edge(pol)
        bars = ax_c.bar(x + offsets[i], means, width=bw,
                        color=_policy_color(pol),
                        edgecolor=ec, linewidth=lw, linestyle=ls,
                        label=CORE_LABEL[pol])
        ax_c.errorbar(x + offsets[i], means,
                      yerr=[lo_err, hi_err], fmt="none",
                      ecolor="black", elinewidth=0.7, capsize=2, zorder=5)
    ax_c.axhline(0, color="grey", lw=0.6, ls=":")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["Fresh", "Static", "Slow", "Perturb.", "Mimicry"],
                         fontsize=12, rotation=25, ha="right")
    ax_c.set_ylabel("Episode-mean return (95% CI)")
    ax_c.set_title("Core policies")
    panel_label(ax_c, "c")

    # ---- (d) mimicry failure mode: ranked horizontal bars ---------------
    # For each of the 10 policies, plot mean return under the mimicry attacker
    # with a bootstrap 95% CI whisker. Sorted best-to-worst so the reader
    # instantly sees the ranking. B7 ablation policies have no ci95 stored in
    # 5d stats.json (they were run as a lightweight ablation sweep), so we
    # compute their CI on the fly from per-episode arrays.
    from numpy.random import default_rng
    def _boot_ci(arr, n=10000, seed=0):
        rng = default_rng(seed)
        idx = rng.integers(0, arr.size, size=(n, arr.size))
        means = arr[idx].mean(axis=1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    rows = []
    for pol in ALL_POLICIES:
        k = f"mimicry::{pol}"
        v = stats.get(k)
        if v is None:
            continue
        mean = float(v["episode_mean"])
        ci = v.get("ci95")
        if ci is None:
            arr = _load_episodes("mimicry", pol)
            if arr.size == 0:
                continue
            ci = list(_boot_ci(arr))
        rows.append((pol, mean, float(ci[0]), float(ci[1])))
        print(f"[mimicry] {pol}: mean={mean:+.2f}  CI=[{ci[0]:+.2f},{ci[1]:+.2f}]")

    # Sort best -> worst (highest mean at top).
    rows.sort(key=lambda r: r[1], reverse=True)
    pols   = [r[0] for r in rows]
    means  = np.array([r[1] for r in rows])
    lo_err = np.array([r[1] - r[2] for r in rows])
    hi_err = np.array([r[3] - r[1] for r in rows])

    ypos = np.arange(len(pols))
    colors = [_policy_color(p) for p in pols]
    edges  = [_policy_edge(p) for p in pols]

    bars = ax_d.barh(ypos, means, height=0.55, color=colors,
                     edgecolor=[e[0] for e in edges],
                     linewidth=[e[1] for e in edges])
    for b, (ec, lw, ls) in zip(bars, edges):
        b.set_linestyle(ls)

    ax_d.errorbar(means, ypos, xerr=[lo_err, hi_err], fmt="none",
                  ecolor="black", elinewidth=0.8, capsize=3, zorder=5)

    xmax = float(means.max() + hi_err.max())
    xmin = float(means.min() - lo_err.max())

    ax_d.axvline(0, color="grey", lw=0.6, ls=":")
    ax_d.set_yticks(ypos)
    ax_d.set_yticklabels([ALL_LABEL[p] for p in pols], fontsize=13)
    ax_d.invert_yaxis()   # best on top
    ax_d.set_xlabel("Episode-mean return under mimicry (95% CI)")
    ax_d.set_title("Mimicry ranking")
    # Extend x-limits so annotations fit.
    ax_d.set_xlim(xmin - 5, xmax + 5)
    panel_label(ax_d, "d")

    # ---- figure legend for policy family colouring ----------------------
    handles = [
        mpatches.Patch(facecolor=PALETTE_LIST[0], edgecolor="black",
                       linewidth=0.6, label="baseline B4 (gated)"),
        mpatches.Patch(facecolor=PALETTE_LIST[3 % len(PALETTE_LIST)],
                       edgecolor="black", linewidth=1.2, linestyle="--",
                       label="ablation B6 (no uncertainty)"),
        mpatches.Patch(facecolor=PALETTE_LIST[2 % len(PALETTE_LIST)],
                       edgecolor="black", linewidth=1.6,
                       label="proposed family (B5, B7)"),
        mpatches.Patch(facecolor=PALETTE_LIST[1 % len(PALETTE_LIST)],
                       edgecolor="black", linewidth=0.6,
                       label="B7 ablation variants (panel d only)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=13)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, "fig4_adversarial_robustness_phase5d")


if __name__ == "__main__":
    main()
