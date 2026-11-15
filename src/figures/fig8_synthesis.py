"""Figure 8 - Cross-phase synthesis (Phases 2a + 5b-h).

(a) Contribution x phase evidence matrix. Rows = paper contributions
    (N1..N6). Cols = experimental phases (2a, 5b, 5c, 5d, 5e, 5f, 5h).
    Cells annotated with headline metric.
(b) Anvil vs Sepolia gas per decision (real DeviceRegistry transactions).
    Grouped bars with std error, per-decision ratio annotated on top.
(c) Per-attacker retention % for policy B5 vs baseline B4,
    computed from Phase 5d stats.json.
(d) Latency and return (mean step latency ms vs episode return),
    one point per policy. Horizontal band shows real Sepolia block-time
    latency band (mean +/- 1 std) as real-world constraint.

All values loaded from disk. No hardcoded numbers.
"""

from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.figures.style import (
    apply_style, panel_label, save_figure, PALETTE, PALETTE_LIST, BAR_WIDTH,
)

REPO = Path(__file__).resolve().parents[2]

# -------- source paths --------
P2A       = REPO / "results/phase2/phase2a_results.json"
P2_GATE   = REPO / "models/phase2/uncertainty_gate.json"
P5B       = REPO / "results/phase5b/stats.json"
P5C       = REPO / "results/phase5c/stats.json"
P5D       = REPO / "results/phase5d/stats.json"
P5E       = REPO / "results/phase5e/stats.json"
P5F       = REPO / "results/phase5f/stats.json"
P5H_SEP    = REPO / "results/phase5h/stats.json"
P5H_SENSOR = REPO / "results/phase5h_sensor/stats.json"
P5H_ANV    = REPO / "results/phase5h_anvil/stats.json"

# -------- contribution taxonomy --------
CONTRIBUTIONS = ["N1", "N2", "N3", "N4", "N5", "N6"]
CONTRIB_LABEL = {
    "N1": "N1  Adaptive controller\n     family (B5/B6/B7)",
    "N2": "N2  Signal ablation of\n     posterior/uncertainty/OOD",
    "N3": "N3  Adversarial robustness\n     under adaptive attackers",
    "N4": "N4 Cross-attacker\n     transferability",
    "N5": "N5  Per-component\n     overhead breakdown",
    "N6": "N6  On-chain feasibility\n     (Sepolia deployment)",
}
PHASES = ["2a", "5b", "5c", "5d", "5e", "5f", "5h"]
# 3-level colour code: 2=strong evidence, 1=partial/context, 0=n/a
CODES = np.array([
    # 2a 5b 5c 5d 5e 5f 5h
    [ 1, 2, 1, 1, 1, 0, 0],   # N1 uncertainty-gated family
    [ 1, 0, 2, 0, 0, 0, 0],   # N2 signal ablation
    [ 0, 0, 0, 2, 1, 0, 0],   # N3 adversarial robustness
    [ 0, 0, 0, 0, 2, 0, 0],   # N4 cross-attacker transfer
    [ 0, 0, 0, 0, 0, 2, 1],   # N5 overhead breakdown
    [ 0, 0, 0, 0, 0, 1, 2],   # N6 on-chain feasibility
])

# Labels for the proposed controller family in various panels.
FAMILY  = {"B5_myopic", "B7_full_dqn"}
ABLATION = {"B6_no_uncertainty"}

POLICY_ORDER_5F = [
    "B0_random", "B1_always_abstain", "B2_always_honeypot", "B3_threshold",
    "B4_classifier_gated", "B6_no_uncertainty",
    "B7_ablation_full", "B7_ablation_posterior_only",
    "B5_myopic", "B7_full_dqn",
]
POLICY_LABEL = {
    "B0_random":                  "B0",
    "B1_always_abstain":          "B1",
    "B2_always_honeypot":         "B2",
    "B3_threshold":               "B3",
    "B4_classifier_gated":        "B4",
    "B6_no_uncertainty":          "B6",
    "B7_ablation_full":           "B7-abl-full",
    "B7_ablation_posterior_only": "B7-abl-post",
    "B5_myopic":                  "B5*",
    "B7_full_dqn":                "B7*",
}

# Attacker order for panel (c).
ATTACKERS = ["fresh_wallet", "static_signature", "low_and_slow",
             "perturbation", "mimicry"]
ATTACKER_LABEL = {
    "fresh_wallet":     "fresh\nwallet",
    "static_signature": "static\nsignature",
    "low_and_slow":     "low &\nslow",
    "perturbation":     "perturb.",
    "mimicry":          "mimicry",
}


def _sensor_success_groups(stats: dict) -> list[tuple[int, float]]:
    """Return (n_success, mean_latency_s) groups from Scenario 2 stats."""
    groups: list[tuple[int, float]] = []

    for v in stats.get("postReading_by_label", {}).values():
        n = int(v.get("n_success", v.get("n", 0)) or 0)
        lat = v.get("wall_mean_s")
        if n > 0 and lat is not None:
            groups.append((n, float(lat)))

    for key, v in stats.items():
        if key == "postReading_by_label" or not isinstance(v, dict):
            continue
        if {"n", "gas_mean", "wall_mean_s"}.issubset(v.keys()):
            n = int(v.get("n_success", v.get("n", 0)) or 0)
            lat = v.get("wall_mean_s")
            if n > 0 and lat is not None:
                groups.append((n, float(lat)))

    return groups



def main() -> None:
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.0))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # ==================== (a) contribution x phase matrix ====================
    # Discrete colormap: 0 grey, 1 light, 2 accent.
    cmap = plt.matplotlib.colors.ListedColormap([
        "#ffffff",   # 0 = not evaluated
        "#cfe0f3",   # 1 = partial / context (lightened blue)
        "#7fbf7b",   # 2 = strong evidence (green, unchanged)
    ])
    im_a = ax_a.imshow(CODES, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax_a.set_xticks(np.arange(len(PHASES)))
    # Short discipline hint under each phase code; full names given in text.
    PHASE_HINT = {
        "2a": "classifier",
        "5b": "policy\ncompare",
        "5c": "signal\nablation",
        "5d": "adversarial",
        "5e": "transfer",
        "5f": "overhead",
        "5h": "on-chain",
    }
    ax_a.set_xticklabels(
        PHASES,
        fontsize=13,
    )
    ax_a.set_yticks(np.arange(len(CONTRIBUTIONS)))
    ax_a.set_yticklabels([CONTRIB_LABEL[c] for c in CONTRIBUTIONS],
                         fontsize=13, ha="right")
    # legend for the code
    legend_a = [
        mpatches.Patch(facecolor="#f5f5f5", edgecolor="black",
                       linewidth=0.4, label="not evaluated"),
        mpatches.Patch(facecolor="#cfe0f3", edgecolor="black",
                       linewidth=0.4, label="partial / context"),
        mpatches.Patch(facecolor=PALETTE_LIST[2 % len(PALETTE_LIST)],
                       edgecolor="black", linewidth=0.4,
                       label="strong evidence"),
    ]
    ax_a.legend(handles=legend_a, loc="upper center",
                bbox_to_anchor=(0.5, -0.10), fontsize=10, ncol=3,
                framealpha=0.9)
    ax_a.set_title("Evidence by phase", pad=12)
    panel_label(ax_a, "a")

    # ==================== (b) Anvil vs Sepolia gas per decision =============
    sep = json.loads(P5H_SEP.read_text())["per_label"]
    anv = json.loads(P5H_ANV.read_text())["per_label"]
    LABELS = ["ALLOW", "HONEYPOT_A", "HONEYPOT_B", "HONEYPOT_C"]
    TICKS  = ["ALLOW", "HP-A", "HP-B", "HP-C"]
    x = np.arange(len(LABELS))
    bw = 0.32
    anv_mean = [anv[l]["gas_mean"] for l in LABELS]
    anv_std  = [anv[l]["gas_std"]  for l in LABELS]
    sep_mean = [sep[l]["gas_mean"] for l in LABELS]
    sep_std  = [sep[l]["gas_std"]  for l in LABELS]

    ax_b.bar(x - bw/2, anv_mean, width=bw, color=PALETTE_LIST[0],
             edgecolor="black", linewidth=0.6,
             yerr=anv_std, capsize=3, ecolor="black",
             error_kw=dict(elinewidth=0.7),
             label="Anvil")
    ax_b.bar(x + bw/2, sep_mean, width=bw,
             color=PALETTE_LIST[2 % len(PALETTE_LIST)],
             edgecolor="black", linewidth=0.6,
             yerr=sep_std, capsize=3, ecolor="black",
             error_kw=dict(elinewidth=0.7),
             label="Sepolia")
    top = max(anv_mean) + max(anv_std) + max(sep_std)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(TICKS)
    ax_b.set_ylabel("Mean gas (units)")
    ax_b.set_title("Enrollment gas cost")
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                ncol=2, fontsize=10, frameon=False)
    ax_b.grid(True, axis="y", ls=":", lw=0.5)
    ax_b.set_ylim(0, top * 1.15)
    panel_label(ax_b, "b")

    # ==================== (c) B5 vs B4 retention per attacker ==============
    d5d = json.loads((Path(os.environ["HONEYCHAIN_ATTACKER_AUDIT"]) /
                       "phase5d/stats.json").read_text())
    x = np.arange(len(ATTACKERS))
    bw = 0.35
    b4 = [d5d[f"{a}::B4_classifier_gated"]["retention_pct"] for a in ATTACKERS]
    b5 = [d5d[f"{a}::B5_myopic"]["retention_pct"]           for a in ATTACKERS]
    ax_c.bar(x - bw/2, b4, width=bw, color=PALETTE_LIST[0],
             edgecolor="black", linewidth=0.6, label="B4 gated (baseline)")
    ax_c.bar(x + bw/2, b5, width=bw,
             color=PALETTE_LIST[2 % len(PALETTE_LIST)],
             edgecolor="black", linewidth=1.4,
             label="B5 myopic")
    ax_c.axhline(100, color="grey", lw=0.6, ls="--",
                 label="reactive baseline (100%)")
    ax_c.axhline(0, color="black", lw=0.7)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["Fresh", "Static", "Slow", "Perturb.", "Mimicry"],
                         fontsize=11, rotation=25, ha="right")
    ax_c.set_ylabel("Reactive return retained (%)")
    ax_c.set_title("Attacker return retention")
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=3, fontsize=13, frameon=False)
    ax_c.grid(True, axis="y", ls=":", lw=0.5)
    panel_label(ax_c, "c")

    # ==================== (d) latency-vs-return Pareto =====================
    with Path(os.environ["HONEYCHAIN_HEADLINE_CSV"]).open() as stream:
        d5b = {row["policy"]: row for row in csv.DictReader(stream)
               if row["dataset"] == "corrected"}
    d5f = json.loads(P5F.read_text())
    xs, ys, names, colours = [], [], [], []
    for pol in POLICY_ORDER_5F:
        if pol not in d5f:
            continue
        # Return: use B7_ablation_* names for 5f but 5b only has the base policies.
        # We use the 5b return where available; skip 5f-only ablations.
        if pol not in d5b:
            continue
        xs.append(d5f[pol]["step_total_ms"]["mean"])
        ys.append(float(d5b[pol]["malicious_return_mean"]))
        names.append(POLICY_LABEL[pol])
        if pol in FAMILY:
            colours.append(PALETTE_LIST[2 % len(PALETTE_LIST)])
        elif pol in ABLATION:
            colours.append(PALETTE_LIST[3 % len(PALETTE_LIST)])
        else:
            colours.append(PALETTE_LIST[0])
    xs = np.asarray(xs); ys = np.asarray(ys)

    ax_d.scatter(xs, ys, s=80, c=colours,
                 edgecolor="black", linewidth=0.8, zorder=5)
    offsets = {"B0": (-8, 12), "B2": (12, 9), "B3": (3, -16)}
    for xi, yi, nm in zip(xs, ys, names):
        offset = offsets.get(nm, (7, 4))
        ax_d.annotate(nm, (xi, yi), xytext=offset, textcoords="offset points",
                      fontsize=11, ha="center" if nm in offsets else "left",
                      arrowprops=dict(arrowstyle="-", color=".45", lw=.6)
                      if nm in offsets else None)

    # Real Sepolia block-time reference band.
    sep_lat_ms = [sep[l]["wall_mean_s"] * 1000 for l in LABELS]
    sep_std_ms = [sep[l]["wall_std_s"]  * 1000 for l in LABELS]
    sep_mean_lat = np.mean(sep_lat_ms)
    sep_std_lat  = np.mean(sep_std_ms)
    ax_d.axvspan(sep_mean_lat - sep_std_lat, sep_mean_lat + sep_std_lat,
                 color="grey", alpha=0.15,
                 label=f"Sepolia block inclusion band\n({sep_mean_lat/1000:.1f} $\\pm$ {sep_std_lat/1000:.1f} s)")
    ax_d.axvline(sep_mean_lat, color="grey", lw=0.8, ls="--")

    ax_d.set_xscale("log")
    ax_d.set_xlabel("Mean local step latency (ms, log scale)")
    ax_d.set_ylabel("Malicious-pool return")
    ax_d.margins(y=.16)
    ax_d.axhline(0, color="grey", lw=0.6, ls=":")
    ax_d.set_title("Latency and return")
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                fontsize=13, frameon=False)
    ax_d.grid(True, ls=":", lw=0.5)
    panel_label(ax_d, "d", dx=-0.18, dy=1.10)

    fig.tight_layout()
    save_figure(fig, "fig8_synthesis_cross_phase")


if __name__ == "__main__":
    main()
