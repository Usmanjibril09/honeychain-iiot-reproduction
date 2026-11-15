"""Figure 7 - Sepolia deployment measurements (Phase 5h).

(a) Gas per decision label - grouped bars: DR contract vs SA postReading,
    plus SA enroll as its own group.
(b) Wall-clock latency per decision label - same grouped layout.
(c) Raw per-transaction gas: one dot per successful tx, ordered by
    (scenario, label), showing full variance.
(d) Raw per-transaction latency: same layout as (c).

All values loaded from disk (stats.json + tx_log.jsonl). No hardcoded numbers.
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
    apply_style, save_figure, PALETTE, PALETTE_LIST, BAR_WIDTH,
)

REPO = Path(__file__).resolve().parents[2]
PHASE5H_DR = REPO / "results/phase5h"
PHASE5H_SA = REPO / "results/phase5h_sensor"

LABEL_ORDER = ["ALLOW", "HONEYPOT_A", "HONEYPOT_B", "HONEYPOT_C"]
LABEL_TICK  = ["ALLOW", "HP-A", "HP-B", "HP-C"]

# Colours: DR neutral, SA enroll accent, SA postReading warmer.
COL_DR       = PALETTE_LIST[0]
COL_SA_ENROLL = PALETTE_LIST[2 % len(PALETTE_LIST)]
COL_SA_POST   = PALETTE_LIST[4 % len(PALETTE_LIST)]


def _load_tx_log(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    apply_style()

    dr_stats = json.loads((PHASE5H_DR / "stats.json").read_text())
    sa_stats = json.loads((PHASE5H_SA / "stats.json").read_text())
    dr_log   = _load_tx_log(PHASE5H_DR / "tx_log.jsonl")
    sa_log   = _load_tx_log(PHASE5H_SA / "tx_log.jsonl")

    dr_addr = dr_stats["deployment"]["device_registry"]
    sa_addr = sa_stats["deployment"]["sensor_attestation"]
    print(f"[DR] contract: {dr_addr}")
    print(f"[SA] contract: {sa_addr}")
    print(f"[DR] tx_log rows: {len(dr_log)}")
    print(f"[SA] tx_log rows: {len(sa_log)}")

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.4))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # ---- (a) mean gas per label, grouped bars ---------------------------
    x = np.arange(len(LABEL_ORDER))
    bw = 0.28

    dr_means = [dr_stats["per_label"][l]["gas_mean"] for l in LABEL_ORDER]
    dr_stds  = [dr_stats["per_label"][l]["gas_std"]  for l in LABEL_ORDER]
    sa_means = [sa_stats["postReading_by_label"][l]["gas_mean"] for l in LABEL_ORDER]
    sa_stds  = [sa_stats["postReading_by_label"][l]["gas_std"]  for l in LABEL_ORDER]

    ax_a.bar(x - bw/2, dr_means, width=bw,
             color=COL_DR, edgecolor="black", linewidth=0.6,
             yerr=dr_stds, capsize=3, ecolor="black",
             error_kw=dict(elinewidth=0.7),
             label="DR enrollDevice")
    ax_a.bar(x + bw/2, sa_means, width=bw,
             color=COL_SA_POST, edgecolor="black", linewidth=0.6,
             yerr=sa_stds, capsize=3, ecolor="black",
             error_kw=dict(elinewidth=0.7),
             label="SA postReading")

    # SA enroll as separate group to the right (single bar).
    x_enroll = len(LABEL_ORDER) + 0.3
    en = sa_stats["enroll"]
    ax_a.bar([x_enroll], [en["gas_mean"]], width=bw,
             color=COL_SA_ENROLL, edgecolor="black", linewidth=0.6,
             yerr=[en["gas_std"]], capsize=3, ecolor="black",
             error_kw=dict(elinewidth=0.7),
             label="SA enrollDevice")

    ax_a.set_xticks(list(x) + [x_enroll])
    ax_a.set_xticklabels(LABEL_TICK + ["enroll"], fontsize=13)
    ax_a.set_ylabel("Gas used (units)")
    ax_a.set_title("Mean gas per decision (real Sepolia txs)")
    ax_a.legend(loc="upper left", fontsize=13, framealpha=0.9)
    ax_a.grid(True, axis="y", ls=":", lw=0.5)
    ax_a.set_title(f"(a) {ax_a.get_title()}", fontsize=15, pad=9)

    # ---- (b) mean wall latency per label --------------------------------
    dr_wm = [dr_stats["per_label"][l]["wall_mean_s"] for l in LABEL_ORDER]
    dr_ws = [dr_stats["per_label"][l]["wall_std_s"]  for l in LABEL_ORDER]
    sa_wm = [sa_stats["postReading_by_label"][l]["wall_mean_s"] for l in LABEL_ORDER]
    sa_ws = [sa_stats["postReading_by_label"][l]["wall_std_s"]  for l in LABEL_ORDER]

    ax_b.bar(x - bw/2, dr_wm, width=bw,
             color=COL_DR, edgecolor="black", linewidth=0.6,
             yerr=dr_ws, capsize=3, ecolor="black",
             error_kw=dict(elinewidth=0.7),
             label="DR enrollDevice")
    ax_b.bar(x + bw/2, sa_wm, width=bw,
             color=COL_SA_POST, edgecolor="black", linewidth=0.6,
             yerr=sa_ws, capsize=3, ecolor="black",
             error_kw=dict(elinewidth=0.7),
             label="SA postReading")

    ax_b.bar([x_enroll], [en["wall_mean_s"]], width=bw,
             color=COL_SA_ENROLL, edgecolor="black", linewidth=0.6,
             yerr=[en["wall_std_s"]], capsize=3, ecolor="black",
             error_kw=dict(elinewidth=0.7),
             label="SA enrollDevice")

    # Reference: Sepolia nominal block time (~12 s).
    ax_b.axhline(12.0, color="grey", lw=0.8, ls="--",
                 label="Sepolia block time (12 s)")

    ax_b.set_xticks(list(x) + [x_enroll])
    ax_b.set_xticklabels(LABEL_TICK + ["enroll"], fontsize=13)
    ax_b.set_ylabel("Submit-to-inclusion latency (s)")
    ax_b.set_title("Mean wall-clock latency per decision")
    ax_b.legend(loc="lower right", fontsize=13, framealpha=0.9)
    ax_b.grid(True, axis="y", ls=":", lw=0.5)
    ax_b.set_title(f"(b) {ax_b.get_title()}", fontsize=15, pad=9)

    # ---- (c) raw per-tx gas scatter -------------------------------------
    _scatter_raw(ax_c, dr_log, sa_log, field="gas_used",
                 ylabel="Gas used (units)",
                 title="Raw per-transaction gas")
    ax_c.set_title(f"(c) {ax_c.get_title()}", fontsize=15, pad=9)

    # ---- (d) raw per-tx latency scatter ---------------------------------
    _scatter_raw(ax_d, dr_log, sa_log, field="wall_seconds",
                 ylabel="Wall-clock latency (s)",
                 title="Raw per-transaction latency")
    ax_d.axhline(12.0, color="grey", lw=0.8, ls="--",
                 label="Sepolia block time (12 s)")
    ax_d.legend(loc="upper right", fontsize=13, framealpha=0.9)
    ax_d.set_title(f"(d) {ax_d.get_title()}", fontsize=15, pad=9)

    fig.tight_layout()
    save_figure(fig, "fig7_sepolia_phase5h")


def _scatter_raw(ax, dr_log, sa_log, field: str, ylabel: str, title: str) -> None:
    """Plot per-tx dots grouped by (scenario, label). Skipped/errored txs excluded."""
    # Build groups in x order: DR labels, then SA-enroll, then SA postReading labels.
    groups: list[tuple[str, str, str]] = []  # (scenario, label, color)
    for l in LABEL_ORDER:
        groups.append(("DR", l, COL_DR))
    groups.append(("SA-enroll", "enroll", COL_SA_ENROLL))
    for l in LABEL_ORDER:
        groups.append(("SA-post", l, COL_SA_POST))

    xticklabels = [f"DR/{LABEL_TICK[i]}" for i in range(len(LABEL_ORDER))]
    xticklabels += ["SA/enroll"]
    xticklabels += [f"SA/{LABEL_TICK[i]}" for i in range(len(LABEL_ORDER))]

    rng = np.random.default_rng(0)
    for gi, (scen, label, color) in enumerate(groups):
        if scen == "DR":
            vals = [r[field] for r in dr_log
                    if r["label"] == label and r.get("status") == "success"]
        elif scen == "SA-enroll":
            vals = [r[field] for r in sa_log
                    if r.get("phase") == "enroll" and r.get("status") == "success"]
        else:  # SA-post
            vals = [r[field] for r in sa_log
                    if r.get("phase") == "postReading" and r["label"] == label
                    and r.get("status") == "success"]
        if not vals:
            continue
        y = np.asarray(vals, dtype=np.float64)
        x = gi + (rng.random(y.size) - 0.5) * 0.45
        ax.plot(x, y, "o", color=color, markersize=4.2,
                alpha=0.75, markeredgecolor="black", markeredgewidth=0.4)
        # Mean bar overlay.
        ax.plot([gi - 0.22, gi + 0.22], [y.mean(), y.mean()],
                color="black", lw=1.4, zorder=6)

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(xticklabels, rotation=40, ha="right", fontsize=13)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", ls=":", lw=0.5)


if __name__ == "__main__":
    main()
