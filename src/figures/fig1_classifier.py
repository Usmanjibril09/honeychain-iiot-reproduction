"""Figure 1 - Classifier performance (Phase 2a).

(a) ROC per seed with ensemble mean, shaded envelope.
(b) Precision-Recall per seed with ensemble mean, shaded envelope.
(c) Reliability diagram (calibration) for the canonical ensemble seed.
(d) Selective classification: macro-F1 vs coverage for all 6 uncertainty
    configurations, chosen operating point marked.

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
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score
)

from src.figures.style import (
    apply_style, panel_label, save_figure, PALETTE, PALETTE_LIST,
)

REPO = Path(__file__).resolve().parents[2]
PHASE2 = REPO / "results/phase2"
def reliability(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(p, bins) - 1
    ids = np.clip(ids, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = ids == b
        if m.sum() == 0:
            continue
        xs.append(p[m].mean())
        ys.append(y_true[m].mean())
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def main():
    apply_style()
    p2a = json.loads((PHASE2 / "phase2a_results.json").read_text())
    gate = json.loads((PHASE2 / "uncertainty_gate_results.json").read_text())

    seeds = p2a["seeds"]
    canonical_seed = p2a["canonical_seed"]
    inputs = json.loads((PHASE2 / "validation_predictions.json").read_text())
    y_true = np.asarray(inputs["labels"])
    probs_by_seed = {seed: np.asarray(inputs["scores"][str(seed)]) for seed in seeds}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)

    # ---------------- (a) ROC ----------------
    ax = axes[0, 0]
    fprs_grid = np.linspace(0, 1, 200)
    tprs_interp = []
    for i, s in enumerate(seeds):
        fpr, tpr, _ = roc_curve(y_true, probs_by_seed[s])
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE_LIST[i % len(PALETTE_LIST)],
                lw=1.0, alpha=0.55, label=f"seed {s} (AUROC={auroc:.3f})")
        tprs_interp.append(np.interp(fprs_grid, fpr, tpr))
    tprs_arr = np.vstack(tprs_interp)
    tprs_mean = tprs_arr.mean(axis=0)
    tprs_std  = tprs_arr.std(axis=0)
    ax.plot(fprs_grid, tprs_mean, color="black", lw=2.0, label="ensemble mean")
    ax.fill_between(fprs_grid, tprs_mean - tprs_std, tprs_mean + tprs_std,
                    color="black", alpha=0.15, label="+/- 1 std")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.legend(loc="lower right", fontsize=13)
    panel_label(ax, "a")

    # ---------------- (b) PR -----------------
    ax = axes[0, 1]
    recs_grid = np.linspace(0, 1, 200)
    prec_interp = []
    for i, s in enumerate(seeds):
        prec, rec, _ = precision_recall_curve(y_true, probs_by_seed[s])
        ap = average_precision_score(y_true, probs_by_seed[s])
        # sort by recall for interpolation
        order = np.argsort(rec)
        ax.plot(rec, prec, color=PALETTE_LIST[i % len(PALETTE_LIST)],
                lw=1.0, alpha=0.55, label=f"seed {s} (AUPRC={ap:.3f})")
        prec_interp.append(np.interp(recs_grid, rec[order], prec[order]))
    prec_arr = np.vstack(prec_interp)
    prec_mean = prec_arr.mean(axis=0)
    prec_std  = prec_arr.std(axis=0)
    ax.plot(recs_grid, prec_mean, color="black", lw=2.0, label="ensemble mean")
    ax.fill_between(recs_grid, prec_mean - prec_std, prec_mean + prec_std,
                    color="black", alpha=0.15, label="+/- 1 std")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.legend(loc="lower left", fontsize=13)
    panel_label(ax, "b")

    # ---------------- (c) Reliability (canonical seed) ----------------
    ax = axes[1, 0]
    p_can = probs_by_seed[canonical_seed]
    xs, ys, ns = reliability(y_true, p_can, n_bins=10)
    # Perfect calibration line.
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.7, label="perfect")
    # Bin scatter with size proportional to bin count.
    sizes = 20 + (ns / max(ns.max(), 1)) * 260
    ax.scatter(xs, ys, s=sizes, color=PALETTE["primary"],
               alpha=0.75, edgecolor="black", linewidth=0.6,
               label=f"seed {canonical_seed} bins")
    # Confidence band using binomial std per bin.
    p_bin_hat = ys
    band = 1.96 * np.sqrt(p_bin_hat * (1 - p_bin_hat) / np.maximum(ns, 1))
    ax.fill_between(xs, np.clip(ys - band, 0, 1), np.clip(ys + band, 0, 1),
                    color=PALETTE["primary"], alpha=0.18, label="95% Wilson band")
    # Report ensemble ECE from the JSON for accuracy.
    ece = p2a["per_seed_metrics"]["ensemble"][seeds.index(canonical_seed)]["ece"]
    ax.text(0.02, 0.96, f"ECE = {ece:.3f}", transform=ax.transAxes,
            fontsize=13, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Empirical positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=13)
    panel_label(ax, "c")

    # ---------------- (d) Selective classification sweep ----------------
    ax = axes[1, 1]
    sweep = gate["sweep"]
    chosen = gate["chosen"]
    for i, (cfg, points) in enumerate(sweep.items()):
        cov = np.array([p["coverage"] for p in points])
        f1  = np.array([p["selective_macro_f1"] for p in points])
        order = np.argsort(cov)
        ax.plot(cov[order], f1[order],
                marker="o", markersize=3.5, lw=1.2,
                color=PALETTE_LIST[i % len(PALETTE_LIST)],
                alpha=0.85 if cfg == chosen["config"] else 0.55,
                label=cfg.replace("_", " "))
    # Chosen operating point marker.
    ax.scatter([chosen["coverage"]], [chosen["selective_macro_f1"]],
               s=200, marker="*", color="black", zorder=5,
               label="Selected gate")
    # Baseline macro-F1 dashed line.
    ax.axhline(gate["baseline_macro_f1"], color="gray", ls="--", lw=1.0,
               alpha=0.7, label=f"baseline F1 = {gate['baseline_macro_f1']:.3f}")
    # Min-coverage constraint shaded.
    ax.axvspan(0, gate["min_coverage"], color="red", alpha=0.06,
               label=f"coverage < {gate['min_coverage']:.2f} (rejected)")
    ax.set_xlabel("Coverage (fraction retained)")
    ax.set_ylabel("Selective macro-F1")
    ax.set_xlim(0.5, 1.005)
    ax.set_ylim(0.94, 1.005)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              fontsize=11, ncol=2, frameon=False)
    panel_label(ax, "d")

    save_figure(fig, "fig1_classifier_phase2a")


if __name__ == "__main__":
    main()
