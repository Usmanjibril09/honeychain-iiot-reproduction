import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.figures.style import FIG_DIR

root = Path(__file__).resolve().parents[2]
with (root / "results" / "ablation-comparison.csv").open() as stream:
    by_mask = {row["mask"]: row for row in csv.DictReader(stream)}
labels = {
    "full": "Full observation",
    "posterior_only": "Posterior",
    "posterior_uncertainty": "Posterior + uncertainty",
    "posterior_uncertainty_ood": "Posterior + uncertainty + OOD",
    "confidence_only": "Confidence",
    "uncertainty_only": "Uncertainty",
}

fig, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
for side, label, shift, color in (
    ("malicious", "Malicious caller pool", -.15, "#245985"),
    ("benign", "Benign caller pool", .15, "#9a542d"),
):
    rows = [by_mask[mask] for mask in labels]
    ax.errorbar([float(row[f"{side}_mean"]) for row in rows],
                [i + shift for i in range(len(rows))],
                xerr=[float(row[f"{side}_sd"]) for row in rows],
                fmt="o", capsize=2, color=color, label=label)
ax.set_yticks(range(len(labels)), list(labels.values()))
ax.invert_yaxis()
ax.axvline(0, color=".55", linewidth=.7)
ax.grid(axis="x", alpha=.2)
ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.18), ncols=2, frameon=False)
ax.set_xlabel("Mean episode return (bars show run SD)")
fig.savefig(root / "figures" / "fig3_signal_ablation_corrected.pdf")
