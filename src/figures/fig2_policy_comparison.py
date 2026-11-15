import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.figures.style import FIG_DIR

root = Path(__file__).resolve().parents[2]
with (root / "results" / "comparison.csv").open() as stream:
    rows = [row for row in csv.DictReader(stream) if row["dataset"] == "corrected"]

fig, ax = plt.subplots(figsize=(5.6, 3.7), layout="constrained")
for metric, label, shift, color in (
    ("malicious_return", "Malicious caller pool", -.17, "#245985"),
    ("benign_return", "Benign caller pool", .17, "#9a542d"),
):
    ax.errorbar([float(row[metric + "_mean"]) for row in rows],
                [i + shift for i in range(len(rows))],
                xerr=[float(row[metric + "_sd"]) for row in rows],
                fmt="o", capsize=2, color=color, label=label)
ax.set_yticks(range(len(rows)), [row["policy"].split("_")[0] for row in rows])
ax.invert_yaxis()
ax.set_xlabel("Mean episode return (bars show run SD)")
ax.axvline(0, color=".55", linewidth=.7)
ax.grid(axis="x", alpha=.2)
ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.16), ncols=2, frameon=False)
fig.savefig(root / "figures" / "fig2_policy_comparison_corrected.pdf")
