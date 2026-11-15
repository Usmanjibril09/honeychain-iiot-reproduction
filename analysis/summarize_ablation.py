import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

root = Path(sys.argv[1])
writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(["mask","malicious_mean","malicious_sd","benign_mean","benign_sd","pooled_mean","pooled_sd","aer","ber"])
for mask in sorted(root.iterdir()):
    values = []
    for seed in range(3):
        files = [mask/f"s{seed}"/f"episodes_{mode}.jsonl" for mode in ("malicious","benign")]
        if not all(path.exists() for path in files):
            break
        mal, ben = [[json.loads(line) for line in path.read_text().splitlines()] for path in files]
        for label, rows in ((1, mal),(0, ben)):
            assert len(rows) == 50 and all(row["caller_pool_label"] == label for row in rows)
        values.append([mean(row["return"] for row in mal),mean(row["return"] for row in ben),
                       mean(row["return"] for row in mal+ben),
                       *(sum(row["successful_baits"] for row in rows)/sum(row["length"] for row in rows)
                         for rows in (mal,ben))])
    if len(values) == 3:
        columns=list(zip(*values))
        writer.writerow([mask.name, *(f"{value:.6f}" for col in columns[:3] for value in (mean(col),stdev(col))),
                         *(f"{mean(col):.6f}" for col in columns[3:])])
