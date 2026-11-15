import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
parser.add_argument("headline", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()

def episode_returns(case, pattern="s[0-2]/episodes_malicious.jsonl"):
    paths = sorted(case.glob(pattern))
    assert len(paths) == 3, case
    groups = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 50 and all(row["mode"] == "malicious" for row in rows), path
        groups.append([row["return"] for row in rows])
    return np.array(groups)

def summarize(groups):
    returns = groups.ravel()
    rng = np.random.default_rng(0)
    means = rng.choice(returns, size=(10000, len(returns)), replace=True).mean(axis=1)
    return {
        "episode_mean": float(returns.mean()),
        "seed_std": float(groups.mean(axis=1).std(ddof=1)),
        "ci95": np.percentile(means, [2.5, 97.5]).tolist(),
    }

baseline = {}
for policy in ("B4_classifier_gated","B5_myopic","B6_no_uncertainty","B7_full_dqn"):
    is_b7 = policy == "B7_full_dqn"
    source = args.headline / ("b7-corrected" if is_b7 else "phase5b-corrected")
    pattern = "s[0-2]/episodes_malicious.jsonl" if is_b7 else f"s[0-2]/{policy}/episodes_malicious.jsonl"
    baseline[policy] = float(episode_returns(source, pattern).mean())

for phase, pattern, expected in (("phase5d", "*/*", 50), ("phase5e", "train_*/eval_*", 36)):
    base = args.results / phase
    stats = {}
    for case in sorted(base.glob(pattern)):
        if not case.is_dir():
            continue
        left, right = case.relative_to(base).parts
        if phase == "phase5e":
            left, right = left.removeprefix("train_"), right.removeprefix("eval_")
        cell = summarize(episode_returns(case))
        if phase == "phase5d" and right in baseline:
            cell["retention_pct"] = 100 * cell["episode_mean"] / baseline[right]
        stats[f"{left}::{right}"] = cell
    assert len(stats) == expected, phase
    (args.output / phase).mkdir(parents=True, exist_ok=True)
    (args.output / phase / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
