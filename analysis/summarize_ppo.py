"""Summarize matched PPO and temporal Double-DQN evaluation records."""
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1] / "results"

def summarize(folder):
    rows = []
    for seed in range(5):
        groups = []
        for mode, offset in (("malicious", 0), ("benign", 100000)):
            path = folder / f"s{seed}" / f"episodes_{mode}.jsonl"
            episodes = [json.loads(line) for line in path.read_text().splitlines()]
            assert [ep["seed"] for ep in episodes] == list(range(seed + offset, seed + offset + 50))
            assert all(ep["mode"] == mode for ep in episodes)
            groups.append(episodes)
        rows.append([
            *(mean(ep["return"] for ep in group) for group in groups),
            mean(ep["return"] for group in groups for ep in group),
            *(sum(ep["successful_baits"] for ep in group) / sum(ep["length"] for ep in group)
              for group in groups),
        ])
    return [value for col in zip(*rows) for value in (mean(col), stdev(col))]

def main():
    fields = ("malicious_return", "benign_return", "pooled_return", "attack_exposure", "benign_exposure")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["policy", *(f"{field}_{stat}" for field in fields for stat in ("mean", "sd"))])
    for name, folder in (("PPO", "ppo-matched"), ("Temporal Double-DQN (B7)", "temporal-double-dqn")):
        writer.writerow([name, *(f"{value:.6f}" for value in summarize(ROOT / folder))])

if __name__ == "__main__":
    main()
