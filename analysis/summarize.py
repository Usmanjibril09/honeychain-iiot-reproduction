import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

root, corrected_b7 = map(Path, sys.argv[1:3])
policies = ('B0_random', 'B1_always_abstain', 'B2_always_honeypot', 'B3_threshold',
            'B3b_stateful_escalation', 'B4_classifier_gated', 'B5_myopic',
            'B6_no_uncertainty', 'B7_full_dqn')
metrics = ('return', 'malicious_return', 'benign_return', 'AER', 'BER', 'benign_abstention')

def load(folder, mode):
    return [json.loads(line) for line in (folder / f'episodes_{mode}.jsonl').read_text().splitlines()]

def exposure(rows):
    return sum(row['successful_baits'] for row in rows) / sum(row['length'] for row in rows)

writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(['dataset','policy','seeds', *(f'{metric}_{part}' for metric in metrics for part in ('mean','sd'))])
for dataset in ('corrected',):
    for policy in policies:
        values = []
        for seed in range(5):
            key = f's{seed}'
            folder = corrected_b7/key if policy == 'B7_full_dqn' else root/key/policy
            if not all((folder/f'episodes_{mode}.jsonl').exists() for mode in ('malicious','benign')):
                continue
            malicious, benign = (load(folder, mode) for mode in ('malicious','benign'))
            assert len(malicious)==len(benign)==50
            if dataset == 'corrected' and policy != 'B7_full_dqn':
                assert all(row['caller_pool_label']==1 for row in malicious)
                assert all(row['caller_pool_label']==0 for row in benign)
            values.append((mean(r['return'] for r in malicious+benign),mean(r['return'] for r in malicious),
                           mean(r['return'] for r in benign),exposure(malicious),exposure(benign),
                           sum(r['action_counts'].get('0',0) for r in benign)/sum(r['length'] for r in benign)))
        if len(values)==5:
            writer.writerow([dataset,policy,5,*(f'{v:.6f}' for col in zip(*values) for v in (mean(col),stdev(col)))])
