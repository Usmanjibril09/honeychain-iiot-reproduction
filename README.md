# HoneyChain-IIoT evaluation artifact

This public artifact supports verification of the corrected HoneyChain-IIoT
manuscript. It provides the eight quantitative figures, saved evaluation
records, and scripts that recompute the headline, signal-ablation, robustness,
and transfer summaries. Figure 1 is the conceptual architecture and is separate
from the quantitative evaluation.

## Reproduce the figures

Use Python 3.11 in a virtual environment:

```sh
python -m pip install -r requirements.txt
./reproduce.sh
```

Outputs are written to `figures/`. The bundled Dockerfile packages the same
figure workflow.

## Check the underlying numbers

| Manuscript evidence | Inputs and commands |
| --- | --- |
| Fig. 2 and classifier metrics | Validation labels and saved predictions, gate sweep, and held-out metric table in `results/phase2/` |
| Fig. 3 and nine-policy comparison | Corrected episode records in `results/phase5b-corrected/` and `results/b7-corrected/`. Run `python analysis/summarize.py results/phase5b-corrected results/b7-corrected` |
| Fig. 4 and six-mask ablation | Corrected episode records in `results/phase5c-corrected/`. Run `python analysis/summarize_ablation.py results/phase5c-corrected` |
| Figs. 5 and 6, robustness and transfer | Attacker-designated records from `results/phase5d/` and `results/phase5e/`, with the corresponding three corrected reactive seeds as the retention reference. Run `python analysis/reaggregate_attackers.py results results check-output` |
| Fig. 7 and overhead | Timing traces and aggregates in `results/phase5f/` |
| Fig. 8 and chain measurements | Anvil and Sepolia gas and latency exports in `results/phase5h*/` |
| Fig. 9 and cross-phase synthesis | Corrected headline, attacker-branch, overhead and deployment summaries |
| Learning-attacker evaluation | Malicious-only episode records and summary in `results/phase5d_adaptive/` |

The corrected headline uses five controller seeds and 50 episodes per caller
pool per seed. Signal ablation uses three seeds and the same balance. The
robustness and transfer analyses use 50 attacker-designated episodes per cell
and seed, excluding the auxiliary branch because the older evaluator reset
could discard its caller-pool selection. `results/attacker-spot-replays.txt`
records the trajectory checks. `results/comparison.csv` preserves the earlier
summary rows for auditability. The figure generators use the corrected rows.

This package reproduces reported analyses from saved outputs. Fresh training,
classifier inference from input features, and local or public-chain
deployment require the original environment. Raw transactions, caller
identifiers, and credentials are excluded.
