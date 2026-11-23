# HoneyChain-IIoT evaluation artifact

HoneyChain-IIoT is an uncertainty-aware adaptive deception framework for
blockchain-enabled industrial IoT. It estimates caller risk from Ethereum
transaction history and uses classifier disagreement to identify uncertain
predictions. An off-chain reinforcement-learning controller selects responses
for device enrollment, sensor attestation, and maintenance authorization.
An EIP-712 bridge authenticates decisions for on-chain execution.

This repository accompanies the HoneyChain-IIoT manuscript. It provides
evaluation records and scripts for reproducing its quantitative analyses.

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
| Fig. 3 and nine-policy comparison | Episode records in `results/policy-comparison/` and `results/temporal-double-dqn/`. Run `python analysis/summarize.py results/policy-comparison results/temporal-double-dqn` |
| Fig. 4 and six-mask ablation | Episode records in `results/signal-ablation/`. Run `python analysis/summarize_ablation.py results/signal-ablation` |
| Figs. 5 and 6, robustness and transfer | Attacker-designated records from `results/phase5d/` and `results/phase5e/`, with the corresponding three reactive seeds as the retention reference. Run `python analysis/reaggregate_attackers.py results results check-output` |
| Fig. 7 and overhead | Timing traces and aggregates in `results/phase5f/` |
| Fig. 8 and chain measurements | Anvil and Sepolia gas and latency exports in `results/phase5h*/` |
| Fig. 9 and cross-phase synthesis | Headline, attacker-branch, overhead and deployment summaries |
| Matched internal PPO comparison | Five PPO evaluation runs in `results/ppo-matched/` and the same five temporal Double-DQN runs in `results/temporal-double-dqn/`. Run `python analysis/summarize_ppo.py` to reproduce `results/ppo-comparison.csv`. |
| Learning-attacker evaluation | Malicious-only episode records and summary in `results/phase5d_adaptive/` |

The headline comparison uses five controller seeds and 50 episodes per
caller pool per seed. Signal ablation uses three seeds with the same balance.
The robustness and transfer analyses use 50 attacker-designated episodes per
cell and seed. The three matched reactive runs provide the reference for
return retention.

This package reproduces reported analyses from saved outputs. Fresh training,
classifier inference from input features, and local or public-chain
deployment require the original environment. Raw transactions, caller
identifiers, and credentials are excluded.

## Revised figure exports and PPO evaluation

The supplied [policy comparison](figures/manuscript/fig2_policy_comparison.pdf) and [signal ablation](figures/manuscript/fig3_signal_ablation.pdf) are under `figures/manuscript/`. The figure workflow above continues to generate the reproduction plots separately.

The five-seed standalone PPO evaluation is in `results/ppo-policy-evaluation/`. It is a separate run from `results/ppo-matched/`, which remains the source for the matched PPO comparison.
