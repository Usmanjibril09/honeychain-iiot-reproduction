#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HONEYCHAIN_HEADLINE_CSV="$PWD/results/comparison.csv"
export HONEYCHAIN_ATTACKER_AUDIT="$PWD/results/attacker-branch"
for module in fig1_classifier fig2_policy_comparison fig3_signal_ablation \
  fig4_adversarial_robustness fig5_cross_scenario_transfer fig6_overhead \
  fig7_sepolia fig8_synthesis; do
  python -m "src.figures.$module"
done
