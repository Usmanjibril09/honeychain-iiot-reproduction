# HoneyChain-IIoT reproduction

This package regenerates manuscript Figures 2, 3, and 8 from exported evaluation
records and aggregate results. Episode records contain evaluation metrics and
omit source transactions and caller identifiers. The package supports
reproduction of these figures from saved results. End-to-end retraining and
contract deployment require the original experimental environment.

Use Python 3.11 and run from this directory.

```sh
python -m pip install -r requirements.txt
python -m src.figures.fig2_policy_comparison
python -m src.figures.fig3_signal_ablation
python -m src.figures.fig8_synthesis
```

The commands write PDF and PNG files to `figures/`.
