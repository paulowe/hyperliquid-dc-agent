# Experiment 008: Bottleneck Midpoint (96)

## Question
Is the optimal bottleneck between 64 and 128? Or is 128 a sharp threshold?

## Changes from 005
- bottleneck_dim=96 (was 128)
- Also corrected misleading hparams in pipeline code (loss_fn, optimizer, removed early_stopping_epochs)

## Pipeline Review Findings (documented during this experiment)
Before running exp 008, conducted a thorough review of all pipeline components:

1. **Config mismatches fixed**: loss_fn was listed as MSE but hardcoded as MAE; optimizer listed as Adam but is AdamW
2. **Data flow verified**: Scaling, splitting, windowing, and comparison alignment all correct
3. **Scaler independence**: Each threshold path fits its own StandardScaler, but PRICE/vol_quote/cvd_quote have identical mean/std across paths (same raw data). Comparison of PRICE_std predictions is valid.
4. **shift=50 noted**: Model predicts 50 ticks ahead (not next tick). Consistent across arms, suitable for initial experiments.
5. **dsl.Collected ordering**: ParallelFor ordering not guaranteed, but irrelevant for baseline (identical common features) and consistent within each run.

## Results
TBD

## Analysis
TBD
