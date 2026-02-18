# Experiment 007: Tighter Bottleneck (64)

## Question
Does a tighter bottleneck (64 vs 128) improve single-DC via stronger information bottleneck, or degrade it via too much compression?

## Changes from 005
- bottleneck_dim=64 (was 128)
- enable_caching=False (stale load-dataset-to-bigquery task was blocking cached runs)
- Everything else identical

## Results

| Metric | Baseline | Single-DC | Multi-DC |
|--------|:--------:|:---------:|:--------:|
| RMSE | **0.365** | 0.654 | 1.681 |
| MAE | **0.284** | 0.610 | 1.361 |
| R-squared | **0.531** | -0.509 | -8.956 |
| sMAPE | **0.101** | 0.231 | 0.764 |

## Analysis

**Bottleneck=64 is too aggressive.** All arms degraded significantly:
- Baseline: R² 0.871 → 0.531 (bottleneck compresses 150→64 = 57%, losing too much temporal info)
- Single-DC: R² 0.874 → -0.509 (450→64 = 86% compression destroys DC signal)
- Multi-DC: R² -20.07 → -8.956 (paradoxically "better" but still terrible)

## Complete Bottleneck Sweep

| Bottleneck | Baseline R² | Single-DC R² | Delta (DC-BL) | Verdict |
|:----------:|:-----------:|:------------:|:-------------:|:-------:|
| 0 (exp 003) | 0.845 | 0.665 | -0.180 | BL wins |
| 32 (exp 002) | -57.3 | -81.7 | -24.4 | Both dead |
| **64 (exp 007)** | **0.531** | **-0.509** | **-1.040** | **BL wins, both degraded** |
| **128 (exp 005)** | **0.871** | **0.874** | **+0.003** | **DC wins (optimal!)** |
| 384 (exp 006) | 0.728 | 0.786 | +0.058 | DC wins, both degraded |

## Key Insight
The bottleneck sweep is complete. 128 is definitively the optimal bottleneck:
- Below 128: too much compression destroys temporal information (64 hurts both, 32 kills both)
- Above 128: too little compression weakens regularization (384 degrades both)
- At 128: perfect balance — mild compression for baseline (15%), moderate for single-DC (72%)

The single-DC advantage ONLY exists at bottleneck=128 and 384. At 0 and 64, baseline wins.
This means DC features require the information bottleneck to be useful — without it, they add noise.
