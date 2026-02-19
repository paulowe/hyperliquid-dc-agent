# Experiment 011: DC Threshold Sweep (threshold=0.005)

## Question
Does a larger DC threshold (0.005 = 0.5% price change) produce better forecasting than 0.001 (0.1%)?

## Changes from 005
- single_dc_threshold=0.005 (was 0.001)
- shift=50 (same as 005)
- bottleneck_dim=128 (same as 005)
- Training script includes reference_price for directional accuracy

## Hypothesis
- Larger threshold = fewer but more significant DC regime changes
- May provide cleaner regime signal with less noise
- threshold=0.001 produces many events (0.1% = frequent in crypto)
- threshold=0.005 filters out minor fluctuations, capturing only meaningful reversals

## Results

| Metric | Baseline | Single-DC (0.005) | Multi-DC |
|--------|:--------:|:------------------:|:--------:|
| RMSE | | | |
| MAE | | | |
| R-squared | | | |
| Dir. Acc. | | | |

## Analysis
(To be filled after results)
