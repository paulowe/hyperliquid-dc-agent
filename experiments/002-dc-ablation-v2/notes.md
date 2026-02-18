# Experiment 002: DC Feature Ablation v2

## Question
With equal model capacity, regularization, and more data, do DC features improve forecasting?

## Changes from 001
1. **Equal capacity**: Bottleneck layer (Dense(32)) after Flatten projects all feature dims to same size
2. **More data**: 3 months (May 1 - Aug 1, 2025) instead of 7 days
   - Actual coverage: May 22 - Jul 31 (~2.3 months) — no data before May 22 in BQ
   - ~4.1M ticks vs 422K in v1 (~10x more)
   - Monthly: May=221K (partial), Jun=2.0M, Jul=1.9M
3. **Regularization**: Dropout(0.3) + L2(1e-4) to prevent overfitting
4. **More epochs**: 20 (up from 5) with early stopping patience=5

## Results

| Metric | Baseline | Single-DC | Multi-DC |
|--------|:--------:|:---------:|:--------:|
| RMSE | **2.257** | 2.688 | 3.263 |
| MAE | **2.250** | 2.679 | 3.248 |
| R-squared | **-57.3** | -81.7 | -120.9 |
| sMAPE | **0.777** | 0.998 | 1.374 |
| Dir. Accuracy | 0.479 | 0.477 | 0.473 |

Statistical significance (all p=0.0):
- Single-DC vs Baseline: -19.1% worse RMSE
- Multi-DC vs Baseline: -44.6% worse RMSE
- Multi-DC vs Single-DC: -21.4% worse RMSE

## Analysis

**All three arms are catastrophically bad.** R-squared of -57 for the baseline means the model performs 57x worse than simply predicting the mean. This was NOT the case in v1 where baseline achieved R²=0.545.

### What went wrong

The DC feature question is now secondary — the primary issue is that the model can't learn at all with this configuration. Likely causes:

1. **Bottleneck too aggressive**: Projecting 150 flattened values (50 timesteps * 3 features) down to 32 dims may destroy critical temporal structure. The Flatten layer outputs a 150-dim vector for baseline, 450 for single-DC, 1350 for multi-DC — compressing any of these to 32 is an extreme information bottleneck.

2. **Over-regularization**: Dropout(0.3) + L2(1e-4) combined with the bottleneck may leave too little effective capacity. The model can't even fit the baseline (3 features) which worked fine in v1 without regularization.

3. **Scale mismatch with more data**: 4.1M ticks with batch_size=100 means ~41K steps per epoch. With 20 epochs that's 820K gradient updates. Early stopping may have triggered before the model could converge, or the learning rate (0.01) may be too high for this data volume.

### Baseline degraded from v1

| | v1 (7 days) | v2 (2.3 months) |
|---|:-:|:-:|
| Baseline RMSE | 0.359 | 2.257 |
| Baseline R² | 0.545 | -57.3 |

The baseline went from decent to terrible — this is NOT a DC feature issue, it's a training regime issue. The bottleneck + regularization broke even the simple case.

### Key takeaway
The v2 experiment changed too many variables at once (bottleneck + dropout + L2 + data volume + epochs). We can't isolate which change caused the degradation. Need to test incrementally.

## Next Experiment
-> 003: Remove bottleneck and regularization. Keep only the extended data range (3 months). This isolates whether more data alone helps or hurts baseline performance. If baseline recovers to R²>0, THEN add bottleneck in 004.
