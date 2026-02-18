# Experiment 001: DC Feature Ablation v1

## Question
Do Directional Change (DC) features improve BTC-USD price forecasting accuracy?

## Design
3-arm study with identical Dense(64)->Dense(32)->Dense(1) architecture:
- **Baseline**: 3 common features (PRICE_std, vol_quote_std, cvd_quote_std)
- **Single-DC**: 9 features (3 common + 6 DC from threshold=0.001)
- **Multi-DC**: 27 features (3 common + 6 DC * 4 thresholds)

Data: 7 days (Aug 1-8, 2025), ~422K ticks, window=50, shift=50, label_width=1

## Results

| Metric | Baseline | Single-DC | Multi-DC |
|--------|:--------:|:---------:|:--------:|
| RMSE | **0.359** | 0.507 | 1.386 |
| MAE | **0.279** | 0.323 | 1.214 |
| R-squared | **0.545** | 0.095 | -5.774 |
| sMAPE | **0.099** | 0.110 | 0.550 |

Baseline won decisively. Multi-DC has negative R-squared indicating severe overfitting.

## Analysis

The result does NOT conclusively show DC features are useless. Three confounds:

1. **Unequal capacity**: Flatten->Dense means first layer params scale linearly with features (9.6K vs 28.8K vs 86.4K). More params + less data = overfitting.
2. **Tiny data window**: 7 days is insufficient for DC regime patterns to emerge. DC events at threshold=0.015 may only fire a handful of times in a week.
3. **No regularization**: No dropout or weight decay to prevent the larger models from memorizing noise.

## Next Experiment
-> 002-dc-ablation-v2: Add bottleneck projection (equal capacity), extend to 3 months, add dropout + L2 regularization.
