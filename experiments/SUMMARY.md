# Experiment Summary: DC Feature Ablation Study

## Research Question
Do Directional Change (DC) features improve BTC-USD price forecasting accuracy?

## Experiment Matrix

| Exp | Data | Bottleneck | Dropout | L2 | Epochs | Baseline R² | Single-DC R² | Multi-DC R² | Key Finding |
|-----|------|:----------:|:-------:|:--:|:------:|:-----------:|:------------:|:-----------:|-------------|
| 001 | 7 days (422K) | 0 | 0 | 0 | 5 | **0.545** | 0.095 | -5.774 | Baseline wins; DC arms overfit |
| 002 | 3 months (4.1M) | 32 | 0.3 | 1e-4 | 20 | -57.3 | -81.7 | -120.9 | Bottleneck(32) destroyed all arms |
| 003 | 3 months (4.1M) | 0 | 0 | 0 | 20 | **0.845** | 0.665 | -75.4 | More data helps! Baseline excellent |
| 004 | 3 months (4.1M) | 0 | 0.2 | 0 | 20 | -0.291 | -2.229 | -40.6 | Dropout hurts small networks |
| 005 | 3 months (4.1M) | 128 | 0 | 0 | 20 | 0.871 | **0.874** | -20.07 | **Single-DC beats baseline!** |
| 006 | 3 months (4.1M) | 384 | 0 | 0 | 20 | TBD | TBD | TBD | Match compression ratio for multi-DC |

## Key Insights (so far)

1. **More data dramatically improves baseline**: R² went from 0.545 (7 days) to 0.845 (3 months)
2. **Bottleneck(32) is too aggressive**: Compressing 150+ flattened dims to 32 destroys temporal information
3. **Bottleneck(128) is the sweet spot**: Improves BOTH baseline (0.845→0.871) and single-DC (0.665→0.874)
4. **Single-DC beats baseline with bottleneck(128)**: First positive DC result! R² 0.874 vs 0.871, MAE 13% better (p=0.0)
5. **Dropout doesn't work for small dense networks**: Even 0.2 ruins baseline performance
6. **Compression ratio hypothesis**: Single-DC succeeds at 72% compression (450→128). Multi-DC fails at 90% (1350→128). Exp 006 tests multi-DC at 72% (1350→384).
7. **Information bottleneck = implicit feature selection**: Forces the model to retain only the most useful patterns
8. **Multi-DC always overfits without capacity control**: 86.4K–183K params with 27 features catastrophically fails
9. **KFP caching pitfall**: Same table name + different data = stale cache. Use `enable_caching=False` for new data ranges.

## Architecture
```
Model: Flatten -> [Optional: Dense(bottleneck_dim)] -> Dense(64) -> Dense(32) -> Dense(1)
Target: PRICE_std (standardized BTC-USD price)
Window: input_width=50, shift=50, label_width=1
Split: 70/20/10 train/val/test
```

## Feature Sets
- **Baseline** (3): PRICE_std, vol_quote_std, cvd_quote_std
- **Single-DC** (9): 3 common + 6 DC from threshold=0.001
- **Multi-DC** (27): 3 common + 6 DC * 4 thresholds (0.001, 0.005, 0.010, 0.015)

## DC Features (per threshold)
PDCC_Down (binary), OSV_Down_std (continuous), OSV_Up_std (continuous), PDCC2_UP (binary), regime_up (binary), regime_down (binary)
