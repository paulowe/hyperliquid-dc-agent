# Experiment 003: Data Volume Only

## Question
Was v2's catastrophic failure caused by the bottleneck/regularization or by the data volume?

## Changes from 002
- Removed bottleneck_dim (32 -> 0)
- Removed dropout_rate (0.3 -> 0.0)
- Removed l2_reg (1e-4 -> 0.0)
- Everything else identical: same 3-month data, same epochs=20, patience=5, lr=0.01

## Results

| Metric | Baseline | Single-DC | Multi-DC |
|--------|:--------:|:---------:|:--------:|
| RMSE | **0.116** | 0.171 | 2.583 |
| MAE | **0.096** | **0.096** | 2.152 |
| R-squared | **0.845** | 0.665 | -75.4 |
| sMAPE | **0.024** | 0.025 | 0.882 |
| Dir. Accuracy | 0.479 | 0.480 | 0.415 |

Statistical tests:
- Single-DC vs Baseline: RMSE -47% worse, Wilcoxon p=0.0, **t-test p=0.088 (not significant at 0.05!)**
- Multi-DC vs Baseline: RMSE -2123% worse, both tests p=0.0

## Analysis

**Bottleneck was the culprit in v2.** Baseline recovered dramatically:

| | v1 (7d) | v2 (3mo+bottleneck) | v3 (3mo, no bottleneck) |
|---|:-:|:-:|:-:|
| Baseline R² | 0.545 | -57.3 | **0.845** |
| Baseline RMSE | 0.359 | 2.257 | **0.116** |

More data HELPED the baseline — R² improved from 0.545 to 0.845, and RMSE improved 3x from 0.359 to 0.116.

### Key observations

1. **Baseline is excellent with more data** (R²=0.845). The 3 common features (PRICE_std, vol_quote_std, cvd_quote_std) capture price dynamics well.

2. **Single-DC is competitive on MAE** (0.096 vs 0.096) but worse on RMSE (0.171 vs 0.116). The t-test p=0.088 means the MAE difference is NOT statistically significant. The RMSE difference IS significant (Wilcoxon p=0.0), suggesting single-DC has similar median error but more large outlier errors.

3. **Multi-DC is still catastrophically bad** (R²=-75). With 27 features and no regularization, the 86K-param model massively overfits. This is the same capacity problem from v1, amplified.

4. **Capacity matters, but bottleneck was too aggressive.** The single-DC model (28.8K params) is borderline — it almost matches baseline on MAE. The multi-DC model (86.4K params) needs regularization.

## Conclusions
- The bottleneck (projecting 150+ dims to 32) was destroying information, causing v2's catastrophe.
- More data helps — baseline went from R²=0.55 to R²=0.85.
- Single-DC features may be adding marginal value (statistically tied on MAE).
- Multi-DC is overfitting — needs capacity control, but NOT an aggressive bottleneck.

## Next Experiment
-> 004: Add ONLY mild regularization (dropout=0.2, no bottleneck, no L2) to help multi-DC. Keep baseline and single-DC architectures identical. The question becomes: can mild regularization bring multi-DC into competitive range while not hurting baseline/single-DC?
