# Experiment 005: Larger Bottleneck (128)

## Question
With a gentler bottleneck (128 vs 32), can we equalize capacity without destroying baseline performance? Does multi-DC improve?

## Changes from 003
- Added bottleneck_dim=128 (was 0 in 003, was 32 in 002)
- No dropout, no L2 (same as 003)

## Results

| Metric | Baseline | Single-DC | Multi-DC |
|--------|:--------:|:---------:|:--------:|
| RMSE | 0.106 | **0.105** | 1.357 |
| MAE | 0.087 | **0.076** | 1.051 |
| R-squared | 0.871 | **0.874** | -20.07 |
| sMAPE | 0.022 | **0.019** | 0.366 |

Statistical tests (single-DC vs baseline):
- Wilcoxon p-value: 0.0 (significant)
- Paired t-test p-value: 0.0 (significant)
- RMSE improvement: 1.14%

Param counts: baseline=29,697, single-DC=68,097, multi-DC=183,297

## Analysis

**BREAKTHROUGH: Single-DC beats baseline for the first time!**

1. **Single-DC wins on every metric** — R² 0.874 vs 0.871, MAE 13% better, RMSE 1.14% better, all statistically significant (p=0.0).

2. **Bottleneck(128) actually improves both arms vs exp 003** (no bottleneck):
   - Baseline: R² 0.845 → 0.871 (+0.026)
   - Single-DC: R² 0.665 → 0.874 (+0.209, massive!)
   - The information bottleneck acts as implicit regularization, forcing the model to learn compact representations.

3. **Compression ratio matters**:
   - Baseline: 150→128 (15% compression) — mild, preserves almost all info → R²=0.871
   - Single-DC: 450→128 (72% compression) — moderate, forces feature selection → R²=0.874
   - Multi-DC: 1350→128 (90% compression) — too aggressive, destroys signal → R²=-20.07

4. **Multi-DC improved but still bad**: R²=-20.07 vs -75.4 (exp 003). The bottleneck helps but 90% compression is too much for 27 features.

## Key Insight
The information bottleneck is a form of implicit feature selection. At 72% compression, single-DC features are forced to compete for limited representational capacity, and DC features win enough representation to improve forecasting. At 90%, multi-DC features overwhelm the bottleneck.

## Next Experiment
→ 006: Test bottleneck=384 to give multi-DC the same 72% compression ratio that made single-DC succeed. (1350→384 = 72% compression, matching single-DC's ratio in this experiment.)
