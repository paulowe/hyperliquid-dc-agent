# Experiment 006: Matched Compression Ratio (bottleneck=384)

## Question
Does multi-DC succeed when given the same compression ratio (72%) that made single-DC beat baseline in exp 005?

## Changes from 005
- bottleneck_dim=384 (was 128)
- Everything else identical (no dropout, no L2, 20 epochs)

## Results

| Metric | Baseline | Single-DC | Multi-DC |
|--------|:--------:|:---------:|:--------:|
| RMSE | 0.154 | **0.137** | 3.784 |
| MAE | 0.140 | **0.119** | 3.678 |
| R-squared | 0.728 | **0.786** | -162.9 |
| sMAPE | 0.036 | **0.029** | 1.718 |

Statistical tests (single-DC vs baseline):
- Wilcoxon p-value: 0.0 (significant)
- Paired t-test p-value: 0.0 (significant)
- RMSE improvement: **11.1%** (much larger than exp 005's 1.14%)

Pipeline status: FAILED (due to load-dataset-to-bigquery — cached from old compiled JSON). Compare-forecasts succeeded.

## Analysis

1. **Single-DC beats baseline by 11.1% RMSE** — the largest margin yet, confirming DC features add real predictive value.

2. **Both arms degraded vs exp 005** (bottleneck=128):
   - Baseline: R² 0.871 → 0.728 (-0.143)
   - Single-DC: R² 0.874 → 0.786 (-0.088)
   - Bottleneck=384 is too large. For baseline (150 dims), 384 is an expansion that adds unnecessary parameters. For single-DC (450 dims), 15% compression provides less information bottleneck regularization than 72%.

3. **Multi-DC fails catastrophically even at matched compression** (R²=-162.9, worse than exp 005's -20.07). This decisively shows the extra DC thresholds (0.005, 0.010, 0.015) add noise, not signal. The compression ratio wasn't the problem — the features themselves are noisy.

4. **Information bottleneck strength vs model quality trade-off**: Tighter bottleneck → stronger regularization → potentially better generalization, but can also lose signal. Sweet spot is bottleneck=128 for both baseline and single-DC.

## Conclusions on Bottleneck Dimension

| Bottleneck | Baseline R² | Single-DC R² | Single-DC margin |
|:----------:|:-----------:|:------------:|:----------------:|
| 0 (exp 003) | 0.845 | 0.665 | -0.180 (worse) |
| 128 (exp 005) | **0.871** | **0.874** | +0.003 (better) |
| 384 (exp 006) | 0.728 | 0.786 | +0.058 (better) |

Key pattern: bottleneck always helps single-DC more than baseline, but optimal value is ~128.

## Next Experiment
→ 007: Try bottleneck=64 to test whether even tighter compression helps (more aggressive info bottleneck).
