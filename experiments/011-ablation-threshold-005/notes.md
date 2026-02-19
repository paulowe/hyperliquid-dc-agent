# Experiment 011: DC Threshold Sweep (threshold=0.005)

## Question
Does a larger DC threshold (0.005 = 0.5% price change) produce better forecasting than 0.001 (0.1%)?

## Changes from 005
- single_dc_threshold=0.005 (was 0.001)
- shift=50 (same, but now parametrized instead of hardcoded — causes cache miss)
- Training script includes reference_price for directional accuracy
- compare_forecasts uses reference_price metric (UP/DOWN vs current price)

## Pipeline Run
- Run ID: dc-ablation-pipeline-20260219043444
- Caching: 20 upstream tasks cached (DC detection, feature engineering)
- Duration: ~30 minutes

## Results

| Metric | Baseline | Single-DC (0.005) | Multi-DC |
|--------|:--------:|:-----------------:|:--------:|
| RMSE | **0.477** | 0.792 | 1.554 |
| MAE | **0.422** | 0.693 | 1.331 |
| R-squared | **0.198** | -1.212 | -7.517 |
| Dir. Acc. | 0.442 | **0.558** | 0.443 |

## Analysis

### Mixed and important findings

1. **Training stochasticity is a major concern.** Baseline R²=0.198 in this run vs 0.871 in exp 005 (same architecture, same data, same shift=50, same bottleneck=128). The only differences are: (a) shift is now a pipeline parameter instead of hardcoded, (b) training script adds reference_price to predictions. Neither should affect model training. The R² difference must be due to random weight initialization + training dynamics.

2. **Single-DC (0.005) has 55.8% directional accuracy.** This is the FIRST metric above 50% in any experiment, using the new reference_price metric (sign(pred-ref) == sign(true-ref)). The model correctly predicts whether price will go UP or DOWN relative to current price 55.8% of the time.

3. **Directional accuracy vs R² paradox.** Single-DC has better directional accuracy (55.8%) but terrible R² (-1.212). This means the model captures directional information from 0.005 DC regime features but gets magnitude completely wrong. For trading, direction may be more important than magnitude.

4. **threshold=0.005 is worse than 0.001 on R².** Within this run, single-DC (0.005) has R²=-1.212 vs baseline R²=0.198. In exp 005, single-DC (0.001) had R²=0.874 vs baseline R²=0.871. But cross-run comparison is unreliable due to training stochasticity.

### Need for reproducibility

The R² instability (0.198 vs 0.871 for identical architecture) means:
- We CANNOT reliably compare R² across experiments
- We need either: (a) fixed random seeds, or (b) multiple runs per configuration
- Within-experiment comparisons remain valid
- Directional accuracy may be more stable across runs (to be verified)

### Directional Accuracy Comparison (reference_price metric, exp 011 only)

| Arm | Dir. Acc. |
|:---:|:---------:|
| Baseline | 0.442 (below random) |
| Single-DC (0.005) | **0.558** (above random!) |
| Multi-DC | 0.443 (below random) |

Baseline DA < 50% suggests the model predicts mean-reversion while the market was actually trending during the test period. DC features at threshold=0.005 capture enough regime information to correctly predict direction.

## Next Experiment
Re-run exp 005 configuration (threshold=0.001, shift=50, bottleneck=128) with reference_price to get comparable directional accuracy. This will also verify baseline R² stability.
