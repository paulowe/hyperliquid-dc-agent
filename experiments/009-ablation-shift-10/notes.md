# Experiment 009: Prediction Horizon (shift=10)

## Question
Does a shorter prediction horizon (10 ticks gap vs 50) improve forecasting accuracy? Does it change the relative value of DC features?

## Changes from 005 (best result so far)
- shift=10 (was 50)
- input_width and label_width unchanged (50, 1)
- bottleneck_dim=128 (same as 005)
- Added shift, input_width, label_width as pipeline parameters for future flexibility

## Pipeline Run
- Run ID: dc-ablation-pipeline-20260219031034
- Caching: enabled (DC detection, feature engineering, splitting all cached)
- Duration: ~35 minutes (vs ~60+ for full pipeline)

## Results

| Metric | Baseline | Single-DC | Multi-DC |
|--------|:--------:|:---------:|:--------:|
| RMSE | **0.171** | 0.212 | 0.745 |
| MAE | **0.101** | 0.127 | 0.675 |
| R-squared | **0.897** | 0.842 | -0.960 |
| Dir. Acc. | 0.497 | 0.495 | 0.498 |

## Analysis

**Shorter horizon is easier to predict, but DC features hurt at this timescale.**

### Shift Comparison (all with bottleneck=128)

| Shift | Baseline R² | Single-DC R² | Delta (DC-BL) | Best |
|:-----:|:-----------:|:------------:|:-------------:|:----:|
| 10 | **0.897** | 0.842 | -0.055 | Baseline |
| 50 | 0.871 | **0.874** | +0.003 | Single-DC |

### Key Observations

1. **Baseline R²=0.897 is the highest absolute R² in any experiment.** Shorter prediction horizons are inherently easier because the target is more correlated with recent input.

2. **DC features HURT at shift=10.** Single-DC R²=0.842 is 6% worse than baseline R²=0.897 (RMSE -23.7%). At shift=50, DC was marginally better (+0.003). This is a reversal.

3. **DC regime info operates at a specific timescale.** DC events (threshold=0.001 = 0.1% price change) happen over many ticks. At shift=10 (~seconds), the regime info is stale noise. At shift=50 (~minutes), the regime has time to "play out" and provides useful context.

4. **R² at short horizons may be misleading.** At shift=10, the target (tick[60]) is very close to the last input (tick[49]). A naive "predict last seen price" model would score high R² by autocorrelation. The R²=0.897 may not indicate useful predictive power — just proximity.

5. **Directional accuracy still ~50%** for all arms and both shifts. The tick-by-tick metric (np.diff) is uninformative. The reference_price improvement (committed locally) will fix this in future experiments.

## Implications for Trading

- At shift=10, baseline > DC features. The model doesn't need regime info for very short-term prediction.
- At shift=50, DC > baseline (marginally). Regime info helps at medium-term horizons.
- The natural next question: does DC advantage grow at longer horizons (shift=100)?
- Before running more shifts, the reference_price directional accuracy metric should be deployed to evaluate whether ANY of these models correctly predict UP/DOWN relative to current price.

## Next Experiment
Experiment 010: shift=25 (interpolate between 10 and 50 to find crossover) OR shift=100 (test if DC advantage grows at longer horizons).
