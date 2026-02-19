# Experiment 010: Longer Prediction Horizon (shift=100)

## Question
Does DC feature advantage grow at longer prediction horizons? Is there a monotonic relationship between shift and DC value?

## Changes from 009
- shift=100 (was 10)
- Training script now outputs reference_price in predictions CSV
- compare_forecasts computes directional accuracy as sign(pred-ref) == sign(true-ref)

## Hypothesis
Based on the pattern from exps 005 and 009:
- shift=10: DC delta = -0.055 (DC hurts)
- shift=50: DC delta = +0.003 (DC helps marginally)
- shift=100: DC delta = ??? (expect positive, possibly larger)

If DC regime events operate at a specific timescale, longer horizons should give the regime more time to express, potentially increasing DC advantage. However, longer horizons are also harder to predict overall, so R² may drop.

## Results

| Metric | Baseline | Single-DC | Multi-DC |
|--------|:--------:|:---------:|:--------:|
| RMSE | | | |
| MAE | | | |
| R-squared | | | |
| Dir. Acc. | | | |

## Analysis
(To be filled after results)
