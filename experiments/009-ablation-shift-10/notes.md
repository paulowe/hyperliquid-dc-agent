# Experiment 009: Prediction Horizon (shift=10)

## Question
Does a shorter prediction horizon (10 ticks gap vs 50) improve forecasting accuracy? Does it change the relative value of DC features?

## Changes from 005 (best result so far)
- shift=10 (was 50)
- input_width and label_width unchanged (50, 1)
- bottleneck_dim=128 (same as 005)
- Added shift, input_width, label_width as pipeline parameters for future flexibility

## Hypothesis
- Shorter horizons should be easier to predict (higher R²)
- DC regime information may be more/less relevant at different time horizons
- The model predicts tick[60] given ticks[0..49], vs tick[100] in previous experiments

## Key context
- Exp 005 directional accuracy was ~47.7% for all arms (essentially random)
- The directional accuracy metric measures tick-by-tick direction of consecutive predictions, which is noisy
- R² is the primary comparison metric for arm comparison

## Results

| Metric | Baseline | Single-DC | Multi-DC |
|--------|:--------:|:---------:|:--------:|
| RMSE | | | |
| MAE | | | |
| R-squared | | | |

## Analysis
(To be filled after results)
