# Experiment 014: Threshold=0.005 with Full Determinism

## Question
Is the DA=55.8% result from exp 011 (threshold=0.005) reproducible with deterministic training?

## Changes from 013
- single_dc_threshold=0.005 (was 0.001)

## Pipeline Run
- Run ID: dc-ablation-pipeline-20260219060551
- Duration: ~25 minutes (upstream cached)

## Results

| Metric | Baseline | Single-DC (0.005) | Multi-DC |
|--------|:--------:|:-----------------:|:--------:|
| RMSE | **0.599** | 0.907 | 1.690 |
| MAE | **0.555** | 0.694 | 1.454 |
| R-squared | **-0.266** | -1.901 | -9.071 |
| Dir. Acc. | 0.441 | **0.534** | 0.441 |

## Analysis

### 1. DETERMINISM IS WORKING — baseline and multi-DC are bit-for-bit identical to exp 013

| Metric | Exp 013 | Exp 014 | Match? |
|--------|:-------:|:-------:|:------:|
| BL R² | -0.2664489035 | -0.2664489035 | EXACT |
| BL RMSE | 0.5994269067 | 0.5994269067 | EXACT |
| BL DA | 0.4408390147 | 0.4408390147 | EXACT |
| Multi R² | -9.0706574930 | -9.0706574930 | EXACT |
| Multi RMSE | 1.6903292632 | 1.6903292632 | EXACT |

This is the strongest possible evidence that deterministic training is working.
The baseline and multi-DC arms have identical hparams and input data between exp 013
and exp 014, and produce bit-for-bit identical results.

### 2. DA=53.4% confirms directional signal at threshold=0.005

| Experiment | Threshold | DA | Determinism |
|:----------:|:---------:|:--:|:-----------:|
| Exp 011 | 0.005 | **0.558** | No |
| **Exp 014** | **0.005** | **0.534** | **Yes** |
| Exp 013 | 0.001 | 0.519 | Yes |
| Exp 012 | 0.001 | 0.439 | No (basic seed) |

Both deterministic runs (013, 014) show DA > 50% for DC features.
threshold=0.005 gives slightly higher DA (53.4%) than 0.001 (51.9%).

### 3. Threshold=0.005 trades R² for DA

| Config | R² | DA | RMSE |
|--------|:--:|:--:|:----:|
| Baseline | -0.266 | 0.441 | 0.599 |
| DC 0.001 (exp 013) | **0.039** | 0.519 | **0.522** |
| DC 0.005 (exp 014) | -1.901 | **0.534** | 0.907 |

Higher threshold captures more directional info (higher DA) but at the cost of magnitude
accuracy (worse R² and RMSE). This is consistent: larger DC thresholds trigger on bigger
price moves, which are more directionally meaningful but harder to predict in magnitude.

### 4. For trading, DA may matter more than R²

If the model correctly predicts direction 53.4% of the time with threshold=0.005,
a simple directional trading strategy could be profitable:
- Signal: predicted direction (UP/DOWN relative to reference_price)
- Entry: market order in predicted direction
- Exit: at shift=50 ticks
- Expected edge: (0.534 * avg_win) - (0.466 * avg_loss)

The key question is whether avg_win > avg_loss. With symmetric wins/losses, 53.4% DA
gives a positive expected value.

## Key Implications

1. **Determinism confirmed**: Identical hparams + data = identical results (bit-for-bit)
2. **DA > 50% is robust**: Both thresholds show DA above random with determinism
3. **threshold=0.005 gives best DA** (53.4%) but worst RMSE among DC arms
4. **threshold=0.001 gives best R²** (0.039) but lower DA (51.9%)
5. **Baseline DA < 50%**: Consistently 44.1% — model predicts mean-reversion while market trends

## Next Experiment
Test threshold=0.010 to complete the DA trend (does DA keep improving with larger thresholds?).
Then decide on production config based on DA vs RMSE trade-off.
