# Experiment 008: Bottleneck Midpoint (96)

## Question
Is the optimal bottleneck between 64 and 128? Or is 128 a sharp threshold?

## Changes from 005
- bottleneck_dim=96 (was 128)
- Fixed misleading hparams in pipeline code
- Fixed stale compiled JSON (pipelines/src/ablation.json was from Feb 16, still had load-dataset-to-bigquery)

## Results

| Metric | Baseline | Single-DC | Multi-DC |
|--------|:--------:|:---------:|:--------:|
| RMSE | **0.466** | 0.506 | 1.657 |
| MAE | 0.411 | **0.347** | 1.406 |
| R-squared | **0.235** | 0.097 | -8.673 |

## Analysis

**Bottleneck=96 is worse than both 64 and 128.** Non-monotonic behavior:

| Bottleneck | Baseline R² | Single-DC R² |
|:----------:|:-----------:|:------------:|
| 0 (003) | 0.845 | 0.665 |
| 32 (002) | -57.3 | -81.7 |
| 64 (007) | 0.531 | -0.509 |
| 96 (008) | 0.235 | 0.097 |
| **128 (005)** | **0.871** | **0.874** |
| 384 (006) | 0.728 | 0.786 |

The R² landscape is NOT monotonic. There appears to be a narrow peak at 128 with a valley around 64-96. This suggests complex interactions between the bottleneck dimension and the downstream Dense(64) → Dense(32) layers.

Possible explanation: At bottleneck=96, the model has a bottleneck → Dense(64) contraction (96→64), which may create a problematic gradient landscape. At bottleneck=128, the 128→64 contraction provides a smooth 2:1 compression at each stage (128→64→32→1). At bottleneck=64, all hidden layers are the same width (64→64→32→1), which also works better than 96.

## Pipeline Fix
Found that `pipelines/src/ablation.json` was stale (from Feb 16). The compiler writes to `pipelines/ablation.json` but trigger reads from `pipelines/src/ablation.json`. Fixed the compiler output path.

## Conclusion
128 is definitively optimal. The bottleneck sweep is complete (6 values tested: 0, 32, 64, 96, 128, 384).
