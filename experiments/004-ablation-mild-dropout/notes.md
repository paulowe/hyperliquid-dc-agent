# Experiment 004: Mild Dropout

## Question
Can mild dropout (0.2) bring multi-DC into competitive range without hurting baseline/single-DC?

## Changes from 003
- Added dropout_rate=0.2 (was 0.0)
- Everything else identical

## Results

| Metric | Baseline | Single-DC | Multi-DC |
|--------|:--------:|:---------:|:--------:|
| RMSE | 0.336 | 0.531 | 1.906 |
| MAE | 0.246 | 0.436 | 1.489 |
| R-squared | -0.291 | -2.229 | -40.6 |
| sMAPE | 0.063 | 0.115 | 0.521 |

## Analysis

**Dropout hurt everything.** Compare to 003 (no dropout):

| Model | 003 R² | 004 R² (dropout=0.2) | Impact |
|-------|:------:|:--------------------:|--------|
| Baseline | **0.845** | -0.291 | Destroyed |
| Single-DC | **0.665** | -2.229 | Destroyed |
| Multi-DC | -75.4 | -40.6 | Slightly less bad |

Key findings:
1. **Dropout 0.2 is too aggressive** for this architecture. Even the baseline (9.6K params) can't learn with dropout.
2. **Multi-DC improved slightly** (-75 -> -41) but is still terrible. Regularization alone won't fix 86.4K params.
3. The Dense layers are already small (64, 32). Dropping 20% of 64 neurons leaves very little capacity.

## Conclusion
Dropout is not the right regularization strategy for these small dense networks. The model needs every neuron to learn useful representations. Even 0.2 is too much.

## Next Experiment
-> 005: Fair capacity comparison using a larger bottleneck (128 dims). This preserves most temporal structure for baseline (150->128, mild compression) while significantly reducing multi-DC capacity (1350->128). No dropout, no L2.
