# Experiment 012: Seed Verification (re-run exp 005 with seed=42)

## Question
Does adding random seed control (seed=42) produce reproducible R² across runs?

## Changes from 011
- single_dc_threshold=0.001 (was 0.005)
- random_seed=42 added to hparams (seeds TF, NumPy, Python random)
- Training caching disabled (force re-train with seed)

## Pipeline Run
- Run ID: dc-ablation-pipeline-20260219051337
- Duration: ~20 minutes (upstream cached)

## Results

| Metric | Baseline | Single-DC (0.001) | Multi-DC |
|--------|:--------:|:-----------------:|:--------:|
| RMSE | **0.453** | 0.522 | 2.220 |
| MAE | **0.394** | 0.448 | 2.013 |
| R-squared | **0.278** | 0.039 | -16.378 |
| Dir. Acc. | 0.443 | 0.439 | 0.439 |

## Analysis

### 1. Seed did NOT fix stochasticity

Baseline R² across 3 runs of the SAME architecture (shift=50, bottleneck=128):

| Run | Seed | Baseline R² | Single-DC R² | Notes |
|-----|:----:|:-----------:|:------------:|-------|
| Exp 005 | none | 0.871 | 0.874 | DC barely wins (+0.003) |
| Exp 011 | none | 0.198 | -1.212 | BL wins (threshold=0.005 for DC) |
| **Exp 012** | **42** | **0.278** | **0.039** | BL wins; seed gave 3rd different value |

Basic `tf.random.set_seed()` + `np.random.seed()` + `random.seed()` is INSUFFICIENT.
Known TF issues: `shuffle(reshuffle_each_iteration=True)` without explicit seed,
non-deterministic ops, PYTHONHASHSEED not set.

### 2. "DC beats baseline" from exp 005 was likely noise

Within-experiment comparisons flip:
- Exp 005: DC R² 0.874 > BL 0.871 → DC wins by +0.003
- Exp 012: DC R² 0.039 < BL 0.278 → BL wins by +0.239

Same configuration, different random initialization. The +0.003 margin from exp 005
is far within the noise band (~0.6 R² variance across runs).

### 3. Directional accuracy ~44% for all arms with threshold=0.001

All three arms cluster around 43-44% DA (below random). Compared to exp 011 where
single-DC (threshold=0.005) achieved 55.8%, threshold=0.001 provides no directional signal.

### 4. Multi-DC catastrophically bad again

R² = -16.378 confirms multi-DC overfitting is consistent and NOT a stochasticity artifact.

## Critical Implications

1. **Cannot trust cross-experiment R² comparisons with < 0.5 margin**
2. **Need full TF determinism**: `tf.keras.utils.set_random_seed()` + `tf.config.experimental.enable_op_determinism()` + explicit shuffle seed
3. **The exp 005 "DC beats baseline" result is unreliable** — within noise
4. **threshold=0.005 directional signal is real**: DA=55.8% vs ~44% is too large to be noise
5. **Multi-DC conclusion stands**: Catastrophic failure is consistent across all runs

## Next Experiment
Implement full TF determinism (`enable_op_determinism()` + explicit shuffle seed) and
re-run to verify reproducibility before continuing threshold sweep.
