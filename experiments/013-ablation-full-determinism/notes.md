# Experiment 013: Full TF Determinism Verification

## Question
Does `tf.config.experimental.enable_op_determinism()` + explicit shuffle seed produce
reproducible results?

## Changes from 012
- `tf.keras.utils.set_random_seed(42)` replaces manual seed calls
- `tf.config.experimental.enable_op_determinism()` forces all TF ops to be deterministic
- Explicit `seed=42` passed to `tf.data.Dataset.shuffle()`
- `PYTHONHASHSEED=42` set as environment variable

## Key Comparisons
- **vs exp 012**: If results differ, the extra determinism measures are needed
- **Absolute R²**: Can we get back to 0.87 range, or is 0.28 the deterministic result?
- **Reproducibility**: A future re-run with same config should give identical results

## Pipeline Run
- Run ID: (pending)
- Duration: (pending)

## Results
(pending)
