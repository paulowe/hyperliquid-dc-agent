# Experiment 012: Seed Verification (re-run exp 005 with seed=42)

## Question
Does adding random seed control (seed=42) produce reproducible R² across runs?

## Changes from 011
- single_dc_threshold=0.001 (was 0.005)
- random_seed=42 added to hparams (seeds TF, NumPy, Python random)
- Training caching disabled (force re-train with seed)

## Key Comparisons
- **R² stability**: Compare baseline R² to exp 005 (0.871) and exp 011 (0.198)
- **Reference_price DA**: First directional accuracy measurement for threshold=0.001
- **Reproducibility**: If this run and a future re-run give same R², seeds work

## Pipeline Run
- Run ID: (pending)
- Duration: (pending)

## Results
(pending)
