# Experiment Tracking

Each experiment is stored in a numbered directory: `NNN-<descriptive-name>/`.

## Structure

```
NNN-<name>/
  config.yaml     # Frozen pipeline params (dates, hparams, thresholds)
  results.json    # Raw output from compare_forecasts (copied from GCS)
  notes.md        # Analysis, learnings, link to next experiment
```

## Convention

- Sequential numbering (001, 002, ...)
- Each experiment documents what changed from the previous one and why
- `results.json` is copied from GCS after the pipeline run completes
- `notes.md` is filled in after reviewing results
