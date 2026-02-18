# Experiment 006: Matched Compression Ratio (bottleneck=384)

## Question
Does multi-DC succeed when given the same compression ratio (72%) that made single-DC beat baseline in exp 005?

## Changes from 005
- bottleneck_dim=384 (was 128)
- Everything else identical (no dropout, no L2, 20 epochs)

## Rationale
In exp 005, single-DC succeeded at 72% compression (450→128). Multi-DC failed at 90% compression (1350→128). This experiment tests whether matching the compression ratio (1350→384 = 72%) allows multi-DC to learn useful patterns from the additional DC thresholds.

If multi-DC still fails at 72% compression, it means the extra thresholds (0.005, 0.010, 0.015) add noise rather than signal.

## Results
TBD

## Analysis
TBD
