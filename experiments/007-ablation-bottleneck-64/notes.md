# Experiment 007: Tighter Bottleneck (64)

## Question
Does a tighter bottleneck (64 vs 128) improve single-DC via stronger information bottleneck, or degrade it via too much compression?

## Changes from 005
- bottleneck_dim=64 (was 128)
- Everything else identical

## Bottleneck Sweep Context
| Bottleneck | Baseline R² | Single-DC R² | Single-DC vs Baseline |
|:----------:|:-----------:|:------------:|:---------------------:|
| 0 (exp 003) | 0.845 | 0.665 | -0.180 (worse) |
| **64 (this)** | TBD | TBD | TBD |
| 128 (exp 005) | 0.871 | 0.874 | +0.003 (better) |
| 384 (exp 006) | 0.728 | 0.786 | +0.058 (better) |

## Results
TBD

## Analysis
TBD
