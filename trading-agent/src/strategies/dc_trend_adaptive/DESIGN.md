# DC Trend-Adaptive Strategy — Design Document

## Problem Statement

MFE analysis of HYPE 14-day BQ telemetry (48 trades, 1.17M ticks) reveals a
massive directional asymmetry destroying profitability:

| Direction | Trades | Win Rate | Avg Captured | Avg MFE | Avg MAE | Sum Captured |
|-----------|--------|----------|-------------|---------|---------|-------------|
| LONG      | 24     | **79%**  | +1.49%      | 3.10%   | 1.43%   | **+35.8%**  |
| SHORT     | 24     | **21%**  | -1.36%      | 1.38%   | **3.36%** | **-32.7%** |

Shorts have avg MAE 2.4x their MFE — price consistently moves against short
entries because HYPE is in a structural uptrend. The existing DC Adaptive
strategy has no awareness of trend direction; its RegimeDetector only classifies
volatility/rate, not direction.

**Net effect**: Longs make +$35.8%, shorts lose -$32.7%. The strategy is net
break-even despite the long side being highly profitable — shorts wipe out all gains.

## Solution: Guard 4 — Trend Direction Filter

Add a fourth guard to the existing DC Adaptive stack that detects macro trend
direction from DC sensor event flow and blocks or reduces counter-trend entries.

```
DCTrendAdaptiveStrategy (extends DCAdaptiveStrategy)
├── LiveDCDetector          (reused — event detection)
├── TrailingRiskManager     (reused — SL/TP/trailing exits)
├── RegimeDetector          (reused — Guard 1: blocks choppy)
├── OvershootTracker        (reused — Guard 2: adapts TP)
├── LossStreakGuard         (reused — Guard 3: circuit breaker)
└── TrendDirectionFilter    (NEW  — Guard 4: blocks counter-trend)
```

**Key constraint**: ZERO changes to existing `dc_adaptive/` module. The trend
filter is a new strategy layer that imports and extends, never modifies.

## Architecture

### Module Structure

```
strategies/dc_trend_adaptive/
├── __init__.py
├── trend_direction_filter.py      # Guard 4 component
├── dc_trend_adaptive_strategy.py  # Extends DCAdaptiveStrategy
├── config.py                      # DCTrendAdaptiveConfig
├── backtest_compare.py            # Three-way comparison
├── trend_bridge.py                # Live trading bridge
└── DESIGN.md                      # This document

tests/strategies/dc_trend_adaptive/
├── test_trend_direction_filter.py  # 25 tests
├── test_dc_trend_adaptive_strategy.py  # 20 tests
└── test_backtest_compare.py        # 4 tests
```

### Signal Flow

```
Tick arrives (price, timestamp)
│
├─ 1. LiveDCDetector.process_tick(price, ts)
│     → DC events at both sensor (0.4%) and trade (2.0%) thresholds
│
├─ 2. Route sensor events to BOTH:
│     ├─ RegimeDetector.record_event(direction, ts)     [Guard 1]
│     └─ TrendDirectionFilter.record_event(dir, ts, tmv) [Guard 4]
│                                                    ▲
│                                         TMV = abs(end-start)/start
│                                         (larger moves weigh more)
│
├─ 3. Check close-on-trend-flip (Ch 6 JC1/JC2):
│     └─ If position opposes dominant trend → emit CLOSE signal
│
├─ 4. TrailingRiskManager.update(price, ts)
│     → check for SL/TP exits
│     → on exit: LossStreakGuard.record_trade(is_win, ts)
│
└─ 5. Entry Logic (PDCC event at trade threshold)
      ├─ Gate: long_only / short_only?        → skip if wrong direction
      ├─ Gate 1: RegimeDetector.should_trade?  → skip if choppy
      ├─ Gate 3: LossStreakGuard.should_trade?  → skip if cooling down
      ├─ Gate 4: TrendFilter.should_trade_direction? → skip/reduce if counter-trend
      ├─ Gate: Already in position same direction? → skip
      ├─ Gate: Cooldown elapsed? → skip (unless reversal)
      └─ Emit BUY/SELL signal (with optional size reduction)
```

### How Guards Interact

```
                          ┌──────────────────────────────────────┐
      Price tick ───────▶ │         DC Detection (all δ)        │
                          └──────────────┬───────────────────────┘
                                         │
                    ┌────────────────────┬┴────────────────────┐
                    │ Sensor events      │                     │ Trade events
                    │ (δ = 0.4%)         │                     │ (δ = 2.0%)
                    ▼                    ▼                     ▼
            ┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐
            │ Guard 1:     │   │ Guard 4:        │   │ Overshoot        │
            │ RegimeDetect │   │ TrendDirection  │   │ Tracker          │
            │              │   │ Filter          │   │ (adaptive TP)    │
            │ Rate+consist │   │ TMV-weighted    │   │                  │
            │ → choppy?    │   │ bias → trend?   │   │ Guard 2          │
            └──────┬───────┘   └────────┬────────┘   └──────────────────┘
                   │                    │
                   ▼                    ▼
            ┌─────────────────────────────────────────────────┐
            │              Entry Decision                     │
            │                                                 │
            │  if choppy ─────────────────────→ SKIP          │
            │  if loss_streak cooldown ───────→ SKIP          │
            │  if counter-trend + block ──────→ SKIP          │
            │  if counter-trend + reduce ─────→ ENTER (½ sz)  │
            │  else ──────────────────────────→ ENTER          │
            └─────────────────────────────────────────────────┘
                                 │
                                 ▼
            ┌─────────────────────────────────────────────────┐
            │         TrailingRiskManager                     │
            │  TP = adaptive (from Guard 2)                   │
            │  SL = normal (or tighter if counter-trend)      │
            │  Exit → record win/loss in Guard 3              │
            └─────────────────────────────────────────────────┘
```

## Theoretical Foundation (Chen & Tsang)

The TrendDirectionFilter adapts concepts from **Chapter 5** of "Detecting
Regime Change in Computational Finance" (Chen & Tsang, 2020).

### TMV-Weighted Bias (Ch 5)

The book's **TMV indicator** (Total price Movement Value) measures the total
absolute price displacement within a DC trend. We adapt it for directional
bias detection:

```
Simple bias:       B = sum(d_i) / N
TMV-weighted bias: B = sum(d_i × tmv_i) / sum(tmv_i)
```

Where `d_i` is +1 (up) or -1 (down), and `tmv_i = abs(end - start) / start`.

**Why TMV-weighted beats simple count**: In a HYPE uptrend, we might see 5 up
events and 4 down events (simple bias = 0.11, below threshold). But the up
events have TMV = 2-3% while down events have TMV = 0.5-1%. TMV-weighted
bias = ~0.65, correctly identifying the uptrend. The book showed TMV is the
primary discriminator between normal and abnormal regimes.

### B-Simple vs B-Strict Decision Rules (Ch 5)

The book defines two decision rules:

| Rule | Our Adaptation | When |
|------|---------------|------|
| **B-Simple** (highest probability wins) | `counter_trend_action="block"` | If bias > min_consistency, block counter-trend |
| **B-Strict** (require confidence > threshold₂) | `counter_trend_action="reduce"` | If bias > strict_threshold (0.8), reduce size. Below that but above min_consistency → allow full size |

### Close-on-Trend-Flip (Ch 6 JC1/JC2)

From Chapter 6 (Algorithmic Trading Based on Regime Change Tracking), the
**JC2 algorithm** closes all positions when the regime changes. We adapt this:

```
If close_on_trend_flip=True:
  On every sensor event, check if the current position opposes
  the newly dominant trend. If so, emit a protective CLOSE signal.
```

**Rationale**: JC2 showed comparable returns to JC1 with smaller drawdowns.
The primary value is drawdown reduction, not increased win rate.

### What We Didn't Use (And Why)

| Book Concept | Status | Reason |
|-------------|--------|--------|
| T indicator (Time for completion) | Skipped | Adds complexity without clear benefit for direction |
| Full Naive Bayes Classifier | Simplified | Would require labeled training data for "uptrend" vs "downtrend" |
| Unfinished trend tracking (§5.2.1) | Skipped | Sensor threshold (0.4%) already 5x finer than trade (2%), giving frequent completed events |
| JC1 strategy switching | Skipped | Would require different entry logic per regime |

## Guard 4: TrendDirectionFilter

### Core Algorithm

```python
class TrendDirectionFilter:
    def record_event(self, direction: int, timestamp: float, tmv: float = 1.0):
        """Record sensor DC event with TMV weight."""

    def bias(self, timestamp: float) -> float:
        """Signed directional bias in [-1, +1]."""

    def dominant_trend(self, timestamp: float) -> Optional[str]:
        """'up', 'down', or None (no clear trend)."""

    def should_trade_direction(self, side: str, ts: float) -> (bool, float):
        """(allowed, size_multiplier) — core decision method."""
```

### Data Source

Fed from **sensor** DC events (same as RegimeDetector) — they're 5x more
frequent than trade events, giving faster trend detection. Uses a separate,
longer lookback window (15 min vs 10 min) because trend direction needs more
stability than volatility classification.

### Key Differences from RegimeDetector

| Aspect | RegimeDetector | TrendDirectionFilter |
|--------|---------------|---------------------|
| Uses | `abs(bias)` (magnitude only) | **Signed bias** (direction matters) |
| Purpose | Detect volatility regime | Detect trend direction |
| Weights | Equal per event | TMV-weighted (larger moves count more) |
| Confidence | Single threshold | Two tiers (B-Simple / B-Strict) |
| Lookback | 600s (10 min) | 900s (15 min, more stable) |
| Output | choppy/trending/quiet | up/down/None |

## Additional Features

### Close-on-Trend-Flip

When enabled (`close_on_trend_flip=True`, default), the strategy checks
after each sensor event update whether the current position has become
counter-trend. If so, it generates a protective CLOSE signal immediately.

```
Holding SHORT → trend flips to UP → emit CLOSE("trend_flip_protective")
Holding LONG  → trend flips to DOWN → emit CLOSE("trend_flip_protective")
```

### Asymmetric Stop-Loss

If `counter_trend_sl_pct` is set, counter-trend trades get a tighter SL
than trend-aligned trades. This limits damage on trades that go through
the filter with `action="reduce"`.

### Nuclear Options

- `long_only=True`: Skip ALL short signals (regardless of trend filter)
- `short_only=True`: Skip ALL long signals
- These bypass the trend filter entirely — use when the directional
  thesis is absolute.

## Parameter Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trend_lookback_seconds` | 900 | Trend detection window (15 min) |
| `trend_min_events` | 5 | Min sensor events before trend active |
| `trend_min_consistency` | 0.6 | Directional bias threshold (0-1) |
| `trend_bias_mode` | `tmv_weighted` | `simple` or `tmv_weighted` |
| `trend_strict_threshold` | 0.8 | B-Strict confidence for "reduce" mode |
| `counter_trend_action` | `block` | `block` / `reduce` / `allow` |
| `counter_trend_size_fraction` | 0.5 | Size multiplier when action=reduce |
| `counter_trend_sl_pct` | None | Tighter SL for counter-trend (None = normal) |
| `close_on_trend_flip` | True | Close position if trend reverses against it |
| `long_only` | False | Skip all short signals |
| `short_only` | False | Skip all long signals |

All DC Adaptive parameters (Guards 1-3) are inherited unchanged.

## Usage

### Live Trading

```bash
# Launch on sub-account with recommended HYPE parameters
screen -dmS sub-trend bash -c 'uv run --package hyperliquid-trading-bot python \
  -m strategies.dc_trend_adaptive.trend_bridge \
  --symbol HYPE --threshold 0.02 \
  --vault-address 0x969b... \
  --compound --compound-fraction 0.9 \
  --sensor-threshold 0.004 --position-size 10 \
  --sl-pct 0.018 --tp-pct 0.008 \
  --trail-pct 0.5 --min-profit-to-trail-pct 0.008 \
  --backstop-sl-pct 0.05 --backstop-tp-pct 0.05 --leverage 10 \
  --trend-lookback-seconds 900 --trend-min-events 5 --trend-min-consistency 0.6 \
  --counter-trend-action block \
  --lookback-seconds 600 --choppy-rate-threshold 4.0 --trending-consistency 0.6 \
  --tp-fraction 0.4 --min-tp-pct 0.005 \
  --os-window-size 20 --os-min-samples 5 \
  --max-consecutive-losses 3 --base-cooldown-seconds 300 \
  --telemetry --yes 2>&1 | tee /tmp/sub_trend.log'

# Observe only (no trades)
uv run --package hyperliquid-trading-bot python \
  -m strategies.dc_trend_adaptive.trend_bridge \
  --symbol HYPE --observe-only --duration 60
```

### Backtesting

```bash
# Three-way comparison: Overshoot vs Adaptive vs Trend-Adaptive
make backtest-trend-adaptive symbol=HYPE days=14

# JSON output for analysis
make backtest-trend-adaptive-json symbol=HYPE days=14
```

### Testing

```bash
make test-trend-adaptive    # 49 tests across 3 test files
```

### Status Logging

The bridge logs trend info every 100 ticks:

```
Tick #500 | price=31.20 | trend=UP(0.72) regime=trending tp=0.42% |
  skip_trend=5 skip_chop=2 skip_loss=0 | signals=8 trades=3 | elapsed=1200s
```

### Telemetry

MOMENTUM_UPDATE events include trend filter fields:

```json
{
  "regime": "trending",
  "event_rate": 2.1,
  "adaptive_tp": 0.0042,
  "dominant_trend": "up",
  "trend_bias": 0.72,
  "trend_bias_mode": "tmv_weighted",
  "trend_avg_tmv_up": 0.0185,
  "trend_avg_tmv_down": 0.0062
}
```

## Design Decisions

### Why sensor events feed the trend filter (not trade events)

Trade-threshold events (2.0%) are too infrequent for timely trend detection —
HYPE produces ~4/day at δ=2.0%. Sensor events at δ=0.4% fire ~20x more
frequently, giving the trend filter enough data points to detect direction
shifts within minutes rather than hours.

### Why TMV-weighted is the default (not simple count)

In sideways markets with occasional strong directional bursts, simple event
counts can show balanced (no trend) while TMV-weighted correctly identifies
that one direction's moves are consistently larger. This matches the book's
finding that TMV is the primary regime discriminator.

### Why the default action is "block" (not "reduce")

The MFE data shows counter-trend shorts have avg MAE of 3.36% vs MFE of 1.38%.
Reducing size to 50% still loses — the trades are structurally wrong, not just
oversized. Blocking entirely is the correct default. "Reduce" mode is available
for markets with less extreme directional asymmetry.

### Why close_on_trend_flip defaults to True

From Ch 6: JC2 (close on regime change) showed comparable returns with
significantly smaller drawdowns. When holding a SHORT in a newly-detected
uptrend, the expected value of staying in the trade is negative. Early exit
limits damage from the most common loss scenario.

### Why the lookback is 900s (not matching RegimeDetector's 600s)

Trend direction should be more stable than volatility classification:
- Volatility can spike and drop within minutes → short lookback (600s)
- Trend direction changes less frequently → longer lookback (900s)
  prevents false trend flips from brief counter-moves

## Test Coverage

49 tests across 3 test files:

- `test_trend_direction_filter.py` — 25 tests: bias modes (simple + TMV),
  B-Simple/B-Strict rules, lookback purge, consistency boundaries, counter-
  trend detection, status/telemetry output
- `test_dc_trend_adaptive_strategy.py` — 20 tests: trend blocking, aligned
  entry, reduce mode, nuclear modes, close-on-trend-flip (3 scenarios),
  asymmetric SL, reversal blocking, status/lifecycle
- `test_backtest_compare.py` — 4 tests: three-way comparison, directional
  breakdown, uptrend counter-trend reduction
