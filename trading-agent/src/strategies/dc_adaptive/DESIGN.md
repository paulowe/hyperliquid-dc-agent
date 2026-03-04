# DC Adaptive Strategy — Design Document

## Problem Statement

The current DC Overshoot strategy lost $12.64 over 3 days on HYPE (Mar 1-4, 2026).
Root cause analysis reveals three critical weaknesses:

1. **Whipsaw vulnerability**: 26 reversal closes, 100% losers, -$22.66.
   When SL equals the threshold, any choppy market creates a guaranteed loss
   machine as the bot flip-flops between long and short.

2. **Fixed take-profit**: TP=1.5% was above the 75th percentile of actual
   overshoots (median overshoot was only 0.8%). Most trades never reached TP
   and exited via trailing SL instead.

3. **No loss-streak protection**: The bot kept trading during extended choppy
   periods, accumulating losses trade after trade with no circuit breaker.

The backtest sweep confirmed: **0% of parameter combinations were profitable**
during this 3-day period. No static config can handle regime changes.

## Solution: Three Adaptive Guards

The DC Adaptive strategy keeps the proven DC overshoot entry/exit mechanics
but wraps them with three guards that prevent trading in unfavorable conditions.

### Architecture

```
DCAdaptiveStrategy (extends TradingStrategy)
├── LiveDCDetector          (reused — event detection)
├── TrailingRiskManager     (reused — SL/TP/trailing exits)
├── RegimeDetector          (NEW — blocks trades in choppy markets)
├── OversshootTracker       (NEW — adapts TP to recent market behavior)
└── LossStreakGuard         (NEW — circuit breaker after consecutive losses)
```

### Guard 1: RegimeDetector

**Purpose**: Detect choppy markets and block entries.

**How it works**:
- Observes DC events at a fast "sensor" threshold (e.g., 0.004)
- Tracks event rate (events/minute) and directional consistency over a
  rolling window (default 10 minutes)
- Classifies market as: `trending`, `choppy`, or `quiet`

**Classification logic**:
- `choppy`: event_rate > threshold AND direction_consistency < threshold
  (lots of events, but no clear direction — the whipsaw regime)
- `trending`: direction_consistency >= threshold
  (events mostly agree on direction)
- `quiet`: fewer than 3 events in the lookback window

**Impact**: When regime is `choppy`, the strategy skips new entries entirely.
This would have prevented the 26 reversal losses from Mar 1-4.

**Parameters**:
- `sensor_threshold`: DC threshold for regime sensing (default: 0.004)
- `lookback_seconds`: Rolling window for event analysis (default: 600)
- `choppy_rate_threshold`: Events/min above which market is considered
  fast-moving (default: 4.0)
- `trending_consistency_threshold`: Minimum directional consistency to be
  considered trending (default: 0.6, range 0-1)

### Guard 2: OversshootTracker

**Purpose**: Adapt TP to recent actual overshoot magnitudes.

**How it works**:
- After each completed DC lifecycle (DC event followed by opposite DC event),
  measures the actual overshoot magnitude
- Maintains a rolling window of the last N overshoots (default: 20)
- Computes adaptive TP as a fraction of the median overshoot

**Adaptive TP formula**:
```
adaptive_tp = max(p50(recent_overshoots) * tp_fraction, min_tp)
```
Where:
- `tp_fraction` = 0.8 (target 80% of median, so >50% of trades reach it)
- `min_tp` = 0.003 (floor at 0.3% to always cover fees)

**How overshoots are measured**:
- On each DC event, check if there's a previous DC event in the opposite
  direction. The overshoot is the price extension beyond the previous DC
  confirmation before this reversal.
- Example: Price drops 1.5% (DC down confirmed at $99), then continues to
  $98, then reverses 1.5% up (DC up confirmed at $99.47). The overshoot
  from the down DC was ($99 - $98) / $99 = 1.01%.

**Impact**: On HYPE (Mar 1-4), median overshoot was 0.8%. Adaptive TP would
have been ~0.64% vs fixed 1.5% — significantly more reachable.

**Parameters**:
- `window_size`: Number of recent overshoots to track (default: 20)
- `min_samples`: Minimum overshoots before adapting (default: 5, use
  default_tp_pct until then)
- `tp_fraction`: Fraction of median overshoot to use as TP (default: 0.8)
- `min_tp_pct`: Floor for adaptive TP (default: 0.003)
- `default_tp_pct`: TP to use before enough samples (default: 0.005)

### Guard 3: LossStreakGuard

**Purpose**: Circuit breaker after consecutive losses.

**How it works**:
- Tracks consecutive losing trades
- After `max_consecutive` losses (default: 3), enters cooldown
- Cooldown duration = `base_cooldown * consecutive_losses` (escalating)
- Resets to zero after any winning trade

**Cooldown formula**:
```
cooldown_seconds = base_cooldown * consecutive_losses
```
Where `base_cooldown` = 300 seconds (5 minutes).

After 3 losses: 15 min cooldown. After 4 losses: 20 min. After 5: 25 min.

**Impact**: Prevents the pattern seen on Mar 1 where the bot had 5
consecutive losses totaling -$2.11, and later a 4-loss streak of -$7.76.

**Parameters**:
- `max_consecutive_losses`: Threshold to trigger cooldown (default: 3)
- `base_cooldown_seconds`: Cooldown per loss count (default: 300)

## Signal Flow

```
Tick arrives (price, timestamp)
│
├─ 1. LiveDCDetector.process_tick(price, ts)
│     → DC events (PDCC_Down, PDCC2_UP, OSV_*)
│
├─ 2. RegimeDetector.record_event(direction, ts)  [for sensor events]
│     RegimeDetector.classify(ts) → regime
│
├─ 3. OversshootTracker.record_overshoot(magnitude) [on DC events]
│     OversshootTracker.adaptive_tp() → current TP target
│
├─ 4. TrailingRiskManager.update(price, ts)
│     → check for SL/TP exits
│     → on exit: LossStreakGuard.record_trade(is_win, ts)
│
└─ 5. Entry Logic (PDCC event at trade threshold)
      ├─ Gate: RegimeDetector.should_trade(ts)? → skip if choppy
      ├─ Gate: LossStreakGuard.should_trade(ts)? → skip if cooling down
      ├─ Gate: Already in position same direction? → skip
      ├─ Gate: Cooldown elapsed? → skip (unless reversal)
      ├─ Update TrailingRiskManager TP to adaptive_tp
      └─ Emit BUY/SELL signal
```

## Integration with Existing Infrastructure

### Backtesting
- Create `AdaptiveBacktestEngine` that extends `BacktestEngine`
- Feeds candles through `DCAdaptiveStrategy` instead of `DCOvershootStrategy`
- Same trade recording, fee accounting, metrics computation
- Add sweep over new parameters (sensor_threshold, tp_fraction, etc.)

### Live Trading
- Create `adaptive_bridge.py` — same WebSocket/execution pattern as
  `live_bridge.py`
- Add CLI flags for new parameters
- Reuse backstop order logic, telemetry, reconciliation

### Telemetry
- Reuse existing telemetry collector
- Add new event types: REGIME_CHANGE, GUARD_SKIP, ADAPTIVE_TP_UPDATE
- Allows post-hoc analysis of when guards activated

## New Files

```
trading-agent/src/strategies/dc_adaptive/
├── __init__.py
├── config.py               # DCAdaptiveConfig dataclass
├── dc_adaptive_strategy.py # Main strategy class
├── regime_detector.py       # RegimeDetector component
├── overshoot_tracker.py     # OversshootTracker component
├── loss_streak_guard.py     # LossStreakGuard component
└── adaptive_bridge.py       # Live trading bridge

trading-agent/tests/strategies/dc_adaptive/
├── __init__.py
├── test_regime_detector.py
├── test_overshoot_tracker.py
├── test_loss_streak_guard.py
├── test_dc_adaptive_strategy.py
└── test_adaptive_bridge.py
```

## Parameter Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| symbol | HYPE | Asset to trade |
| threshold | 0.015 | DC threshold for trade signals |
| sensor_threshold | 0.004 | DC threshold for regime sensing |
| sl_pct | 0.02 | Initial stop loss (wider than threshold) |
| default_tp_pct | 0.005 | TP before adaptive kicks in |
| tp_fraction | 0.8 | Fraction of median overshoot for TP |
| min_tp_pct | 0.003 | Floor for adaptive TP |
| trail_pct | 0.3 | Trailing SL lock-in fraction |
| min_profit_to_trail_pct | 0.003 | Min profit before trailing |
| lookback_seconds | 600 | Regime detection window |
| choppy_rate_threshold | 4.0 | Events/min for "choppy" |
| trending_consistency | 0.6 | Direction agreement for "trending" |
| max_consecutive_losses | 3 | Losses before cooldown |
| base_cooldown_seconds | 300 | Cooldown duration multiplier |
| os_window_size | 20 | Overshoots to track for adaptive TP |
| backstop_sl_pct | 0.05 | Exchange-level emergency SL |
| backstop_tp_pct | 0.05 | Exchange-level emergency TP |
| leverage | 10 | Position leverage |
| position_size_usd | 50 | Position size in USD |

## Expected Improvements

Based on Mar 1-4 HYPE data:
- **Regime filter** would have blocked ~15 of the 26 reversal-close losses
  (those during high-rate choppy periods), saving ~$13
- **Adaptive TP** at ~0.64% instead of 1.5% would have captured more
  take-profit exits (from 29 to ~35), adding ~$2
- **Loss streak guard** would have prevented ~5 trades during the two
  major loss streaks, saving ~$3

Combined estimated improvement: $18+ savings on a $12.64 loss = net positive.

## Test Strategy

TDD approach — write tests before implementation:

1. **Unit tests** for each guard component (regime, overshoot, loss streak)
2. **Integration test** for full strategy signal flow
3. **Backtest comparison** against baseline DC Overshoot on same data
4. **Edge case tests** (empty data, single tick, no DC events, etc.)
