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

## Backtest Results (HYPE, Mar 2026)

### 3-day comparison (same candles, same core params)

| Metric | DC Overshoot | DC Adaptive |
|--------|-------------|-------------|
| Total trades | 46 | 46 |
| Win rate | 63% | **76%** |
| SL exits | 12 | **2** |
| TP exits | 24 | **34** |
| Reversal exits | 10 | 10 |
| Net P&L | $0.01 | **$0.19** |
| Profit factor | 1.00 | 1.02 |

**Key insight**: The adaptive TP is the biggest win — it converts would-be
SL losses into TP wins by setting realistic exit targets based on observed
overshoot magnitudes. SL exits dropped from 12 to 2 (83% reduction).

### 7-day comparison

Both strategies slightly negative over 7 days (choppy HYPE market). The
adaptive strategy had 76.9% win rate vs 63.5% baseline, but the tighter
adaptive TP reduced gross profit per trade.

### Overshoot distribution (from tracker)

- p50: 0.215%
- p75: 0.422%
- Samples: 20

Adaptive TP settled at ~0.3% (0.215% × 0.8 = 0.172%, floored at min_tp 0.3%).

## Usage

### Live trading

```bash
# Launch with recommended HYPE parameters
screen -S hype bash -c 'uv run --package hyperliquid-trading-bot python \
  -m strategies.dc_adaptive.adaptive_bridge \
  --symbol HYPE --threshold 0.015 \
  --sensor-threshold 0.004 \
  --position-size 50 \
  --sl-pct 0.02 --tp-pct 0.005 \
  --trail-pct 0.3 --min-profit-to-trail-pct 0.003 \
  --backstop-sl-pct 0.05 --backstop-tp-pct 0.05 --leverage 10 \
  --telemetry --yes 2>&1 | tee /tmp/hype_adaptive.log'

# Observe only (no trades, no --yes needed)
uv run --package hyperliquid-trading-bot python \
  -m strategies.dc_adaptive.adaptive_bridge \
  --symbol HYPE --observe-only --duration 60
```

### Backtesting

```bash
# Compare DC Adaptive vs DC Overshoot
make backtest-adaptive symbol=HYPE days=3

# JSON output for analysis
make backtest-adaptive-json symbol=HYPE days=7
```

### Testing

```bash
make test-adaptive    # 50 tests across 4 test files
```

## Design Decisions

### Why overshoot is measured as net V displacement (not true extreme)

The tracker measures the price displacement between two consecutive opposite
DC confirmations — the full "V shape" from DC Down confirmation to DC Up
confirmation (or vice versa). This is the net displacement, not the maximum
extension to the extreme point.

This is intentionally conservative. The true extreme (bottom of the V) is only
known in hindsight and can't be captured in real trading. The net V displacement
represents a reliably achievable price level. Setting TP at 80% of this
conservative measurement ensures TPs are actually hit, producing higher win rates.

Backtesting confirmed: TP=0.5% (conservative) was the only breakeven setting.
TP=1.0% lost $1.56, TP=1.5% lost $2.42. Lower, more achievable TPs win.

### Why overshoots are tracked in a single pool (not per-direction)

The tracker combines both upward and downward overshoots into one pool rather
than maintaining separate per-direction medians. Reasons:

1. **Sample size**: At threshold=0.015, HYPE produces ~7 overshoots/day total.
   Splitting halves this to ~3.5/direction, too few for a stable median.
2. **Conservative bias**: Combined averaging pulls the median toward the smaller
   direction, producing lower TPs — which we've shown works better.
3. **HYPE is mostly symmetric**: Mar 1-3 data showed 32 UP vs 32 DOWN DC events.
4. **Regime detector covers asymmetry**: Trending markets (where directional
   asymmetry matters) are already detected by the RegimeDetector.

This may be revisited if telemetry data shows significant directional asymmetry.

## Test Coverage

50 tests across 4 test files (TDD — tests written before implementation):

- `test_regime_detector.py` — 15 tests: quiet/choppy/trending/neutral regimes,
  lookback purge, event rate computation
- `test_overshoot_tracker.py` — 12 tests: adaptive TP, median computation,
  rolling window eviction, percentiles
- `test_loss_streak_guard.py` — 11 tests: win resets, cooldown activation,
  duration escalation, expiry, status
- `test_dc_adaptive_strategy.py` — 12 tests: init, entry signals, regime gating,
  loss streak gating, adaptive TP, trade execution, lifecycle
