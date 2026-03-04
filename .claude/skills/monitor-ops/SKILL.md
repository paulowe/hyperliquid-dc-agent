---
description: Monitor live trading operations — review trades, analyze performance, fix bugs, restart bots, and propose improvements autonomously
user_invocable: true
argument: optional symbol (default HYPE) and duration (e.g., "HYPE 8h", "SOL 4h")
---

# Monitor Operations

Autonomous trading operations monitor. Reviews live trades, checks account health, analyzes strategy performance, identifies and fixes bugs, restarts bots when needed, documents findings, and commits changes. Designed to run while the user is away.

## How to Use

- `/monitor-ops` — monitor HYPE for ~8 hours (default)
- `/monitor-ops HYPE 4h` — monitor HYPE for 4 hours
- `/monitor-ops SOL 2h` — monitor SOL for 2 hours

## Core Principles

1. **Act independently** — make intelligent decisions without asking
2. **Fix bugs when found** — run tests, verify, restart bots
3. **Don't break things** — never force-close profitable positions
4. **Document everything** — update DESIGN.md, commit, push
5. **Report in EST timezone** — the user thinks in EST (UTC - 5)

## Workflow

### Phase 1: Initial Assessment (first 10 minutes)

Run these in parallel to understand current state:

#### 1a. Account Status

```bash
uv run --package hyperliquid-trading-bot python3 << 'PYEOF'
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path('trading-agent/.env'), override=True)
from hyperliquid.info import Info

info = Info('https://api.hyperliquid.xyz', skip_ws=True)
wallet = os.environ.get('MAINNET_ACCOUNT_ADDRESS')

state = info.user_state(wallet)
acct_val = float(state["marginSummary"]["accountValue"])
margin_used = float(state["marginSummary"]["totalMarginUsed"])
print(f"Account: ${acct_val:.2f} | Margin: ${margin_used:.2f}")

for pos in state.get("assetPositions", []):
    p = pos["position"]
    sz = float(p["szi"])
    if sz == 0:
        continue
    side = "LONG" if sz > 0 else "SHORT"
    entry = float(p["entryPx"])
    upnl = float(p["unrealizedPnl"])
    print(f"  {p['coin']} {side} sz={abs(sz)} entry={entry:.4f} uPnL=${upnl:.2f}")

if not any(float(p['position']['szi']) != 0 for p in state.get('assetPositions', [])):
    print("  No open positions")
PYEOF
```

#### 1b. Running Bot Status

```bash
screen -ls 2>&1
```

Check bot logs for each running screen:

```bash
# DC Adaptive (primary)
tail -20 /tmp/hype_adaptive.log 2>/dev/null
# DC Overshoot (legacy)
tail -20 /tmp/hype-bot.log 2>/dev/null
# Multi-scale
tail -20 /tmp/sol_ms_bot.log 2>/dev/null
```

#### 1c. Recent Trade History

```bash
uv run --package hyperliquid-trading-bot python -m trade_review.cli \
  --symbol <SYMBOL> --hours 24
```

#### 1d. Record Starting Account Value

Note the account value at the start of monitoring. All subsequent reports should reference this baseline.

### Phase 2: Analyze Performance

#### Trade-Level Analysis

From the trade review output, compute:

- **Win rate** (net and gross)
- **Average win vs average loss** — is the strategy taking profits correctly?
- **Exit type distribution** — SL/TP/trailing/reversal breakdown
- **Fee impact** — fees as % of gross profit (>50% = threshold too small)
- **Consecutive losses** — max losing streak
- **Hold time distribution** — are trades held too long or too short?

#### Bot Health Checks

From the bot log, verify:

- **Ticks flowing** — bot is receiving WebSocket data
- **Reconnect count** — stable connection (rc=0 or rc=1 is fine, rc>5 is concerning)
- **Regime classification** — is the market trending/neutral/choppy/quiet?
- **Adaptive TP** — has it adapted from default? What value?
- **Guard activity** — any choppy skips or loss guard activations?
- **No errors** — grep for ERROR, Exception, Traceback in logs

```bash
grep -i "error\|exception\|traceback" /tmp/hype_adaptive.log 2>/dev/null | tail -10
```

#### Identify Issues

Check for these problems (ranked by severity):

1. **Bot not running** — screen session dead, no recent ticks
2. **Stuck position** — position open for >2 hours with no exit signal
3. **Excessive losses** — account down >5% from start of monitoring
4. **Adaptive TP at floor** — adaptive_tp=0.300% means tracker contaminated
5. **High reconnect rate** — rc>5 suggests network instability
6. **Fee dominance** — fees eating >50% of gross profit

### Phase 3: Take Action (if needed)

#### If Bot Is Down

Restart it:

```bash
screen -dmS hype bash -c 'uv run --package hyperliquid-trading-bot python \
  -m strategies.dc_adaptive.adaptive_bridge \
  --symbol HYPE --threshold 0.015 \
  --sensor-threshold 0.004 \
  --position-size 50 \
  --sl-pct 0.02 --tp-pct 0.005 \
  --trail-pct 0.3 --min-profit-to-trail-pct 0.003 \
  --backstop-sl-pct 0.05 --backstop-tp-pct 0.05 \
  --leverage 10 \
  --lookback-seconds 600 --choppy-rate-threshold 4.0 --trending-consistency 0.6 \
  --tp-fraction 0.4 --min-tp-pct 0.003 \
  --os-window-size 20 --os-min-samples 5 \
  --max-consecutive-losses 3 --base-cooldown-seconds 300 \
  --telemetry --yes 2>&1 | tee /tmp/hype_adaptive.log'
```

Wait 30 seconds then verify ticks are flowing:

```bash
sleep 30 && tail -5 /tmp/hype_adaptive.log
```

#### If Bug Found in Strategy Code

1. Read the relevant source files to understand the issue
2. Write a fix
3. Run tests: `uv run --package hyperliquid-trading-bot pytest trading-agent/tests/strategies/dc_adaptive/ -v`
4. If tests pass, stop the bot (`screen -S hype -X stuff $'\003'`), wait for clean shutdown
5. Restart with fixed code
6. Commit and push the fix

#### If Position Is Stuck or Account At Risk

- **DO NOT** force-close positions unless account is down >10% from baseline
- If a position is profitable and seems stuck, let the trailing stop do its job
- If a position is deeply underwater (>-2%), check if SL should have triggered — may be a bug
- Only manually close via the exchange as a last resort

#### If Profitable Position and Market Reversing

- Trust the trailing stop mechanism — it locks in profits automatically
- The trailing TP at 0.5% and trailing SL at 30% lock-in are calibrated
- Manual intervention is only needed if the bot appears to not be processing ticks

### Phase 4: Monitoring Loop

After initial assessment, enter a monitoring loop. Check every 30-60 minutes:

```bash
# Quick status (run every 30 min)
grep -E "(DC Event|DC Adaptive ENTRY|DC Adaptive EXIT|Close filled)" /tmp/hype_adaptive.log 2>/dev/null | tail -10
tail -3 /tmp/hype_adaptive.log 2>/dev/null
```

```bash
# Account check (run every 1-2 hours)
uv run --package hyperliquid-trading-bot python3 << 'PYEOF'
from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv(Path('trading-agent/.env'), override=True)
from hyperliquid.info import Info
info = Info('https://api.hyperliquid.xyz', skip_ws=True)
wallet = os.environ.get('MAINNET_ACCOUNT_ADDRESS')
state = info.user_state(wallet)
av = float(state["marginSummary"]["accountValue"])
print(f"Account: ${av:.2f}")
positions = [p for p in state.get("assetPositions", []) if float(p["position"]["szi"]) != 0]
for p in positions:
    pos = p["position"]
    side = "LONG" if float(pos["szi"]) > 0 else "SHORT"
    print(f"  {pos['coin']} {side} sz={pos['szi']} entry={pos['entryPx']} pnl={pos['unrealizedPnl']}")
if not positions:
    print("  Flat")
PYEOF
```

For each new trade observed, log it in a running tally:

| # | Time (EST) | Side | Entry | Exit | PnL | Exit Reason | Account |
|---|------------|------|-------|------|-----|-------------|---------|

**Timezone conversion**: All times from logs are UTC. Convert to EST by subtracting 5 hours.

### Phase 5: Improvement Analysis

After observing several trades (or at the end of the monitoring session), analyze patterns:

#### What to Look For

1. **Reversal losses** — are reversals consistently losing? Consider proposing to skip reversals and just close
2. **SL exits** — if SL exits > 30% of total, SL may be too tight relative to threshold
3. **TP hit rate** — if TP exits are rare, adaptive TP may need recalibration
4. **Hold times** — very short holds (<1 min) suggest noise; very long holds (>1 hour) suggest TP too ambitious
5. **Regime detector** — is it actually blocking trades during choppy periods? If skip_chop=0 for hours, the sensor threshold may need tuning
6. **Directional bias** — are longs or shorts systematically better/worse?

#### Run Backtest Comparison

```bash
uv run --package hyperliquid-trading-bot python -m strategies.dc_adaptive.backtest_compare \
  --symbol <SYMBOL> --days 3
```

Compare live performance against backtest expectations. If they diverge significantly, investigate why.

### Phase 6: Document and Commit

Before the monitoring session ends:

1. **Update DESIGN.md** if any new findings or calibration changes were made
2. **Update memory** if patterns were discovered that should persist across sessions
3. **Commit all changes** with descriptive messages
4. **Push to remote**

### Phase 7: Final Report

Present a complete monitoring report to the user (in EST timezone):

```
## Monitoring Report — <SYMBOL> (<date>)

### Session: <start_time EST> - <end_time EST> (<duration>)

### Account
- Start: $X.XX → End: $X.XX (net +/- $X.XX)

### Trades (N total)
| # | Time (EST) | Side | Entry | Exit | Net P&L | Exit Reason |
|---|------------|------|-------|------|---------|-------------|
| ... |

### Win Rate: X/Y (Z%)
- TP exits: N | Trailing SL exits: N | SL exits: N | Reversals: N

### Bot Health
- Uptime: X hours | Ticks: N | Reconnects: N
- Regime: mostly quiet/neutral/choppy/trending
- Adaptive TP: X.XX% (adapted/default)
- Guards: N choppy skips, N loss guard skips

### Issues Found & Fixed
- (list any bugs found, fixes applied, code changes)

### Observations & Proposals
- (patterns noticed, improvement ideas for next session)

### Commits
- `abc1234` — Description
```

## Decision Framework

### When to restart the bot
- Bot screen is dead (no screen session)
- Bot stopped receiving ticks (last tick >5 min old)
- Code fix was applied that affects live behavior
- Bot is stuck in an error loop

### When to NOT restart
- Bot has an open profitable position (let trailing stop handle it)
- Market is volatile and a restart would miss events
- No actual issue found (don't restart "just because")

### When to manually close a position
- Account down >10% from start AND position is still losing
- Bot appears to have stopped processing (no new ticks) AND position is open
- Backstop SL/TP on exchange will catch emergencies — manual close is rarely needed

### When to commit changes
- After any code fix
- After updating documentation
- After discovering and documenting a new insight
- Before ending the monitoring session

## Parameter Reference (DC Adaptive, HYPE)

| Parameter | Current Value | Notes |
|-----------|--------------|-------|
| threshold | 0.015 | Trade-level DC threshold |
| sensor-threshold | 0.004 | Regime detection |
| sl-pct | 0.02 | 1.33x threshold |
| tp-pct | 0.005 | Default before adaptive |
| tp-fraction | 0.4 | 40% of median overshoot |
| trail-pct | 0.3 | Lock in 70% of gains |
| min-profit-to-trail | 0.003 | Min before trailing |
| backstop-sl/tp | 0.05 | Emergency safety net |
| leverage | 10 | Cross margin |
| position-size | 50 | USD per trade |

## Useful Log Patterns

```bash
# All DC events and trades
grep -E "(DC Event|DC Adaptive ENTRY|DC Adaptive EXIT|Close filled)" /tmp/hype_adaptive.log

# Errors only
grep -i "error\|exception\|traceback" /tmp/hype_adaptive.log

# Regime changes
grep "regime=" /tmp/hype_adaptive.log | grep -v "regime=quiet" | tail -20

# Guard activations
grep "SKIPPED" /tmp/hype_adaptive.log

# Reconnections
grep "Reconnect\|resubscribed" /tmp/hype_adaptive.log

# Trade count summary
grep -c "DC Adaptive ENTRY" /tmp/hype_adaptive.log
grep -c "DC Adaptive EXIT" /tmp/hype_adaptive.log
grep -c "Close filled" /tmp/hype_adaptive.log
```
