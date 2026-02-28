---
description: Review live trade P&L from Hyperliquid and suggest strategy parameter adjustments
user_invocable: true
argument: symbol and optional time range (e.g., HYPE, or "HYPE --hours 48", or "SOL --days 7")
---

# Review Live Trades

Analyze real trade history from Hyperliquid to evaluate strategy performance and suggest parameter improvements. Fetches actual fill data from the exchange, runs a backtest parameter sweep over the same period, and produces concrete recommendations to make the strategy profitable.

## How to Use

The user invokes this with `/review-trades <SYMBOL>` or `/review-trades <SYMBOL> --hours 48` or `/review-trades <SYMBOL> --days 7`.

## Workflow

### Step 1: Fetch Live Trade Data (JSON)

For delegation setups, fills are on the main wallet (not the API wallet). `MAINNET_ACCOUNT_ADDRESS` is already configured in `.env`.

```bash
uv run --package hyperliquid-trading-bot python -m trade_review.cli \
  --symbol $ARGUMENTS --json
```

If the user specified a time range, it will be included in `$ARGUMENTS`. Default is 24 hours.

If no trades are returned, try expanding the time range or adding `--wallet <address>`:

```bash
uv run --package hyperliquid-trading-bot python -m trade_review.cli \
  --symbol $ARGUMENTS --days 7 --json
```

### Step 2: Run Backtest Sweep (Same Period)

**Always** run a parameter sweep over the same time period to identify what *would* have been profitable. Use the same `--days` as Step 1.

```bash
uv run --package hyperliquid-trading-bot python -m backtesting.cli \
  --symbol <SYMBOL> --days <same_period> --sweep --json
```

Then analyze the sweep results in Python to extract:

```bash
uv run --package hyperliquid-trading-bot python -c "
import json, sys
from collections import defaultdict

data = json.load(sys.stdin)
results = data['results']
by_threshold = defaultdict(list)
for r in results:
    by_threshold[r['threshold']].append(r)

for t in sorted(by_threshold.keys()):
    runs = by_threshold[t]
    profitable = sum(1 for r in runs if r['net_pnl_usd'] > 0)
    avg_pnl = sum(r['net_pnl_usd'] for r in runs) / len(runs)
    avg_trades = sum(r['total_trades'] for r in runs) / len(runs)
    avg_fees = sum(r['total_fees_usd'] for r in runs) / len(runs)
    print(f'threshold={t}: {profitable}/{len(runs)} profitable ({profitable/len(runs)*100:.0f}%)')
    print(f'  avg net P&L: \${avg_pnl:.2f}, avg trades: {avg_trades:.0f}, avg fees: \${avg_fees:.2f}')
" < <(uv run --package hyperliquid-trading-bot python -m backtesting.cli \
  --symbol <SYMBOL> --days <same_period> --sweep --json 2>/dev/null)
```

### Step 3: Diagnose — Identify Root Causes

Parse the JSON output from Step 1 and compare against the sweep in Step 2. Run a diagnostic analysis to identify the primary issue(s):

#### Diagnostic Checklist

Run these checks in order — they are ranked by severity:

1. **Fee dominance check**: Is `fee_pct_of_gross > 100%`?
   - Fees exceed gross profit → the strategy is guaranteed to lose money
   - Root cause: threshold too small — signals are not strong enough to overcome fees
   - Check sweep: what % of configs are profitable at the user's threshold vs higher thresholds?

2. **Fee erosion check**: Is `fee_pct_of_gross > 50%`? Are `trades_eaten_by_fees > 20%` of total?
   - Many individually-profitable trades become net losers after fees
   - Root cause: threshold or position size too small relative to fee structure
   - The round-trip fee is 0.07% (0.035% per side). Threshold must be significantly larger.

3. **Win rate gap check**: Is `win_rate_gross - win_rate_net > 15%`?
   - Strategy has good directional accuracy but fees destroy many winners
   - This confirms fee problem — increase threshold to trade on stronger signals

4. **SL dominance check**: Are SL exits > 60% of total exits?
   - Stop loss is too tight — getting stopped out before signals develop
   - SL should be >= threshold, ideally 1-2x threshold

5. **TP unreachability check**: Is TP >> threshold (e.g., tp_pct > 5x threshold)?
   - TP target is unrealistic given the signal magnitude
   - TP should be 0.3-1x threshold for DC Overshoot

6. **Churn check**: Is `avg_hold_seconds < 60`?
   - Trades are opening and closing too quickly (noise)
   - Increase threshold to filter noise

7. **Backstop collision check**: Are `backstop_sl_pct` or `backstop_tp_pct` close to or smaller than `sl_pct`/`tp_pct`?
   - Backstop should be the emergency safety net, not the primary exit
   - Backstop should be 3-5x the primary SL/TP

8. **Drawdown vs P&L check**: Is `max_drawdown_usd > |net_pnl_usd| * 3`?
   - Risk-reward is poor — reduce leverage or tighten SL

#### Additional Analysis

Also compute these from the trade-level data:
- **Notional per trade**: `entry_price * size` — is position size being fully utilized?
- **Side balance**: LONG vs SHORT count and P&L — is there a directional bias?
- **Daily P&L**: Group by exit day to spot time-based patterns
- **Consecutive losses**: Max losing streak — indicates if strategy was in a bad regime
- **Price movement distribution**: How many trades moved < fee threshold (0.07%)?

### Step 4: Generate Recommendations

For each issue found in the diagnostic, provide a concrete fix:

| Diagnostic Issue | Parameter to Change | Recommended Value | Why |
|-----------------|-------------------|-------------------|-----|
| Fee dominance (>100%) | `threshold` | 0.01-0.015 | Need larger price moves to overcome fees |
| Fee erosion (>50%) | `threshold` | At least 0.008 | Trades need more room to profit |
| SL too tight | `sl-pct` | 1-2x threshold | Let signals develop before stopping out |
| TP unreachable | `tp-pct` | 0.3-1x threshold | Take profits at realistic targets |
| No trailing | `trail-pct` | 0.3 | Lock in 70% of gains consistently |
| No min-profit-to-trail | `min-profit-to-trail-pct` | 0.001-0.002 | Don't trail until minimum profit reached |
| Backstop too tight | `backstop-sl-pct`, `backstop-tp-pct` | 0.05 | Emergency safety net, not primary exit |

**Cross-reference with the backtest sweep**: Always show the user what the sweep found as the optimal config, including:
- The #1 config by net P&L and its metrics
- The best risk-adjusted config (highest profit factor with net P&L > $0)
- How many configs are profitable at each threshold level
- Whether the user's current threshold has ANY profitable combinations

### Step 5: Present Analysis Report

Structure the output as a clear report with these sections:

1. **Performance Summary Table** — Key metrics in a markdown table
2. **Root Cause Analysis** — The 1-3 most impactful issues from the diagnostic
3. **Backtest Comparison** — Table showing profitability by threshold + top configs
4. **Specific Parameter Changes** — Table: current value → recommended value → why
5. **Ready-to-Run Command** — The optimized `live_bridge.py` command

### Step 6: Suggest Optimized Command

Generate a ready-to-run `live_bridge.py` command with the recommended parameters:

```bash
screen -S <symbol_lower> bash -c 'uv run --package hyperliquid-trading-bot python \
  trading-agent/src/strategies/dc_overshoot/live_bridge.py \
  --symbol <SYMBOL> --threshold <recommended> \
  --position-size <keep_user_value> \
  --sl-pct <recommended> --tp-pct <recommended> \
  --trail-pct <recommended> \
  --min-profit-to-trail-pct <recommended> \
  --backstop-sl-pct 0.05 --backstop-tp-pct 0.05 --leverage 10 --yes'
```

### Step 7 (Optional): Backtest Validation

If the user wants to validate the specific recommended config:

```bash
uv run --package hyperliquid-trading-bot python -m backtesting.cli \
  --symbol <SYMBOL> --threshold <recommended> \
  --sl-pct <recommended> --tp-pct <recommended> \
  --trail-pct <recommended> --min-profit-to-trail-pct <recommended> \
  --days <same_period> --json
```

## Key Metrics Reference

- **Net P&L**: Profit after all exchange fees (0.035% taker fee per side)
- **Win rate (net vs gross)**: Net accounts for fees; gross does not. Large gap = fees eating profits.
- **Profit factor**: Sum of winning trades / sum of losing trades. >1.5 is acceptable, >2.0 is good, >5.0 is excellent.
- **Max drawdown**: Worst peak-to-trough on cumulative P&L curve.
- **Fee impact (%)**: Fees as percentage of gross profit. >50% means threshold is too small. >100% means guaranteed losses.
- **Trades eaten by fees**: Trades profitable before fees but negative after. High count = threshold too small.
- **Exit type breakdown**: SL/TP/reversal distribution reveals strategy behavior.
- **Avg hold time**: How long positions are held. Compare to expected for the threshold.

## Parameter Relationships (DC Overshoot)

These relationships determine profitability:

- **Threshold** is the most important parameter. It determines signal strength.
  - Too small (<0.005): generates noise trades eaten by fees
  - Sweet spot: 0.01-0.015 for most symbols
  - Too large (>0.03): too few trades, slow capital deployment
- **SL** should be 0.5-2x threshold. Tighter = more trades stopped out. Wider = larger losses.
- **TP** should be 0.3-1x threshold. Too wide = never reached. Too tight = caps upside.
- **Trail** at 0.3 is consistently optimal. Locks in 70% of gains once min profit is reached.
- **Min-profit-to-trail**: 0.001-0.002 works best. Prevents trailing before meaningful profit.
- **Backstop SL/TP**: Should be 3-5x primary SL/TP. Emergency crash protection only.
- **Position size**: Does not affect win rate, but larger positions amplify both gains and losses.
- **Leverage**: 10x is standard. Higher leverage = higher drawdown risk.

## Known Insights from Previous Analysis

- **HYPE (Feb 2026)**: threshold=0.002 → 0/216 configs profitable; threshold=0.015 → 146/216 (68%) profitable
- **SOL**: threshold=0.015 makes 100% of parameter combinations profitable
- Lower thresholds (0.002-0.008) generate many small trades that get eaten by fees
- **SL% around 1.0-1.5%** works best for most symbols
- **TP% around 0.5%** works best — takes profits before reversals
- **Trail = 0.3** is consistently optimal across all parameter sweeps
- Fee impact: At $100 position size, round-trip fee is ~$0.07. Trades need >$0.07 profit to be net positive.
- **Gross win rate gap**: A strategy can be 55%+ accurate and still lose money if threshold is too small

## Caveats

- Exit reasons (SL/TP/reversal) are inferred from price movement, not from actual strategy labels
- Fills from other strategies or manual trades on the same wallet will be included — filtered by symbol only
- Partial fills are paired using weighted average entry price
- This reviews closed trades only; open positions are reported separately
- Fee calculation assumes taker fees (0.035%) — if the strategy uses limit orders, actual fees may be lower
- The Hyperliquid API may limit the number of fills returned per request
- Backtest results are historical and don't guarantee future performance, but threshold patterns are remarkably consistent
