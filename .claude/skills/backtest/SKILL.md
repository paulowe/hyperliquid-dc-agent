---
description: Backtest DC Overshoot strategy on historical data and find profitable parameters
user_invocable: true
argument: symbol (e.g., SOL, BTC, ETH)
---

# Backtest DC Overshoot Strategy

Run backtests and parameter sweeps for the DC Overshoot trading strategy on Hyperliquid historical data. This skill fetches real candle data, simulates the strategy with fee accounting, and identifies profitable parameter configurations.

## How to Use

The user invokes this with `/backtest <SYMBOL>` (e.g., `/backtest SOL`).

## Workflow

### Step 1: Single Backtest with Current/Default Parameters

Run a quick single backtest first to establish a baseline:

```bash
uv run --package hyperliquid-trading-bot python -m backtesting.cli \
  --symbol $ARGUMENTS --threshold 0.015 --sl-pct 0.015 --tp-pct 0.005 \
  --trail-pct 0.5 --min-profit-to-trail-pct 0.002 --days 7
```

### Step 2: Parameter Sweep

Run a full parameter sweep to find the best configuration:

```bash
uv run --package hyperliquid-trading-bot python -m backtesting.cli \
  --symbol $ARGUMENTS --sweep --days 7
```

Or with JSON output for programmatic analysis:

```bash
uv run --package hyperliquid-trading-bot python -m backtesting.cli \
  --symbol $ARGUMENTS --sweep --days 7 --json
```

### Step 3: Analyze & Recommend

After the sweep completes, analyze the results and provide:

1. **Top 3 configurations** — show threshold, SL, TP, trail params, net P&L, win rate, profit factor
2. **Pattern insights** — what parameter ranges are consistently profitable
3. **Risk assessment** — max drawdown, fee impact, trade frequency
4. **Recommended live command** — ready-to-run `live_bridge.py` command with the best params

### Step 4: Generate Live Trading Command

For the best configuration, output the full command:

```bash
uv run --package hyperliquid-trading-bot python \
  trading-agent/src/strategies/dc_overshoot/live_bridge.py \
  --symbol <SYMBOL> --threshold <best_threshold> \
  --position-size <size> \
  --sl-pct <best_sl> --tp-pct <best_tp> \
  --trail-pct <best_trail> \
  --min-profit-to-trail-pct <best_min_trail> \
  --backstop-sl-pct 0.05 --leverage 10 --yes
```

## Key Metrics to Explain

- **Net P&L**: Profit after all fees (0.035% taker fee per side)
- **Win rate (net vs gross)**: Net accounts for fees; gross does not. Large gap = fees eating profits.
- **Profit factor**: Sum of winning trades / sum of losing trades. >2.0 is good, >5.0 is excellent.
- **Max drawdown**: Worst peak-to-trough on cumulative P&L curve.
- **Trades eaten by fees**: Trades profitable before fees but negative after. High count = threshold too small.
- **Net P&L/day**: Annualized proxy for strategy viability.

## Known Insights from Previous Sweeps

- **Threshold = 0.015 (1.5%)** makes 100% of parameter combinations profitable for SOL
- Lower thresholds (0.002–0.008) generate too many small trades that get eaten by fees
- **SL% around 1.0–1.5%** works best — tight enough to limit losses, wide enough to avoid noise exits
- **TP% around 0.5–1.0%** works best — takes profits quickly before reversals
- **Trail = 0.5** is consistently optimal across all sweeps
- Fee impact: At $100 position size, round-trip fee is ~$0.07. Trades need >$0.07 profit to be net positive.

## Caveats to Mention

- Backtests use 1-minute candle closes, not tick-by-tick — real fills may differ slightly
- No slippage modeling — real orders may get slightly worse prices
- Past performance does not guarantee future results
- Market regime changes (trending vs ranging) affect strategy performance
- Recommended: re-run sweep weekly to adapt to current market conditions
