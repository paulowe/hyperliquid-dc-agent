# DC Overshoot Bot: Parameter Guide

How every parameter works, how they interact with leverage, and what you actually see on the Hyperliquid UI.

---

## The key thing to understand

**All `--sl-pct`, `--tp-pct`, and `--backstop-sl-pct` values are raw price moves.**
Your actual P&L on margin is multiplied by leverage.

```
Margin P&L = price move % x leverage
```

So with 10x leverage:

| Raw price move | Your margin P&L |
|----------------|-----------------|
| 0.1% against you | -1.0% on UI |
| 0.3% against you | -3.0% on UI |
| 0.5% against you | -5.0% on UI |
| 5.0% against you | -50% on UI |
| 10% against you | -100% (liquidation) |

**When you set `--sl-pct 0.005`, you're saying "exit if price moves 0.5% against me".
At 10x leverage, that's a 5% loss on your margin. That's what the UI shows.**

---

## Parameters reference

### `--threshold` (default: 0.001 = 0.1%)

The DC (Directional Change) threshold. This controls how big a price move must be
before the algorithm detects a trend reversal and enters a trade.

- **Lower threshold** (e.g., 0.001): More frequent trades, smaller expected moves
- **Higher threshold** (e.g., 0.005): Fewer trades, larger expected moves

**DC theory says**: the expected overshoot after a directional change roughly equals
the threshold. So with `--threshold 0.001`, expect ~0.1% moves after each signal.

This is the anchor for all other parameters. Everything else should be sized relative to it.

### `--sl-pct` (default: 0.003 = 0.3%)

**Initial stop loss** — the maximum raw price move against you before the software
closes the position. This is checked every tick (~1 second).

Example with `--sl-pct 0.003`:
- LONG entry @ 67,000 → SL at 66,799 (0.3% below)
- SHORT entry @ 67,000 → SL at 67,201 (0.3% above)

**What you see on the UI at 10x leverage**: -3% margin loss when SL hits.

**How to think about it**: This is your "I was wrong" exit. DC theory says overshoot ~ threshold,
so if price moves 3x the threshold against you, the signal was probably wrong. Cut it.

| Leverage | `--sl-pct 0.003` loss | `--sl-pct 0.005` loss |
|----------|----------------------|----------------------|
| 3x | -0.9% | -1.5% |
| 5x | -1.5% | -2.5% |
| 10x | -3.0% | -5.0% |
| 20x | -6.0% | -10.0% |
| 40x | -12.0% | -20.0% |

### `--tp-pct` (default: 0.002 = 0.2%)

**Initial take profit** — the raw price move in your favor that triggers an immediate
close. This fires once and closes the whole position.

Example with `--tp-pct 0.10`:
- SHORT entry @ 67,000 → TP at 60,300 (10% below)
- LONG entry @ 67,000 → TP at 73,700 (10% above)

**Important behavior**: TP also moves! When the trailing ratchet is active and price
keeps pushing in your favor, the TP pushes further away from you. This means TP acts
as a ceiling/floor that you approach only during very large sustained moves.

**In practice with a wide TP (e.g., 10%)**: You will almost never hit TP directly.
The ratcheting SL will close you out first as price pulls back. The wide TP is
intentional — it lets the ratcheting SL be your profit-taking mechanism.

### `--trail-pct` (default: 0.5 = 50%)

**Trail lock-in percentage** — when the position is profitable and making new highs
(or lows for shorts), the SL ratchets to lock in this fraction of the profit.

Example with `--trail-pct 0.5`:
```
SHORT entry @ 67,000
Price drops to 66,800 (new low, 0.3% profit)

Ratcheted SL = entry - (entry - low) x trail_pct
             = 67,000 - (67,000 - 66,800) x 0.5
             = 67,000 - 100
             = 66,900

If price bounces back to 66,900 → exit with ~0.15% profit
(instead of the original SL at 67,201 which would be a loss)
```

**Higher trail_pct** (e.g., 0.7): Tighter trailing — locks in more profit but exits sooner
on small pullbacks. Less room for the trade to breathe.

**Lower trail_pct** (e.g., 0.3): Looser trailing — gives more room for pullbacks but
locks in less profit. Better for volatile assets.

**This is the primary profit-taking mechanism.** Not the TP.

### `--min-profit-to-trail-pct` (not a CLI arg — set in config, default: 0.001 = 0.1%)

The minimum raw profit before trailing ratchet activates. Below this threshold,
SL stays at the initial level and doesn't move.

**Why it exists**: Without this, noise could ratchet the SL to a tiny profit that
immediately gets hit. This ensures the position has room to develop before locking in.

**Current default** (0.001 = 0.1%): Trailing starts once you're 0.1% in profit.
At 10x leverage, that's 1% margin profit before ratcheting begins.

### `--position-size` (default: 50)

Notional value in USD per trade. The bot calculates the asset quantity as:

```
size = position_size_usd / current_price
```

Example: `--position-size 13` with BTC at $67,000 → 0.000194 BTC per trade.

**This is NOT your margin**. With 10x leverage, $13 notional requires ~$1.30 margin.

### `--leverage` (default: 3)

Sets cross-leverage on your Hyperliquid account for the symbol. This is an
account-level setting, not per-order.

**Liquidation distance**: approximately `1 / leverage` as a price move.

| Leverage | Approx. liquidation distance |
|----------|------------------------------|
| 3x | ~33% price move |
| 5x | ~20% price move |
| 10x | ~10% price move |
| 20x | ~5% price move |
| 40x | ~2.5% price move |

**Your backstop SL must be tighter than liquidation**, otherwise the exchange
liquidates you before the backstop fires and you pay a liquidation fee.

### `--backstop-sl-pct` (default: 0.10 = 10%)

**Exchange-level hard stop loss** — a trigger order placed directly on Hyperliquid.
This only fires if the bridge (your computer/bot) crashes or disconnects.

The software trailing SL handles normal exits. The backstop is a safety net.

```
     Normal operation           Bridge crash
     ────────────────           ────────────
     Software SL exits     →   Backstop SL exits
     (every tick check)         (exchange trigger order)
```

**Must be tighter than liquidation:**

| Leverage | Liquidation | Good backstop | Bad backstop |
|----------|-------------|---------------|--------------|
| 3x | ~33% | 10% | 40% |
| 5x | ~20% | 10% | 25% |
| 10x | ~10% | 5% | 15% |
| 20x | ~5% | 3% | 10% |
| 40x | ~2.5% | 1.5% | 5% |

### `--backstop-tp-pct` (default: 0.10 = 10%)

**Exchange-level hard take profit** — a trigger order placed directly on Hyperliquid.
This only fires if the bridge (your computer/bot) crashes or disconnects while
the position is in profit.

The software trailing SL handles normal profit-taking. The backstop TP is a safety net
that ensures profits are captured even if the bot goes offline.

```
     Normal operation           Bridge crash while in profit
     ────────────────           ────────────────────────────
     Trailing SL exits      →  Backstop TP exits
     (ratchets to lock in)      (exchange trigger order)
```

**Must be wider than software TP**, otherwise the exchange closes the position
before the trailing mechanism gets a chance to let profits run.

Example with `--backstop-tp-pct 0.10`:
- LONG entry @ 67,000 → backstop TP trigger at 73,700 (10% above)
- SHORT entry @ 67,000 → backstop TP trigger at 60,300 (10% below)

### `--duration` (default: 30)

How long the bot runs in minutes. When duration expires:

- If **no position open**: bot shuts down immediately
- If **position open**: bot keeps running until the trailing RM closes it (SL, TP, or reversal)

### `--observe-only`

Runs the full strategy (DC detection, signal generation) but does not place any
orders. Use this to observe how the bot would trade without risking money.

---

## The layers of protection

```
                        ┌──────────────────────────┐
   Layer 1 (software)   │  Trailing SL / TP         │  Checks every tick (~1s)
                        │  --sl-pct / --tp-pct      │  Ratchets to lock in profit
                        └────────────┬─────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              │   ┌──────────────────▼─────────────────┐    │
   Layer 2    │   │  Backstop SL     --backstop-sl-pct │    │
   (exchange) │   │  Fires if bot offline (loss side)  │    │
              │   └────────────────────────────────────┘    │
              │                                             │
              │   ┌────────────────────────────────────┐    │
              │   │  Backstop TP     --backstop-tp-pct │    │
              │   │  Fires if bot offline (profit side)│    │
              │   └────────────────────────────────────┘    │
              └──────────────────────┬──────────────────────┘
                                     │
                        ┌────────────▼─────────────┐
   Layer 3 (exchange)   │  Liquidation              │  Forced close by exchange
                        │  ~1/leverage              │  You lose everything + fee
                        └──────────────────────────┘
```

**Loss side must be ordered by distance**: `sl-pct < backstop-sl-pct < 1/leverage`

**Profit side**: `backstop-tp-pct > tp-pct` so trailing has room to work

---

## Example: lifecycle of a SHORT trade

Config: `--threshold 0.001 --sl-pct 0.003 --tp-pct 0.10 --trail-pct 0.5 --leverage 10`

```
1. DC detects PDCC_Down → SELL signal
   Entry: SHORT @ 67,000
   Initial SL: 67,201 (0.3% above = 3% margin loss at 10x)
   Initial TP: 60,300 (10% below = 100% margin profit at 10x)
   Backstop SL: 73,700 (10% above, on exchange)
   Backstop TP: 60,300 (10% below, on exchange)

2. Price drops to 66,933 (0.1% profit, 1% on UI)
   → min_profit_to_trail (0.1%) reached
   → Ratchet SL down: 67,000 - (67,000 - 66,933) x 0.5 = 66,967
   → New SL = 66,967 (now only 0.05% above low, not 0.3%)
   → TP also pushes lower

3. Price drops further to 66,800 (0.3% profit, 3% on UI)
   → Ratchet SL: 67,000 - (67,000 - 66,800) x 0.5 = 66,900
   → SL is now 66,900 — ABOVE entry! Profit is locked in.

4. Price bounces to 66,900
   → SL hit → CLOSE position
   → Realized profit: 0.15% raw = 1.5% on margin

   Compare: without ratcheting, SL was at 67,201 — you would
   have given back all profit and lost 0.3% (3% on margin).
```

---

## Configuration recipes

### Conservative (learning / small account)

```bash
--threshold 0.001 --sl-pct 0.003 --tp-pct 0.10 \
--trail-pct 0.5 --leverage 3 --backstop-sl-pct 0.10 --backstop-tp-pct 0.10 \
--position-size 13
```

- SL hit = 0.9% margin loss (manageable)
- Backstop SL at 10%, backstop TP at 10%, liquidation at ~33% (plenty of room)
- Small position size

### Moderate (confident in signals)

```bash
--threshold 0.001 --sl-pct 0.003 --tp-pct 0.10 \
--trail-pct 0.5 --leverage 10 --backstop-sl-pct 0.05 --backstop-tp-pct 0.10 \
--position-size 50
```

- SL hit = 3% margin loss
- Backstop SL at 5%, backstop TP at 10%, liquidation at ~10% (backstop fires first)
- Larger position

### Aggressive (experienced, high conviction)

```bash
--threshold 0.005 --sl-pct 0.010 --tp-pct 0.10 \
--trail-pct 0.5 --leverage 10 --backstop-sl-pct 0.05 --backstop-tp-pct 0.10 \
--position-size 100
```

- Higher threshold = fewer but larger trades
- Wider SL (1%) matches bigger threshold = 10% margin loss
- Same backstop/leverage safety

---

## Quick sanity check for any config

Before running, verify:

1. **SL margin loss is acceptable**: `sl_pct x leverage` = your worst normal loss
2. **Backstop SL < liquidation**: `backstop_sl_pct < 1/leverage`
3. **SL is proportional to threshold**: `sl_pct` should be 2-5x `threshold`
4. **TP is wide**: Let the ratcheting SL do the profit-taking work
5. **Position size is affordable**: `position_size / leverage` = margin used per trade
6. **Backstop TP > software TP**: `backstop_tp_pct > tp_pct` so trailing has room to work
