---
description: Validate DC Overshoot bot configuration for safety and profitability before trading
user_invocable: true
argument: CLI args (e.g., --symbol SOL --threshold 0.015 --sl-pct 0.015 --leverage 10)
---

# Validate DC Overshoot Configuration

Check whether a DC Overshoot bot configuration is safe and profitable before going live. Catches liquidation risks, fee traps, and suboptimal parameter relationships.

## How to Use

The user invokes this with `/validate-dc <args>` where args are the same CLI flags used with `live_bridge.py`.

## Workflow

### Step 1: Run Validation

```bash
uv run --package hyperliquid-trading-bot python \
  trading-agent/src/strategies/dc_overshoot/live_bridge.py \
  $ARGUMENTS --validate
```

If no arguments provided, validate the current recommended defaults:

```bash
uv run --package hyperliquid-trading-bot python \
  trading-agent/src/strategies/dc_overshoot/live_bridge.py \
  --symbol SOL --threshold 0.015 --sl-pct 0.015 --tp-pct 0.005 \
  --trail-pct 0.5 --min-profit-to-trail-pct 0.002 \
  --backstop-sl-pct 0.05 --leverage 10 --validate
```

### Step 2: Interpret Results

The validator outputs three levels:

- **ERRORS** — Must fix. The config will cause liquidation or guaranteed losses. Exit code 1.
- **WARNINGS** — Review recommended. The config is suboptimal but won't immediately fail.
- **INFO** — Educational context about margin impact, fees, and protection layers.

### Step 3: Explain & Fix

For each error/warning, explain to the user:
1. **What's wrong** — the specific relationship that's broken
2. **Why it matters** — what would happen in practice
3. **How to fix** — concrete parameter change recommendation

## Validation Rules Reference

### Errors
| Code | Rule |
|------|------|
| E1 | SL >= liquidation distance (1/leverage). You get liquidated before SL fires. |
| E2 | Backstop SL >= liquidation distance. Exchange crash protection is useless. |
| E3 | Software SL >= backstop SL. Backstop fires before software SL. |
| E4 | Position size <= 0. |
| E5 | Leverage outside 1-50x range. |

### Warnings
| Code | Rule |
|------|------|
| W1 | TP < round-trip fees. Every TP exit loses money. |
| W2 | SL < DC threshold. Noise will stop you out before overshoot completes. |
| W3 | Backstop within 20% of liquidation. Dangerously close to liquidation. |
| W4 | TP < 4x single-side fee. Thin profit margin. |
| W5 | SL * leverage > 20%. High margin loss per stop. |
| W6 | min_profit_to_trail < fee. Trailing may ratchet on fee-noise. |

## Key Relationships

The three-layer protection must be ordered:
```
Software SL < Backstop SL < Liquidation (~1/leverage)
```

Example at 10x leverage:
- Software SL: 1.5% (fires first, software-level)
- Backstop SL: 5.0% (exchange trigger order, crash protection)
- Liquidation: ~10% (final safety net, avoid at all costs)
