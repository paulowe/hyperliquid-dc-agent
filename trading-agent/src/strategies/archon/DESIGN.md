# Archon Strategy — Claude-Augmented DC Trading

## Overview

Archon is a trading strategy that combines **Directional Change (DC) event detection** for timing with **Claude AI intelligence** for entry/exit decisions. DC events tell us *when* the market moved significantly; Claude tells us *whether and how* to trade.

## Architecture

```
WebSocket Ticks → DC Detector → DC Events (timing signals)
                       ↓                    ↓
               Sensor events         Trade events (2% threshold)
                       ↓                    ↓
               Regime Detector      Context Builder → Market Summary
                                         ↓
                                  Claude Reasoner → Structured Decision (JSON)
                                         ↓
                              TrailingRiskManager → Position Exits
```

### Signal Flow

1. **Tick arrives** from Hyperliquid WebSocket
2. **DC Detector** processes tick through sensor (0.4%) and trade (2.0%) thresholds
3. **Sensor events** feed the regime detector (quiet/neutral/choppy/trending)
4. **Trade events** (PDCC_Down or PDCC2_UP) trigger analysis:
   - Context Builder creates a market snapshot (prices, DC history, trade results, regime)
   - Claude Reasoner analyzes context and returns a structured JSON decision
   - Direction filter and confidence threshold gate the decision
5. **Entry signals** open positions via TrailingRiskManager
6. **Every tick**: TrailingRiskManager checks SL/TP/trailing exits

### Claude Decision Format

```json
{
    "action": "enter_long" | "enter_short" | "close" | "skip",
    "confidence": 0.0 - 1.0,
    "reasoning": "brief explanation",
    "tp_pct": 0.008,
    "sl_pct": 0.015
}
```

### Heuristic Fallback

When Claude is unavailable (no API key, rate limited, timeout), the strategy falls back to rule-based logic:
- Enter LONG on PDCC2_UP in quiet/neutral regime
- Close on opposing DC event
- Skip in choppy regime
- Skip after 3+ consecutive losses
- Adaptive TP from overshoot distribution

## Modules

| File | Purpose |
|------|---------|
| `config.py` | Configuration dataclass with validation |
| `context.py` | Market context builder — rolling windows of ticks, DC events, trades |
| `reasoner.py` | Claude AI integration + heuristic fallback |
| `strategy.py` | Main strategy: DC detection → async decision → signal emission |
| `bridge.py` | Live bridge: WebSocket → strategy → trade execution |

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| threshold | 2.0% | Trade-level DC threshold |
| sensor | 0.4% | Regime detection threshold |
| sl_pct | 1.5% | Stop loss |
| tp_pct | 0.8% | Default take profit (adaptive overrides) |
| trail_pct | 35% | Lock in 35% of gains |
| min_confidence | 60% | Minimum confidence to act on Claude decision |
| direction | long | Long-only by default (HYPE uptrend) |
| model | claude-haiku-4-5 | Fast, cheap model for real-time decisions |
| max_calls/hour | 30 | Cost control (~$0.05/day at current rates) |

## Key Design Decisions

### Why long-only?

BQ telemetry analysis (128 trades, Feb-Mar 2026):
- HYPE longs: +25.16% net (profitable)
- HYPE shorts: -32.55% net (bleeding money)
- HYPE has been in a sustained uptrend since early March

### Why 2.0% threshold?

Backtest sweep (HYPE 3.6-day, long-only):
- 1.0%: -$3.07, 36.7% WR — too many SL exits
- 1.5%: +$0.01, 53.8% WR — breakeven
- **2.0%: +$0.38, 77.8% WR, PF 1.24** — best config
- 2.5%: too few trades

### Why Claude?

Fixed rules can't adapt to:
- Changing market regimes within a single session
- Nuanced pattern recognition (e.g., DC event quality varies)
- Dynamic TP/SL sizing based on context
- Integration of multiple weak signals into a strong decision

Claude adds contextual intelligence on top of proven DC mechanics.

## Sub-Account

- Name: **Archon0325**
- Address: `0xe614e36a3e7c99d3682360c4f160433fe59cb6be`
- Status: Created, no funds (observe-only mode)

## Running

```bash
# Observe-only (heuristic mode)
uv run --package hyperliquid-trading-bot python -m strategies.archon.bridge \
    --symbol HYPE --observe-only --no-ai --long-only

# Observe-only (Claude mode — requires ANTHROPIC_API_KEY)
uv run --package hyperliquid-trading-bot python -m strategies.archon.bridge \
    --symbol HYPE --observe-only --long-only

# Live trading on sub-account
uv run --package hyperliquid-trading-bot python -m strategies.archon.bridge \
    --symbol HYPE --vault-address 0xe614... --long-only --yes
```

## Testing

```bash
make test-archon  # or:
uv run --package hyperliquid-trading-bot pytest trading-agent/tests/strategies/archon/ -v
```

46 tests covering config, context, reasoner (heuristic + Claude parsing), strategy integration.
