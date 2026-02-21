# Observer Agent — Multi-threshold Concurrent Trading Orchestrator

The observer agent autonomously runs multiple observe-only DC Overshoot sessions at different thresholds concurrently, collects performance data, and uses AI reasoning (Claude Agent SDK) to recommend the best configuration for current market conditions.

## Quick Start

```bash
# Run one exploration round: 6 thresholds, 30 min each, all concurrent
uv run --package hyperliquid-trading-bot python \
    trading-agent/src/agents/observer/cli.py explore --symbol SOL --duration 30

# Same but skip Claude API (use heuristic ranking instead)
uv run --package hyperliquid-trading-bot python \
    trading-agent/src/agents/observer/cli.py explore --symbol SOL --duration 30 --no-ai
```

## Commands

### `explore` — Run one exploration round

Launches N observe-only sessions concurrently, waits for them to complete, collects JSON reports, and analyzes which threshold performed best.

```bash
uv run --package hyperliquid-trading-bot python \
    trading-agent/src/agents/observer/cli.py explore \
    --symbol SOL \
    --duration 30 \
    --thresholds 0.005 0.01 0.015 0.02 \
    --position-size 100 \
    --leverage 10 \
    --no-ai
```

| Flag | Default | Description |
|------|---------|-------------|
| `--symbol` | SOL | Trading symbol |
| `--duration` | 30 | Minutes per session |
| `--thresholds` | 0.002 0.004 0.006 0.008 0.01 0.015 | DC thresholds to test |
| `--position-size` | 100 | Simulated position size USD |
| `--leverage` | 10 | Simulated leverage |
| `--no-ai` | false | Skip Claude analysis, use heuristic ranking |

### `loop` — Continuous explore/exploit loop

Runs exploration rounds back-to-back. After each round, logs the best config and starts the next round.

```bash
# Run 3 rounds then stop
uv run --package hyperliquid-trading-bot python \
    trading-agent/src/agents/observer/cli.py loop --symbol SOL --rounds 3

# Run forever (Ctrl+C to stop)
uv run --package hyperliquid-trading-bot python \
    trading-agent/src/agents/observer/cli.py loop --symbol SOL
```

| Flag | Default | Description |
|------|---------|-------------|
| `--rounds` | 0 | Number of rounds (0 = infinite) |
| (same as explore) | | All explore flags work here too |

### `status` — Show current state

```bash
uv run --package hyperliquid-trading-bot python \
    trading-agent/src/agents/observer/cli.py status
```

Shows: current phase, active sessions, completed rounds, and recommendations.

### `stop` — Stop all running sessions

```bash
uv run --package hyperliquid-trading-bot python \
    trading-agent/src/agents/observer/cli.py stop
```

Sends SIGINT to all active observe-only sessions for graceful shutdown.

## How It Works

### Architecture

```
┌─────────────────────────────────────────────┐
│             Observer Agent CLI                │
│  explore / loop / status / stop              │
└──────────────────┬──────────────────────────┘
                   │
      ┌────────────┼────────────────┐
      │            │                │
┌─────▼──────┐ ┌──▼──────────┐ ┌──▼──────────┐
│ Session    │ │ Result      │ │ Claude      │
│ Manager    │ │ Collector   │ │ Reasoner    │
│            │ │             │ │             │
│ Spawns N   │ │ Reads JSON  │ │ AI analysis │
│ subprocs   │ │ reports     │ │ or heuristic│
└────────────┘ └─────────────┘ └─────────────┘
      │
┌─────▼──────────────────────────────────────┐
│  live_bridge.py --observe-only             │
│  --json-report state/reports/session_X.json│
│                                             │
│  N concurrent subprocesses                  │
└─────────────────────────────────────────────┘
```

### Explore/Exploit Flow

Each exploration round:

1. **EXPLORE** — Launch N observe-only `live_bridge.py` subprocesses (one per threshold)
2. **WAIT** — All sessions run concurrently for `--duration` minutes
3. **COLLECT** — Gather JSON reports from each session
4. **ANALYZE** — Compute simulated PnL, then either:
   - Call Claude Agent SDK for AI reasoning about which config is best
   - Or use heuristic ranking (sort by net PnL + win rate) with `--no-ai`
5. **RECOMMEND** — Log the best config with a ready-to-use live trading command

### Simulated PnL

Each observe-only session generates signals (BUY/SELL/CLOSE) without executing trades. The collector replays these signals to estimate what PnL would have been, including:
- Entry/exit price differences
- Taker fees (0.035% per side on Hyperliquid)
- Win/loss counting

### State Persistence

State is saved to `trading-agent/state/observer_state.json`:
- Current phase (idle/exploring/analyzing)
- Active session PIDs
- History of completed rounds with recommendations

## Default Thresholds

The default exploration grid uses thresholds from the backtest sweep module:

| Threshold | Meaning | Typical Use Case |
|-----------|---------|-----------------|
| 0.002 | 0.2% | High-frequency, BTC in calm markets |
| 0.004 | 0.4% | Medium-frequency |
| 0.006 | 0.6% | Balanced |
| 0.008 | 0.8% | Lower-frequency |
| 0.010 | 1.0% | Conservative |
| 0.015 | 1.5% | Very conservative, best for volatile assets |

Each threshold gets a paired SL (= threshold) and TP (= 0.5 * threshold) as starting points.

## Per-Asset Recommendations

Based on backtest sweeps:

| Asset | Recommended Thresholds | Notes |
|-------|----------------------|-------|
| BTC | 0.002 - 0.006 | Most liquid, smaller thresholds work |
| ETH | 0.003 - 0.008 | Slightly wider than BTC |
| SOL | 0.006 - 0.015 | Best at 0.015 per backtest sweep |
| HYPE | 0.008 - 0.02 | Low liquidity, needs wide thresholds |

## Claude Agent SDK Integration

When `--no-ai` is not set, the agent uses Claude Agent SDK (`claude-agent-sdk`) to reason about results:

- Sends structured session data (signal counts, simulated PnL, win rates) as a prompt
- Claude analyzes which threshold/config combination is optimal
- Returns: best config, reasoning, confidence score, suggestions for next round

If the Claude API is unavailable (no API key, network error), the agent automatically falls back to heuristic ranking.

**Requirements for AI mode:**
- `claude-agent-sdk` installed (included in dependencies)
- Anthropic API key configured (via `ANTHROPIC_API_KEY` env var or Claude Code auth)

## File Structure

```
trading-agent/src/agents/observer/
    __init__.py
    config.py       # SessionConfig, ObserverConfig dataclasses
    session.py      # SessionManager: subprocess lifecycle
    collector.py    # ResultCollector: parse JSON, compute PnL
    reasoner.py     # ClaudeReasoner: AI analysis + heuristic fallback
    state.py        # StateManager: JSON file persistence
    cli.py          # CLI entry point (explore/loop/status/stop)
    __main__.py     # python -m agents.observer support

trading-agent/tests/agents/observer/
    test_config.py
    test_session.py
    test_collector.py
    test_reasoner.py
    test_state.py
    test_cli.py
```

## Example Output

```
======================================================================
EXPLORATION ROUND 1
======================================================================
Sessions: 6 | Duration: 30 min | Thresholds: [0.002, 0.004, 0.006, 0.008, 0.01, 0.015]
Launched 6 sessions: ['a1b2c3d4', 'e5f6g7h8', ...]
Waiting for sessions to complete (up to 32 minutes)...
All sessions complete: {'a1b2c3d4': 'rc=0', ...}
Collected 6 reports (of 6 sessions)
  threshold=0.0020 | signals=45 | trades=18 | net=$-0.32 | 8W/10L
  threshold=0.0040 | signals=22 | trades=9  | net=$+0.45 | 5W/4L
  threshold=0.0060 | signals=15 | trades=6  | net=$+0.78 | 4W/2L
  threshold=0.0080 | signals=10 | trades=4  | net=$+0.92 | 3W/1L
  threshold=0.0100 | signals=7  | trades=3  | net=$+0.65 | 2W/1L
  threshold=0.0150 | signals=4  | trades=2  | net=$+1.15 | 2W/0L
----------------------------------------------------------------------
ANALYSIS RESULT
----------------------------------------------------------------------
Best threshold: 0.015
Confidence: 82%
Reasoning: Threshold 0.015 had highest net PnL with perfect win rate...
Recommended live command:
  uv run --package hyperliquid-trading-bot python \
    trading-agent/src/strategies/dc_overshoot/live_bridge.py \
    --symbol SOL --threshold 0.015 --sl-pct 0.015 --tp-pct 0.0075 ...
======================================================================
```
