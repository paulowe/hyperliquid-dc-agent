# Long Run

Run a task autonomously for an extended period with periodic check-ins. Designed for operations that need continuous local VM access: monitoring bots, managing live trades, iterating on strategy, running experiments.

## How to Use

- `/long-run <task description> <duration>` — run the described task
- Duration examples: `2h`, `4h`, `8h`, `12h` (default: 4h)

### Examples

```
/long-run monitor archon scanner, go live when validated, optimize prompts 8h
/long-run backtest all symbols with parameter sweep and find best configs 2h
/long-run watch for DC events and analyze claude decisions 4h
/long-run run experiments comparing threshold values across assets 3h
```

## Behavior

The skill runs a continuous loop on the local VM with access to:
- Screen sessions (start/stop/restart bots)
- Log files (`/tmp/*.log`)
- Hyperliquid API (account state, positions, prices)
- Strategy code (read, edit, test, commit, push)
- BigQuery telemetry
- Claude CLI proxy (for AI-augmented decisions)

### Loop Structure

Every cycle (default 30 minutes, adjustable based on task):

1. **Assess** — Check what changed since last cycle (prices, events, positions, logs)
2. **Decide** — Based on the user's task description, decide what action to take
3. **Act** — Execute the action (restart bot, update code, run backtest, etc.)
4. **Record** — Log what was done and why to `.claude/agents/topics/`
5. **Sleep** — Wait for next cycle

### Safety Rails

- If a running bot dies, restart it before doing anything else
- If account drawdown exceeds 20%, switch to observe-only
- Never force-close profitable positions
- Commit code changes before restarting bots
- Run tests before deploying code changes

### Adaptive Behavior

The loop adjusts based on what's happening:
- **Quiet market (no events):** Longer sleep intervals, focus on analysis/backtesting
- **Active trading:** Shorter intervals, focus on position monitoring
- **Bug detected:** Immediate fix → test → deploy cycle
- **Pattern discovered:** Update prompts/parameters → backtest → validate → deploy

## ARGUMENTS

Parse the user's message for:
- **task**: What to do (everything before the duration)
- **duration**: How long to run (last token if it matches `\d+h`)
- **interval**: How often to check (default 30m, can be specified as `every 10m`)

## Implementation

```python
import time

# Parse args from ARGUMENTS
task = "<user's task description>"
duration_hours = 4  # from args
interval_minutes = 30  # default

end_time = time.time() + duration_hours * 3600
cycle = 0

while time.time() < end_time:
    cycle += 1
    # 1. Assess current state
    # 2. Decide what to do based on task
    # 3. Act
    # 4. Record
    # 5. Sleep
    time.sleep(interval_minutes * 60)
```

The agent should interpret the task description and use judgment about what specific actions to take each cycle. It has full access to all tools and the codebase.
