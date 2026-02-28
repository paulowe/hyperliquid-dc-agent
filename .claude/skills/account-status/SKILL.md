---
description: Check Hyperliquid account value, open positions, and running bot status
user_invocable: true
---

# Account Status

Show current account state on Hyperliquid: account value, open positions, running bot screens, and historical account value.

## How to Use

- `/account-status` — current snapshot
- `/account-status` with questions like "what was my account value 24 hours ago?" or "start of day P&L"

## Workflow

### Step 1: Fetch Account State

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
withdrawable = float(state["withdrawable"])

print(f"Wallet: {wallet}")
print(f"Account Value: ${acct_val:,.2f}")
print(f"Margin Used: ${margin_used:,.2f}")
print(f"Withdrawable: ${withdrawable:,.2f}")
print()

positions = []
for pos in state.get("assetPositions", []):
    p = pos["position"]
    sz = float(p["szi"])
    if sz == 0:
        continue
    coin = p["coin"]
    entry = float(p["entryPx"])
    upnl = float(p["unrealizedPnl"])
    lev = p.get("leverage", {})
    lev_val = lev.get("value", "?")
    lev_type = lev.get("type", "")
    side = "LONG" if sz > 0 else "SHORT"
    margin = float(p.get("marginUsed", 0))
    liq = p.get("liquidationPx")
    positions.append({
        "coin": coin, "side": side, "size": abs(sz), "entry": entry,
        "upnl": upnl, "leverage": f"{lev_val}x {lev_type}",
        "margin": margin, "liq": liq,
    })

if positions:
    print("Open Positions:")
    for pos in positions:
        print(f"  {pos['coin']} | {pos['side']} {pos['size']} @ {pos['entry']:.4f}")
        print(f"    uPnL: ${pos['upnl']:,.2f} | Leverage: {pos['leverage']} | Margin: ${pos['margin']:,.2f} | Liq: {pos['liq']}")
else:
    print("No open positions")

# Open orders
orders = info.open_orders(wallet)
if orders:
    print(f"\nOpen Orders ({len(orders)}):")
    for o in orders:
        side = "BUY" if o.get("side") == "B" else "SELL"
        print(f"  {o.get('coin', '?')} {side} {o.get('sz', '?')} @ {o.get('limitPx', '?')} | OID: {o.get('oid', '?')}")
else:
    print("\nNo open orders")
PYEOF
```

### Step 2: Check Running Bot Screens

```bash
screen -ls 2>&1
```

For each running screen, show the last few log lines to indicate bot state:

- **HYPE bot**: `tail -5 /tmp/hype-bot.log 2>/dev/null`
- **SOL bot**: `tail -5 /tmp/sol-ms-bot.log 2>/dev/null` or `tail -5 /tmp/sol_ms_bot.log 2>/dev/null`

### Step 3: Historical Account Value (if requested)

If the user asks about account value at a prior time (e.g., "24 hours ago", "start of day", "yesterday"), calculate it by working backwards from the current value using realized P&L from fills.

**Formula:** `account_value_then = current_value - net_realized_pnl_since_then`

Where `net_realized_pnl = sum(closedPnl) - sum(fee)` for all fills in the period.

Determine the lookback period from the user's question:
- "start of day" / "today" → midnight EST (05:00 UTC) of current day
- "24 hours ago" → now minus 24 hours
- "yesterday" → midnight EST yesterday to midnight EST today
- "this week" → Monday 00:00 EST

Then fetch fills and compute:

```bash
uv run --package hyperliquid-trading-bot python3 << 'PYEOF'
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os

load_dotenv(Path('trading-agent/.env'), override=True)
from hyperliquid.info import Info

info = Info('https://api.hyperliquid.xyz', skip_ws=True)
wallet = os.environ.get('MAINNET_ACCOUNT_ADDRESS')

# Current account value
state = info.user_state(wallet)
current_value = float(state["marginSummary"]["accountValue"])

# Set the lookback start time (adjust per user request)
# Examples:
#   24 hours ago:
#     start = datetime.now(timezone.utc) - timedelta(hours=24)
#   Start of day (midnight EST):
#     est = timezone(timedelta(hours=-5))
#     now_est = datetime.now(est)
#     start = now_est.replace(hour=0, minute=0, second=0, microsecond=0)
est = timezone(timedelta(hours=-5))
now_est = datetime.now(est)
start = now_est.replace(hour=0, minute=0, second=0, microsecond=0)  # start of day EST
start_ms = int(start.timestamp() * 1000)

fills = info.user_fills_by_time(wallet, start_ms)

total_closed_pnl = sum(float(f.get("closedPnl", "0")) for f in fills)
total_fees = sum(float(f.get("fee", "0")) for f in fills)
net_realized = total_closed_pnl - total_fees

# Approximate past value
past_value = current_value - net_realized

# Per-coin breakdown
coins = {}
for f in fills:
    coin = f.get("coin", "?")
    if coin not in coins:
        coins[coin] = {"pnl": 0.0, "fees": 0.0, "fills": 0}
    coins[coin]["pnl"] += float(f.get("closedPnl", "0"))
    coins[coin]["fees"] += float(f.get("fee", "0"))
    coins[coin]["fills"] += 1

label = start.strftime("%b %d %I:%M %p EST")
print(f"Period: {label} -> now")
print(f"Account value then:  ${past_value:,.2f}")
print(f"Account value now:   ${current_value:,.2f}")
print(f"Change:              ${net_realized:+,.2f}")
print(f"  Realized PnL:      ${total_closed_pnl:+,.2f}")
print(f"  Fees paid:         ${total_fees:,.2f}")
print(f"  Total fills:       {len(fills)}")
print()
if coins:
    print("Breakdown by coin:")
    for coin, data in sorted(coins.items()):
        net = data["pnl"] - data["fees"]
        print(f"  {coin}: ${net:+,.2f} net ({data['fills']} fills, ${data['fees']:,.2f} fees)")
PYEOF
```

### Step 4: Present Summary

Present the information as a clean summary:

**Account Overview:**
| Field | Value |
|-------|-------|
| Account Value | $X.XX |
| Margin Used | $X.XX |
| Available | $X.XX |

**Open Positions** (if any): table with coin, side, size, entry, uPnL, ROE%, leverage

**Running Bots**: list active screen sessions with their latest status (price, PnL, trade count)

**Open Orders** (if any): backstop SL/TP orders on the exchange

**Historical** (if requested):
| Period | Start Value | Current Value | Change |
|--------|-------------|---------------|--------|
| Today (since midnight EST) | $X.XX | $X.XX | +/- $X.XX |

With per-coin breakdown showing realized PnL and fees.

**Note:** Historical value is approximate — calculated as `current_value - net_realized_pnl`. This does not account for deposits, withdrawals, or funding payments in the period. If these occurred, the estimate will be off.
