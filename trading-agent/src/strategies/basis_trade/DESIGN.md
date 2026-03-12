# Basis Trade Strategy — Design Document

## Trade Idea

**Category: Cash-and-carry ARBITRAGE (delta-neutral)**

A basis trade simultaneously holds a long spot position and a short perpetual futures position on the same asset (HYPE). Since the two legs move in opposite directions, the trade is **delta-neutral** — it profits not from price direction, but from **funding rate payments**.

On Hyperliquid, funding is paid hourly. When funding is positive (the common case in bullish markets), **short perp holders receive payments from long holders**. By holding an offsetting spot position, we eliminate directional risk and collect this yield.

### Why This Works

Perpetual futures have no expiry date, so they use funding rates to keep the perp price anchored to the spot price. When sentiment is bullish:
- Traders pile into long perps → perp price trades above spot
- Funding rate goes positive → longs pay shorts
- This incentivizes shorts and reduces longs, pulling the perp price back

The basis trade systematically captures this structural premium.

### Trade Mechanics

```
Entry:
  1. Buy X HYPE spot  (long exposure)
  2. Short X HYPE perp (short exposure, equal notional)
  → Net delta = 0

Ongoing:
  - Every hour, receive funding payment = rate × notional (when rate > 0)
  - Price moves cancel out: spot +$1, perp -$1 → net $0

Exit:
  1. Sell X HYPE spot
  2. Close X HYPE perp short
  → Realize: cumulative funding - total fees
```

### Not Speculation, Not Hedging

| Category | Description | This Trade? |
|----------|-------------|-------------|
| **Speculation** | Directional bet on price | No — delta-neutral |
| **Hedging** | Protecting an existing exposure | No — no pre-existing position |
| **Arbitrage** | Exploiting a structural price differential | **Yes** — captures funding rate premium between spot and perp markets |

This is specifically **cash-and-carry arbitrage**, a well-established strategy in traditional finance (Treasury basis trade, crypto basis trade).

## Target Profit

### At Current HYPE Funding Rates

| Metric | Anchor Rate | Elevated Rate |
|--------|------------|---------------|
| Hourly funding | 0.00125%/h | 0.05%/h |
| Daily rate | 0.03%/day | 1.2%/day |
| Annualized | ~11% APR | ~438% APR |

### Projected Returns ($3 Capital)

With $3 capital in unified mode, max per-leg notional = $2.73 (at 10x leverage):

| Scenario | Daily Funding | Monthly | Break-even |
|----------|--------------|---------|------------|
| Anchor (0.01%/8h) | $0.008 | $0.24 | ~5 days |
| Moderate (0.03%/8h) | $0.025 | $0.74 | ~2 days |
| Elevated (0.1%/8h) | $0.082 | $2.46 | ~12 hours |

**Break-even** = time to recoup round-trip fees (~0.14% on both legs).

### Scaling Projections

| Capital | Per-Leg | Daily (anchor) | Monthly (anchor) | Monthly (elevated) |
|---------|---------|----------------|------------------|--------------------|
| $3 | $2.73 | $0.008 | $0.24 | $2.46 |
| $50 | $45.45 | $0.14 | $4.09 | $40.91 |
| $500 | $454.55 | $1.36 | $40.91 | $409.09 |
| $5,000 | $4,545 | $13.64 | $409.09 | $4,090.91 |

Returns scale linearly with capital — no diminishing returns until position size impacts market.

## Risk Analysis

### Risk Sources

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Negative funding** | Medium | Exit after N consecutive negative hours |
| **Execution slippage** | Low | Use limit orders, stagger entry |
| **Leg risk** (one leg fills, other doesn't) | Medium | Use IOC orders, small size |
| **Spot/perp basis divergence** | Low | Minimal at small size |
| **Exchange risk** | Low-Medium | Sub-account isolation, small allocation |
| **Liquidation** (perp leg) | Low | Delta-neutral + moderate leverage |

### Key Risk: Negative Funding

Funding rates aren't always positive. During bearish periods, shorts pay longs. The strategy handles this with:

1. **Entry gate**: Only enter when funding has been positive for N consecutive hours
2. **Exit gate**: Close when funding is negative for N consecutive hours
3. **Max loss cap**: Hard stop on cumulative negative funding

### What This Strategy Does NOT Risk

- **Directional loss**: Price can move 50% in either direction — net P&L from price = $0
- **Liquidation from price moves**: Both legs offset, so margin usage stays stable
- **Impermanent loss**: This is not an LP position

## Architecture

```
strategies/basis_trade/
├── config.py            # BasisTradeConfig — all tunable parameters
├── funding_monitor.py   # FundingMonitor — tracks rates, streaks, payments
├── basis_strategy.py    # BasisTradeStrategy — state machine + decisions
└── basis_bridge.py      # Live bridge — executes on sub-account via vault_address
```

### State Machine

```
IDLE ──(funding attractive for N hours)──→ ENTERING
ENTERING ──(both legs filled)──→ ACTIVE
ACTIVE ──(funding negative / target hit / max time)──→ EXITING
EXITING ──(both legs closed)──→ IDLE
Any ──stop()──→ STOPPED
```

### Sub-Account Isolation

The basis trade runs on the "Basis trade" sub-account (`0x969b...`), completely isolated from the DC Adaptive bot on the master account. The bridge uses `vault_address` in the SDK to route orders through the master wallet's API key to the sub-account.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | HYPE | Asset to trade |
| `spot_pair` | @107 | HYPE/USDC spot pair on Hyperliquid |
| `position_size_usd` | 3.0 | Total capital allocated |
| `leverage` | 10 | Perp leg leverage |
| `min_funding_rate` | 0.0001 | Min hourly rate to enter (0.01%/h ≈ 88% APR) |
| `min_funding_hours` | 3 | Consecutive hours above min to enter |
| `exit_funding_rate` | -0.00005 | Exit threshold (slightly negative) |
| `exit_funding_hours` | 6 | Consecutive hours below exit to close |
| `slippage_tolerance` | 0.002 | Max slippage for market orders |

## Testing

```bash
# Run all basis trade tests (52 tests)
uv run --package hyperliquid-trading-bot pytest trading-agent/tests/strategies/basis_trade/ -v
```
