---
description: Transfer funds between master account and sub-accounts on Hyperliquid
user_invocable: true
---

# Sub-Account Transfer

Transfer USD between the master Hyperliquid account and sub-accounts, or check sub-account balances.

## How to Use

- `/sub-transfer status` — show balances for master and all sub-accounts
- `/sub-transfer $5 to sub` — transfer $5 from master to sub-account
- `/sub-transfer $3 from sub` — withdraw $3 from sub-account to master
- `/sub-transfer $5 to sub 0xABC...` — transfer to a specific sub-account address

## Workflow

### Step 1: Determine the Action

Parse the user's message to determine:
- **status**: show balances only
- **deposit**: transfer from master → sub-account (`is_deposit=True`)
- **withdraw**: transfer from sub-account → master (`is_deposit=False`)
- **amount**: dollar amount to transfer
- **sub-account address**: defaults to `SUB_ACCOUNT_ADDRESS` from `.env` (`0x969b8f653dbd7168931a6ab5478195851d2ef021`), or use a specific address if provided

### Step 2: Check Balances (always)

Always show current balances before any transfer:

```bash
uv run --package hyperliquid-trading-bot python -m utils.sub_account_transfer --status
```

### Step 3: Execute Transfer (if requested)

**Transfer to sub-account:**
```bash
uv run --package hyperliquid-trading-bot python -m utils.sub_account_transfer \
  --to-sub <SUB_ADDRESS> --amount <AMOUNT>
```

**Withdraw from sub-account:**
```bash
uv run --package hyperliquid-trading-bot python -m utils.sub_account_transfer \
  --from-sub <SUB_ADDRESS> --amount <AMOUNT>
```

Default sub-account address: `0x969b8f653dbd7168931a6ab5478195851d2ef021` ("Basis trade")

### Step 4: Verify

After any transfer, run `--status` again to confirm the new balances.

### Step 5: Present Summary

Show a clean before/after summary:

| Account | Before | After |
|---------|--------|-------|
| Master | $X.XX | $Y.YY |
| Basis trade (sub) | $X.XX | $Y.YY |

## Safety Checks

- Confirm the transfer amount doesn't exceed the source account's available balance
- If the user requests more than 50% of the master account's available balance, warn them before proceeding
- Always verify balances after the transfer

## Reference

- Sub-accounts don't have private keys — the master account's API wallet signs all transactions
- `usd` parameter in the SDK is in **micro-USD** (1 USD = 1,000,000) — the CLI script handles conversion
- Sub-account address: `0x969b8f653dbd7168931a6ab5478195851d2ef021` (env var: `SUB_ACCOUNT_ADDRESS`)
- Master account: `0xFfe839343E3c51F04501392b0430CD6cE940597C` (env var: `MAINNET_ACCOUNT_ADDRESS`)
