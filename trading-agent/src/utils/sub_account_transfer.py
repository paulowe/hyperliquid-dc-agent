"""Transfer funds between master account and sub-account on Hyperliquid.

Usage:
    # Transfer $5 to sub-account "THE SIX"
    uv run --package hyperliquid-trading-bot python -m utils.sub_account_transfer \
        --to-sub 0x969b8f653dbd7168931a6ab5478195851d2ef021 --amount 5

    # Withdraw $3 from sub-account back to master
    uv run --package hyperliquid-trading-bot python -m utils.sub_account_transfer \
        --from-sub 0x969b8f653dbd7168931a6ab5478195851d2ef021 --amount 3

    # Check sub-account balances
    uv run --package hyperliquid-trading-bot python -m utils.sub_account_transfer --status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
_SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC_DIR))

from dotenv import load_dotenv
load_dotenv(_SRC_DIR.parent / ".env", override=True)


def get_exchange():
    """Create Exchange instance from the master wallet (signs sub-account txns)."""
    from hyperliquid.exchange import Exchange
    from eth_account import Account

    private_key = os.environ.get("HYPERLIQUID_MAINNET_PRIVATE_KEY")
    if not private_key:
        print("ERROR: HYPERLIQUID_MAINNET_PRIVATE_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    wallet = Account.from_key(private_key)
    base_url = "https://api.hyperliquid.xyz"
    return Exchange(wallet, base_url)


def get_info():
    """Create Info instance for read-only queries."""
    from hyperliquid.info import Info
    return Info("https://api.hyperliquid.xyz", skip_ws=True)


def show_status():
    """Show master and all sub-account balances."""
    info = get_info()
    main_wallet = os.environ.get("MAINNET_ACCOUNT_ADDRESS", "")

    # List sub-accounts
    subs = info.query_sub_accounts(main_wallet)
    if not subs:
        print("No sub-accounts found.")
        return

    # Master account balance
    master_state = info.user_state(main_wallet)
    master_val = float(master_state["marginSummary"]["accountValue"])
    master_avail = float(master_state["withdrawable"])
    print(f"Master account ({main_wallet[:10]}...): ${master_val:,.2f} (${master_avail:,.2f} available)")
    print()

    # Sub-account balances
    for sub in subs:
        name = sub.get("name", "?")
        addr = sub.get("subAccountUser", "")
        try:
            sub_state = info.user_state(addr)
            sub_val = float(sub_state["marginSummary"]["accountValue"])
            sub_avail = float(sub_state["withdrawable"])
            print(f"  [{name}] ({addr[:10]}...): ${sub_val:,.2f} (${sub_avail:,.2f} available)")

            # Show positions if any
            for pos in sub_state.get("assetPositions", []):
                p = pos["position"]
                sz = float(p["szi"])
                if sz != 0:
                    coin = p["coin"]
                    entry = float(p["entryPx"])
                    upnl = float(p["unrealizedPnl"])
                    side = "LONG" if sz > 0 else "SHORT"
                    print(f"    {coin} {side} {abs(sz)} @ {entry:.4f} | uPnL: ${upnl:,.2f}")
        except Exception as e:
            print(f"  [{name}] ({addr[:10]}...): error querying: {e}")


def transfer(sub_address: str, amount_usd: float, is_deposit: bool):
    """Transfer USD between master and sub-account.

    Args:
        sub_address: Sub-account L1 address
        amount_usd: Amount in USD (e.g., 5.0 for $5)
        is_deposit: True = master → sub, False = sub → master
    """
    exchange = get_exchange()

    # Convert to micro-USD (integer)
    usd_micro = int(amount_usd * 1_000_000)

    direction = "master → sub" if is_deposit else "sub → master"
    print(f"Transferring ${amount_usd:.2f} ({direction})")
    print(f"  Sub-account: {sub_address}")
    print(f"  Amount (micro-USD): {usd_micro}")

    result = exchange.sub_account_transfer(
        sub_account_user=sub_address,
        is_deposit=is_deposit,
        usd=usd_micro,
    )

    print(f"  Result: {json.dumps(result, indent=2)}")

    status = result.get("status", "")
    if status == "ok":
        print(f"\nTransfer successful!")
    else:
        print(f"\nTransfer may have failed — check result above.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Transfer funds to/from Hyperliquid sub-accounts")
    parser.add_argument("--to-sub", type=str, help="Sub-account address to deposit INTO")
    parser.add_argument("--from-sub", type=str, help="Sub-account address to withdraw FROM")
    parser.add_argument("--amount", type=float, help="Amount in USD")
    parser.add_argument("--status", action="store_true", help="Show balances for all sub-accounts")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.to_sub and args.from_sub:
        print("ERROR: Use either --to-sub or --from-sub, not both", file=sys.stderr)
        sys.exit(1)

    if not args.to_sub and not args.from_sub:
        print("ERROR: Specify --to-sub or --from-sub (or --status to check balances)", file=sys.stderr)
        sys.exit(1)

    if not args.amount or args.amount <= 0:
        print("ERROR: --amount must be a positive number", file=sys.stderr)
        sys.exit(1)

    if args.to_sub:
        transfer(args.to_sub, args.amount, is_deposit=True)
    else:
        transfer(args.from_sub, args.amount, is_deposit=False)


if __name__ == "__main__":
    main()
