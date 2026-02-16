"""
Test script for modifying a spot order to see how it appears in WebSocket.
Finds an open spot order and modifies its price or size.
"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.signing import ModifyRequest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from price_and_size_util import format_order

load_dotenv()

BASE_URL = os.getenv("HYPERLIQUID_TESTNET_PUBLIC_BASE_URL")

def build_token_index(spot_meta: dict) -> dict[str, dict]:
    return {
        token["name"]: token
        for token in spot_meta.get("tokens", [])
    }


async def modify_spot_order():
    """Modify a spot order"""
    print("Modify Spot Order Test")
    print("=" * 40)

    private_key = os.getenv("HYPERLIQUID_TESTNET_PRIVATE_KEY")
    if not private_key:
        print("❌ Missing HYPERLIQUID_TESTNET_PRIVATE_KEY in .env file")
        return

    try:
        wallet = Account.from_key(private_key)
        exchange = Exchange(wallet, BASE_URL)
        info = Info(BASE_URL, skip_ws=True)
        spot_data = info.spot_meta_and_asset_ctxs()
        spot_meta = spot_data[0]
        # O(1) lookup of token by name
        token_index = build_token_index(spot_meta)

        print(f"📱 Wallet: {wallet.address}")

        # Get open orders using environment wallet address
        wallet_address = os.getenv("TESTNET_WALLET_ADDRESS") or wallet.address
        open_orders = info.open_orders(wallet_address)
        print(f"📋 Found {len(open_orders)} open orders")

        if not open_orders:
            print("❌ No open orders to modify")
            print("💡 Run place_order.py first to create an order")
            return

        # Find the first spot order
        spot_order = None
        for order in open_orders:
            coin = order.get("coin", "")
            if coin.startswith("@") or "/" in coin:  # Spot order indicators
                spot_order = order
                break

        if not spot_order:
            print("❌ No spot orders found to modify")
            print("💡 Only perpetual orders are open")
            return

        order_id = spot_order.get("oid")
        coin_field = spot_order.get("coin")
        base_token = coin_field.split("/")[0]
        side = "BUY" if spot_order.get("side") == "B" else "SELL"
        current_size = float(spot_order.get("sz", 0))
        current_price = float(spot_order.get("limitPx", 0))
        sz_decimals = token_index[base_token].get("szDecimals", 6)

        print(f"🎯 Found spot order to modify:")
        print(f"   ID: {order_id}")
        print(f"   Current: {side} {current_size} {coin_field} @ ${current_price}")

        # Calculate new values
        price_modifier = (
            0.97 if side == "BUY" else 1.01
        )  # Make buy orders cheaper, sell orders more expensive
        new_price = round(current_price * price_modifier, 6)
        _, new_price = format_order(current_size, new_price, sz_decimals=sz_decimals, is_spot=True, is_buy=side == "BUY")

        print(f"   New: {side} {current_size} {coin_field} @ ${new_price}")

        # Create modify request
        print(f"🔄 Modifying order {order_id}...")
        result = exchange.modify_order(
            oid=order_id,
            name=coin_field,
            is_buy=(side == "BUY"),
            sz=current_size,
            limit_px=new_price,
            order_type={"limit": {"tif": "Gtc"}},
            reduce_only=False,
        )

        print(f"📋 Modify result:")
        print(json.dumps(result, indent=2))

        statuses = (
            result
            .get("response", {})
            .get("data", {})
            .get("statuses", [])
        )
        has_error = any("error" in s for s in statuses)

        if result and result.get("status") == "ok" and not has_error:
            print(f"✅ Order {order_id} modified successfully!")
            print(f"🔍 Monitor this modification in your WebSocket stream")
        else:
            print(f"❌ Modify failed: {result}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(modify_spot_order())
