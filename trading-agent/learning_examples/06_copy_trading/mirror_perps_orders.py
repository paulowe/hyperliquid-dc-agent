"""
Mirror perps orders from a leader wallet with fixed sizing.
Monitors leader's perp orders and places corresponding orders for follower.

NOTE:
- hyperliquid Exchange/Info methods are synchronous; treat them as blocking.
"""

import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass
from typing import Dict, Optional, Literal

from dotenv import load_dotenv
import websockets
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.signing import OrderType as HLOrderType

load_dotenv()

# Configuration
WS_URL = os.getenv("HYPERLIQUID_TESTNET_PUBLIC_WS_URL")
BASE_URL = os.getenv("HYPERLIQUID_TESTNET_PUBLIC_BASE_URL")

# For tests, you can use the same wallet as a leader and follower.
# Follower's orders will be ignored in the mirroring logic.
LEADER_ADDRESS = os.getenv("TESTNET_WALLET_ADDRESS")
FIXED_ORDER_VALUE_USDC = 15.0
FOLLOWER_LEVERAGE = int(os.getenv("FOLLOWER_LEVERAGE", "1"))

running = False
order_mappings: Dict[int, int] = {}  # leader_order_id -> follower_order_id

# ----------------------------
# Utilities / validation
# ----------------------------

def signal_handler(signum, frame, stop_event: asyncio.Event):
    """Handle Ctrl+C gracefully"""
    del signum, frame  # Unused parameters
    print("\nShutting down...")
    stop_event.set()


def detect_market_type(coin_field):
    """Detect market type from coin field"""
    if coin_field.startswith("@"):
        return "SPOT"
    elif "/" in coin_field:
        return "SPOT"
    else:
        return "PERP"


def is_perp_order(coin_field: str) -> bool:
    """Check if order is for perpetual trading - basic format validation only"""
    if not coin_field or coin_field == "N/A":
        return False

    market_type = detect_market_type(coin_field)
    if market_type != "PERP":
        return False

    # Perp orders are direct symbols (BTC, ETH, SOL, etc.)
    # Reject if it starts with @ (spot index) or contains / (spot pair)
    if coin_field.startswith("@"):
        return False
    if "/" in coin_field:
        return False

    return True


# ----------------------------
# Perp metadata cache
# ----------------------------

@dataclass
class PerpCache:
    # Cache perp meta and a few derived maps to avoid repeated meta calls
    # These are loaded once and reused.
    meta: Optional[dict] = None
    # symbol (e.g. "BTC") -> szDecimals (int)
    symbol_to_szdec: Optional[Dict[str, int]] = None

perp_cache = PerpCache()


async def load_perp_cache(info: Info) -> None:
    """
    Slow-ish (HTTP) but called once at startup.
    Uses asyncio.to_thread because info.meta is synchronous.
    """
    global perp_cache
    if perp_cache.meta is not None:
        return

    def _fetch():
        return info.meta()

    meta = await asyncio.to_thread(_fetch)
    if not isinstance(meta, dict):
        raise RuntimeError("Unexpected meta response")

    universe = meta.get("universe", []) or []
    symbol_to_szdec: Dict[str, int] = {}

    for asset in universe:
        name = asset.get("name")
        sz_dec = asset.get("szDecimals")
        if isinstance(name, str) and isinstance(sz_dec, int):
            symbol_to_szdec[name] = sz_dec

    perp_cache.meta = meta
    perp_cache.symbol_to_szdec = symbol_to_szdec


async def get_perp_price_and_szdec(info: Info, coin_field: str) -> Optional[dict]:
    """
    Returns: {"price": float, "szDecimals": int, "coin": "BTC"}
    Price comes from info.all_mids().
    szDecimals comes from cached perp meta.
    """
    if not is_perp_order(coin_field):
        return None

    # Ensure cache is loaded
    if not perp_cache.symbol_to_szdec:
        await load_perp_cache(info)

    # Fetch latest price from all_mids (this is HTTP, so run in thread)
    def _fetch_prices():
        return info.all_mids()

    all_prices = await asyncio.to_thread(_fetch_prices)
    if not isinstance(all_prices, dict):
        return None

    price_str = all_prices.get(coin_field)
    if not price_str:
        print(f"⚠️ Perp symbol {coin_field} not found in price feed")
        return None

    try:
        price = float(price_str)
    except (ValueError, TypeError):
        print(f"⚠️ Invalid price for {coin_field}: {price_str}")
        return None

    if price <= 0:
        print(f"⚠️ No perp price for {coin_field} (price={price})")
        return None

    sz_dec = perp_cache.symbol_to_szdec.get(coin_field, 4)  # type: ignore[union-attr]
    if sz_dec is None:
        print(f"⚠️ Perp symbol {coin_field} not found in metadata")
        return None

    return {"price": price, "szDecimals": sz_dec, "coin": coin_field}


# ----------------------------
# Action model (fast path output)
# ----------------------------

ActionType = Literal["PLACE", "CANCEL"]

@dataclass(frozen=True)
class Action:
    type: ActionType
    leader_oid: int
    coin: str
    side: Optional[str] = None   # "B"/"A" for PLACE
    limit_px: Optional[float] = None


# ----------------------------
# Slow path execution (thread-offloaded)
# ----------------------------

async def execute_place(exchange: Exchange, info: Info, action: Action) -> Optional[int]:
    """
    Places follower order for a leader open event.
    Runs blocking Exchange.order in a thread to avoid blocking the event loop.
    Sets leverage from config before placing order.
    """
    assert action.type == "PLACE"
    assert action.side in ("B", "A")
    assert action.limit_px is not None

    asset_info = await get_perp_price_and_szdec(info, action.coin)
    if not asset_info:
        print(f"❌ Could not get asset info for {action.coin}")
        return None

    price = float(action.limit_px)
    if price <= 0:
        print(f"❌ Invalid price in action: {price}")
        return None

    sz_dec = int(asset_info["szDecimals"])
    order_size = round(FIXED_ORDER_VALUE_USDC / price, sz_dec)
    if order_size <= 0:
        print(f"❌ Invalid order size calculated for {action.coin}")
        return None

    # Set leverage from config before placing order
    if FOLLOWER_LEVERAGE > 0:
        def _update_leverage():
            return exchange.update_leverage(
                leverage=FOLLOWER_LEVERAGE,
                name=action.coin,
                is_cross=True,
            )

        try:
            leverage_result = await asyncio.to_thread(_update_leverage)
            if leverage_result and leverage_result.get("status") == "ok":
                print(f"⚙️ Set leverage to {FOLLOWER_LEVERAGE}x for {action.coin}")
            else:
                print(f"⚠️ Failed to set leverage: {leverage_result}")
        except Exception as e:
            print(f"⚠️ Error setting leverage: {e}")

    is_buy = action.side == "B"
    leverage_str = f" ({FOLLOWER_LEVERAGE}x)" if FOLLOWER_LEVERAGE > 0 else ""
    print(f"🔄 Placing follower order: {'BUY' if is_buy else 'SELL'} {order_size} {action.coin} @ ${price}{leverage_str}")

    def _place_sync():
        return exchange.order(
            name=action.coin,
            is_buy=is_buy,
            sz=order_size,
            limit_px=price,
            order_type=HLOrderType({"limit": {"tif": "Gtc"}}),
            reduce_only=False,
        )

    result = await asyncio.to_thread(_place_sync)

    if result and result.get("status") == "ok":
        statuses = result.get("response", {}).get("data", {}).get("statuses", []) or []
        if statuses:
            s0 = statuses[0]
            if "resting" in s0:
                oid = s0["resting"]["oid"]
                print(f"✅ Follower order placed! ID: {oid}")
                return oid
            if "filled" in s0:
                oid = s0["filled"]["oid"]
                print(f"✅ Follower order filled immediately! ID: {oid}")
                return oid

    print(f"❌ Failed to place follower order: {result}")
    return None


async def execute_cancel(exchange: Exchange, action: Action, follower_oid: int) -> bool:
    """
    Cancels follower order.
    Runs blocking Exchange.cancel in a thread to avoid blocking the event loop.
    """
    assert action.type == "CANCEL"

    print(f"🔄 Cancelling follower order ID: {follower_oid}")

    def _cancel_sync():
        return exchange.cancel(name=action.coin, oid=follower_oid)

    result = await asyncio.to_thread(_cancel_sync)

    if result and result.get("status") == "ok":
        print("✅ Follower order cancelled successfully")
        return True

    print(f"❌ Failed to cancel follower order: {result}")
    return False


# ----------------------------
# Fast path parsing (no network)
# ----------------------------

def extract_actions_from_message(data: dict) -> list[Action]:
    """
    FAST PATH: only parse/filter. Do NOT call exchange/info here.

    Produces a list of Actions to execute in order.
    """
    actions: list[Action] = []
    channel = data.get("channel")

    if channel == "subscriptionResponse":
        print("✅ WebSocket subscription confirmed")
        return actions

    # NOTE: Hyperliquid channel names can differ. You subscribed to:
    # - orderUpdates
    # - userEvents
    # Some payloads come in as "orderUpdates" and "user" or "userEvents".
    # We handle both.
    if channel == "orderUpdates":
        orders = data.get("data", []) or []
        for order_update in orders:
            order = order_update.get("order", {}) or {}
            status = order_update.get("status", "unknown")
            coin_field = order.get("coin", "")
            if not is_perp_order(coin_field):
                continue

            leader_oid = order.get("oid")
            if not isinstance(leader_oid, int):
                continue

            # Skip follower orders (if using same wallet)
            if leader_oid in order_mappings.values():
                continue

            side = order.get("side")
            limit_px = order.get("limitPx")

            # Skip filled orders - we get better fill info from user channel
            if status != "filled":
                print(
                    f"LEADER ORDER {str(status).upper()}: {side} {order.get('sz')} "
                    f"{coin_field} @ {limit_px} (ID: {leader_oid})"
                )

            if status == "open":
                try:
                    if limit_px is None:
                        continue
                    limit_px_float = float(limit_px)
                    actions.append(
                        Action(
                            type="PLACE",
                            leader_oid=leader_oid,
                            coin=coin_field,
                            side=str(side),
                            limit_px=limit_px_float,
                        )
                    )
                except (ValueError, TypeError):
                    continue

            elif status == "canceled":
                actions.append(Action(type="CANCEL", leader_oid=leader_oid, coin=coin_field))

    elif channel in ("user", "userEvents"):
        # Fills are just logged (fast path)
        user_data = data.get("data", {}) or {}
        fills = user_data.get("fills", []) or []
        for fill in fills:
            coin_field = fill.get("coin", "N/A")
            if not is_perp_order(coin_field):
                continue

            fill_oid = fill.get("oid")
            # Skip follower fills if they correspond to mapped follower oids
            if fill_oid and fill_oid in order_mappings.values():
                continue

            side = "BUY" if fill.get("side") == "B" else "SELL"
            print(f"LEADER FILL: {side} {fill.get('sz')} {coin_field} @ {fill.get('px')}")

    return actions


# ----------------------------
# Main loop
# ----------------------------

async def monitor_and_mirror_perp_orders():
    """Connect to WebSocket and monitor leader's perp order activity."""
    global running, order_mappings

    private_key = os.getenv("HYPERLIQUID_TESTNET_PRIVATE_KEY")
    if not private_key:
        print("❌ Missing HYPERLIQUID_TESTNET_PRIVATE_KEY in .env file")
        return

    # Initialize follower trading components
    try:
        wallet = Account.from_key(private_key)
        exchange = Exchange(wallet, BASE_URL)
        info = Info(BASE_URL, skip_ws=True)
        print(f"✅ Follower wallet initialized: {wallet.address}")
    except Exception as e:
        print(f"❌ Failed to initialize follower wallet: {e}")
        return

    # Preload perp meta cache once (helps latency later)
    try:
        await load_perp_cache(info)
    except Exception as e:
        print(f"⚠️ Could not preload perp cache (will retry lazily): {e}")

    print(f"🔗 Connecting to {WS_URL}")
    # Use an explicit stop event for clean shutdown + reconnect loop exit
    stop_event = asyncio.Event()

    def _sigint_handler(signum, frame):
        del signum, frame
        print("\nShutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    reconnect_delay_s = 2
    attempt = 0

    while not stop_event.is_set():
        attempt += 1
        print(f"🔄 Reconnecting (attempt {attempt})...")

        # Bounded queues: if you fall behind, you'll know (queue grows / drops)
        message_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=2000)   # raw WS messages
        action_queue: asyncio.Queue[Action] = asyncio.Queue(maxsize=2000) # parsed actions

        # Basic metrics
        recv_count = 0
        last_metrics_time = time.time()

        running = True

        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            ) as websocket:
                print("✅ WebSocket connected!")

                # Subscribe each time reconnect
                await websocket.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {"type": "orderUpdates", "user": LEADER_ADDRESS},
                }))
                await websocket.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {"type": "userEvents", "user": LEADER_ADDRESS},
                }))

                print(f"📊 Monitoring PERPS orders for leader: {LEADER_ADDRESS}")
                print(f"💰 Fixed order value: ${FIXED_ORDER_VALUE_USDC} USDC per order")
                print(f"⚙️ Follower leverage: {FOLLOWER_LEVERAGE}x")
                print(f"👤 Follower wallet: {wallet.address}")
                print("=" * 80)

                # --- Task 0: Hyperliquid heartbeat (JSON ping)
                async def hl_heartbeat():
                    while running and not stop_event.is_set():
                        try:
                            await websocket.send(json.dumps({"method": "ping"}))
                        except Exception as e:
                            print(f"💔 Heartbeat send failed: {e}")
                            break
                        await asyncio.sleep(25)  # < 60s server idle timeout

                # --- Task 1: WS receive (FAST PATH)
                async def message_receiver():
                    nonlocal recv_count, last_metrics_time
                    global running
                    try:
                        async for message in websocket:
                            if not running or stop_event.is_set():
                                break

                            # Drop pong frames
                            try:
                                d = json.loads(message)
                                if d.get("channel") == "pong":
                                    # print("🏓 pong")
                                    continue
                            except Exception:
                                pass

                            await message_queue.put(message)
                            recv_count += 1

                            now = time.time()
                            if now - last_metrics_time >= 2.0:
                                last_metrics_time = now
                                print(f"📥 recv={recv_count} mq={message_queue.qsize()} aq={action_queue.qsize()} maps={len(order_mappings)}")
                    except Exception as e:
                        print(f"❌ Receiver error: {e}")
                    finally:
                        # IMPORTANT: if receiver stops, everything should stop (or reconnect)
                        running = False
                        print("🔌 Receiver ended (WS likely closed).")

                # --- Task 2: parse messages into actions (FAST PATH)
                async def message_parser():
                    while running and not stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(message_queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                        try:
                            data = json.loads(raw)
                            actions = extract_actions_from_message(data)
                            for act in actions:
                                # backpressure if action queue is full
                                await action_queue.put(act)
                        except json.JSONDecodeError:
                            print("⚠️ Received invalid JSON")
                        except Exception as e:
                            print(f"❌ Parse error: {e}")
                        finally:
                            message_queue.task_done()

                # --- Task 3: execute actions sequentially (SLOW PATH, thread-offloaded)
                async def action_executor():
                    while running and not stop_event.is_set():
                        try:
                            act = await asyncio.wait_for(action_queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                        try:
                            if act.type == "PLACE":
                                follower_oid = await execute_place(exchange, info, act)
                                if follower_oid:
                                    order_mappings[act.leader_oid] = follower_oid
                                    print(f"Mapped {act.leader_oid} -> {follower_oid}")

                            elif act.type == "CANCEL":
                                follower_oid = order_mappings.get(act.leader_oid)
                                if follower_oid:
                                    await execute_cancel(exchange, act, follower_oid)
                                    order_mappings.pop(act.leader_oid, None)

                        except Exception as e:
                            print(f"❌ Execution error: {e}")
                        finally:
                            action_queue.task_done()

                await asyncio.gather(
                    hl_heartbeat(),
                    message_receiver(),
                    message_parser(),
                    action_executor(),
                )

        except websockets.exceptions.ConnectionClosed as e:
            print(f"🔌 WebSocket connection closed: {e}")
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
        finally:
            running = False
            print("👋 Disconnected")
            print(f"📊 Current order mappings: {len(order_mappings)} active")
        
        # Exit loop if user requested shutdown
        if stop_event.is_set():
            print("👋 Shutdown requested")
            break

        # Sleep for reconnect delay
        await asyncio.sleep(reconnect_delay_s)

    print("🛑 Stopped reconnect loop. Bye.")

async def main():
    print("Hyperliquid Perp Order Mirror")
    print("=" * 40)

    if not WS_URL or not BASE_URL:
        print("❌ Missing required environment variables:")
        print("   HYPERLIQUID_TESTNET_PUBLIC_WS_URL")
        print("   HYPERLIQUID_TESTNET_PUBLIC_BASE_URL")
        return

    if not LEADER_ADDRESS or LEADER_ADDRESS == "0x...":
        print("❌ Please set LEADER_ADDRESS in the script")
        return

    await monitor_and_mirror_perp_orders()


if __name__ == "__main__":
    asyncio.run(main())
