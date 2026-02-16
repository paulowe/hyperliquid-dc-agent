"""
Hyperliquid Spot + Perp Order Mirror with Resync

This script mirrors both SPOT and PERP orders from a leader wallet to a follower
wallet on Hyperliquid. It:

- Listens to the leader's orderUpdates and userEvents over WebSocket.
- Mirrors "open" orders as follower orders with fixed USDC notional.
- Cancels follower orders when the corresponding leader order is canceled.
- Handles both SPOT and PERP using a unified flow and market-type detection.
- Applies leverage ONLY for PERP markets.
- Uses a client order ID (cloid) convention to survive process/VM crashes and
  to resync follower state with leader state after reconnects.

Resync strategy:
- Follower orders have cloid = "copy_{leader_oid}".
- On startup and before every WS reconnect:
  - Fetch leader open orders.
  - Fetch follower open orders.
  - Cancel follower copy-orders whose leader is no longer open.
  - Place follower orders for any open leader orders that have no follower copy-order.
  - Rebuild in-memory leader→follower order_mappings.

This avoids “cancel-all-and-rebuild” and ensures follower state matches leader
state even after WS disconnects or process restarts.
"""

import asyncio
import json
import os
import signal
import struct
import time
from dataclasses import dataclass
from typing import Dict, Optional, Literal, List, TypedDict

from dotenv import load_dotenv
import websockets
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.signing import OrderType as HLOrderType
from hyperliquid.utils.types import Cloid


load_dotenv()

# ============================================================
# Configuration
# ============================================================

WS_URL = os.getenv("HYPERLIQUID_TESTNET_PUBLIC_WS_URL")
BASE_URL = os.getenv("HYPERLIQUID_TESTNET_PUBLIC_BASE_URL")

# For tests, you can use the same wallet as a leader and follower.
# Follower's orders will be ignored in the mirroring logic (we skip by cloid).
LEADER_ADDRESS = os.getenv("TESTNET_WALLET_ADDRESS")

# Fixed notional per order in USDC
FIXED_ORDER_VALUE_USDC = float(os.getenv("FIXED_ORDER_VALUE_USDC", "15.0"))

# Leverage only applies to PERP orders
FOLLOWER_LEVERAGE = int(os.getenv("FOLLOWER_LEVERAGE", "0"))

# Client order ID prefix for follower copy-orders
# All follower orders managed by this bot will have cloid = "copy_{leader_oid}"
COPY_ORDER_PREFIX = "copy_"
# 8-byte magic to mark bot-managed orders (must be exactly 8 bytes)
CLOID_MAGIC = b"COPYBOT1"  # change if you want, but keep 8 bytes

# Risk limits configuration
MAX_TOTAL_EXPOSURE_USDC = float(os.getenv("MAX_TOTAL_EXPOSURE_USDC", "1000.0"))
MAX_SINGLE_POSITION_USDC = float(os.getenv("MAX_SINGLE_POSITION_USDC", "500.0"))
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "10.0"))
DRIFT_THRESHOLD_PCT = float(os.getenv("DRIFT_THRESHOLD_PCT", "5.0"))
ENABLE_AUTO_POSITION_SYNC = os.getenv("ENABLE_AUTO_POSITION_SYNC", "false").lower() == "true"

# Global runtime flags/state
running = False
kill_switch_triggered = False

# Mapping: leader_order_id -> follower_order_id
# This is in-memory and rebuilt from exchange state on resync.
order_mappings: Dict[int, int] = {}


# ============================================================
# Risk Limits and Monitor
# ============================================================

@dataclass
class RiskLimits:
    max_total_exposure_usdc: float = MAX_TOTAL_EXPOSURE_USDC
    max_single_position_usdc: float = MAX_SINGLE_POSITION_USDC
    max_drawdown_pct: float = MAX_DRAWDOWN_PCT
    drift_threshold_pct: float = DRIFT_THRESHOLD_PCT
    enable_auto_position_sync: bool = ENABLE_AUTO_POSITION_SYNC


@dataclass
class FillState:
    leader_oid: int
    coin: str
    leader_filled_qty: float = 0.0
    follower_filled_qty: float = 0.0
    leader_total_qty: float = 0.0
    follower_total_qty: float = 0.0

    def leader_fill_pct(self) -> float:
        if self.leader_total_qty <= 0:
            return 0.0
        return (self.leader_filled_qty / self.leader_total_qty) * 100

    def follower_fill_pct(self) -> float:
        if self.follower_total_qty <= 0:
            return 0.0
        return (self.follower_filled_qty / self.follower_total_qty) * 100

    def drift_pct(self) -> float:
        return abs(self.leader_fill_pct() - self.follower_fill_pct())


class RiskMonitor:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.high_water_mark: float = 0.0
        self.initial_equity: Optional[float] = None
        self.current_equity: float = 0.0
        self.total_exposure: float = 0.0
        self.position_exposures: Dict[str, float] = {}

    def update_equity(self, equity: float) -> None:
        if self.initial_equity is None:
            self.initial_equity = equity
        self.current_equity = equity
        if equity > self.high_water_mark:
            self.high_water_mark = equity

    def update_exposure(self, coin: str, exposure_usdc: float) -> None:
        self.position_exposures[coin] = abs(exposure_usdc)
        self.total_exposure = sum(self.position_exposures.values())

    def drawdown_pct(self) -> float:
        if self.high_water_mark <= 0:
            return 0.0
        return ((self.high_water_mark - self.current_equity) / self.high_water_mark) * 100

    def check_limits(self) -> tuple[bool, str]:
        if self.total_exposure > self.limits.max_total_exposure_usdc:
            return False, f"Total exposure ${self.total_exposure:.2f} exceeds limit ${self.limits.max_total_exposure_usdc:.2f}"

        for coin, exposure in self.position_exposures.items():
            if exposure > self.limits.max_single_position_usdc:
                return False, f"{coin} exposure ${exposure:.2f} exceeds limit ${self.limits.max_single_position_usdc:.2f}"

        dd = self.drawdown_pct()
        if dd > self.limits.max_drawdown_pct:
            return False, f"Drawdown {dd:.2f}% exceeds limit {self.limits.max_drawdown_pct:.2f}%"

        return True, "OK"

    def should_trigger_kill_switch(self) -> tuple[bool, str]:
        ok, reason = self.check_limits()
        if not ok:
            return True, reason
        return False, ""


risk_limits = RiskLimits()
risk_monitor = RiskMonitor(risk_limits)
fill_states: Dict[int, FillState] = {}
recent_activity: Dict[str, float] = {}  # coin -> timestamp of last order/fill


# ============================================================
# Utilities / Market Type Detection & Validation
# ============================================================

def signal_handler(signum, frame, stop_event: asyncio.Event) -> None:
    """
    Handle Ctrl+C gracefully by setting the stop_event.

    This is used to break out of the reconnect loop and shut down cleanly.
    """
    del signum, frame  # Unused parameters
    print("\nShutting down...")
    stop_event.set()


def detect_market_type(coin_field: str) -> str:
    """
    Detect market type from coin field.

    Rules (Hyperliquid-style):
    - SPOT:
        - starts with '@' (index form, e.g. "@12")
        - or contains '/' (pair, e.g. "PURR/USDC")
    - PERP:
        - otherwise (e.g. "BTC", "ETH", "SOL", ...)

    Returns:
        "SPOT", "PERP", or "UNKNOWN"
    """
    if not isinstance(coin_field, str) or not coin_field:
        return "UNKNOWN"
    if coin_field.startswith("@"):
        return "SPOT"
    if "/" in coin_field:
        return "SPOT"
    return "PERP"


def is_spot_order(coin_field: str) -> bool:
    """
    Check if order is for spot trading - basic format validation only.

    We intentionally do not do an exhaustive validation here; just enough to
    distinguish from perps and reject obviously invalid forms.
    """
    if not coin_field or coin_field == "N/A":
        return False

    market_type = detect_market_type(coin_field)
    if market_type != "SPOT":
        return False

    # Basic format validation for @index, e.g. "@12"
    if coin_field.startswith("@"):
        try:
            asset_index = int(coin_field[1:])
            # Only reject obviously invalid indices
            if asset_index < 0:
                return False
        except ValueError:
            return False

    return True


def is_perp_order(coin_field: str) -> bool:
    """
    Check if order is for perpetual trading - basic format validation only.

    Perp orders:
      - DO NOT start with '@'
      - DO NOT contain '/'
      - Are plain symbols like "BTC", "ETH", "SOL", etc.
    """
    if not coin_field or coin_field == "N/A":
        return False

    market_type = detect_market_type(coin_field)
    if market_type != "PERP":
        return False

    # Perp orders are direct symbols (BTC, ETH, SOL, etc.)
    if coin_field.startswith("@"):
        return False
    if "/" in coin_field:
        return False

    return True


def is_supported_order(coin_field: str) -> bool:
    """
    Unified check: we only consider orders that are either valid SPOT or PERP.

    This function is used in the fast path to filter out unsupported instruments.
    """
    return is_spot_order(coin_field) or is_perp_order(coin_field)

# ============================================================
# Cloid utilities
# ============================================================

def make_cloid_from_leader_oid(leader_oid: int) -> Cloid:
    if leader_oid < 0 or leader_oid > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"leader_oid out of uint64 range: {leader_oid}")

    raw_bytes = CLOID_MAGIC + struct.pack(">Q", leader_oid)  # 8 + 8 = 16 bytes
    raw_hex = "0x" + raw_bytes.hex()                         # 32 hex chars after 0x
    return Cloid(raw_hex)

def parse_leader_oid_from_cloid(raw_cloid: str) -> Optional[int]:
    # raw_cloid comes back from user_state as a string like "0x...."
    if not isinstance(raw_cloid, str) or not raw_cloid.startswith("0x"):
        return None
    hexpart = raw_cloid[2:]
    if len(hexpart) != 32:
        return None
    try:
        b = bytes.fromhex(hexpart)
    except ValueError:
        return None
    if len(b) != 16 or b[:8] != CLOID_MAGIC:
        return None
    return struct.unpack(">Q", b[8:])[0]

# ============================================================
# Spot metadata cache
# ============================================================

@dataclass
class SpotCache:
    """
    Cache for spot metadata and derived maps.

    This is loaded once and reused to avoid repeated heavy metadata calls.
    """
    meta: Optional[dict] = None
    # pair_name (e.g. "PURR/USDC") -> index (int)
    name_to_index: Optional[Dict[str, int]] = None
    # index (int) -> szDecimals (int) for base token
    index_to_szdec: Optional[Dict[int, int]] = None


spot_cache = SpotCache()


def _build_spot_maps(spot_meta: dict) -> tuple[Dict[str, int], Dict[int, int]]:
    """
    Build helper maps for spot markets from the spot metadata.

    Returns:
        (name_to_index, index_to_szdec)
    """
    universe = spot_meta.get("universe", []) or []
    tokens = spot_meta.get("tokens", []) or []

    name_to_index: Dict[str, int] = {}
    index_to_szdec: Dict[int, int] = {}

    for pair in universe:
        idx = pair.get("index")
        name = pair.get("name")
        if isinstance(idx, int) and isinstance(name, str):
            name_to_index[name] = idx

        # Determine szDecimals from base token index (pair["tokens"][0])
        sz_dec = 6
        tok_ixs = pair.get("tokens") or []
        if tok_ixs and isinstance(tok_ixs[0], int) and tok_ixs[0] < len(tokens):
            sz_dec = tokens[tok_ixs[0]].get("szDecimals", 6)
        if isinstance(idx, int):
            index_to_szdec[idx] = sz_dec

    return name_to_index, index_to_szdec


async def load_spot_cache(info: Info) -> None:
    """
    Load the spot metadata cache once.

    info.spot_meta_and_asset_ctxs is synchronous, so we run it in a thread.

    This should be called at startup; lazy reloads will happen if needed.
    """
    global spot_cache
    if spot_cache.meta is not None:
        return

    def _fetch():
        return info.spot_meta_and_asset_ctxs()

    spot_data = await asyncio.to_thread(_fetch)
    if not (isinstance(spot_data, list) and len(spot_data) >= 2):
        raise RuntimeError("Unexpected spot_meta_and_asset_ctxs response")

    spot_meta = spot_data[0]
    name_to_index, index_to_szdec = _build_spot_maps(spot_meta)

    spot_cache.meta = spot_meta
    spot_cache.name_to_index = name_to_index
    spot_cache.index_to_szdec = index_to_szdec


async def get_spot_price_and_szdec(info: Info, coin_field: str) -> Optional[dict]:
    """
    Resolve spot asset price and size decimals.

    Args:
        info: Hyperliquid Info client.
        coin_field: Either "@index" or "PAIR/USDC".

    Returns:
        dict with:
            - "price": float (midPx/markPx)
            - "szDecimals": int
            - "coin": str (normalized "@index")
        or None if lookup fails or coin_field is not valid spot.
    """
    if not is_spot_order(coin_field):
        return None

    # Normalize "PAIR/USDC" -> "@index" using cached mapping
    if "/" in coin_field:
        if not spot_cache.name_to_index:
            await load_spot_cache(info)
        idx = spot_cache.name_to_index.get(coin_field)  # type: ignore[union-attr]
        if idx is None:
            print(f"⚠️ Spot pair {coin_field} not found in universe")
            return None
        coin_field = f"@{idx}"

    if not coin_field.startswith("@"):
        return None

    # Fetch latest price from ctxs (this is HTTP, so run in thread)
    index = int(coin_field[1:])

    def _fetch_ctxs():
        return info.spot_meta_and_asset_ctxs()

    spot_data = await asyncio.to_thread(_fetch_ctxs)
    if not (isinstance(spot_data, list) and len(spot_data) >= 2):
        return None

    asset_ctxs = spot_data[1]
    if index >= len(asset_ctxs):
        print(f"⚠️ Spot index {coin_field} out of range (max: @{len(asset_ctxs)-1})")
        return None

    ctx = asset_ctxs[index]
    price = float(ctx.get("midPx", ctx.get("markPx", 0)) or 0)

    if price <= 0:
        print(
            f"⚠️ No spot price for {coin_field} "
            f"(midPx={ctx.get('midPx')}, markPx={ctx.get('markPx')})"
        )
        return None

    if not spot_cache.index_to_szdec:
        await load_spot_cache(info)
    sz_dec = spot_cache.index_to_szdec.get(index, 6)  # type: ignore[union-attr]

    return {"price": price, "szDecimals": sz_dec, "coin": coin_field}


# ============================================================
# Perp metadata cache
# ============================================================

@dataclass
class PerpCache:
    """
    Cache for perp metadata and derived maps.

    This is loaded once and reused to avoid repeated heavy metadata calls.
    """
    meta: Optional[dict] = None
    # symbol (e.g. "BTC") -> szDecimals (int)
    symbol_to_szdec: Optional[Dict[str, int]] = None


perp_cache = PerpCache()


async def load_perp_cache(info: Info) -> None:
    """
    Load the perp metadata cache once.

    info.meta is synchronous, so we run it in a thread.
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
    Resolve perp asset price and size decimals.

    Args:
        info: Hyperliquid Info client.
        coin_field: Perp symbol, e.g. "BTC", "ETH".

    Returns:
        dict with:
            - "price": float (from info.all_mids())
            - "szDecimals": int
            - "coin": str (same as coin_field)
        or None if lookup fails or coin_field is not valid perp.
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


# ============================================================
# Unified asset info retrieval
# ============================================================

async def get_asset_price_and_szdec(info: Info, coin_field: str) -> Optional[dict]:
    """
    Unified asset info lookup for both SPOT and PERP markets.

    Args:
        info: Hyperliquid Info client.
        coin_field: Raw coin field as seen in order updates.

    Returns:
        dict with:
            - "price": float
            - "szDecimals": int
            - "coin": str      (normalized symbol: "@idx" for spot, "BTC" for perp)
            - "market_type": "SPOT" | "PERP"
        or None if not supported or lookup fails.
    """
    market_type = detect_market_type(coin_field)
    if market_type == "SPOT":
        ret = await get_spot_price_and_szdec(info, coin_field)
    elif market_type == "PERP":
        ret = await get_perp_price_and_szdec(info, coin_field)
    else:
        return None

    if not ret:
        return None

    ret["market_type"] = market_type
    return ret


# ============================================================
# Action model (fast path output)
# ============================================================

ActionType = Literal["PLACE", "CANCEL", "FILL"]


@dataclass(frozen=True)
class Action:
    """
    Lightweight representation of a leader-driven action for the slow path.

    Attributes:
        type: "PLACE", "CANCEL", or "FILL".
        leader_oid: Leader order ID this action refers to.
        coin: Symbol/coin field as emitted by the leader order.
        side: "B" or "A" for PLACE/FILL actions (buy/ask).
        limit_px: Limit price for PLACE actions.
        fill_sz: Size filled (for FILL actions).
        fill_px: Fill price (for FILL actions).
    """
    type: ActionType
    leader_oid: int
    coin: str
    side: Optional[str] = None
    limit_px: Optional[float] = None
    fill_sz: Optional[float] = None
    fill_px: Optional[float] = None


# ============================================================
# Slow path execution (thread-offloaded)
# ============================================================

async def execute_place(exchange: Exchange, info: Info, action: Action) -> Optional[int]:
    """
    Place follower order for a leader "open" event.

    This is the SLOW path:
    - Uses Info for asset info (price + szDecimals).
    - Computes follower order size based on FIXED_ORDER_VALUE_USDC.
    - For PERP, updates leverage (if FOLLOWER_LEVERAGE > 0).
    - Places the order via Exchange.order in a thread pool.
    - Attaches a client order ID (cloid) that encodes the leader_oid:
        cloid = "copy_{leader_oid}"

    Returns:
        follower_oid (int) if placed successfully, otherwise None.
    """
    assert action.type == "PLACE"
    assert action.side in ("B", "A")
    assert action.limit_px is not None

    asset_info = await get_asset_price_and_szdec(info, action.coin)
    if not asset_info:
        print(f"❌ Could not get asset info for {action.coin}")
        return None

    market_type = asset_info["market_type"]
    price = float(action.limit_px)
    if price <= 0:
        print(f"❌ Invalid price in action: {price}")
        return None

    sz_dec = int(asset_info["szDecimals"])
    order_size = round(FIXED_ORDER_VALUE_USDC / price, sz_dec)
    if order_size <= 0:
        print(f"❌ Invalid order size calculated for {action.coin}")
        return None

    # For PERPs only, set leverage from config before placing order
    leverage_str = ""
    if market_type == "PERP" and FOLLOWER_LEVERAGE > 0:
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
        leverage_str = f" ({FOLLOWER_LEVERAGE}x)"
    else:
        # SPOT or leverage disabled => no leverage call
        leverage_str = ""

    is_buy = action.side == "B"
    mtype_short = "SPOT" if market_type == "SPOT" else "PERP"

    # cloid encodes leader_oid for resync
    cloid = make_cloid_from_leader_oid(action.leader_oid)

    print(
        f"🔄 Placing follower {mtype_short} order: "
        f"{'BUY' if is_buy else 'SELL'} {order_size} {action.coin} @ ${price}{leverage_str} "
        f"(cloid={cloid.to_raw()})"
    )

    def _place_sync():
        # NOTE: Hyperliquid Exchange.order is expected to support cloid.
        return exchange.order(
            name=action.coin,
            is_buy=is_buy,
            sz=order_size,
            limit_px=price,
            order_type=HLOrderType({"limit": {"tif": "Gtc"}}),
            reduce_only=False,
            cloid=cloid,
        )

    result = await asyncio.to_thread(_place_sync)

    if result and result.get("status") == "ok":
        statuses = result.get("response", {}).get("data", {}).get("statuses", []) or []
        if statuses:
            s0 = statuses[0]
            if "resting" in s0:
                oid = s0["resting"]["oid"]
                order_value = order_size * price
                risk_monitor.update_exposure(action.coin, order_value)
                recent_activity[action.coin] = time.time()
                print(f"✅ Follower order placed! ID: {oid}")
                return oid
            if "filled" in s0:
                oid = s0["filled"]["oid"]
                order_value = order_size * price
                risk_monitor.update_exposure(action.coin, order_value)
                recent_activity[action.coin] = time.time()
                print(f"✅ Follower order filled immediately! ID: {oid}")
                return oid

    print(f"❌ Failed to place follower order: {result}")
    return None


async def execute_cancel(exchange: Exchange, action: Action, follower_oid: int) -> bool:
    """
    Cancel follower order corresponding to a leader CANCEL event.

    This is the SLOW path:
    - Runs Exchange.cancel in a thread pool to avoid blocking the event loop.

    Returns:
        True if cancel succeeded, False otherwise.
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


async def handle_fill(action: Action) -> None:
    """
    Handle a leader fill event by updating fill state tracking.

    This tracks:
    - Leader filled quantity for the order
    - Drift between leader and follower fill percentages
    """
    assert action.type == "FILL"
    assert action.fill_sz is not None

    leader_oid = action.leader_oid
    fill_sz = action.fill_sz

    if leader_oid not in fill_states:
        fill_states[leader_oid] = FillState(
            leader_oid=leader_oid,
            coin=action.coin,
        )

    state = fill_states[leader_oid]
    state.leader_filled_qty += fill_sz

    drift = state.drift_pct()
    if drift > risk_limits.drift_threshold_pct:
        print(
            f"⚠️ Fill drift warning: {action.coin} leader={state.leader_fill_pct():.1f}% "
            f"follower={state.follower_fill_pct():.1f}% drift={drift:.1f}%"
        )


def update_follower_fill(leader_oid: int, fill_sz: float) -> None:
    """Update follower fill quantity for drift tracking."""
    if leader_oid not in fill_states:
        return
    fill_states[leader_oid].follower_filled_qty += fill_sz


def init_fill_state(
    leader_oid: int,
    coin: str,
    leader_sz: float,
    follower_sz: float,
) -> None:
    """Initialize fill state when placing a follower order."""
    fill_states[leader_oid] = FillState(
        leader_oid=leader_oid,
        coin=coin,
        leader_total_qty=leader_sz,
        follower_total_qty=follower_sz,
    )


# ============================================================
# Resync helpers: open orders and state reconciliation
# ============================================================

class SimpleOrder(TypedDict, total=False):
    """
    Minimal view of an open order used for resync.

    Fields mirror what we need from user_state:
        oid: int (order ID)
        coin: str
        side: str ("B" or "A")
        limitPx: float
        sz: float
        cloid: str (client order ID, may include COPY_ORDER_PREFIX)
        leader_oid: int (only for parsed follower orders; not present on exchange)
    """
    oid: int
    coin: str
    side: str
    limitPx: float
    sz: float
    cloid: str
    leader_oid: int


async def fetch_open_orders(info: Info, address: str) -> List[SimpleOrder]:
    """
    Fetch open orders for a given address from Hyperliquid using info.open_orders().

    Returns:
        List[SimpleOrder] with normalized fields.
    """

    def _fetch():
        return info.open_orders(address)

    raw_orders = await asyncio.to_thread(_fetch)
    open_orders: List[SimpleOrder] = []

    if not isinstance(raw_orders, list):
        print(f"⚠️ Unexpected open_orders response for {address}: {type(raw_orders)}")
        return open_orders

    for o in raw_orders:
        if not isinstance(o, dict):
            continue
        try:
            oid = int(o.get("oid"))
            coin = str(o.get("coin"))
            side = str(o.get("side"))
            limit_px = float(o.get("limitPx"))
            sz = float(o.get("sz"))
            cloid = o.get("cloid", "") or ""
        except (TypeError, ValueError):
            continue

        open_orders.append(
            SimpleOrder(
                oid=oid,
                coin=coin,
                side=side,
                limitPx=limit_px,
                sz=sz,
                cloid=cloid,
            )
        )

    return open_orders


async def resync_state(
    exchange: Exchange,
    info: Info,
    leader_address: str,
    follower_address: str,
) -> None:
    """
    Reconcile follower open orders with leader open orders.

    Goals:
    - Cancel follower copy-orders whose corresponding leader order is no longer open.
    - Place follower orders for any currently open leader orders that are missing.
    - Rebuild global 'order_mappings' from the live exchange state.

    This function is safe to run:
    - On cold start (process/VM restart).
    - Before each WebSocket (re)connection attempt.

    It uses a 16-byte hex cloid encoding (magic prefix + leader_oid) to map follower orders back
    to leader orders after crashes.
    """
    global order_mappings

    print("🔄 Starting resync_state() ...")

    # 1) Fetch leader + follower open orders
    leader_open = await fetch_open_orders(info, leader_address)
    follower_open_all = await fetch_open_orders(info, follower_address)

    # 2) Build leader map: leader_oid -> SimpleOrder
    leader_by_oid: Dict[int, SimpleOrder] = {}
    for lo in leader_open:
        oid = lo.get("oid")
        if isinstance(oid, int):
            leader_by_oid[oid] = lo

    # 3) From follower orders, keep only those managed by this bot (cloid magic)
    follower_by_leader_oid: Dict[int, SimpleOrder] = {}
    extra_follower_orders: List[SimpleOrder] = []

    for fo in follower_open_all:
        raw_cloid = fo.get("cloid", "") or ""
        leader_oid = parse_leader_oid_from_cloid(raw_cloid)
        if leader_oid is None:
            # Not managed by this bot; leave it alone.
            continue

        # If multiple follower orders share same leader_oid, keep first, mark rest as extra
        if leader_oid in follower_by_leader_oid:
            extra_follower_orders.append(fo)
        else:
            fo["leader_oid"] = leader_oid  # type: ignore[typeddict-item]
            follower_by_leader_oid[leader_oid] = fo

    # 4) Cancel follower orders whose leader is no longer open (orphans) and duplicates
    cancel_targets: List[SimpleOrder] = []

    for leader_oid, fo in follower_by_leader_oid.items():
        if leader_oid not in leader_by_oid:
            cancel_targets.append(fo)

    cancel_targets.extend(extra_follower_orders)

    if cancel_targets:
        print(f"🔧 Resync: cancelling {len(cancel_targets)} orphan/duplicate follower orders ...")

    for fo in cancel_targets:
        try:
            leader_oid = fo.get("leader_oid")
            if not isinstance(leader_oid, int):
                # If we don't know the leader_oid, just log and skip
                print(f"⚠️ Skipping cancel for follower oid={fo.get('oid')} (no leader_oid)")
                continue

            follower_oid = fo.get("oid")
            coin = fo.get("coin", "")
            if not isinstance(follower_oid, int) or not isinstance(coin, str):
                continue

            act = Action(
                type="CANCEL",
                leader_oid=leader_oid,
                coin=coin,
            )
            ok = await execute_cancel(exchange, act, follower_oid)
            if not ok:
                print(f"⚠️ Cancel failed for follower oid={follower_oid}, leader_oid={leader_oid}")
        except Exception as e:
            print(f"❌ Error cancelling follower order during resync: {e}")

    # 5) Ensure every leader open order has a follower mirror
    print("🔧 Resync: ensuring follower mirrors all leader open orders ...")

    new_mappings: Dict[int, int] = {}

    for leader_oid, lo in leader_by_oid.items():
        coin = lo.get("coin", "")
        side = lo.get("side", "")
        limit_px = lo.get("limitPx")

        if not isinstance(coin, str) or not isinstance(side, str) or limit_px is None:
            continue

        try:
            limit_px_f = float(limit_px)
        except (TypeError, ValueError):
            continue

        existing_f = follower_by_leader_oid.get(leader_oid)

        if existing_f:
            # Optionally sanity-check (coin/side/price within epsilon) here.
            f_oid = existing_f.get("oid")
            if isinstance(f_oid, int):
                new_mappings[leader_oid] = f_oid
            continue

        # Missing follower: place new order now
        print(
            f"🔧 Resync: placing missing follower order for leader_oid={leader_oid} "
            f"{side} {coin} @ {limit_px_f}"
        )

        act = Action(
            type="PLACE",
            leader_oid=leader_oid,
            coin=coin,
            side=side,
            limit_px=limit_px_f,
        )
        try:
            follower_oid = await execute_place(exchange, info, act)
            if follower_oid:
                new_mappings[leader_oid] = follower_oid
        except Exception as e:
            print(f"❌ Error placing follower order during resync (leader_oid={leader_oid}): {e}")

    # 6) Swap in the new in-memory mapping
    order_mappings = new_mappings
    print(f"✅ Resync complete. Tracked mappings: {len(order_mappings)}")


# ============================================================
# Position reconciliation
# ============================================================

class PositionInfo(TypedDict, total=False):
    coin: str
    szi: float
    entry_px: float
    unrealized_pnl: float
    margin_used: float
    position_value: float


async def fetch_positions(
    info: Info,
    address: str,
) -> Dict[str, PositionInfo]:
    """
    Fetch current positions for an address.

    Returns a dict mapping coin -> PositionInfo.
    """

    def _fetch():
        return info.user_state(address)

    user_state = await asyncio.to_thread(_fetch)
    positions: Dict[str, PositionInfo] = {}

    if not isinstance(user_state, dict):
        return positions

    asset_positions = user_state.get("assetPositions", []) or []

    for ap in asset_positions:
        if not isinstance(ap, dict):
            continue

        pos = ap.get("position", {}) or {}
        coin = pos.get("coin", "")
        if not coin:
            continue

        try:
            szi = float(pos.get("szi", 0))
            entry_px = float(pos.get("entryPx", 0))
            unrealized_pnl = float(pos.get("unrealizedPnl", 0))
            margin_used = float(pos.get("marginUsed", 0))
            position_value = abs(szi * entry_px)
        except (ValueError, TypeError):
            continue

        if abs(szi) > 0:
            positions[coin] = PositionInfo(
                coin=coin,
                szi=szi,
                entry_px=entry_px,
                unrealized_pnl=unrealized_pnl,
                margin_used=margin_used,
                position_value=position_value,
            )

    return positions


async def fetch_account_equity(info: Info, address: str) -> float:
    """Fetch account equity (account value) for an address."""

    def _fetch():
        return info.user_state(address)

    user_state = await asyncio.to_thread(_fetch)

    if not isinstance(user_state, dict):
        return 0.0

    try:
        margin_summary = user_state.get("marginSummary", {}) or {}
        return float(margin_summary.get("accountValue", 0))
    except (ValueError, TypeError):
        return 0.0


async def reconcile_positions(
    exchange: Exchange,
    info: Info,
    leader_address: str,
    follower_address: str,
) -> Dict[str, float]:
    """
    Compare leader and follower positions, detect drift.

    Returns a dict of coin -> drift_pct for positions with significant drift.
    """
    global kill_switch_triggered

    print("🔄 Reconciling positions...")

    leader_positions = await fetch_positions(info, leader_address)
    follower_positions = await fetch_positions(info, follower_address)
    follower_equity = await fetch_account_equity(info, follower_address)

    risk_monitor.update_equity(follower_equity)

    for coin, fpos in follower_positions.items():
        risk_monitor.update_exposure(coin, fpos.get("position_value", 0))

    should_kill, reason = risk_monitor.should_trigger_kill_switch()
    if should_kill and not kill_switch_triggered:
        print(f"🚨 KILL SWITCH TRIGGERED: {reason}")
        kill_switch_triggered = True
        await trigger_kill_switch(exchange, info, follower_address)
        return {}

    drifts: Dict[str, float] = {}
    all_coins = set(leader_positions.keys()) | set(follower_positions.keys())
    now = time.time()
    grace_period = 30  # seconds to skip warnings after recent activity

    for coin in all_coins:
        # Skip drift check for coins with recent order activity (API latency grace)
        last_activity = recent_activity.get(coin, 0)
        if now - last_activity < grace_period:
            continue

        leader_pos = leader_positions.get(coin)
        follower_pos = follower_positions.get(coin)

        leader_szi = leader_pos.get("szi", 0) if leader_pos else 0
        follower_szi = follower_pos.get("szi", 0) if follower_pos else 0

        if leader_szi == 0 and follower_szi == 0:
            continue

        leader_direction = 1 if leader_szi > 0 else (-1 if leader_szi < 0 else 0)
        follower_direction = 1 if follower_szi > 0 else -1 if follower_szi < 0 else 0

        if leader_direction != follower_direction and leader_direction != 0:
            print(
                f"⚠️ Direction mismatch {coin}: "
                f"leader={'LONG' if leader_direction > 0 else 'SHORT'} "
                f"follower={'LONG' if follower_direction > 0 else 'SHORT' if follower_direction < 0 else 'FLAT'}"
            )
            drifts[coin] = 100.0
            continue

        if leader_szi == 0 and follower_szi != 0:
            print(f"⚠️ Orphan position {coin}: follower has {follower_szi}, leader flat")
            drifts[coin] = 100.0

            if risk_limits.enable_auto_position_sync:
                print(f"🔧 Auto-closing orphan position {coin}")
                await close_position(exchange, coin, follower_szi)
            continue

        if leader_szi != 0 and abs(leader_szi) > 0:
            drift_pct = abs(1 - (abs(follower_szi) / abs(leader_szi))) * 100
        else:
            drift_pct = 0

        if drift_pct > risk_limits.drift_threshold_pct:
            print(
                f"⚠️ Position drift {coin}: leader={leader_szi:.6f} "
                f"follower={follower_szi:.6f} drift={drift_pct:.1f}%"
            )
            drifts[coin] = drift_pct

    if drifts:
        print(f"📊 Position drift summary: {len(drifts)} coins with drift")
    else:
        print("✅ Position reconciliation complete, no significant drift")

    return drifts


async def close_position(exchange: Exchange, coin: str, szi: float) -> bool:
    """Close a position by placing a market order in the opposite direction."""
    is_buy = szi < 0
    sz = abs(szi)

    print(f"🔄 Closing position: {'BUY' if is_buy else 'SELL'} {sz} {coin}")

    def _close():
        return exchange.order(
            name=coin,
            is_buy=is_buy,
            sz=sz,
            limit_px=0,
            order_type=HLOrderType({"limit": {"tif": "Ioc"}}),
            reduce_only=True,
        )

    try:
        result = await asyncio.to_thread(_close)
        if result and result.get("status") == "ok":
            print(f"✅ Position closed: {coin}")
            return True
        print(f"❌ Failed to close position {coin}: {result}")
        return False
    except Exception as e:
        print(f"❌ Error closing position {coin}: {e}")
        return False


async def trigger_kill_switch(
    exchange: Exchange,
    info: Info,
    follower_address: str,
) -> None:
    """
    Emergency shutdown: cancel all orders and optionally close all positions.
    """
    global kill_switch_triggered
    kill_switch_triggered = True

    print("🚨 KILL SWITCH: Cancelling all orders...")

    def _cancel_all():
        return exchange.cancel_all()

    try:
        result = await asyncio.to_thread(_cancel_all)
        print(f"Cancel all result: {result}")
    except Exception as e:
        print(f"❌ Error cancelling orders: {e}")

    if risk_limits.enable_auto_position_sync:
        print("🚨 KILL SWITCH: Closing all positions...")
        positions = await fetch_positions(info, follower_address)
        for coin, pos in positions.items():
            szi = pos.get("szi", 0)
            if abs(szi) > 0:
                await close_position(exchange, coin, szi)

    print("🛑 Kill switch complete. Bot is halted.")


# ============================================================
# Fast path parsing (no network)
# ============================================================

def extract_actions_from_message(data: dict) -> list[Action]:
    """
    FAST PATH: only parse/filter WebSocket messages into Actions.

    This function must NOT perform any network I/O (no Info/Exchange calls).

    It produces a list of Actions to execute in order, for both SPOT and PERP:
    - PLACE on "open" orders
    - CANCEL on "canceled" orders

    It also:
    - Skips follower orders by checking cloid prefix (in case leader and follower
      share the same wallet for testing).
    - Skips PLACE if the leader_oid is already known in order_mappings, to avoid
      duplicate mirroring after resync.
    """
    actions: list[Action] = []
    channel = data.get("channel")

    if channel == "subscriptionResponse":
        print("✅ WebSocket subscription confirmed")
        return actions

    # You subscribed to:
    # - orderUpdates
    # - userEvents
    # Some payloads come in as "orderUpdates" and "user" or "userEvents".
    if channel == "orderUpdates":
        orders = data.get("data", []) or []
        for order_update in orders:
            order = order_update.get("order", {}) or {}
            status = order_update.get("status", "unknown")
            coin_field = order.get("coin", "")

            if not is_supported_order(coin_field):
                continue

            # Skip follower orders (if using same wallet) by cloid prefix
            cloid = order.get("cloid", "") or ""
            if parse_leader_oid_from_cloid(cloid) is not None:
                # This is one of OUR follower mirror orders
                continue

            leader_oid = order.get("oid")
            if not isinstance(leader_oid, int):
                continue

            # If we've already mirrored this leader_oid (via resync or earlier),
            # skip additional PLACE events to avoid duplicates.
            if leader_oid in order_mappings:
                # Still log if you want to see updates, but don't create new actions.
                continue

            side = order.get("side")
            limit_px = order.get("limitPx")

            market_type = detect_market_type(coin_field)
            mtype_short = "SPOT" if market_type == "SPOT" else "PERP"

            # Skip noisy "filled" logs here; fills logged on user channel
            if status != "filled":
                print(
                    f"LEADER {mtype_short} ORDER {str(status).upper()}: {side} {order.get('sz')} "
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
                actions.append(
                    Action(
                        type="CANCEL",
                        leader_oid=leader_oid,
                        coin=coin_field,
                    )
                )

    elif channel in ("user", "userEvents"):
        user_data = data.get("data", {}) or {}
        fills = user_data.get("fills", []) or []
        for fill in fills:
            coin_field = fill.get("coin", "N/A")
            if not is_supported_order(coin_field):
                continue

            fill_oid = fill.get("oid")
            if not isinstance(fill_oid, int):
                continue

            # Skip follower fills - track them separately for drift detection
            if fill_oid in order_mappings.values():
                for leader_oid, foid in order_mappings.items():
                    if foid == fill_oid:
                        try:
                            fill_sz = float(fill.get("sz", 0))
                            update_follower_fill(leader_oid, fill_sz)
                        except (ValueError, TypeError):
                            pass
                        break
                continue

            # Skip fills from orders we're not tracking
            cloid = fill.get("cloid", "") or ""
            if parse_leader_oid_from_cloid(cloid) is not None:
                continue

            side_raw = fill.get("side", "")
            side = "BUY" if side_raw == "B" else "SELL"
            market_type = detect_market_type(coin_field)
            mtype_short = "SPOT" if market_type == "SPOT" else "PERP"

            try:
                fill_sz = float(fill.get("sz", 0))
                fill_px = float(fill.get("px", 0))
            except (ValueError, TypeError):
                continue

            print(
                f"LEADER {mtype_short} FILL: {side} {fill_sz} "
                f"{coin_field} @ {fill_px} (oid={fill_oid})"
            )

            # Create FILL action for leader fills we're tracking
            if fill_oid in order_mappings:
                actions.append(
                    Action(
                        type="FILL",
                        leader_oid=fill_oid,
                        coin=coin_field,
                        side=side_raw,
                        fill_sz=fill_sz,
                        fill_px=fill_px,
                    )
                )

    return actions


# ============================================================
# Main loop (shared WS, unified flow)
# ============================================================

async def monitor_and_mirror_orders() -> None:
    """
    Main control loop:
    - Initializes follower wallet, Exchange, and Info clients.
    - Preloads metadata caches.
    - Performs an initial resync_state() (cold start).
    - Enters a reconnect loop for the WebSocket connection:
        - Before each WS connection, resync_state() runs again.
        - Inside WS, spawns:
            - hl_heartbeat (ping)
            - message_receiver (fast path, raw WS -> queue)
            - message_parser (fast path, JSON -> Actions)
            - action_executor (slow path, Actions -> Exchange calls)
    """
    global running, order_mappings, kill_switch_triggered

    private_key = os.getenv("HYPERLIQUID_TESTNET_PRIVATE_KEY")
    if not private_key:
        print("❌ Missing HYPERLIQUID_TESTNET_PRIVATE_KEY in .env file")
        return

    if not BASE_URL:
        print("❌ Missing HYPERLIQUID_TESTNET_PUBLIC_BASE_URL in environment")
        return

    # Initialize follower trading components
    try:
        wallet = Account.from_key(private_key)
        exchange = Exchange(wallet, BASE_URL)
        info = Info(BASE_URL, skip_ws=True)
        follower_address = wallet.address
        print(f"✅ Follower wallet initialized: {wallet.address}")
    except Exception as e:
        print(f"❌ Failed to initialize follower wallet: {e}")
        return

    # Preload both perp and spot meta caches once (helps latency later)
    try:
        await load_perp_cache(info)
    except Exception as e:
        print(f"⚠️ Could not preload perp cache (will retry lazily): {e}")

    try:
        await load_spot_cache(info)
    except Exception as e:
        print(f"⚠️ Could not preload spot cache (will retry lazily): {e}")

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

        # Re-resync before each WS attempt to be robust to downtime
        if kill_switch_triggered:
            print("🛑 Kill switch triggered, stopping reconnect loop")
            break

        try:
            await resync_state(exchange, info, LEADER_ADDRESS, follower_address)
            await reconcile_positions(exchange, info, LEADER_ADDRESS, follower_address)
        except Exception as e:
            print(f"⚠️ Resync before WS connect failed: {e}")

        # Bounded queues: if you fall behind, you'll see queue size grow
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

                # Subscribe each time we reconnect
                await websocket.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {"type": "orderUpdates", "user": LEADER_ADDRESS},
                }))
                await websocket.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {"type": "userEvents", "user": LEADER_ADDRESS},
                }))

                print(f"📊 Monitoring SPOT + PERP orders for leader: {LEADER_ADDRESS}")
                print(f"💰 Fixed order value: ${FIXED_ORDER_VALUE_USDC} USDC per order")
                print(f"⚙️ Follower leverage (PERPS only): {FOLLOWER_LEVERAGE}x")
                print(f"👤 Follower wallet: {wallet.address}")
                print("--- Risk Limits ---")
                print(f"🛡️ Max total exposure: ${MAX_TOTAL_EXPOSURE_USDC}")
                print(f"🛡️ Max single position: ${MAX_SINGLE_POSITION_USDC}")
                print(f"🛡️ Max drawdown: {MAX_DRAWDOWN_PCT}%")
                print(f"🛡️ Drift threshold: {DRIFT_THRESHOLD_PCT}%")
                print(f"🛡️ Auto position sync: {ENABLE_AUTO_POSITION_SYNC}")
                print("=" * 80)

                # --- Task 0: Hyperliquid heartbeat (JSON ping)
                async def hl_heartbeat():
                    """
                    Periodic heartbeat ping to keep the WebSocket connection alive.

                    If sending the ping fails, this task exits and the connection
                    will eventually be torn down and reconnected.
                    """
                    while running and not stop_event.is_set():
                        try:
                            await websocket.send(json.dumps({"method": "ping"}))
                        except Exception as e:
                            print(f"💔 Heartbeat send failed: {e}")
                            break
                        await asyncio.sleep(25)  # < 60s server idle timeout

                # --- Task 1: WS receive (FAST PATH)
                async def message_receiver():
                    """
                    Receive messages from WebSocket and push raw JSON strings
                    into the message_queue.

                    This is the main "fast path" that must NOT perform any
                    blocking or slow operations.
                    """
                    nonlocal recv_count, last_metrics_time
                    global running
                    try:
                        async for message in websocket:
                            if not running or stop_event.is_set():
                                break

                            # Drop pong frames early
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
                                print(
                                    f"📥 recv={recv_count} "
                                    f"mq={message_queue.qsize()} "
                                    f"aq={action_queue.qsize()} "
                                    f"maps={len(order_mappings)}"
                                )
                    except Exception as e:
                        print(f"❌ Receiver error: {e}")
                    finally:
                        # IMPORTANT: if receiver stops, everything should stop (or reconnect)
                        running = False
                        print("🔌 Receiver ended (WS likely closed).")

                # --- Task 2: parse messages into actions (FAST PATH)
                async def message_parser():
                    """
                    Parse raw JSON messages from the message_queue into Actions.

                    This is still "fast path": only JSON decode + in-memory logic.
                    It must never perform network I/O.
                    """
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
                    """
                    Consume Actions from action_queue and execute them sequentially.

                    This is the only place that calls Exchange/Info (slow path).
                    All HTTP/SDK calls are offloaded using asyncio.to_thread.
                    """
                    while running and not stop_event.is_set():
                        try:
                            act = await asyncio.wait_for(action_queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                        try:
                            if kill_switch_triggered:
                                print("🛑 Kill switch active, skipping action")
                                continue

                            if act.type == "PLACE":
                                ok, reason = risk_monitor.check_limits()
                                if not ok:
                                    print(f"⚠️ Risk limit exceeded: {reason}")
                                    continue

                                follower_oid = await execute_place(
                                    exchange, info, act
                                )
                                if follower_oid:
                                    order_mappings[act.leader_oid] = follower_oid
                                    print(
                                        f"Mapped leader OID {act.leader_oid} "
                                        f"-> follower OID {follower_oid}"
                                    )

                            elif act.type == "CANCEL":
                                follower_oid = order_mappings.get(act.leader_oid)
                                if follower_oid:
                                    await execute_cancel(exchange, act, follower_oid)
                                    order_mappings.pop(act.leader_oid, None)
                                    fill_states.pop(act.leader_oid, None)

                            elif act.type == "FILL":
                                await handle_fill(act)

                        except Exception as e:
                            print(f"❌ Execution error: {e}")
                        finally:
                            action_queue.task_done()

                async def position_monitor():
                    """
                    Periodic position reconciliation and risk monitoring.

                    Runs every 60 seconds to detect drift and check risk limits.
                    """
                    while running and not stop_event.is_set():
                        await asyncio.sleep(60)
                        if not running or stop_event.is_set():
                            break

                        try:
                            await reconcile_positions(
                                exchange, info, LEADER_ADDRESS, follower_address
                            )
                            if kill_switch_triggered:
                                print("🛑 Kill switch triggered, stopping")
                                break
                        except Exception as e:
                            print(f"⚠️ Position monitor error: {e}")

                await asyncio.gather(
                    hl_heartbeat(),
                    message_receiver(),
                    message_parser(),
                    action_executor(),
                    position_monitor(),
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

        # Sleep for reconnect delay before attempting to connect again
        await asyncio.sleep(reconnect_delay_s)

    print("🛑 Stopped reconnect loop. Bye.")


async def main() -> None:
    """
    Entry point for the unified Hyperliquid Spot + Perp Order Mirror.
    """
    print("Hyperliquid Spot + Perp Order Mirror")
    print("=" * 40)

    if not WS_URL or not BASE_URL:
        print("❌ Missing required environment variables:")
        print("   HYPERLIQUID_TESTNET_PUBLIC_WS_URL")
        print("   HYPERLIQUID_TESTNET_PUBLIC_BASE_URL")
        return

    if not LEADER_ADDRESS or LEADER_ADDRESS == "0x...":
        print("❌ Please set TESTNET_WALLET_ADDRESS (LEADER_ADDRESS) in the environment")
        return

    await monitor_and_mirror_orders()


if __name__ == "__main__":
    asyncio.run(main())
