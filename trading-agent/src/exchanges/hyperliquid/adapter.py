"""
Hyperliquid Exchange Adapter

Clean implementation of Hyperliquid integration using the exchange interface.
Technical implementation separated from business logic.
"""

from typing import Dict, List, Optional, Any
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import time

from interfaces.exchange import (
    ExchangeAdapter,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Balance,
    MarketInfo,
)
from core.endpoint_router import get_endpoint_router

# Default precision used when an asset isn't found in the cache
_DEFAULT_PRECISION = {"sz_decimals": 2, "is_spot": False}

# Perps allow 6 total significant decimal digits, spot allows 8
_MAX_DECIMALS_PERP = 6
_MAX_DECIMALS_SPOT = 8


class HyperliquidAdapter(ExchangeAdapter):
    """
    Hyperliquid DEX adapter implementation

    Handles all Hyperliquid-specific technical details while implementing
    the clean exchange interface that strategies can use.
    """

    def __init__(self, private_key: str, testnet: bool = True):
        super().__init__("Hyperliquid")
        self.private_key = private_key
        self.testnet = testnet
        self.paper_trading = False

        # Hyperliquid SDK components (will be initialized on connect)
        self.info = None
        self.exchange = None

        # Per-asset precision rules: {asset_name: {sz_decimals, is_spot}}
        self._precision_cache: Dict[str, Dict[str, Any]] = {}

        # Spot pair name → universe index (for price lookups)
        self._spot_pair_to_index: Dict[str, int] = {}

        # Endpoint router for smart routing
        self.endpoint_router = get_endpoint_router(testnet)

    # ---- Precision helpers ----

    def _build_precision_cache(self) -> None:
        """Populate _precision_cache from meta() (perps).

        Call this after self.info is available.  Spot assets are added
        separately by _build_spot_precision_cache.
        """
        if self.info is None:
            return
        try:
            meta = self.info.meta()
            for asset_info in meta.get("universe", []):
                name = asset_info.get("name")
                if name:
                    self._precision_cache[name] = {
                        "sz_decimals": int(asset_info.get("szDecimals", 2)),
                        "is_spot": False,
                    }
        except Exception:
            pass  # cache stays empty; defaults used

    def _build_spot_precision_cache(self) -> None:
        """Populate precision cache and index map from spot_meta_and_asset_ctxs."""
        if self.info is None:
            return
        try:
            spot_data = self.info.spot_meta_and_asset_ctxs()
            if not (isinstance(spot_data, list) and len(spot_data) >= 2):
                return
            spot_meta = spot_data[0]
            tokens = spot_meta.get("tokens", [])

            # Build token index → szDecimals mapping
            token_sz: Dict[int, int] = {}
            token_names: Dict[int, str] = {}
            for t in tokens:
                idx = t.get("index")
                if isinstance(idx, int):
                    token_sz[idx] = int(t.get("szDecimals", 6))
                    token_names[idx] = t.get("name", "")

            for pair in spot_meta.get("universe", []):
                pair_name = pair.get("name")
                pair_index = pair.get("index")
                tok_idxs = pair.get("tokens", [])
                if not pair_name or not isinstance(pair_index, int):
                    continue

                # szDecimals comes from the base token
                base_idx = tok_idxs[0] if tok_idxs else None
                sz_dec = token_sz.get(base_idx, 6) if base_idx is not None else 6

                self._precision_cache[pair_name] = {
                    "sz_decimals": sz_dec,
                    "is_spot": True,
                }
                self._spot_pair_to_index[pair_name] = pair_index
        except Exception:
            pass

    @staticmethod
    def _is_spot(asset: str) -> bool:
        """Detect spot pairs by format: contains '/' or starts with '@'."""
        return "/" in asset or asset.startswith("@")

    def _get_precision_rules(self, asset: str) -> Dict[str, Any]:
        """Return precision dict for *asset*, falling back to defaults."""
        return self._precision_cache.get(asset, dict(_DEFAULT_PRECISION))

    def _max_price_decimals(self, asset: str) -> int:
        """Max price decimal places for *asset*."""
        rules = self._get_precision_rules(asset)
        max_dec = _MAX_DECIMALS_SPOT if rules["is_spot"] else _MAX_DECIMALS_PERP
        return max(0, max_dec - rules["sz_decimals"])

    def _round_size(self, asset: str, size: float) -> float:
        """Round order size to the asset's szDecimals (ROUND_DOWN)."""
        rules = self._get_precision_rules(asset)
        sz_dec = rules["sz_decimals"]
        d = Decimal(str(size)).quantize(
            Decimal(10) ** -sz_dec, rounding=ROUND_DOWN,
        )
        return float(d)

    def _round_price(self, asset: str, price: float, is_buy: bool) -> float:
        """Round order price respecting Hyperliquid's decimal rules.

        Buyers round DOWN (safer), sellers round UP.
        Non-integer prices are limited to 5 significant figures.
        """
        rounding = ROUND_DOWN if is_buy else ROUND_UP
        price_dec = Decimal(str(price))
        max_dp = self._max_price_decimals(asset)

        # Try from max decimals down to 0, first valid candidate wins
        for dp in range(max_dp, -1, -1):
            candidate = price_dec.quantize(
                Decimal(10) ** -dp, rounding=rounding,
            )
            # Integer prices always OK
            if candidate == candidate.to_integral_value():
                return float(candidate)
            # Non-integer limited to 5 significant figures
            if len(candidate.normalize().as_tuple().digits) <= 5:
                return float(candidate)

        # Fallback: integer price
        return float(price_dec.to_integral_value(rounding=rounding))

    async def connect(self) -> bool:
        """Connect to Hyperliquid with smart endpoint routing"""
        try:
            # Import here to avoid dependency issues
            from hyperliquid.info import Info
            from hyperliquid.exchange import Exchange
            from eth_account import Account

            # Get the info endpoint from router
            info_url = self.endpoint_router.get_endpoint_for_method("user_state")
            if not info_url:
                raise RuntimeError("No healthy info endpoint available")

            # Get the exchange endpoint from router
            exchange_url = self.endpoint_router.get_endpoint_for_method("cancel_order")
            if not exchange_url:
                raise RuntimeError("No healthy exchange endpoint available")

            # Remove /info and /exchange suffixes (SDK adds them automatically)
            info_base_url = (
                info_url.replace("/info", "")
                if info_url.endswith("/info")
                else info_url
            )
            exchange_base_url = (
                exchange_url.replace("/exchange", "")
                if exchange_url.endswith("/exchange")
                else exchange_url
            )

            # Create wallet from private key
            wallet = Account.from_key(self.private_key)

            # Initialize SDK components with proper endpoint routing
            self.info = Info(info_base_url, skip_ws=True)
            self.exchange = Exchange(wallet, exchange_base_url)

            # Test connection
            user_state = self.info.user_state(self.exchange.wallet.address)

            self.is_connected = True

            # Build precision cache from market metadata
            self._build_precision_cache()
            self._build_spot_precision_cache()

            print(
                f"✅ Connected to Hyperliquid ({'testnet' if self.testnet else 'mainnet'})"
            )
            print(f"📡 Info endpoint: {info_url}")
            print(f"💱 Exchange endpoint: {exchange_url}")
            print(f"🔑 Wallet address: {self.exchange.wallet.address}")
            return True

        except Exception as e:
            print(f"❌ Failed to connect to Hyperliquid: {e}")
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Hyperliquid"""
        self.is_connected = False
        self.info = None
        self.exchange = None
        print("🔌 Disconnected from Hyperliquid")

    async def get_balance(self, asset: str) -> Balance:
        """Get account balance for an asset"""
        if not self.is_connected:
            raise RuntimeError("Not connected to exchange")

        try:
            user_state = self.info.user_state(self.exchange.wallet.address)

            # Find asset balance
            for balance_info in user_state.get("balances", []):
                coin = balance_info.get("coin", "")
                if coin == asset:
                    total = float(balance_info.get("total", 0))
                    hold = float(balance_info.get("hold", 0))
                    available = total - hold

                    return Balance(
                        asset=asset, available=available, locked=hold, total=total
                    )

            # Asset not found, return zero balance
            return Balance(asset=asset, available=0.0, locked=0.0, total=0.0)

        except Exception as e:
            raise RuntimeError(f"Failed to get {asset} balance: {e}")

    async def get_market_price(self, asset: str) -> float:
        """Get current market price for perp or spot asset."""
        if not self.is_connected:
            raise RuntimeError("Not connected to exchange")

        try:
            if self._is_spot(asset):
                return self._get_spot_price(asset)

            # Perp price from all_mids
            all_mids = self.info.all_mids()
            if asset in all_mids:
                return float(all_mids[asset])
            raise ValueError(f"Asset {asset} not found in market data")

        except Exception as e:
            raise RuntimeError(f"Failed to get {asset} price: {e}")

    def _get_spot_price(self, pair_name: str) -> float:
        """Fetch spot price from spot_meta_and_asset_ctxs context."""
        idx = self._spot_pair_to_index.get(pair_name)
        if idx is None:
            raise ValueError(f"Spot pair {pair_name} not found in metadata")

        spot_data = self.info.spot_meta_and_asset_ctxs()
        if not (isinstance(spot_data, list) and len(spot_data) >= 2):
            raise RuntimeError("Unexpected spot_meta_and_asset_ctxs response")

        ctxs = spot_data[1]
        if idx >= len(ctxs):
            raise ValueError(f"Spot pair index {idx} out of range")

        ctx = ctxs[idx]
        price = float(ctx.get("midPx") or ctx.get("markPx") or 0)
        if price <= 0:
            raise ValueError(f"No valid price for {pair_name}")
        return price

    async def place_order(self, order: Order) -> str:
        """Place an order on Hyperliquid"""
        if not self.is_connected:
            raise RuntimeError("Not connected to exchange")

        try:
            is_buy = order.side == OrderSide.BUY

            from hyperliquid.utils.signing import OrderType as HLOrderType

            # Round size using dynamic precision for this asset
            rounded_size = self._round_size(order.asset, order.size)

            # Enforce minimum size (1 unit at the lowest decimal)
            rules = self._get_precision_rules(order.asset)
            min_size = 10 ** -rules["sz_decimals"]
            rounded_size = max(rounded_size, min_size)

            if order.order_type == OrderType.MARKET:
                # Market order via IOC limit with ±1% slippage
                market_price = await self.get_market_price(order.asset)
                slippage_price = market_price * (1.01 if is_buy else 0.99)
                adjusted_price = self._round_price(order.asset, slippage_price, is_buy)
                result = self.exchange.order(
                    name=order.asset,
                    is_buy=is_buy,
                    sz=rounded_size,
                    limit_px=adjusted_price,
                    order_type=HLOrderType({"limit": {"tif": "Ioc"}}),
                    reduce_only=False,
                )
            else:
                # Limit order (GTC)
                rounded_price = self._round_price(order.asset, order.price, is_buy)
                result = self.exchange.order(
                    name=order.asset,
                    is_buy=is_buy,
                    sz=rounded_size,
                    limit_px=rounded_price,
                    order_type=HLOrderType({"limit": {"tif": "Gtc"}}),
                    reduce_only=False,
                )

            # Extract order ID from result (resting or filled)
            if result and "status" in result and result["status"] == "ok":
                if "response" in result and "data" in result["response"]:
                    response_data = result["response"]["data"]
                    if "statuses" in response_data and response_data["statuses"]:
                        status_info = response_data["statuses"][0]
                        if "resting" in status_info:
                            return str(status_info["resting"]["oid"])
                        if "filled" in status_info:
                            return str(status_info["filled"]["oid"])

            raise RuntimeError(f"Failed to place order: {result}")

        except Exception as e:
            raise RuntimeError(f"Failed to place {order.side.value} order: {e}")

    async def cancel_order(self, exchange_order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected:
            raise RuntimeError("Not connected to exchange")

        try:
            # Convert to int (Hyperliquid uses integer order IDs)
            oid = int(exchange_order_id)

            # Find the asset name for this order by querying open orders
            open_orders = self.info.open_orders(self.exchange.wallet.address)
            target_order = None

            for order in open_orders:
                if order.get("oid") == oid:
                    target_order = order
                    break

            if not target_order:
                print(f"❌ Order {exchange_order_id} not found in open orders")
                return False

            asset_name = target_order.get("coin")
            if not asset_name:
                print(f"❌ Could not determine asset for order {exchange_order_id}")
                return False

            # Use the correct SDK method: cancel(name, oid)
            result = self.exchange.cancel(name=asset_name, oid=oid)

            # Check if cancellation was successful
            if result and isinstance(result, dict) and result.get("status") == "ok":
                response_data = result.get("response", {}).get("data", {})
                statuses = response_data.get("statuses", [])

                if statuses and statuses[0] == "success":
                    print(f"✅ Order {exchange_order_id} cancelled successfully")
                    return True
                else:
                    print(f"❌ Cancel failed with status: {statuses}")
                    return False
            else:
                print(f"❌ Cancel request failed: {result}")
                return False

        except Exception as e:
            print(f"❌ Error cancelling order {exchange_order_id}: {e}")
            return False

    async def get_order_status(self, exchange_order_id: str) -> Order:
        """Query order status from the Hyperliquid API."""
        if not self.is_connected:
            raise RuntimeError("Not connected to exchange")

        try:
            oid = int(exchange_order_id)
            result = self.info.query_order_by_oid(
                self.exchange.wallet.address, oid,
            )

            if not result:
                # Order not found — return PENDING as safe fallback
                return Order(
                    id=exchange_order_id,
                    asset="",
                    side=OrderSide.BUY,
                    size=0.0,
                    order_type=OrderType.LIMIT,
                    status=OrderStatus.PENDING,
                    exchange_order_id=exchange_order_id,
                )

            order_data = result.get("order", {})
            raw_status = result.get("status", "")

            # Map Hyperliquid statuses to our enum
            status_map = {
                "open": OrderStatus.SUBMITTED,
                "filled": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
                "triggered": OrderStatus.SUBMITTED,
                "rejected": OrderStatus.REJECTED,
                "marginCanceled": OrderStatus.CANCELLED,
            }
            status = status_map.get(raw_status, OrderStatus.PENDING)

            side = OrderSide.BUY if order_data.get("side") == "B" else OrderSide.SELL

            return Order(
                id=exchange_order_id,
                asset=order_data.get("coin", ""),
                side=side,
                size=float(order_data.get("sz", 0)),
                order_type=OrderType.LIMIT,
                price=float(order_data.get("limitPx", 0)),
                status=status,
                exchange_order_id=exchange_order_id,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to get order status for {exchange_order_id}: {e}")

    async def get_market_info(self, asset: str) -> MarketInfo:
        """Get market information for perp or spot asset."""
        if not self.is_connected:
            raise RuntimeError("Not connected to exchange")

        try:
            if self._is_spot(asset):
                return self._get_spot_market_info(asset)

            # Perp market info from meta()
            meta = self.info.meta()
            for asset_info in meta.get("universe", []):
                if asset_info.get("name") == asset:
                    sz_dec = int(asset_info.get("szDecimals", 2))
                    return MarketInfo(
                        symbol=asset,
                        base_asset=asset,
                        quote_asset="USD",
                        min_order_size=10 ** -sz_dec,
                        price_precision=self._max_price_decimals(asset),
                        size_precision=sz_dec,
                        is_active=True,
                    )
            raise ValueError(f"Asset {asset} not found")

        except Exception as e:
            raise RuntimeError(f"Failed to get market info for {asset}: {e}")

    def _get_spot_market_info(self, pair_name: str) -> MarketInfo:
        """Build MarketInfo from cached spot metadata."""
        rules = self._get_precision_rules(pair_name)
        sz_dec = rules["sz_decimals"]

        # Parse base/quote from pair name (e.g. "PURR/USDC")
        parts = pair_name.split("/")
        base = parts[0] if len(parts) >= 1 else pair_name
        quote = parts[1] if len(parts) >= 2 else "USDC"

        return MarketInfo(
            symbol=pair_name,
            base_asset=base,
            quote_asset=quote,
            min_order_size=10 ** -sz_dec,
            price_precision=self._max_price_decimals(pair_name),
            size_precision=sz_dec,
            is_active=True,
        )

    async def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        if not self.is_connected:
            return []

        try:
            open_orders = self.info.open_orders(self.exchange.wallet.address)
            orders = []

            for order_info in open_orders:
                order = Order(
                    id=str(order_info.get("oid", "")),
                    asset=order_info.get("coin", ""),
                    side=OrderSide.BUY
                    if order_info.get("side") == "B"
                    else OrderSide.SELL,
                    size=float(order_info.get("sz", 0)),
                    order_type=OrderType.LIMIT,  # Hyperliquid default
                    price=float(order_info.get("limitPx", 0)),
                    status=OrderStatus.SUBMITTED,
                    exchange_order_id=str(order_info.get("oid", "")),
                )
                orders.append(order)

            return orders

        except Exception as e:
            print(f"❌ Error getting open orders: {e}")
            return []

    async def set_leverage(self, asset: str, leverage: int, is_cross: bool = True) -> bool:
        """Set leverage for a perpetual asset.

        Spot assets are skipped (leverage not applicable).
        Returns True on success, False on failure.
        """
        if not self.is_connected:
            return False

        # Leverage doesn't apply to spot
        if self._is_spot(asset):
            return True

        try:
            result = self.exchange.update_leverage(
                leverage=leverage,
                name=asset,
                is_cross=is_cross,
            )
            if result and result.get("status") == "ok":
                return True
            return False
        except Exception as e:
            print(f"Error setting leverage for {asset}: {e}")
            return False

    async def get_user_fills(self, start_time: int) -> List[Dict[str, Any]]:
        """Get user fill history since start_time (epoch seconds).

        Returns raw fill dicts from the Hyperliquid API. Each fill has
        keys: coin, side, px, sz, oid, time, closedPnl, crossed.
        """
        if not self.is_connected:
            return []

        try:
            # SDK expects milliseconds
            start_ms = start_time * 1000 if start_time < 1e12 else start_time
            fills = self.info.user_fills_by_time(
                self.exchange.wallet.address,
                start_ms,
            )
            return fills if fills else []
        except Exception as e:
            print(f"Error getting user fills: {e}")
            return []

    async def health_check(self) -> bool:
        """Check connection health"""
        if not self.is_connected:
            return False

        try:
            # Simple health check - get account state
            self.info.user_state(self.exchange.wallet.address)
            return True
        except Exception:
            return False

    async def get_positions(self) -> List["Position"]:
        """Get all current positions from Hyperliquid"""
        if not self.is_connected:
            return []

        try:
            # Import Position here to avoid circular imports
            from interfaces.strategy import Position

            # Get user state which includes positions
            user_state = self.info.user_state(self.exchange.wallet.address)
            positions = []

            # Parse positions from user state
            if "assetPositions" in user_state:
                for pos_info in user_state["assetPositions"]:
                    if float(pos_info.get("position", {}).get("szi", 0)) != 0:
                        position_size = float(pos_info["position"]["szi"])
                        entry_price = float(pos_info["position"]["entryPx"] or 0)

                        # Get current price for PnL calculation
                        current_price = await self.get_market_price(
                            pos_info["position"]["coin"]
                        )
                        current_value = abs(position_size) * current_price

                        # Calculate unrealized PnL
                        if entry_price > 0:
                            unrealized_pnl = position_size * (
                                current_price - entry_price
                            )
                        else:
                            unrealized_pnl = 0.0

                        position = Position(
                            asset=pos_info["position"]["coin"],
                            size=position_size,
                            entry_price=entry_price,
                            current_value=current_value,
                            unrealized_pnl=unrealized_pnl,
                            timestamp=time.time(),
                        )
                        positions.append(position)

            return positions

        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            return []

    async def close_position(self, asset: str, size: Optional[float] = None) -> bool:
        """Close a position by placing a reduce-only IOC order.

        Uses dynamic precision from the cached meta() data.
        """
        if not self.is_connected:
            return False

        try:
            from hyperliquid.utils.signing import OrderType as HLOrderType

            # Get current positions to determine position details
            positions = await self.get_positions()
            target_position = None

            for pos in positions:
                if pos.asset == asset:
                    target_position = pos
                    break

            if not target_position:
                print(f"❌ No position found for {asset}")
                return False

            # Determine close size
            if size is None:
                close_size = abs(target_position.size)
            else:
                close_size = min(size, abs(target_position.size))

            # Closing = opposite side of current position
            is_buy = target_position.size < 0  # Short → buy to close

            # Slippage-adjusted price via dynamic precision
            market_price = await self.get_market_price(asset)
            slippage_price = market_price * (1.01 if is_buy else 0.99)
            limit_price = self._round_price(asset, slippage_price, is_buy)
            close_size = self._round_size(asset, close_size)

            result = self.exchange.order(
                name=asset,
                is_buy=is_buy,
                sz=close_size,
                limit_px=limit_price,
                order_type=HLOrderType({"limit": {"tif": "Ioc"}}),
                reduce_only=True,
            )

            if result and result.get("status") == "ok":
                print(f"✅ Position close order placed: {close_size} {asset}")
                return True
            else:
                print(f"❌ Failed to close position: {result}")
                return False

        except Exception as e:
            print(f"❌ Error closing position {asset}: {e}")
            return False

    async def get_account_metrics(self) -> Dict[str, Any]:
        """Get account-level metrics for risk assessment"""
        if not self.is_connected:
            return {
                "total_value": 0.0,
                "total_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "drawdown_pct": 0.0,
            }

        try:
            # Get user state
            user_state = self.info.user_state(self.exchange.wallet.address)

            # Calculate account metrics
            total_value = 0.0
            unrealized_pnl = 0.0

            # Get cross margin summary for total account value
            if "crossMarginSummary" in user_state:
                margin_summary = user_state["crossMarginSummary"]
                total_value = float(margin_summary.get("accountValue", 0))
                unrealized_pnl = float(margin_summary.get("totalMarginUsed", 0))

            # Get positions for detailed PnL
            positions = await self.get_positions()
            position_pnl = sum(pos.unrealized_pnl for pos in positions)

            # Calculate drawdown (simplified - would need historical high water mark)
            # For now, use unrealized PnL as proxy
            total_pnl = position_pnl

            # Estimate drawdown percentage (this would be more sophisticated in production)
            if total_value > 0:
                drawdown_pct = (
                    max(0, -total_pnl / total_value * 100) if total_pnl < 0 else 0.0
                )
            else:
                drawdown_pct = 0.0

            return {
                "total_value": total_value,
                "total_pnl": total_pnl,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": 0.0,  # Would need to track this separately
                "drawdown_pct": drawdown_pct,
                "positions_count": len(positions),
                "largest_position_pct": max(
                    [abs(pos.current_value) / total_value * 100 for pos in positions],
                    default=0.0,
                )
                if total_value > 0
                else 0.0,
            }

        except Exception as e:
            print(f"❌ Error getting account metrics: {e}")
            return {
                "total_value": 0.0,
                "total_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "drawdown_pct": 0.0,
            }
