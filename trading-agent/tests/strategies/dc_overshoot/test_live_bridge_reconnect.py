"""Tests for WebSocket reconnection and state reconciliation in live_bridge.

Tests cover:
- reconcile_on_reconnect() for all position state scenarios
- Water mark updates (only in favorable direction)
- Backstop OID management across reconnects
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path (same as live_bridge.py does)
_SRC_DIR = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from interfaces.strategy import Position
from strategies.dc_overshoot.trailing_risk_manager import TrailingRiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SL_PCT = 0.003
TP_PCT = 0.002
TRAIL_PCT = 0.5


def make_rm(asset: str = "HYPE") -> TrailingRiskManager:
    """Create a TrailingRiskManager with standard test config."""
    return TrailingRiskManager(
        asset=asset,
        initial_stop_loss_pct=SL_PCT,
        initial_take_profit_pct=TP_PCT,
        trail_pct=TRAIL_PCT,
    )


def make_strategy_mock(rm: TrailingRiskManager) -> MagicMock:
    """Create a mock strategy with a real TrailingRiskManager."""
    strategy = MagicMock()
    strategy._trailing_rm = rm
    return strategy


def make_adapter_mock(
    positions: list[Position] | None = None,
    market_price: float = 100.0,
) -> AsyncMock:
    """Create a mock adapter with configurable positions and market price."""
    adapter = AsyncMock()
    adapter.get_positions = AsyncMock(return_value=positions or [])
    adapter.get_market_price = AsyncMock(return_value=market_price)
    return adapter


def make_position(
    asset: str = "HYPE",
    size: float = 1.0,
    entry_price: float = 100.0,
) -> Position:
    """Create a Position dataclass for testing."""
    return Position(
        asset=asset,
        size=size,  # positive = long, negative = short
        entry_price=entry_price,
        current_value=abs(size) * entry_price,
        unrealized_pnl=0.0,
        timestamp=1000.0,
    )


# Import reconcile_on_reconnect after path setup
from strategies.dc_overshoot.live_bridge import reconcile_on_reconnect


# ---------------------------------------------------------------------------
# Scenario 1: Strategy has position, exchange has same position
# ---------------------------------------------------------------------------

class TestReconcilePositionIntact:
    """When both strategy and exchange agree on position, preserve state."""

    @pytest.mark.asyncio
    async def test_long_position_intact_returns_same_backstop(self):
        rm = make_rm()
        rm.open_position("LONG", 100.0, 1.0)
        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(
            positions=[make_position(size=1.0, entry_price=100.0)],
            market_price=101.0,
        )

        result = await reconcile_on_reconnect(adapter, strategy, "HYPE", 12345)

        assert result == 12345  # backstop preserved
        assert rm.has_position
        assert rm.side == "LONG"

    @pytest.mark.asyncio
    async def test_short_position_intact(self):
        rm = make_rm()
        rm.open_position("SHORT", 100.0, 1.0)
        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(
            positions=[make_position(size=-1.0, entry_price=100.0)],
            market_price=99.0,
        )

        result = await reconcile_on_reconnect(adapter, strategy, "HYPE", 99999)

        assert result == 99999
        assert rm.has_position
        assert rm.side == "SHORT"


# ---------------------------------------------------------------------------
# Scenario 2: Strategy has position, exchange does NOT
# ---------------------------------------------------------------------------

class TestReconcilePositionClosedDuringOutage:
    """When exchange position is gone (backstop SL fired), clear strategy state."""

    @pytest.mark.asyncio
    async def test_long_closed_clears_state(self):
        rm = make_rm()
        rm.open_position("LONG", 100.0, 1.0)
        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(positions=[])  # No positions on exchange

        result = await reconcile_on_reconnect(adapter, strategy, "HYPE", 12345)

        assert result is None  # backstop cleared
        assert not rm.has_position

    @pytest.mark.asyncio
    async def test_short_closed_clears_state(self):
        rm = make_rm()
        rm.open_position("SHORT", 100.0, 1.0)
        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(positions=[])

        result = await reconcile_on_reconnect(adapter, strategy, "HYPE", 77777)

        assert result is None
        assert not rm.has_position


# ---------------------------------------------------------------------------
# Scenario 3: No position on either side
# ---------------------------------------------------------------------------

class TestReconcileNoPositionBothSides:
    """When neither side has a position, no-op."""

    @pytest.mark.asyncio
    async def test_no_position_both_sides(self):
        rm = make_rm()  # No position opened
        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(positions=[])

        result = await reconcile_on_reconnect(adapter, strategy, "HYPE", None)

        assert result is None
        assert not rm.has_position


# ---------------------------------------------------------------------------
# Scenario 4: Strategy has NO position, exchange HAS position
# ---------------------------------------------------------------------------

class TestReconcileExternalPosition:
    """When exchange has a position the strategy doesn't know about, log and ignore."""

    @pytest.mark.asyncio
    async def test_external_position_ignored(self):
        rm = make_rm()  # No position in strategy
        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(
            positions=[make_position(size=2.0, entry_price=50.0)],
        )

        result = await reconcile_on_reconnect(adapter, strategy, "HYPE", None)

        assert result is None
        assert not rm.has_position  # Strategy state unchanged


# ---------------------------------------------------------------------------
# Scenario 5: Side mismatch between strategy and exchange
# ---------------------------------------------------------------------------

class TestReconcileSideMismatch:
    """When strategy thinks LONG but exchange is SHORT (or vice versa), clear state."""

    @pytest.mark.asyncio
    async def test_strategy_long_exchange_short_clears(self):
        rm = make_rm()
        rm.open_position("LONG", 100.0, 1.0)
        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(
            positions=[make_position(size=-1.0, entry_price=100.0)],  # SHORT
        )

        result = await reconcile_on_reconnect(adapter, strategy, "HYPE", 12345)

        assert result is None  # backstop cleared
        assert not rm.has_position

    @pytest.mark.asyncio
    async def test_strategy_short_exchange_long_clears(self):
        rm = make_rm()
        rm.open_position("SHORT", 100.0, 1.0)
        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(
            positions=[make_position(size=1.0, entry_price=100.0)],  # LONG
        )

        result = await reconcile_on_reconnect(adapter, strategy, "HYPE", 55555)

        assert result is None
        assert not rm.has_position


# ---------------------------------------------------------------------------
# Scenario 6: Water mark updates on reconnect (favorable direction only)
# ---------------------------------------------------------------------------

class TestReconcileWaterMarkUpdates:
    """Water marks should ratchet in favorable direction during outage gap."""

    @pytest.mark.asyncio
    async def test_long_hwm_updated_when_price_higher(self):
        rm = make_rm()
        rm.open_position("LONG", 100.0, 1.0)
        # Simulate hwm at 101 before disconnect
        rm._high_water_mark = 101.0

        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(
            positions=[make_position(size=1.0, entry_price=100.0)],
            market_price=103.0,  # Price moved up during outage
        )

        await reconcile_on_reconnect(adapter, strategy, "HYPE", 111)

        assert rm._high_water_mark == 103.0  # Updated to current price

    @pytest.mark.asyncio
    async def test_short_lwm_updated_when_price_lower(self):
        rm = make_rm()
        rm.open_position("SHORT", 100.0, 1.0)
        rm._low_water_mark = 99.0

        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(
            positions=[make_position(size=-1.0, entry_price=100.0)],
            market_price=97.0,  # Price dropped during outage
        )

        await reconcile_on_reconnect(adapter, strategy, "HYPE", 222)

        assert rm._low_water_mark == 97.0  # Updated


# ---------------------------------------------------------------------------
# Scenario 7: Water marks do NOT lower (only ratchet in favorable direction)
# ---------------------------------------------------------------------------

class TestReconcileWaterMarkNoWeaken:
    """Water marks should NOT move in unfavorable direction."""

    @pytest.mark.asyncio
    async def test_long_hwm_not_lowered(self):
        rm = make_rm()
        rm.open_position("LONG", 100.0, 1.0)
        rm._high_water_mark = 105.0  # High before disconnect

        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(
            positions=[make_position(size=1.0, entry_price=100.0)],
            market_price=102.0,  # Price pulled back during outage
        )

        await reconcile_on_reconnect(adapter, strategy, "HYPE", 333)

        assert rm._high_water_mark == 105.0  # Stays at old high

    @pytest.mark.asyncio
    async def test_short_lwm_not_raised(self):
        rm = make_rm()
        rm.open_position("SHORT", 100.0, 1.0)
        rm._low_water_mark = 95.0  # Low before disconnect

        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(
            positions=[make_position(size=-1.0, entry_price=100.0)],
            market_price=98.0,  # Price bounced up during outage
        )

        await reconcile_on_reconnect(adapter, strategy, "HYPE", 444)

        assert rm._low_water_mark == 95.0  # Stays at old low


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestReconcileEdgeCases:
    """Edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_get_positions_failure_preserves_state(self):
        """If REST call fails, keep current state (safe default)."""
        rm = make_rm()
        rm.open_position("LONG", 100.0, 1.0)
        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock()
        adapter.get_positions = AsyncMock(side_effect=Exception("Network error"))

        result = await reconcile_on_reconnect(adapter, strategy, "HYPE", 555)

        assert result == 555  # backstop preserved
        assert rm.has_position  # State unchanged

    @pytest.mark.asyncio
    async def test_get_market_price_failure_still_reconciles(self):
        """If market price fetch fails, position check still works."""
        rm = make_rm()
        rm.open_position("LONG", 100.0, 1.0)
        rm._high_water_mark = 101.0
        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(
            positions=[make_position(size=1.0, entry_price=100.0)],
        )
        adapter.get_market_price = AsyncMock(side_effect=Exception("Timeout"))

        result = await reconcile_on_reconnect(adapter, strategy, "HYPE", 666)

        # Position check succeeds, water mark update skipped gracefully
        assert result == 666
        assert rm.has_position
        assert rm._high_water_mark == 101.0  # Unchanged (price fetch failed)

    @pytest.mark.asyncio
    async def test_position_for_different_symbol_ignored(self):
        """Exchange has position for a different symbol — treated as no position."""
        rm = make_rm()
        rm.open_position("LONG", 100.0, 1.0)
        strategy = make_strategy_mock(rm)
        adapter = make_adapter_mock(
            positions=[make_position(asset="BTC", size=0.5, entry_price=90000.0)],
        )

        result = await reconcile_on_reconnect(adapter, strategy, "HYPE", 777)

        # Exchange has BTC position, not HYPE — strategy HYPE position is gone
        assert result is None
        assert not rm.has_position
