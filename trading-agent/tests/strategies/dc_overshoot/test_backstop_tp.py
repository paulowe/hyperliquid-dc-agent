"""Tests for backstop take-profit placement and cancellation in live_bridge.

Mirrors the backstop SL pattern: exchange-level trigger order that captures
profit if the bot crashes while a position is in profit.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path
_SRC_DIR = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from strategies.dc_overshoot.live_bridge import (
    cancel_backstop_tp,
    place_backstop_tp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_adapter_mock(response=None):
    """Create a mock adapter with configurable order response."""
    adapter = MagicMock()
    adapter._round_size.return_value = 1.0
    if response is None:
        response = {
            "status": "ok",
            "response": {"data": {"statuses": [{"resting": {"oid": 42}}]}},
        }
    adapter.exchange.order.return_value = response
    return adapter


# ---------------------------------------------------------------------------
# LONG backstop TP: trigger above entry, sell to close
# ---------------------------------------------------------------------------

class TestPlaceBackstopTpLong:
    """Backstop TP for LONG: sell trigger above entry price."""

    @pytest.mark.asyncio
    async def test_returns_oid_on_success(self):
        adapter = make_adapter_mock()
        oid = await place_backstop_tp(adapter, "BTC", "LONG", 100_000.0, 1.0, 0.10)
        assert oid == 42

    @pytest.mark.asyncio
    async def test_is_sell_to_close_long(self):
        adapter = make_adapter_mock()
        await place_backstop_tp(adapter, "BTC", "LONG", 100_000.0, 1.0, 0.10)

        call = adapter.exchange.order.call_args
        assert call.kwargs["is_buy"] is False  # Sell to close long

    @pytest.mark.asyncio
    async def test_reduce_only(self):
        adapter = make_adapter_mock()
        await place_backstop_tp(adapter, "BTC", "LONG", 100_000.0, 1.0, 0.10)

        call = adapter.exchange.order.call_args
        assert call.kwargs["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_trigger_above_entry(self):
        """LONG TP trigger = entry * (1 + backstop_tp_pct)."""
        adapter = make_adapter_mock()
        await place_backstop_tp(adapter, "BTC", "LONG", 100_000.0, 1.0, 0.10)

        call = adapter.exchange.order.call_args
        order_type = call.kwargs["order_type"]
        # HLOrderType is a plain dict: {"trigger": {"triggerPx": ..., ...}}
        trigger_px = order_type["trigger"]["triggerPx"]
        assert trigger_px == pytest.approx(110_000.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_uses_tpsl_tp(self):
        """Must use tpsl='tp' not tpsl='sl'."""
        adapter = make_adapter_mock()
        await place_backstop_tp(adapter, "BTC", "LONG", 100_000.0, 1.0, 0.10)

        call = adapter.exchange.order.call_args
        order_type = call.kwargs["order_type"]
        assert order_type["trigger"]["tpsl"] == "tp"

    @pytest.mark.asyncio
    async def test_limit_below_trigger_for_sell(self):
        """For LONG TP (sell), limit_px should be below trigger (slippage room)."""
        adapter = make_adapter_mock()
        await place_backstop_tp(adapter, "BTC", "LONG", 100_000.0, 1.0, 0.10)

        call = adapter.exchange.order.call_args
        limit_px = call.kwargs["limit_px"]
        order_type = call.kwargs["order_type"]
        trigger_px = order_type["trigger"]["triggerPx"]
        assert limit_px < trigger_px


# ---------------------------------------------------------------------------
# SHORT backstop TP: trigger below entry, buy to close
# ---------------------------------------------------------------------------

class TestPlaceBackstopTpShort:
    """Backstop TP for SHORT: buy trigger below entry price."""

    @pytest.mark.asyncio
    async def test_returns_oid_on_success(self):
        adapter = make_adapter_mock()
        oid = await place_backstop_tp(adapter, "BTC", "SHORT", 100_000.0, 1.0, 0.10)
        assert oid == 42

    @pytest.mark.asyncio
    async def test_is_buy_to_close_short(self):
        adapter = make_adapter_mock()
        await place_backstop_tp(adapter, "BTC", "SHORT", 100_000.0, 1.0, 0.10)

        call = adapter.exchange.order.call_args
        assert call.kwargs["is_buy"] is True  # Buy to close short

    @pytest.mark.asyncio
    async def test_trigger_below_entry(self):
        """SHORT TP trigger = entry * (1 - backstop_tp_pct)."""
        adapter = make_adapter_mock()
        await place_backstop_tp(adapter, "BTC", "SHORT", 100_000.0, 1.0, 0.10)

        call = adapter.exchange.order.call_args
        order_type = call.kwargs["order_type"]
        # HLOrderType is a plain dict
        trigger_px = order_type["trigger"]["triggerPx"]
        assert trigger_px == pytest.approx(90_000.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_limit_above_trigger_for_buy(self):
        """For SHORT TP (buy), limit_px should be above trigger."""
        adapter = make_adapter_mock()
        await place_backstop_tp(adapter, "BTC", "SHORT", 100_000.0, 1.0, 0.10)

        call = adapter.exchange.order.call_args
        limit_px = call.kwargs["limit_px"]
        order_type = call.kwargs["order_type"]
        trigger_px = order_type["trigger"]["triggerPx"]
        assert limit_px > trigger_px


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPlaceBackstopTpEdgeCases:
    """Immediate fill, errors, unexpected responses."""

    @pytest.mark.asyncio
    async def test_immediate_fill_returns_none(self):
        """If TP triggers immediately (price already past), return None."""
        adapter = make_adapter_mock(response={
            "status": "ok",
            "response": {"data": {"statuses": [{"filled": {"oid": 99}}]}},
        })
        oid = await place_backstop_tp(adapter, "BTC", "LONG", 100_000.0, 1.0, 0.10)
        assert oid is None

    @pytest.mark.asyncio
    async def test_unexpected_response_returns_none(self):
        adapter = make_adapter_mock(response={"status": "err", "response": "failed"})
        oid = await place_backstop_tp(adapter, "BTC", "LONG", 100_000.0, 1.0, 0.10)
        assert oid is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        """If order placement raises, return None gracefully."""
        adapter = MagicMock()
        adapter._round_size.side_effect = Exception("Network error")
        oid = await place_backstop_tp(adapter, "BTC", "LONG", 100_000.0, 1.0, 0.10)
        assert oid is None


# ---------------------------------------------------------------------------
# Cancel backstop TP
# ---------------------------------------------------------------------------

class TestCancelBackstopTp:
    """Tests for cancel_backstop_tp()."""

    @pytest.mark.asyncio
    async def test_cancel_calls_exchange(self):
        adapter = MagicMock()
        await cancel_backstop_tp(adapter, "BTC", 42)
        adapter.exchange.cancel.assert_called_once_with("BTC", 42)

    @pytest.mark.asyncio
    async def test_cancel_none_is_noop(self):
        adapter = MagicMock()
        await cancel_backstop_tp(adapter, "BTC", None)
        adapter.exchange.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_exception_does_not_raise(self):
        adapter = MagicMock()
        adapter.exchange.cancel.side_effect = Exception("API error")
        # Should not raise
        await cancel_backstop_tp(adapter, "BTC", 42)
