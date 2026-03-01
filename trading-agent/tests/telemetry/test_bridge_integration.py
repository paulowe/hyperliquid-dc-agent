"""Tests for DC event callback integration with strategies."""

from __future__ import annotations

import time
from typing import Any

import pytest

from interfaces.strategy import MarketData, SignalType
from strategies.dc_overshoot.dc_overshoot_strategy import DCOvershootStrategy
from strategies.dc_overshoot.multi_scale_strategy import MultiScaleDCStrategy


def make_config(**overrides) -> dict[str, Any]:
    """Strategy config with small threshold for easy triggering."""
    cfg = {
        "symbol": "BTC",
        "dc_thresholds": [[0.001, 0.001]],
        "position_size_usd": 50.0,
        "max_position_size_usd": 200.0,
        "initial_stop_loss_pct": 0.003,
        "initial_take_profit_pct": 0.002,
        "trail_pct": 0.5,
        "cooldown_seconds": 0,
        "max_open_positions": 1,
        "log_events": False,
    }
    cfg.update(overrides)
    return cfg


def make_multi_scale_config(**overrides) -> dict[str, Any]:
    """Multi-scale config with sensor + trade thresholds."""
    cfg = {
        "symbol": "BTC",
        "sensor_thresholds": [[0.0005, 0.0005]],
        "trade_threshold": [0.001, 0.001],
        "position_size_usd": 50.0,
        "max_position_size_usd": 200.0,
        "initial_stop_loss_pct": 0.003,
        "initial_take_profit_pct": 0.002,
        "trail_pct": 0.5,
        "min_profit_to_trail_pct": 0.0,
        "cooldown_seconds": 0,
        "max_open_positions": 1,
        "log_events": False,
        "log_sensor_events": False,
        "momentum_alpha": 0.5,
        "min_momentum_score": 0.0,  # Accept all signals for testing
    }
    cfg.update(overrides)
    return cfg


def make_md(price: float, ts: float) -> MarketData:
    return MarketData(asset="BTC", price=price, volume_24h=0.0, timestamp=ts)


def drive_price_down(strategy, start_price=100_000.0, steps=20):
    """Feed monotonically decreasing ticks to trigger PDCC_Down."""
    price = start_price
    ts = 1000.0
    collected_events: list[dict] = []
    for _ in range(steps):
        price *= 0.99985
        ts += 1.0
        strategy.generate_signals(make_md(price, ts), [], 100_000.0)
    return price, ts


def drive_price_up(strategy, start_price, ts_start, steps=20):
    """Feed monotonically increasing ticks to trigger PDCC2_UP."""
    price = start_price
    ts = ts_start
    for _ in range(steps):
        price *= 1.00015
        ts += 1.0
        strategy.generate_signals(make_md(price, ts), [], 100_000.0)
    return price, ts


class TestDCEventCallback:
    """DCOvershootStrategy fires optional callback on DC events."""

    def test_callback_not_set_by_default(self) -> None:
        s = DCOvershootStrategy(make_config())
        assert s._on_dc_event is None

    def test_set_dc_event_callback(self) -> None:
        s = DCOvershootStrategy(make_config())
        events = []
        s.set_dc_event_callback(lambda e: events.append(e))
        assert s._on_dc_event is not None

    def test_callback_fires_on_dc_event(self) -> None:
        s = DCOvershootStrategy(make_config())
        s.start()
        events = []
        s.set_dc_event_callback(lambda e: events.append(e))

        # Drive price down to trigger PDCC_Down
        drive_price_down(s)

        assert len(events) >= 1
        assert events[0]["event_type"] in ("PDCC_Down", "PDCC2_UP", "OSV_Down", "OSV2_UP")

    def test_callback_receives_event_dict_fields(self) -> None:
        s = DCOvershootStrategy(make_config())
        s.start()
        events = []
        s.set_dc_event_callback(lambda e: events.append(e))

        drive_price_down(s)

        event = events[0]
        assert "event_type" in event
        assert "start_price" in event
        assert "end_price" in event
        assert "threshold_down" in event
        assert "threshold_up" in event

    def test_callback_exception_does_not_propagate(self) -> None:
        s = DCOvershootStrategy(make_config())
        s.start()

        def bad_callback(e):
            raise RuntimeError("telemetry crash")

        s.set_dc_event_callback(bad_callback)

        # Should not raise despite callback error
        price, ts = drive_price_down(s)
        # Strategy should continue to work
        signals = s.generate_signals(make_md(price * 0.99, ts + 1), [], 100_000.0)
        # No crash is the assertion

    def test_callback_fires_for_non_entry_events(self) -> None:
        """Callback fires for ALL DC events, including OSV events that don't produce signals."""
        s = DCOvershootStrategy(make_config())
        s.start()
        events = []
        s.set_dc_event_callback(lambda e: events.append(e))

        # Drive down → PDCC_Down, then up → may get OSV and PDCC2_UP
        price, ts = drive_price_down(s)
        drive_price_up(s, price, ts)

        event_types = {e["event_type"] for e in events}
        # Should have at least 2 different event types
        assert len(event_types) >= 2


class TestMultiScaleCallback:
    """MultiScaleDCStrategy fires callback with is_sensor flag."""

    def test_callback_fires_with_is_sensor_flag(self) -> None:
        s = MultiScaleDCStrategy(make_multi_scale_config())
        s.start()
        events = []
        s.set_dc_event_callback(lambda e: events.append(e))

        # Drive price to trigger events
        drive_price_down(s)

        assert len(events) >= 1
        # All events should have is_sensor field
        for e in events:
            assert "is_sensor" in e

    def test_callback_exception_suppressed(self) -> None:
        s = MultiScaleDCStrategy(make_multi_scale_config())
        s.start()
        s.set_dc_event_callback(lambda e: (_ for _ in ()).throw(ValueError("boom")))

        # Should not crash
        drive_price_down(s)
