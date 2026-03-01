"""Tests for telemetry event types and serialization."""

from __future__ import annotations

import json

import pytest

from telemetry.events import EventType, TelemetryEvent


class TestEventType:
    """EventType enum covers all telemetry event categories."""

    def test_all_event_types_present(self) -> None:
        expected = {
            "SESSION_START",
            "SESSION_END",
            "TICK",
            "TICK_SNAPSHOT",
            "DC_EVENT",
            "MOMENTUM_UPDATE",
            "SIGNAL",
            "TRADE_ENTRY",
            "TRADE_EXIT",
            "FILL",
            "RECONNECT",
            "ACCOUNT_SNAPSHOT",
        }
        actual = {e.name for e in EventType}
        assert actual == expected

    def test_values_are_snake_case(self) -> None:
        for e in EventType:
            assert e.value == e.name.lower()


class TestTelemetryEvent:
    """TelemetryEvent dataclass serialization."""

    def _make_event(self, **overrides) -> TelemetryEvent:
        defaults = {
            "event_type": EventType.TRADE_ENTRY.value,
            "timestamp": 1709000000.123,
            "session_id": "abc123def456",
            "symbol": "HYPE",
            "bridge_type": "single_scale",
            "payload": {"side": "LONG", "entry_price": 28.67},
        }
        defaults.update(overrides)
        return TelemetryEvent(**defaults)

    def test_to_ndjson_returns_valid_json(self) -> None:
        event = self._make_event()
        line = event.to_ndjson()
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    def test_to_ndjson_is_single_line(self) -> None:
        event = self._make_event()
        line = event.to_ndjson()
        assert "\n" not in line

    def test_to_ndjson_contains_all_fields(self) -> None:
        event = self._make_event()
        parsed = json.loads(event.to_ndjson())
        assert parsed["event_type"] == "trade_entry"
        assert parsed["timestamp"] == 1709000000.123
        assert parsed["session_id"] == "abc123def456"
        assert parsed["symbol"] == "HYPE"
        assert parsed["bridge_type"] == "single_scale"
        assert parsed["payload"]["side"] == "LONG"
        assert parsed["payload"]["entry_price"] == 28.67

    def test_payload_round_trips(self) -> None:
        payload = {
            "threshold": 0.015,
            "sl_pct": 0.015,
            "nested": {"a": 1, "b": [2, 3]},
        }
        event = self._make_event(payload=payload)
        parsed = json.loads(event.to_ndjson())
        assert parsed["payload"] == payload

    def test_empty_payload(self) -> None:
        event = self._make_event(payload={})
        parsed = json.loads(event.to_ndjson())
        assert parsed["payload"] == {}

    def test_payload_with_none_values(self) -> None:
        event = self._make_event(payload={"momentum_score": None, "regime": None})
        parsed = json.loads(event.to_ndjson())
        assert parsed["payload"]["momentum_score"] is None

    def test_different_event_types_serialize(self) -> None:
        for et in EventType:
            event = self._make_event(event_type=et.value)
            parsed = json.loads(event.to_ndjson())
            assert parsed["event_type"] == et.value
