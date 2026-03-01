"""Telemetry event definitions — typed structures for all bot telemetry."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class EventType(Enum):
    """All telemetry event types."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TICK = "tick"
    TICK_SNAPSHOT = "tick_snapshot"
    DC_EVENT = "dc_event"
    MOMENTUM_UPDATE = "momentum_update"
    SIGNAL = "signal"
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    FILL = "fill"
    RECONNECT = "reconnect"
    ACCOUNT_SNAPSHOT = "account_snapshot"


@dataclass
class TelemetryEvent:
    """Base telemetry event. All events share these fields."""

    event_type: str  # EventType.value
    timestamp: float  # epoch seconds
    session_id: str  # unique per bot run
    symbol: str
    bridge_type: str  # "single_scale" or "multi_scale"
    payload: dict[str, Any]  # event-specific data

    def to_ndjson(self) -> str:
        """Serialize to single-line JSON (NDJSON format)."""
        return json.dumps(asdict(self), default=str)
