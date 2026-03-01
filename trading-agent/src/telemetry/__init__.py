"""Trading telemetry: structured event collection for strategy mining."""

from .collector import NullCollector, TelemetryCollector
from .events import EventType, TelemetryEvent

__all__ = [
    "EventType",
    "NullCollector",
    "TelemetryCollector",
    "TelemetryEvent",
]
