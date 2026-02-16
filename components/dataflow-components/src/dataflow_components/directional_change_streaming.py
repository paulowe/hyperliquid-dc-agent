"""Apache Beam streaming pipeline for directional change detection.

This module reads tick data from a Pub/Sub topic, applies the directional change
algorithm incrementally per instrument, and publishes detected events to another
Pub/Sub topic. It is designed to run on Cloud Dataflow in streaming mode.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import apache_beam as beam
from apache_beam.coders import coders
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions, StandardOptions
from apache_beam.io.gcp.pubsub import PubsubMessage
from apache_beam.pvalue import TaggedOutput
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.utils.timestamp import Timestamp


@dataclass
class TrendSnapshot:
    """Serializable snapshot of the directional change state for one threshold."""

    threshold_down: float
    threshold_up: float
    # Down-trend tracking
    max_price: Optional[float] = None
    max_price_time: Optional[float] = None  # epoch seconds
    down_active: bool = False
    down_pdcc_price: Optional[float] = None
    down_pdcc_time: Optional[float] = None
    down_min_price: Optional[float] = None
    down_min_time: Optional[float] = None
    down_osv_price: Optional[float] = None
    down_osv_time: Optional[float] = None
    down_max_osv: float = 0.0
    # Up-trend tracking
    min_price2: Optional[float] = None
    min_price2_time: Optional[float] = None
    up_active: bool = False
    up_pdcc_price: Optional[float] = None
    up_pdcc_time: Optional[float] = None
    up_max_price: Optional[float] = None
    up_max_time: Optional[float] = None
    up_osv_price: Optional[float] = None
    up_osv_time: Optional[float] = None
    up_max_osv: float = 0.0

    def initialize(self, price: float, event_ts: float) -> None:
        if self.max_price is None:
            self.max_price = price
            self.max_price_time = event_ts
        if self.min_price2 is None:
            self.min_price2 = price
            self.min_price2_time = event_ts


@dataclass
class Tick:
    symbol: str
    price: float
    event_time: datetime
    raw: Dict

    @property
    def event_timestamp(self) -> float:
        return self.event_time.timestamp()


class ParseTickDoFn(beam.DoFn):
    """Parse raw Pub/Sub messages into keyed tick structures."""

    def __init__(
        self,
        *,
        symbol_field: str,
        price_field: str,
        timestamp_field: str,
    ) -> None:
        self._symbol_field = symbol_field
        self._price_field = price_field
        self._timestamp_field = timestamp_field
        self._decode_errors = beam.metrics.Metrics.counter(
            "directional_change", "parse_errors"
        )

    def process(self, message) -> Iterable:
        try:
            payload_bytes: bytes
            if isinstance(message, PubsubMessage):
                payload_bytes = message.data
            elif isinstance(message, bytes):
                payload_bytes = message
            else:
                payload_bytes = str(message).encode("utf-8")

            record = json.loads(payload_bytes.decode("utf-8"))
            symbol = record[self._symbol_field]
            price = float(record[self._price_field])
            event_time = _coerce_timestamp(record[self._timestamp_field])

            keyed = (
                symbol,
                {
                    "price": price,
                    "event_ts": event_time.timestamp(),
                    "raw": record,
                },
            )
            yield beam.window.TimestampedValue(keyed, Timestamp(event_time.timestamp()))
        except Exception as exc:  # pylint: disable=broad-except
            self._decode_errors.inc()
            error_payload = {
                "error": str(exc),
                "payload": payload_bytes.decode("utf-8", errors="replace"),
            }
            yield TaggedOutput(
                "errors", json.dumps(error_payload).encode("utf-8")
            )


class DirectionalChangeDetector(beam.DoFn):
    """Stateful DoFn that emits directional change events per instrument."""

    def __init__(self, thresholds: List[Tuple[float, float]]):
        self._thresholds = [TrendSnapshot(down, up) for down, up in thresholds]
        self._event_counter = beam.metrics.Metrics.counter(
            "directional_change", "events_emitted"
        )

    _STATE_SPEC = ReadModifyWriteStateSpec(
        name="dc_state", coder=coders.PickleCoder()
    )

    def process(
        self,
        element,
        timestamp=beam.DoFn.TimestampParam,
        state=beam.DoFn.StateParam(_STATE_SPEC),
    ):
        symbol, tick_dict = element
        state_value: Optional[Dict[str, TrendSnapshot]] = state.read()
        tick = Tick(
            symbol=symbol,
            price=tick_dict["price"],
            event_time=datetime.fromtimestamp(tick_dict["event_ts"], tz=timezone.utc),
            raw=tick_dict,
        )

        if state_value is None:
            state_value = {}

        all_events: List[Dict] = []
        for snapshot_template in self._thresholds:
            key = f"{snapshot_template.threshold_down}:{snapshot_template.threshold_up}"
            snapshot = state_value.get(key)
            if snapshot is None:
                snapshot = TrendSnapshot(
                    threshold_down=snapshot_template.threshold_down,
                    threshold_up=snapshot_template.threshold_up,
                )
                state_value[key] = snapshot
            snapshot.initialize(tick.price, tick.event_timestamp)
            events = self._update_snapshot(snapshot, tick)
            for evt in events:
                evt["symbol"] = symbol
                evt["threshold_down"] = snapshot.threshold_down
                evt["threshold_up"] = snapshot.threshold_up
                evt["event_timestamp"] = tick.event_timestamp
                evt["ingested_at"] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
            all_events.extend(events)

        state.write(state_value)
        for event in all_events:
            self._event_counter.inc()
            yield event

    def _update_snapshot(self, snapshot: TrendSnapshot, tick: Tick) -> List[Dict]:
        price = tick.price
        event_ts = tick.event_timestamp
        events: List[Dict] = []

        # Section A: Detect start of downward trend
        if not snapshot.down_active and snapshot.max_price is not None:
            price_drop = snapshot.max_price - price
            if (
                snapshot.max_price > price
                and price_drop >= snapshot.max_price * snapshot.threshold_down
            ):
                events.append(
                    self._make_event(
                        event_type="PDCC_Down",
                        start_time=snapshot.max_price_time,
                        end_time=event_ts,
                        start_price=snapshot.max_price,
                        end_price=price,
                    )
                )
                snapshot.down_active = True
                snapshot.down_pdcc_price = price
                snapshot.down_pdcc_time = event_ts
                snapshot.down_min_price = price
                snapshot.down_min_time = event_ts
                snapshot.down_osv_price = price
                snapshot.down_osv_time = event_ts
                snapshot.down_max_osv = 0.0
                snapshot.up_active = False
                snapshot.up_pdcc_price = None
                snapshot.up_pdcc_time = None
                snapshot.up_max_price = None
                snapshot.up_max_time = None
                if snapshot.up_osv_time is not None and snapshot.up_pdcc_time is not None:
                    events.append(
                        self._make_event(
                            event_type="OSV_Up",
                            start_time=snapshot.up_pdcc_time,
                            end_time=snapshot.up_osv_time,
                            start_price=snapshot.up_pdcc_price,
                            end_price=snapshot.up_osv_price,
                            extra={"max_osv": snapshot.up_max_osv},
                        )
                    )
                snapshot.up_osv_price = None
                snapshot.up_osv_time = None
                snapshot.up_max_osv = 0.0
                snapshot.min_price2 = None
                snapshot.min_price2_time = None

            if snapshot.max_price is None or price > snapshot.max_price:
                snapshot.max_price = price
                snapshot.max_price_time = event_ts
        elif not snapshot.down_active:
            snapshot.max_price = price
            snapshot.max_price_time = event_ts

        # Section B: Overshoot tracking while in down trend
        if snapshot.down_active and snapshot.down_pdcc_price:
            if snapshot.down_min_price is None or price <= snapshot.down_min_price or snapshot.up_active:
                snapshot.down_min_price = price
                snapshot.down_min_time = event_ts
                snapshot.down_osv_price = price
                snapshot.down_osv_time = event_ts

            if price > snapshot.down_min_price:
                overshoot = (snapshot.down_pdcc_price - price) / (
                    snapshot.down_pdcc_price * snapshot.threshold_down
                )
                snapshot.down_max_osv = max(snapshot.down_max_osv, overshoot)
                if overshoot < 0 and snapshot.up_active:
                    snapshot.down_active = False
                    snapshot.down_min_price = None
                    snapshot.down_min_time = None
                    snapshot.down_pdcc_price = None
                    snapshot.down_pdcc_time = None
                    snapshot.max_price = price
                    snapshot.max_price_time = event_ts

        # Section C: Detect start of upward trend
        if not snapshot.up_active and snapshot.min_price2 is not None:
            price_rise = price - snapshot.min_price2
            if (
                price > snapshot.min_price2
                and price_rise >= snapshot.min_price2 * snapshot.threshold_up
            ):
                events.append(
                    self._make_event(
                        event_type="PDCC2_UP",
                        start_time=snapshot.min_price2_time,
                        end_time=event_ts,
                        start_price=snapshot.min_price2,
                        end_price=price,
                    )
                )
                snapshot.up_active = True
                snapshot.up_pdcc_price = price
                snapshot.up_pdcc_time = event_ts
                snapshot.up_max_price = price
                snapshot.up_max_time = event_ts
                snapshot.up_osv_price = price
                snapshot.up_osv_time = event_ts
                snapshot.up_max_osv = 0.0
                if snapshot.down_pdcc_time is not None and snapshot.down_osv_time is not None:
                    events.append(
                        self._make_event(
                            event_type="OSV_Down",
                            start_time=snapshot.down_pdcc_time,
                            end_time=snapshot.down_osv_time,
                            start_price=snapshot.down_pdcc_price,
                            end_price=snapshot.down_osv_price,
                            extra={"max_osv": snapshot.down_max_osv},
                        )
                    )
                snapshot.down_active = False
                snapshot.down_pdcc_price = None
                snapshot.down_pdcc_time = None
                snapshot.down_min_price = None
                snapshot.down_min_time = None
                snapshot.down_osv_price = None
                snapshot.down_osv_time = None
                snapshot.down_max_osv = 0.0
                snapshot.max_price = None
                snapshot.max_price_time = None

            if snapshot.min_price2 is None or price < snapshot.min_price2:
                snapshot.min_price2 = price
                snapshot.min_price2_time = event_ts
        elif not snapshot.up_active:
            snapshot.min_price2 = price
            snapshot.min_price2_time = event_ts

        # Section D: Overshoot tracking while in up trend
        if snapshot.up_active and snapshot.up_pdcc_price:
            if price >= snapshot.up_max_price or snapshot.down_active:
                snapshot.up_max_price = price
                snapshot.up_max_time = event_ts
                snapshot.up_osv_price = price
                snapshot.up_osv_time = event_ts
            elif price < snapshot.up_max_price:
                overshoot = (price - snapshot.up_pdcc_price) / (
                    snapshot.up_pdcc_price * snapshot.threshold_up
                )
                snapshot.up_max_osv = max(snapshot.up_max_osv, overshoot)
                if overshoot < 0 and snapshot.down_active:
                    snapshot.up_active = False
                    snapshot.up_pdcc_price = None
                    snapshot.up_pdcc_time = None
                    snapshot.up_max_price = None
                    snapshot.up_max_time = None
                    snapshot.min_price2 = price
                    snapshot.min_price2_time = event_ts

        return [evt for evt in events if evt["start_time"] and evt["end_time"]]

    @staticmethod
    def _make_event(
        *,
        event_type: str,
        start_time: Optional[float],
        end_time: Optional[float],
        start_price: Optional[float],
        end_price: Optional[float],
        extra: Optional[Dict] = None,
    ) -> Dict:
        payload = {
            "event_type": event_type,
            "start_time": _to_rfc3339(start_time),
            "end_time": _to_rfc3339(end_time),
            "start_price": start_price,
            "end_price": end_price,
        }
        if extra:
            payload.update(extra)
        return payload


def _to_rfc3339(epoch_seconds: Optional[float]) -> Optional[str]:
    if epoch_seconds is None:
        return None
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _parse_thresholds(raw_values: Iterable[str]) -> List[Tuple[float, float]]:
    thresholds: List[Tuple[float, float]] = []
    for raw in raw_values:
        parts = raw.split(":")
        if len(parts) == 1:
            val = float(parts[0])
            thresholds.append((val, val))
        elif len(parts) == 2:
            down, up = parts
            thresholds.append((float(down), float(up)))
        else:
            raise ValueError(
                f"Invalid threshold specification '{raw}'. Use '0.01' or '0.01:0.015'."
            )
    if not thresholds:
        raise ValueError("At least one threshold must be provided.")
    return thresholds

def _coerce_timestamp(raw_value) -> datetime:
    if isinstance(raw_value, (int, float)):
        return datetime.fromtimestamp(float(raw_value), tz=timezone.utc)
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Unable to parse timestamp '{raw_value}'") from exc
    raise ValueError(f"Unsupported timestamp value: {raw_value!r}")


def run(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_topic", help="Pub/Sub topic to read ticks from.")
    parser.add_argument(
        "--input_subscription",
        help="Pub/Sub subscription to read ticks from (exclusive with topic).",
    )
    parser.add_argument("--output_topic", required=True, help="Pub/Sub topic for events.")
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        action="append",
        required=True,
        help="Directional change threshold. Use 'value' or 'down:up'. Repeat for multiple thresholds.",
    )
    parser.add_argument(
        "--price_field",
        default="price",
        help="JSON field containing the numeric price.",
    )
    parser.add_argument(
        "--timestamp_field",
        default="timestamp",
        help="JSON field containing the event timestamp.",
    )
    parser.add_argument(
        "--symbol_field",
        default="symbol",
        help="JSON field containing the instrument identifier.",
    )
    parser.add_argument(
        "--dead_letter_topic",
        default=None,
        help="Optional Pub/Sub topic to send records that fail parsing.",
    )

    known_args, pipeline_args = parser.parse_known_args(argv)
    thresholds = _parse_thresholds(known_args.thresholds)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(StandardOptions).streaming = True
    pipeline_options.view_as(SetupOptions).save_main_session = True

    detector = DirectionalChangeDetector(thresholds)

    with beam.Pipeline(options=pipeline_options) as pipeline:
        if known_args.input_subscription:
            if known_args.input_topic:
                raise ValueError(
                    "Provide either --input_topic or --input_subscription, not both."
                )
            raw_messages = pipeline | "ReadSubscription" >> beam.io.ReadFromPubSub(
                subscription=known_args.input_subscription,
                with_attributes=True,
            )
        elif known_args.input_topic:
            raw_messages = pipeline | "ReadTopic" >> beam.io.ReadFromPubSub(
                topic=known_args.input_topic,
                with_attributes=True,
            )
        else:
            raise ValueError("Either --input_topic or --input_subscription must be specified.")

        parse_results = (
            raw_messages
            | "ParseTicks"
            >> beam.ParDo(
                ParseTickDoFn(
                    symbol_field=known_args.symbol_field,
                    price_field=known_args.price_field,
                    timestamp_field=known_args.timestamp_field,
                )
            ).with_outputs("errors", main="ticks")
        )

        ticks = parse_results.ticks
        errors = parse_results.errors

        events = ticks | "DetectDirectionalChange" >> beam.ParDo(detector)

        serialized_events = (
            events
            | "Serialize" >> beam.Map(lambda evt: json.dumps(evt).encode("utf-8"))
        )

        _ = serialized_events | "WriteEvents" >> beam.io.WriteToPubSub(
            topic=known_args.output_topic
        )

        if known_args.dead_letter_topic:
            _ = errors | "WriteDeadLetter" >> beam.io.WriteToPubSub(
                topic=known_args.dead_letter_topic
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
