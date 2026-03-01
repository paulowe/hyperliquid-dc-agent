"""Tests for TelemetryCollector (GCS → BQ pipeline + local fallback) and NullCollector."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from telemetry.collector import (
    NullCollector,
    TelemetryCollector,
    _EVENT_TABLE_MAP,
    _BUFFER_SIZES,
    _flatten_event,
    _get_table,
)
from telemetry.events import EventType, TelemetryEvent


# ---------------------------------------------------------------------------
# NullCollector
# ---------------------------------------------------------------------------


class TestNullCollector:
    """NullCollector discards everything with zero overhead."""

    def test_emit_does_nothing(self) -> None:
        c = NullCollector()
        c.emit(EventType.TRADE_ENTRY, {"side": "LONG"})

    def test_flush_does_nothing(self) -> None:
        c = NullCollector()
        c.flush()

    def test_close_does_nothing(self) -> None:
        c = NullCollector()
        c.close()

    def test_session_id_is_empty(self) -> None:
        c = NullCollector()
        assert c.session_id == ""


# ---------------------------------------------------------------------------
# Event → table mapping
# ---------------------------------------------------------------------------


class TestEventTableMapping:
    """Every EventType maps to a known BQ table."""

    def test_all_event_types_mapped(self) -> None:
        for et in EventType:
            assert et in _EVENT_TABLE_MAP, f"{et.name} not in _EVENT_TABLE_MAP"

    def test_tick_maps_to_ticks(self) -> None:
        assert _get_table(EventType.TICK) == "ticks"
        assert _get_table(EventType.TICK_SNAPSHOT) == "ticks"

    def test_dc_event_maps_to_dc_events(self) -> None:
        assert _get_table(EventType.DC_EVENT) == "dc_events"
        assert _get_table(EventType.MOMENTUM_UPDATE) == "dc_events"

    def test_signal_maps_to_signals(self) -> None:
        assert _get_table(EventType.SIGNAL) == "signals"

    def test_trade_maps_to_trades(self) -> None:
        assert _get_table(EventType.TRADE_ENTRY) == "trades"
        assert _get_table(EventType.TRADE_EXIT) == "trades"

    def test_fill_maps_to_fills(self) -> None:
        assert _get_table(EventType.FILL) == "fills"

    def test_session_maps_to_sessions(self) -> None:
        assert _get_table(EventType.SESSION_START) == "sessions"
        assert _get_table(EventType.SESSION_END) == "sessions"
        assert _get_table(EventType.RECONNECT) == "sessions"

    def test_account_snapshot_maps(self) -> None:
        assert _get_table(EventType.ACCOUNT_SNAPSHOT) == "account_snapshots"

    def test_ticks_buffer_size_is_1000(self) -> None:
        assert _BUFFER_SIZES["ticks"] == 1000

    def test_low_volume_tables_buffer_is_1(self) -> None:
        """All non-tick tables flush immediately (buffer_size=1)."""
        for table in ("trades", "signals", "fills", "sessions", "dc_events", "account_snapshots"):
            assert _BUFFER_SIZES[table] == 1, f"{table} should have buffer_size=1"


# ---------------------------------------------------------------------------
# _flatten_event
# ---------------------------------------------------------------------------


class TestFlattenEvent:
    """_flatten_event produces BQ-compatible flat NDJSON (no nested payload)."""

    def test_common_fields_included(self) -> None:
        event = TelemetryEvent(
            event_type="trade_entry",
            timestamp=1700000000.0,
            session_id="abc123",
            symbol="HYPE",
            bridge_type="single_scale",
            payload={"side": "LONG", "price": 28.67},
        )
        parsed = json.loads(_flatten_event(event))
        assert parsed["timestamp"] == 1700000000.0
        assert parsed["session_id"] == "abc123"
        assert parsed["symbol"] == "HYPE"
        assert parsed["event_type"] == "trade_entry"
        assert parsed["bridge_type"] == "single_scale"

    def test_payload_fields_merged_at_top_level(self) -> None:
        event = TelemetryEvent(
            event_type="fill",
            timestamp=1700000000.0,
            session_id="abc123",
            symbol="SOL",
            bridge_type="multi_scale",
            payload={"price": 145.0, "size": 10.0, "fee": 0.05},
        )
        parsed = json.loads(_flatten_event(event))
        # Payload fields at top level, not nested
        assert parsed["price"] == 145.0
        assert parsed["size"] == 10.0
        assert parsed["fee"] == 0.05
        assert "payload" not in parsed

    def test_payload_can_override_common_fields(self) -> None:
        """DC event payloads have their own event_type (e.g., 'PDCC_Down')."""
        event = TelemetryEvent(
            event_type="dc_event",
            timestamp=1700000000.0,
            session_id="abc123",
            symbol="HYPE",
            bridge_type="single_scale",
            payload={"event_type": "PDCC_Down", "start_price": 28.0},
        )
        parsed = json.loads(_flatten_event(event))
        # Payload's event_type overrides the common one
        assert parsed["event_type"] == "PDCC_Down"

    def test_output_is_single_line_json(self) -> None:
        event = TelemetryEvent(
            event_type="tick",
            timestamp=1700000000.0,
            session_id="abc123",
            symbol="HYPE",
            bridge_type="single_scale",
            payload={"price": 28.5},
        )
        line = _flatten_event(event)
        assert "\n" not in line
        json.loads(line)  # valid JSON


# ---------------------------------------------------------------------------
# TelemetryCollector — basic properties
# ---------------------------------------------------------------------------


class TestCollectorProperties:
    """Session ID, event counts, etc."""

    def _make(self, tmp_path: Path, **kw) -> TelemetryCollector:
        defaults = {"symbol": "HYPE", "bridge_type": "single_scale", "local_dir": tmp_path, "gcs_bucket": ""}
        defaults.update(kw)
        return TelemetryCollector(**defaults)

    def test_session_id_is_12_hex(self, tmp_path: Path) -> None:
        c = self._make(tmp_path)
        assert len(c.session_id) == 12
        int(c.session_id, 16)  # validates hex

    def test_session_id_unique(self, tmp_path: Path) -> None:
        c1 = self._make(tmp_path)
        c2 = self._make(tmp_path)
        assert c1.session_id != c2.session_id

    def test_event_count_starts_at_zero(self, tmp_path: Path) -> None:
        c = self._make(tmp_path)
        assert c.event_count == 0

    def test_event_count_increments(self, tmp_path: Path) -> None:
        c = self._make(tmp_path)
        c.emit(EventType.TICK, {"price": 100})
        c.emit(EventType.TICK, {"price": 101})
        c.emit(EventType.SIGNAL, {"type": "buy"})
        assert c.event_count == 3


# ---------------------------------------------------------------------------
# TelemetryCollector — local writes (no GCS)
# ---------------------------------------------------------------------------


class TestLocalWrites:
    """With no GCS bucket configured, events flush to local files."""

    def _make(self, tmp_path: Path, **kw) -> TelemetryCollector:
        defaults = {
            "symbol": "HYPE",
            "bridge_type": "single_scale",
            "local_dir": tmp_path,
            "gcs_bucket": "",  # disable GCS
        }
        defaults.update(kw)
        return TelemetryCollector(**defaults)

    def test_emit_and_flush_writes_flat_ndjson(self, tmp_path: Path) -> None:
        """Events are written as flat NDJSON (no nested payload wrapper)."""
        c = self._make(tmp_path)
        # trades buffer=1, so each emit auto-flushes
        c.emit(EventType.TRADE_ENTRY, {"side": "LONG", "price": 28.67})
        c.emit(EventType.TRADE_EXIT, {"side": "LONG", "pnl": 5.87})

        # Both map to "trades" table
        trades_dir = tmp_path / "trades" / "HYPE"
        ndjson_files = list(trades_dir.rglob("*.ndjson"))
        assert len(ndjson_files) == 1

        lines = ndjson_files[0].read_text().strip().split("\n")
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        # Flat format: fields at top level
        assert parsed["symbol"] == "HYPE"
        assert parsed["side"] == "LONG"
        assert parsed["price"] == 28.67
        assert parsed["event_type"] == "trade_entry"
        # No nested payload
        assert "payload" not in parsed

    def test_different_event_types_write_to_separate_files(self, tmp_path: Path) -> None:
        c = self._make(tmp_path)
        c.emit(EventType.TICK, {"price": 100})
        c.emit(EventType.SIGNAL, {"type": "buy"})
        c.flush()

        ticks_files = list((tmp_path / "ticks").rglob("*.ndjson"))
        signals_files = list((tmp_path / "signals").rglob("*.ndjson"))
        assert len(ticks_files) == 1
        assert len(signals_files) == 1

    def test_auto_flush_at_buffer_1(self, tmp_path: Path) -> None:
        """Trades, signals, fills, etc. have buffer_size=1 (immediate flush)."""
        c = self._make(tmp_path)
        # Single emit triggers auto-flush
        c.emit(EventType.TRADE_ENTRY, {"i": 0})
        ndjson_files = list((tmp_path / "trades").rglob("*.ndjson"))
        assert len(ndjson_files) == 1

    def test_tick_buffer_is_large(self, tmp_path: Path) -> None:
        """Ticks auto-flush at 1000, not at 100."""
        c = self._make(tmp_path)
        for i in range(100):
            c.emit(EventType.TICK, {"price": i})
        # 100 events should NOT trigger flush (buffer is 1000)
        assert not list((tmp_path / "ticks").rglob("*.ndjson"))
        c.flush()
        ndjson_files = list((tmp_path / "ticks").rglob("*.ndjson"))
        assert len(ndjson_files) == 1

    def test_close_flushes_remaining(self, tmp_path: Path) -> None:
        """close() flushes buffered ticks (only ticks buffer > 1)."""
        c = self._make(tmp_path)
        c.emit(EventType.TICK, {"price": 28.5})
        # Ticks buffer=1000, so 1 tick stays in buffer
        assert not list((tmp_path / "ticks").rglob("*.ndjson"))
        c.close()
        ndjson_files = list((tmp_path / "ticks").rglob("*.ndjson"))
        assert len(ndjson_files) == 1

    def test_close_on_empty_no_files(self, tmp_path: Path) -> None:
        c = self._make(tmp_path)
        c.close()
        assert not list(tmp_path.rglob("*.ndjson"))

    def test_multiple_flushes_append_to_local(self, tmp_path: Path) -> None:
        """Multiple local flushes append to the same session file."""
        c = self._make(tmp_path)
        # Emit 2 ticks then flush, emit 2 more then flush
        c.emit(EventType.TICK, {"price": 100})
        c.emit(EventType.TICK, {"price": 101})
        c.flush()
        c.emit(EventType.TICK, {"price": 102})
        c.flush()

        ndjson_files = list((tmp_path / "ticks").rglob("*.ndjson"))
        lines = ndjson_files[0].read_text().strip().split("\n")
        assert len(lines) == 3

    def test_timestamp_auto_populated(self, tmp_path: Path) -> None:
        c = self._make(tmp_path)
        c.emit(EventType.TRADE_ENTRY, {"side": "LONG"})
        # buffer=1, auto-flushed
        ndjson_files = list((tmp_path / "trades").rglob("*.ndjson"))
        parsed = json.loads(ndjson_files[0].read_text().strip())
        assert parsed["timestamp"] > 1700000000

    def test_bridge_type_in_flat_output(self, tmp_path: Path) -> None:
        """bridge_type is included in flat NDJSON output."""
        c = self._make(tmp_path, bridge_type="multi_scale")
        c.emit(EventType.SESSION_START, {})
        # buffer=1, auto-flushed
        ndjson_files = list((tmp_path / "sessions").rglob("*.ndjson"))
        parsed = json.loads(ndjson_files[0].read_text().strip())
        assert parsed["bridge_type"] == "multi_scale"

    def test_local_path_structure(self, tmp_path: Path) -> None:
        """Local path follows {dir}/{table}/{symbol}/{date}/{session}.ndjson pattern."""
        c = self._make(tmp_path)
        c.emit(EventType.FILL, {"price": 28.5})
        # buffer=1, auto-flushed
        path = c._local_file_path("fills")
        assert "fills" in str(path)
        assert "HYPE" in str(path)
        assert c.session_id in path.name
        assert path.suffix == ".ndjson"

    def test_per_table_event_counts(self, tmp_path: Path) -> None:
        c = self._make(tmp_path)
        c.emit(EventType.TICK, {"price": 100})
        c.emit(EventType.TICK, {"price": 101})
        c.emit(EventType.SIGNAL, {"type": "buy"})
        assert c._event_counts["ticks"] == 2
        assert c._event_counts["signals"] == 1


# ---------------------------------------------------------------------------
# TelemetryCollector — error suppression
# ---------------------------------------------------------------------------


class TestErrorSuppression:
    """All errors are suppressed — telemetry never crashes the bot."""

    def _make(self, tmp_path: Path, **kw) -> TelemetryCollector:
        defaults = {"symbol": "HYPE", "bridge_type": "single_scale", "local_dir": tmp_path, "gcs_bucket": ""}
        defaults.update(kw)
        return TelemetryCollector(**defaults)

    def test_emit_error_suppressed(self, tmp_path: Path) -> None:
        c = self._make(tmp_path)
        with patch("builtins.open", side_effect=OSError("disk full")):
            # buffer=1 triggers auto-flush, which fails — should not raise
            c.emit(EventType.TRADE_ENTRY, {"side": "LONG"})

    def test_flush_error_suppressed(self, tmp_path: Path) -> None:
        c = self._make(tmp_path)
        c.emit(EventType.TICK, {"price": 100})
        with patch("builtins.open", side_effect=OSError("disk full")):
            c.flush()  # Should not raise

    def test_close_error_suppressed(self, tmp_path: Path) -> None:
        c = self._make(tmp_path)
        c.emit(EventType.TICK, {"price": 100})
        with patch("builtins.open", side_effect=OSError("disk full")):
            c.close()  # Should not raise


# ---------------------------------------------------------------------------
# TelemetryCollector — GCS writes (mocked)
# ---------------------------------------------------------------------------


class TestGCSWrites:
    """GCS upload → BQ load with mocked google.cloud clients."""

    def _make_with_gcs(self, tmp_path: Path, mock_gcs_client: MagicMock) -> TelemetryCollector:
        c = TelemetryCollector(
            symbol="SOL",
            bridge_type="multi_scale",
            local_dir=tmp_path,
            gcs_bucket="my-test-bucket",
            gcs_prefix="telemetry",
        )
        # Inject mock GCS client to avoid real network calls
        c._gcs_client = mock_gcs_client
        c._gcs_available = True
        return c

    def _make_mock_gcs(self) -> MagicMock:
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        return mock_client

    def test_gcs_write_on_flush(self, tmp_path: Path) -> None:
        mock_gcs = self._make_mock_gcs()
        c = self._make_with_gcs(tmp_path, mock_gcs)

        c.emit(EventType.TICK, {"price": 145.0})
        c.flush()

        # Verify GCS upload was called
        mock_gcs.bucket.assert_called_with("my-test-bucket")
        mock_blob = mock_gcs.bucket.return_value.blob.return_value
        mock_blob.upload_from_string.assert_called_once()

        # Verify uploaded data is flat NDJSON
        uploaded_data = mock_blob.upload_from_string.call_args[0][0]
        parsed = json.loads(uploaded_data.decode("utf-8").strip())
        assert parsed["price"] == 145.0
        assert parsed["symbol"] == "SOL"
        assert "payload" not in parsed

    def test_unique_blob_per_flush(self, tmp_path: Path) -> None:
        """Each flush creates a new uniquely-named blob (no append)."""
        mock_gcs = self._make_mock_gcs()
        c = self._make_with_gcs(tmp_path, mock_gcs)

        # First flush
        c.emit(EventType.TICK, {"price": 100})
        c.flush()
        # Second flush
        c.emit(EventType.TICK, {"price": 101})
        c.flush()

        # Two separate blobs should have been created
        blob_calls = mock_gcs.bucket.return_value.blob.call_args_list
        paths = [call[0][0] for call in blob_calls]
        assert len(paths) == 2
        assert paths[0] != paths[1]
        # Sequence numbers: _000000 and _000001
        assert "_000000.ndjson" in paths[0]
        assert "_000001.ndjson" in paths[1]

    def test_gcs_blob_path_format(self, tmp_path: Path) -> None:
        mock_gcs = self._make_mock_gcs()
        c = self._make_with_gcs(tmp_path, mock_gcs)
        path = c._gcs_blob_path("ticks")
        # Format: telemetry/ticks/SOL/{date}/{session_id}_{seq:06d}.ndjson
        parts = path.split("/")
        assert parts[0] == "telemetry"
        assert parts[1] == "ticks"
        assert parts[2] == "SOL"
        assert c.session_id in parts[4]
        assert parts[4].endswith("_000000.ndjson")

    def test_gcs_blob_path_increments_sequence(self, tmp_path: Path) -> None:
        mock_gcs = self._make_mock_gcs()
        c = self._make_with_gcs(tmp_path, mock_gcs)
        p1 = c._gcs_blob_path("ticks")
        p2 = c._gcs_blob_path("ticks")
        assert "_000000.ndjson" in p1
        assert "_000001.ndjson" in p2

    def test_gcs_failure_falls_back_to_local(self, tmp_path: Path) -> None:
        mock_gcs = MagicMock()
        mock_gcs.bucket.side_effect = Exception("GCS down")
        c = self._make_with_gcs(tmp_path, mock_gcs)

        c.emit(EventType.FILL, {"price": 28.5})
        c.flush()

        # Should have written to local fallback
        ndjson_files = list((tmp_path / "fills").rglob("*.ndjson"))
        assert len(ndjson_files) == 1

    def test_gcs_unavailable_skips_subsequent_attempts(self, tmp_path: Path) -> None:
        """Once GCS fails on first init, subsequent flushes go straight to local."""
        c = TelemetryCollector(
            symbol="SOL",
            bridge_type="multi_scale",
            local_dir=tmp_path,
            gcs_bucket="my-test-bucket",
            gcs_prefix="telemetry",
        )
        # Inject a failing mock client — first attempt should mark GCS unavailable
        mock_gcs = MagicMock()
        mock_gcs.bucket.side_effect = Exception("GCS down")
        c._gcs_client = mock_gcs
        # _gcs_available stays None (not yet tested)

        # First flush fails GCS → marks unavailable, falls back to local
        c.emit(EventType.TICK, {"price": 100})
        c.flush()
        assert c._gcs_available is False

        # Reset mock to track subsequent calls
        mock_gcs.reset_mock()

        # Second flush should skip GCS entirely and go local
        c.emit(EventType.TICK, {"price": 101})
        c.flush()

        # GCS bucket() should NOT have been called again
        mock_gcs.bucket.assert_not_called()

    def test_no_gcs_bucket_means_local_only(self, tmp_path: Path) -> None:
        """When gcs_bucket is empty, never attempt GCS."""
        c = TelemetryCollector(
            symbol="HYPE",
            bridge_type="single_scale",
            local_dir=tmp_path,
            gcs_bucket="",
        )
        c.emit(EventType.SIGNAL, {"type": "buy"})
        c.flush()

        # Should only write locally
        ndjson_files = list((tmp_path / "signals").rglob("*.ndjson"))
        assert len(ndjson_files) == 1
        assert c._gcs_client is None  # GCS client never initialized

    def test_gcs_content_type_is_ndjson(self, tmp_path: Path) -> None:
        mock_gcs = self._make_mock_gcs()
        c = self._make_with_gcs(tmp_path, mock_gcs)

        c.emit(EventType.DC_EVENT, {"event_type": "PDCC_Down"})
        c.flush()

        mock_blob = mock_gcs.bucket.return_value.blob.return_value
        _, kwargs = mock_blob.upload_from_string.call_args
        assert kwargs["content_type"] == "application/x-ndjson"

    def test_separate_gcs_blobs_per_table(self, tmp_path: Path) -> None:
        mock_gcs = self._make_mock_gcs()
        c = self._make_with_gcs(tmp_path, mock_gcs)

        c.emit(EventType.TICK, {"price": 100})
        c.emit(EventType.SIGNAL, {"type": "buy"})
        c.flush()

        # Should have called bucket.blob() for two different paths
        blob_calls = mock_gcs.bucket.return_value.blob.call_args_list
        paths = [call[0][0] for call in blob_calls]
        tables_written = {p.split("/")[1] for p in paths}
        assert "ticks" in tables_written
        assert "signals" in tables_written


# ---------------------------------------------------------------------------
# TelemetryCollector — BQ load jobs (mocked)
# ---------------------------------------------------------------------------


class TestBQLoadJobs:
    """BQ load job submission after GCS write."""

    def _make_with_gcs_and_bq(
        self, tmp_path: Path, mock_gcs: MagicMock, mock_bq: MagicMock
    ) -> TelemetryCollector:
        c = TelemetryCollector(
            symbol="HYPE",
            bridge_type="single_scale",
            local_dir=tmp_path,
            gcs_bucket="my-bucket",
            gcs_prefix="telemetry",
        )
        c._gcs_client = mock_gcs
        c._gcs_available = True
        c._bq_client = mock_bq
        return c

    def _make_mock_gcs(self) -> MagicMock:
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        return mock_client

    def test_bq_load_submitted_after_gcs_write(self, tmp_path: Path) -> None:
        """After successful GCS write, a BQ load job is submitted."""
        mock_gcs = self._make_mock_gcs()
        mock_bq = MagicMock()
        c = self._make_with_gcs_and_bq(tmp_path, mock_gcs, mock_bq)

        c.emit(EventType.FILL, {"price": 28.5})
        c.flush()

        # BQ load_table_from_uri should have been called
        mock_bq.load_table_from_uri.assert_called_once()
        call_args = mock_bq.load_table_from_uri.call_args
        # First arg is GCS URI
        gcs_uri = call_args[0][0]
        assert gcs_uri.startswith("gs://my-bucket/telemetry/fills/HYPE/")
        # Second arg is table ID
        table_id = call_args[0][1]
        assert "fills" in table_id

    def test_bq_jobs_count_tracked(self, tmp_path: Path) -> None:
        mock_gcs = self._make_mock_gcs()
        mock_bq = MagicMock()
        c = self._make_with_gcs_and_bq(tmp_path, mock_gcs, mock_bq)

        c.emit(EventType.FILL, {"price": 28.5})
        c.emit(EventType.SIGNAL, {"type": "buy"})
        c.flush()

        # Two tables flushed → two BQ jobs
        assert c._bq_jobs_submitted == 2

    def test_bq_load_failure_suppressed(self, tmp_path: Path) -> None:
        """BQ load failure doesn't crash — data is safe in GCS."""
        mock_gcs = self._make_mock_gcs()
        mock_bq = MagicMock()
        mock_bq.load_table_from_uri.side_effect = Exception("BQ down")
        c = self._make_with_gcs_and_bq(tmp_path, mock_gcs, mock_bq)

        c.emit(EventType.FILL, {"price": 28.5})
        c.flush()  # Should not raise

        # GCS write still happened
        mock_gcs.bucket.return_value.blob.return_value.upload_from_string.assert_called_once()
        # BQ job count stays 0
        assert c._bq_jobs_submitted == 0

    def test_no_bq_load_on_local_fallback(self, tmp_path: Path) -> None:
        """When GCS fails and we fall back to local, no BQ load is attempted."""
        mock_gcs = MagicMock()
        mock_gcs.bucket.side_effect = Exception("GCS down")
        mock_bq = MagicMock()
        c = self._make_with_gcs_and_bq(tmp_path, mock_gcs, mock_bq)

        c.emit(EventType.FILL, {"price": 28.5})
        c.flush()

        # BQ should not have been called
        mock_bq.load_table_from_uri.assert_not_called()
        # Data should be in local fallback
        ndjson_files = list((tmp_path / "fills").rglob("*.ndjson"))
        assert len(ndjson_files) == 1

    def test_close_logs_bq_jobs_count(self, tmp_path: Path) -> None:
        """close() summary includes bq_jobs count."""
        mock_gcs = self._make_mock_gcs()
        mock_bq = MagicMock()
        c = self._make_with_gcs_and_bq(tmp_path, mock_gcs, mock_bq)

        c.emit(EventType.FILL, {"price": 28.5})
        c.flush()
        c.close()  # Should not raise, logs summary

        assert c._bq_jobs_submitted == 1
