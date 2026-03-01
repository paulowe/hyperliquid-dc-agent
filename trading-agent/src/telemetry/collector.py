"""TelemetryCollector: buffered NDJSON writer → GCS → BQ load job pipeline.

On each flush:
  1. Write flat NDJSON to a unique GCS blob
  2. Submit a BQ load job from that blob URI (fire-and-forget)
  3. Fall back to local files if GCS is unavailable

Each flush creates a new blob (no append), so BQ can load each independently.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from telemetry.events import EventType, TelemetryEvent

logger = logging.getLogger(__name__)

# Default local fallback directory
_DEFAULT_LOCAL_DIR = Path.home() / ".cache" / "hyperliquid-telemetry"

# Map EventType → GCS/local table prefix (matches BQ table names)
_EVENT_TABLE_MAP: dict[EventType, str] = {
    EventType.TICK: "ticks",
    EventType.TICK_SNAPSHOT: "ticks",
    EventType.DC_EVENT: "dc_events",
    EventType.MOMENTUM_UPDATE: "dc_events",
    EventType.SIGNAL: "signals",
    EventType.TRADE_ENTRY: "trades",
    EventType.TRADE_EXIT: "trades",
    EventType.FILL: "fills",
    EventType.SESSION_START: "sessions",
    EventType.SESSION_END: "sessions",
    EventType.RECONNECT: "sessions",
    EventType.ACCOUNT_SNAPSHOT: "account_snapshots",
}

# Per-table buffer sizes: ticks are high-volume, everything else flushes immediately
_BUFFER_SIZES: dict[str, int] = {
    "ticks": 1000,
    "dc_events": 1,
    "signals": 1,
    "trades": 1,
    "fills": 1,
    "sessions": 1,
    "account_snapshots": 1,
}

_DEFAULT_BUFFER_SIZE = 1


class NullCollector:
    """No-op collector used when telemetry is disabled. Zero overhead."""

    @property
    def session_id(self) -> str:
        return ""

    def emit(self, event_type: EventType, payload: dict[str, Any]) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _get_table(event_type: EventType) -> str:
    """Map an EventType to its BQ/GCS table name."""
    return _EVENT_TABLE_MAP.get(event_type, "misc")


def _flatten_event(event: TelemetryEvent) -> str:
    """Flatten a TelemetryEvent into a BQ-compatible NDJSON line.

    Merges common fields with payload fields. Payload can override common fields
    (e.g., dc_event payload has its own event_type like "PDCC_Down").
    This produces rows that match BQ table schemas directly.
    """
    flat: dict[str, Any] = {
        "timestamp": event.timestamp,
        "session_id": event.session_id,
        "symbol": event.symbol,
        "event_type": event.event_type,
        "bridge_type": event.bridge_type,
    }
    flat.update(event.payload)
    return json.dumps(flat, default=str)


class TelemetryCollector:
    """Collects telemetry events → GCS blobs → BQ load jobs.

    Pipeline per flush:
      1. Buffer fills → write flat NDJSON to unique GCS blob
      2. Submit BQ load job from that GCS URI (fire-and-forget)
      3. If GCS unavailable → local fallback files

    GCS blob: gs://{bucket}/{prefix}/{table}/{symbol}/{date}/{session}_{seq}.ndjson
    Local:    {local_dir}/{table}/{symbol}/{date}/{session}.ndjson

    Reads config from environment variables:
      - TELEMETRY_GCS_BUCKET  (required for GCS)
      - TELEMETRY_GCS_PREFIX  (default: "telemetry")
      - GCP_PROJECT_ID        (for BQ load jobs)
      - TELEMETRY_BQ_DATASET  (default: "trading_telemetry")

    All errors are caught and logged — telemetry never crashes the bot.
    """

    def __init__(
        self,
        symbol: str,
        bridge_type: str,
        local_dir: Path | None = None,
        gcs_bucket: str | None = None,
        gcs_prefix: str | None = None,
    ) -> None:
        self._session_id = uuid.uuid4().hex[:12]
        self._symbol = symbol
        self._bridge_type = bridge_type
        self._date_str = time.strftime("%Y-%m-%d")

        # Local fallback directory
        self._local_dir = local_dir or _DEFAULT_LOCAL_DIR

        # GCS config: explicit args > env vars (None means "use env var")
        self._gcs_bucket = gcs_bucket if gcs_bucket is not None else os.environ.get("TELEMETRY_GCS_BUCKET", "")
        self._gcs_prefix = gcs_prefix if gcs_prefix is not None else os.environ.get("TELEMETRY_GCS_PREFIX", "telemetry")

        # BQ config from env
        self._gcp_project = os.environ.get("GCP_PROJECT_ID", "derivatives-417104")
        self._bq_dataset = os.environ.get("TELEMETRY_BQ_DATASET", "trading_telemetry")

        # Lazy-init clients (only when first flush needs them)
        self._gcs_client = None
        self._bq_client = None
        self._gcs_available: bool | None = None  # None = not yet tested

        # Per-table buffers and flush sequence counters
        self._buffers: dict[str, list[str]] = {}
        self._flush_seqs: dict[str, int] = {}
        self._event_counts: dict[str, int] = {}
        self._total_event_count = 0
        self._bq_jobs_submitted = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def event_count(self) -> int:
        return self._total_event_count

    def emit(self, event_type: EventType, payload: dict[str, Any]) -> None:
        """Emit a telemetry event. Never raises."""
        try:
            event = TelemetryEvent(
                event_type=event_type.value,
                timestamp=time.time(),
                session_id=self._session_id,
                symbol=self._symbol,
                bridge_type=self._bridge_type,
                payload=payload,
            )
            table = _get_table(event_type)
            buf = self._buffers.setdefault(table, [])
            # Store flat NDJSON line (BQ-compatible format)
            buf.append(_flatten_event(event))
            self._event_counts[table] = self._event_counts.get(table, 0) + 1
            self._total_event_count += 1

            # Auto-flush when buffer reaches table-specific threshold
            limit = _BUFFER_SIZES.get(table, _DEFAULT_BUFFER_SIZE)
            if len(buf) >= limit:
                self._flush_table(table)

        except Exception as e:
            logger.debug("Telemetry emit error (suppressed): %s", e)

    def flush(self) -> None:
        """Flush all table buffers. Never raises."""
        for table in list(self._buffers.keys()):
            self._flush_table(table)

    def close(self) -> None:
        """Flush remaining buffers and log summary. Never raises."""
        try:
            self.flush()
            if self._total_event_count > 0:
                counts = ", ".join(
                    f"{t}={c}" for t, c in sorted(self._event_counts.items())
                )
                dest = f"gs://{self._gcs_bucket}" if self._gcs_available else str(self._local_dir)
                logger.info(
                    "Telemetry: %d events → %s (session=%s, bq_jobs=%d) [%s]",
                    self._total_event_count, dest, self._session_id,
                    self._bq_jobs_submitted, counts,
                )
        except Exception as e:
            logger.debug("Telemetry close error (suppressed): %s", e)

    # -- Internal ----------------------------------------------------------

    def _flush_table(self, table: str) -> None:
        """Flush a single table's buffer: GCS write → BQ load → local fallback."""
        buf = self._buffers.get(table)
        if not buf:
            return
        data = "\n".join(buf) + "\n"
        try:
            # Try GCS → BQ pipeline
            if self._gcs_bucket:
                gcs_uri = self._try_gcs_write(table, data)
                if gcs_uri:
                    self._submit_bq_load(table, gcs_uri)
                    buf.clear()
                    return

            # Fallback to local
            self._write_local(table, data)
            buf.clear()
        except Exception as e:
            logger.debug("Telemetry flush error for %s (suppressed): %s", table, e)

    def _gcs_blob_path(self, table: str) -> str:
        """Build unique GCS blob path per flush.

        Format: {prefix}/{table}/{symbol}/{date}/{session_id}_{seq:06d}.ndjson
        """
        seq = self._flush_seqs.get(table, 0)
        self._flush_seqs[table] = seq + 1
        return (
            f"{self._gcs_prefix}/{table}/{self._symbol}/"
            f"{self._date_str}/{self._session_id}_{seq:06d}.ndjson"
        )

    def _try_gcs_write(self, table: str, data: str) -> str | None:
        """Write data to a new GCS blob. Returns the gs:// URI on success, None on failure."""
        if self._gcs_available is False:
            return None

        try:
            if self._gcs_client is None:
                from google.cloud import storage
                self._gcs_client = storage.Client()
                self._gcs_available = True

            bucket = self._gcs_client.bucket(self._gcs_bucket)
            blob_path = self._gcs_blob_path(table)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(data.encode("utf-8"), content_type="application/x-ndjson")
            return f"gs://{self._gcs_bucket}/{blob_path}"

        except Exception as e:
            if self._gcs_available is None:
                logger.warning("GCS unavailable, falling back to local telemetry: %s", e)
                self._gcs_available = False
            else:
                logger.debug("GCS write error (falling back to local): %s", e)
            return None

    def _submit_bq_load(self, table: str, gcs_uri: str) -> None:
        """Submit a BQ load job for the GCS blob. Fire-and-forget — never blocks."""
        try:
            from google.cloud import bigquery

            if self._bq_client is None:
                self._bq_client = bigquery.Client(project=self._gcp_project)

            table_id = f"{self._gcp_project}.{self._bq_dataset}.{table}"
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                ignore_unknown_values=True,
            )
            # Fire-and-forget: job runs server-side, don't call .result()
            self._bq_client.load_table_from_uri(gcs_uri, table_id, job_config=job_config)
            self._bq_jobs_submitted += 1

        except Exception as e:
            # BQ load failure is non-fatal — data is safe in GCS
            logger.debug("BQ load submit error for %s (data safe in GCS): %s", table, e)

    def _local_file_path(self, table: str) -> Path:
        """Build local fallback path: {local_dir}/{table}/{symbol}/{date}/{session_id}.ndjson"""
        return self._local_dir / table / self._symbol / self._date_str / f"{self._session_id}.ndjson"

    def _write_local(self, table: str, data: str) -> None:
        """Write data to local NDJSON file (append mode)."""
        path = self._local_file_path(table)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(data)
