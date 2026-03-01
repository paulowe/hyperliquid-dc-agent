"""Create BigQuery dataset and tables for trading telemetry.

Idempotent: safe to run multiple times. Uses exists_ok=True.
Pass --recreate to drop and recreate all tables (destroys data!).

Usage:
    uv run --package hyperliquid-trading-bot python -m telemetry.setup_bq
    uv run --package hyperliquid-trading-bot python -m telemetry.setup_bq --recreate

Reads from environment:
    GCP_PROJECT_ID       (default: derivatives-417104)
    TELEMETRY_BQ_DATASET (default: trading_telemetry)
    GCP_LOCATION         (default: us-central1)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path for imports
_SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC_DIR))

from google.cloud import bigquery


def get_config() -> dict[str, str]:
    """Read GCP config from environment."""
    return {
        "project_id": os.environ.get("GCP_PROJECT_ID", "derivatives-417104"),
        "dataset_id": os.environ.get("TELEMETRY_BQ_DATASET", "trading_telemetry"),
        "location": os.environ.get("GCP_LOCATION", "us-central1"),
    }


# ---------------------------------------------------------------------------
# Table schemas — aligned with flat NDJSON from collector._flatten_event()
#
# Every event gets: timestamp, session_id, symbol, event_type, bridge_type
# (from _flatten_event common fields). Then payload fields are merged at
# top level. Schemas list the fields we want to capture; BQ load jobs use
# ignore_unknown_values=True so extra fields are silently dropped.
# ---------------------------------------------------------------------------

TABLE_SCHEMAS: dict[str, list[bigquery.SchemaField]] = {
    # High-volume: every price tick (~10/sec). Keep minimal for storage.
    "ticks": [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("session_id", "STRING"),
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("price", "FLOAT64"),
    ],
    # DC events (DC_EVENT) and momentum updates (MOMENTUM_UPDATE)
    "dc_events": [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("session_id", "STRING"),
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("event_type", "STRING"),
        bigquery.SchemaField("bridge_type", "STRING"),
        # DC_EVENT payload fields
        bigquery.SchemaField("start_price", "FLOAT64"),
        bigquery.SchemaField("end_price", "FLOAT64"),
        bigquery.SchemaField("start_time", "FLOAT64"),
        bigquery.SchemaField("threshold_down", "FLOAT64"),
        bigquery.SchemaField("threshold_up", "FLOAT64"),
        bigquery.SchemaField("is_sensor", "BOOL"),
        # MOMENTUM_UPDATE payload fields
        bigquery.SchemaField("score", "FLOAT64"),
        bigquery.SchemaField("regime", "STRING"),
        bigquery.SchemaField("threshold_key", "STRING"),
    ],
    # Strategy signals (BUY, SELL, CLOSE)
    "signals": [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("session_id", "STRING"),
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("event_type", "STRING"),
        bigquery.SchemaField("bridge_type", "STRING"),
        bigquery.SchemaField("signal_type", "STRING"),
        bigquery.SchemaField("price", "FLOAT64"),
        bigquery.SchemaField("size", "FLOAT64"),
        bigquery.SchemaField("reason", "STRING"),
        bigquery.SchemaField("is_reversal", "BOOL"),
        # Multi-scale fields (null for single-scale)
        bigquery.SchemaField("momentum_score", "FLOAT64"),
        bigquery.SchemaField("regime", "STRING"),
    ],
    # Trade entries and exits (separate rows, distinguished by event_type)
    "trades": [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("session_id", "STRING"),
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("event_type", "STRING"),
        bigquery.SchemaField("bridge_type", "STRING"),
        # TRADE_ENTRY fields
        bigquery.SchemaField("side", "STRING"),
        bigquery.SchemaField("entry_price", "FLOAT64"),
        bigquery.SchemaField("size", "FLOAT64"),
        bigquery.SchemaField("is_reversal", "BOOL"),
        bigquery.SchemaField("backstop_sl_pct", "FLOAT64"),
        bigquery.SchemaField("backstop_tp_pct", "FLOAT64"),
        # TRADE_EXIT fields
        bigquery.SchemaField("exit_price", "FLOAT64"),
        bigquery.SchemaField("exit_reason", "STRING"),
        bigquery.SchemaField("sl_at_exit", "FLOAT64"),
        bigquery.SchemaField("tp_at_exit", "FLOAT64"),
        # Multi-scale fields (null for single-scale)
        bigquery.SchemaField("momentum_score", "FLOAT64"),
        bigquery.SchemaField("regime", "STRING"),
    ],
    # Fill events (what the bridge recorded after order execution)
    "fills": [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("session_id", "STRING"),
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("event_type", "STRING"),
        bigquery.SchemaField("bridge_type", "STRING"),
        bigquery.SchemaField("signal_type", "STRING"),
        bigquery.SchemaField("price", "FLOAT64"),
        bigquery.SchemaField("size", "FLOAT64"),
        bigquery.SchemaField("reason", "STRING"),
    ],
    # Session start/end/reconnect events (distinguished by event_type)
    "sessions": [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("session_id", "STRING"),
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("event_type", "STRING"),
        bigquery.SchemaField("bridge_type", "STRING"),
        # SESSION_START fields
        bigquery.SchemaField("config", "JSON"),
        bigquery.SchemaField("network", "STRING"),
        bigquery.SchemaField("observe_only", "BOOL"),
        bigquery.SchemaField("leverage", "INT64"),
        bigquery.SchemaField("backstop_sl_pct", "FLOAT64"),
        bigquery.SchemaField("backstop_tp_pct", "FLOAT64"),
        # SESSION_END fields
        bigquery.SchemaField("duration_seconds", "FLOAT64"),
        bigquery.SchemaField("tick_count", "INT64"),
        bigquery.SchemaField("signal_count", "INT64"),
        bigquery.SchemaField("trade_count", "INT64"),
        bigquery.SchemaField("dc_event_count", "INT64"),
        bigquery.SchemaField("reconnect_count", "INT64"),
        # Multi-scale SESSION_END extras
        bigquery.SchemaField("sensor_event_count", "INT64"),
        bigquery.SchemaField("trade_event_count", "INT64"),
        bigquery.SchemaField("filtered_count", "INT64"),
        bigquery.SchemaField("final_momentum_score", "FLOAT64"),
        bigquery.SchemaField("final_regime", "STRING"),
        # RECONNECT fields
        bigquery.SchemaField("reconnect_number", "INT64"),
    ],
    # Periodic account snapshots (~hourly)
    "account_snapshots": [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("session_id", "STRING"),
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("event_type", "STRING"),
        bigquery.SchemaField("bridge_type", "STRING"),
        bigquery.SchemaField("account_value", "FLOAT64"),
        bigquery.SchemaField("margin_used", "FLOAT64"),
        bigquery.SchemaField("withdrawable", "FLOAT64"),
    ],
}

# Partitioning and clustering per table
TABLE_PARTITION_FIELD: dict[str, str] = {
    "ticks": "timestamp",
    "dc_events": "timestamp",
    "signals": "timestamp",
    "trades": "timestamp",
    "fills": "timestamp",
    "sessions": "timestamp",
    "account_snapshots": "timestamp",
}

TABLE_CLUSTER_FIELDS: dict[str, list[str]] = {
    "ticks": ["symbol"],
    "dc_events": ["symbol"],
    "signals": ["symbol"],
    "trades": ["symbol"],
    "fills": ["symbol"],
}


def setup_dataset_and_tables(recreate: bool = False) -> None:
    """Create BQ dataset and all telemetry tables.

    Args:
        recreate: If True, drop and recreate all tables (destroys data!).
    """
    cfg = get_config()
    client = bigquery.Client(project=cfg["project_id"])
    dataset_ref = f"{cfg['project_id']}.{cfg['dataset_id']}"

    # Create dataset
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = cfg["location"]
    dataset = client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset: {dataset_ref} (location={dataset.location})")

    # Create tables
    for table_name, schema in TABLE_SCHEMAS.items():
        table_id = f"{dataset_ref}.{table_name}"

        if recreate:
            client.delete_table(table_id, not_found_ok=True)
            print(f"  Dropped: {table_name}")

        table = bigquery.Table(table_id, schema=schema)

        # Add time partitioning
        partition_field = TABLE_PARTITION_FIELD.get(table_name)
        if partition_field:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=partition_field,
            )

        # Add clustering
        cluster_fields = TABLE_CLUSTER_FIELDS.get(table_name)
        if cluster_fields:
            table.clustering_fields = cluster_fields

        table = client.create_table(table, exists_ok=True)
        print(f"  Table: {table_name} ({len(schema)} columns)")

    print(f"\nAll {len(TABLE_SCHEMAS)} tables ready in {dataset_ref}")


if __name__ == "__main__":
    do_recreate = "--recreate" in sys.argv
    if do_recreate:
        print("WARNING: --recreate will DROP and recreate all tables!\n")
    setup_dataset_and_tables(recreate=do_recreate)
