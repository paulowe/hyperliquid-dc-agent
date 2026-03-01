"""Load telemetry NDJSON files from GCS into BigQuery.

Lists NDJSON files in the telemetry GCS prefix, loads them into matching
BQ tables, and moves processed files to a processed/ prefix.

Usage:
    uv run --package hyperliquid-trading-bot python -m telemetry.load_to_bq

Reads from environment:
    GCP_PROJECT_ID           (default: derivatives-417104)
    TELEMETRY_GCS_BUCKET     (default: derivatives-417104-pl-assets)
    TELEMETRY_GCS_PREFIX     (default: telemetry)
    TELEMETRY_BQ_DATASET     (default: trading_telemetry)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path for imports
_SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC_DIR))

from google.cloud import bigquery, storage


# Tables that have corresponding GCS prefixes
KNOWN_TABLES = [
    "ticks",
    "dc_events",
    "signals",
    "trades",
    "fills",
    "sessions",
    "account_snapshots",
]


def get_config() -> dict[str, str]:
    """Read GCP config from environment."""
    project_id = os.environ.get("GCP_PROJECT_ID", "derivatives-417104")
    return {
        "project_id": project_id,
        "gcs_bucket": os.environ.get("TELEMETRY_GCS_BUCKET", f"{project_id}-pl-assets"),
        "gcs_prefix": os.environ.get("TELEMETRY_GCS_PREFIX", "telemetry"),
        "bq_dataset": os.environ.get("TELEMETRY_BQ_DATASET", "trading_telemetry"),
    }


def load_table(
    bq_client: bigquery.Client,
    gcs_client: storage.Client,
    cfg: dict[str, str],
    table_name: str,
) -> int:
    """Load all NDJSON files for a table from GCS into BQ.

    Returns number of files loaded.
    """
    bucket = gcs_client.bucket(cfg["gcs_bucket"])
    prefix = f"{cfg['gcs_prefix']}/{table_name}/"
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Filter to .ndjson files only (skip processed/)
    ndjson_blobs = [b for b in blobs if b.name.endswith(".ndjson")]

    if not ndjson_blobs:
        return 0

    table_id = f"{cfg['project_id']}.{cfg['bq_dataset']}.{table_name}"
    loaded = 0

    for blob in ndjson_blobs:
        uri = f"gs://{cfg['gcs_bucket']}/{blob.name}"

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            autodetect=False,  # Use existing table schema
        )

        try:
            load_job = bq_client.load_table_from_uri(uri, table_id, job_config=job_config)
            load_job.result()  # Wait for completion

            # Move to processed/ prefix
            processed_name = blob.name.replace(
                f"{cfg['gcs_prefix']}/",
                f"{cfg['gcs_prefix']}/processed/",
                1,
            )
            bucket.rename_blob(blob, processed_name)
            loaded += 1

        except Exception as e:
            print(f"  ERROR loading {blob.name}: {e}")

    return loaded


def load_all() -> None:
    """Load all telemetry data from GCS into BigQuery."""
    cfg = get_config()
    bq_client = bigquery.Client(project=cfg["project_id"])
    gcs_client = storage.Client(project=cfg["project_id"])

    print(f"Loading telemetry from gs://{cfg['gcs_bucket']}/{cfg['gcs_prefix']}/")
    print(f"Into BigQuery: {cfg['project_id']}.{cfg['bq_dataset']}")
    print()

    total = 0
    for table_name in KNOWN_TABLES:
        count = load_table(bq_client, gcs_client, cfg, table_name)
        if count > 0:
            print(f"  {table_name}: {count} files loaded")
        total += count

    if total == 0:
        print("  No new NDJSON files found to load.")
    else:
        print(f"\nTotal: {total} files loaded into BigQuery")


if __name__ == "__main__":
    load_all()
