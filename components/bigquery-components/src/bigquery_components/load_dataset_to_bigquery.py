# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kfp.v2.dsl import Dataset, Input, component

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery==2.30.0"],
)
def load_dataset_to_bigquery(
    dataset: Input[Dataset],
    # Destination table — either provide fully-qualified table_id OR the 3 parts below.
    destination_table_id: str = "",  # e.g. "my-proj.myds.mytable"
    destination_project_id: str = "",
    destination_dataset_id: str = "",
    destination_table_name: str = "",
    # BigQuery client / job settings
    bq_client_project_id: str = "",
    location: str = "US",
    write_disposition: str = "WRITE_TRUNCATE",  # WRITE_TRUNCATE | WRITE_APPEND | WRITE_EMPTY
    create_disposition: str = "CREATE_IF_NEEDED",  # CREATE_IF_NEEDED | CREATE_NEVER
    source_format: str = "",  # CSV | NEWLINE_DELIMITED_JSON | PARQUET | AVRO | ORC  (optional; will try to infer)
    autodetect: bool = True,
    # Optional schema: JSON string with list of {name, type, mode} like Google samples
    # Example: '[{"name":"f1","type":"STRING","mode":"NULLABLE"}]'
    schema_json: str = "",
    # Common CSV options (only used when source_format == "CSV" or inferred CSV)
    csv_field_delimiter: str = ",",
    csv_quote: str = '"',
    csv_allow_quoted_newlines: bool = True,
    csv_skip_leading_rows: int = 0,
    # If the input dataset is local files (dataset.path), set this to a filename pattern to load (e.g., "*.csv")
    # If empty, the component will try to find a single file under dataset.path.
    local_file_glob: str = "",
):
    """
    Load a KFP Input[Dataset] into a BigQuery table.

    - If dataset.uri starts with "gs://", uses load_table_from_uri (supports wildcards).
    - Else, loads from local files under dataset.path using load_table_from_file.

    Destination table can be provided as a single fully-qualified string, or via
    project/dataset/table triplet.

    Parameters mirror Google's BigQuery samples where possible.
    """
    import json
    import logging
    import os
    from glob import glob
    from pathlib import Path
    from typing import List, Optional

    from google.cloud import bigquery
    from google.cloud.exceptions import GoogleCloudError

    def _resolve_table_id() -> str:
        if destination_table_id:
            return destination_table_id
        if destination_project_id and destination_dataset_id and destination_table_name:
            return f"{destination_project_id}.{destination_dataset_id}.{destination_table_name}"
        raise ValueError(
            "You must set either `destination_table_id` (fully-qualified) "
            "or all of `destination_project_id`, `destination_dataset_id`, and `destination_table_name`."
        )

    def _infer_source_format_from_path(p: str) -> Optional[str]:
        ext = Path(p).suffix.lower()
        return {
            ".csv": "CSV",
            ".json": "NEWLINE_DELIMITED_JSON",
            ".ndjson": "NEWLINE_DELIMITED_JSON",
            ".parquet": "PARQUET",
            ".avro": "AVRO",
            ".orc": "ORC",
        }.get(ext)

    def _parse_schema(schema_str: str) -> Optional[List[bigquery.SchemaField]]:
        if not schema_str:
            return None
        try:
            schema_list = json.loads(schema_str)
            fields = []
            for f in schema_list:
                name = f["name"]
                ftype = f["type"]
                mode = f.get("mode", "NULLABLE")
                fields.append(bigquery.SchemaField(name=name, field_type=ftype, mode=mode))
            return fields
        except Exception as e:
            raise ValueError(f"Invalid schema_json. Expect a JSON array of objects with name/type/mode. Error: {e}")

    def _build_job_config() -> bigquery.LoadJobConfig:
        cfg = bigquery.LoadJobConfig()
        cfg.write_disposition = write_disposition
        cfg.create_disposition = create_disposition

        # Source format / autodetect
        fmt = source_format or ""
        if not fmt:
            # Try to infer from URI or local file extension
            probe = dataset.uri if dataset.uri else ""
            if not probe and Path(dataset.path).exists():
                # if local, try to look at a candidate file
                candidates = []
                if local_file_glob:
                    candidates = glob(os.path.join(dataset.path, local_file_glob))
                else:
                    # pick the first regular file under path
                    for root, _, files in os.walk(dataset.path):
                        for fn in files:
                            candidates.append(os.path.join(root, fn))
                            break
                        if candidates:
                            break
                if candidates:
                    probe = candidates[0]
            inferred = _infer_source_format_from_path(probe) if probe else None
            if inferred:
                fmt = inferred

        if fmt:
            cfg.source_format = getattr(bigquery.SourceFormat, fmt)
        cfg.autodetect = bool(autodetect)

        # Schema (optional)
        schema = _parse_schema(schema_json)
        if schema:
            cfg.schema = schema
            # If schema is provided, you can still keep autodetect True; BigQuery will respect explicit fields.
            # Set to False if you want strict schema-only behavior:
            # cfg.autodetect = False

        # CSV options (if applicable)
        if (fmt or "").upper() == "CSV" or (not fmt and autodetect):
            cfg.field_delimiter = csv_field_delimiter
            cfg.quote_character = csv_quote
            cfg.allow_quoted_newlines = csv_allow_quoted_newlines
            if csv_skip_leading_rows and csv_skip_leading_rows > 0:
                cfg.skip_leading_rows = csv_skip_leading_rows

        return cfg

    # Resolve destination table and client
    table_id = _resolve_table_id()
    client_project = bq_client_project_id or destination_project_id or ""
    if not client_project:
        raise ValueError("Please set `bq_client_project_id` or `destination_project_id`.")
    client = bigquery.Client(project=client_project, location=location)

    job_config = _build_job_config()

    # Decide loading mode based on dataset.uri vs dataset.path
    is_gcs = bool(dataset.uri and dataset.uri.startswith("gs://"))
    logging.info(f"Dataset.uri={dataset.uri}, Dataset.path={dataset.path}, is_gcs={is_gcs}")

    try:
        if is_gcs:
            # Google sample pattern: client.load_table_from_uri(...).result()
            # Supports wildcards in URI.
            uris = dataset.uri.rstrip("/")
            # if caller already gave a specific object/wildcard, keep it
            if not uris.endswith(".csv") and "*" not in uris:
                # Previous writer always creates predictions.csv under that folder
                uris = f"{uris}/*.csv"  # or f"{uris}/*.csv" if you may shard
            logging.info(f"Loading from GCS: {uris} -> {table_id}")
            load_job = client.load_table_from_uri(uris, table_id, job_config=job_config)
            result = load_job.result()  # Waits for job to complete.
            logging.info("Load job completed: %s", result)
        else:
            # Local-files load. Gather files to load.
            src_files: List[str] = []
            base = Path(dataset.path)
            if base.is_file():
                src_files = [str(base)]
            else:
                if local_file_glob:
                    src_files = glob(os.path.join(str(base), local_file_glob))
                else:
                    # Fallback: find the first file under the path
                    for root, _, files in os.walk(str(base)):
                        for fn in files:
                            src_files = [os.path.join(root, fn)]
                            break
                        if src_files:
                            break

            if not src_files:
                raise FileNotFoundError(
                    f"No source files found under dataset.path={dataset.path}. "
                    "Provide files there or set `local_file_glob`."
                )

            # If many files, load them sequentially. (Alternative: concatenate beforehand.)
            for idx, fpath in enumerate(src_files, start=1):
                logging.info(f"({idx}/{len(src_files)}) Loading local file {fpath} -> {table_id}")
                with open(fpath, "rb") as fh:
                    load_job = client.load_table_from_file(fh, table_id, job_config=job_config)
                result = load_job.result()  # Waits for job to complete.
                logging.info("Load job completed: %s", result)

        # Fetch table stats (sample style)
        table = client.get_table(table_id)
        logging.info(
            "Loaded %s rows and %s bytes into %s.",
            table.num_rows,
            table.num_bytes,
            table_id,
        )
        print(f"Loaded {table.num_rows} rows into {table_id}.")
    except GoogleCloudError as e:
        logging.error("BigQuery load failed: %s", e)
        raise
