# Do not abuse this import. It is only used to make the code compatible with Python 3.7.
# TODO: Remove this import when we migrate to Python 3.11.
# Do not use when running pipeline in Vertex AI.
# from __future__ import annotations
from kfp.v2.dsl import component, Input, Output, Dataset
from typing import List, Sequence

# from vertex_components.directional_change_algorithm import DirectionalChange


@component(
    base_image="python:3.11",
    packages_to_install=[
        # Upgrade pip/setuptools so we get the latest PEP 517 hooks
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
        # Force Cython < 3 so build‐time hooks don't look for the removed API
        "cython<3.0.0",
        # Install PyYAML without isolation so it sees our Cython
        "--no-build-isolation",
        "pyyaml==5.4.1",
        # Other runtime deps
        "IPython",
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "tensorflow",
        "google-cloud-storage",
        "plotly",
        "google-cloud-bigquery",
        "uuid",
        "typing",
        "datetime",
        "pyarrow",
    ],
    # base_image="us-central1-docker.pkg.dev/derivatives-417104/dc-vae-kfp-base-repo/dc-vae-base:2025-06-25",
    # install_kfp_package=False,
    # packages_to_install=[]
)
def directional_change_detector(
    df: Input[Dataset],
    # thresholds: List[float],
    thresholds: float,
    # config_gcs_uri: str,
    # experiment_name: List[str],
    # experiment_name: str,
    price_col: str,
    time_col: str,
    project_id: str,
    dataset_id: str,
    # table_id: List[str],
    # table_id: str,
    load_to_bq: bool,
    fast_plot: bool,
    max_event_markers: int,
    directional_change_events: Output[Dataset],
    figures_output: Output[Dataset],
):

    # In the following code, we are recording both upward and downward trends.
    # Note that PDCC2_Up and PDCC_Down refer to the directional change confirmation
    # points for the upward and downward trends, respectively.
    # OSV_Up and OSV_Down represent the overshoot values for the upward and downward
    # trends. Also note that "Upward Update" marks the end of an upward trend,
    # while "Downward Update" indicates the end of a downward trend.

    import uuid
    from typing import Tuple, Union, Optional
    from datetime import datetime, timezone
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import logging
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    from google.cloud import storage
    import json
    import os
    from pathlib import Path
    class DirectionalChange:
        """
        Class to compute and plot Directional Change (DC) events for given thresholds,
        and load summary events into BigQuery with experiment tracking.

        This class preserves the original algorithmic logic written
        by Dr V L Raju Chinthalapati, University of London. It only
        organizes the code and adds configurability for running multiple thresholds
        (downward and upward) in one go, plus experiment tagging in BigQuery.
        """

        def __init__(
            self,
            df: pd.DataFrame,
            thresholds: list,
            *,
            price_col: str = "PRICE",
            time_col: str = "trade_ts",
            experiment_name: str,
            project_id: str = "derivatives-417104",
            dataset_id: str = "coindesk",
            table_id: str = "dc_events",
            load_to_bq: bool = True,
            fast_plot: bool = False,
            max_event_markers: Optional[int] = 500,
            figures_output_path: Optional[str] = None,
        ) -> None:
            """
            Parameters
            ----------
            df : DataFrame
                Must contain price and time columns
                (configurable via `price_col` and `time_col`).
            thresholds : sequence of floats or (down, up) pairs
                If an element is a single float, it is used for both downward and upward
                thresholds. If an element is a (down, up) pair, each direction uses its
                own value.
            price_col : str
                Column name for prices (default 'PRICE').
            time_col : str
                Column name for timestamps (default 'load_time_toronto').
            experiment_name : str
                Human-readable label for this run; stored with the events.
            project_id, dataset_id, table_id : str
                BigQuery destination identifiers.
            load_to_bq : bool
                If True, load summary rows into BigQuery.
                If True, load summary rows into BigQuery.
            """
            # Store inputs
            self.df_original = df.copy()
            self.price_col = price_col
            self.time_col = time_col
            # Basic validation
            missing = [
                c
                for c in [self.price_col, self.time_col]
                if c not in self.df_original.columns
            ]
            if missing:
                raise ValueError(
                    f"DirectionalChange: missing required column(s): {missing}"
                )
            # dc_thresholds = [0.001, 0.005, 0.010, 0.015]
            # It means that by default up and down thresholds are the same
            # threshold_sets = [(0.001,0.001), (0.005,0.005), (0.010,0.010), (0.015,0.015)]
            # Means u can have up and down thresholds separately per DC processing
            self.threshold_sets: list[Tuple[float, float]] = [
                (t, t)
                if not isinstance(t, (tuple, list))
                else (float(t[0]), float(t[1]))
                for t in thresholds
            ]
            self.experiment_name = experiment_name
            self.experiment_id = str(uuid.uuid4())  # auto-generated experiment ID

            # BigQuery parameters
            self.project_id = project_id
            self.dataset_id = dataset_id
            self.table_id = table_id
            self.load_to_bq = load_to_bq
            self.client = bigquery.Client(project=self.project_id)

            # Storage for per-threshold outputs
            # key = (down, up) tuple -> dict of DC events
            # self.events is a dictionary
            # whose keys are (down_threshold, up_threshold) pairs (like (0.01, 0.01)),
            # and whose values are generic dictionaries containing event data.
            self.events: dict[Tuple[float, float], dict] = {}
            # self.figures is a dictionary
            # whose keys are the same (down_threshold, up_threshold) pairs,
            # and whose values are Plotly Figure objects corresponding to those thresholds.
            self.figures: dict[Tuple[float, float], go.Figure] = {}

            # Plot parameter options
            self.fast_plot = fast_plot
            self.max_event_markers = max_event_markers

            # Figure output settings
            self.figures_output_path = figures_output_path

            # Ensure destination table exists with the required schema
            if self.load_to_bq:
                self._ensure_table()

        # ---------------------------------------------------------------------
        # Public API
        # ---------------------------------------------------------------------
        def run_all(self) -> None:
            """Run DC detection for all configured thresholds."""
            for down_up in self.threshold_sets:
                self._process_threshold(down_up)

            # Save all figures if output path is specified
            if self.figures_output_path:
                self._save_all_figures()

        def get_events(self, threshold: Union[float, Tuple[float, float]]):
            """Retrieve raw events data and ancillary outputs for a threshold.

            Accepts a single float (interpreted as (t, t)) or a (down, up) pair.
            Returns a dict with keys: 'downward', 'upward', 'max_OSV', 'max_OSV2', 'df'.
            """
            if not isinstance(threshold, (tuple, list)):
                key = (threshold, threshold)
            else:
                key = (float(threshold[0]), float(threshold[1]))
            return self.events.get(key)

        def get_figure(
            self, threshold: Union[float, Tuple[float, float]]
        ) -> Optional[go.Figure]:
            """Retrieve the Plotly figure for a given threshold (float or pair)."""
            if not isinstance(threshold, (tuple, list)):
                key = (threshold, threshold)
            else:
                key = (float(threshold[0]), float(threshold[1]))
            return self.figures.get(key)

        def query_summary(
            self,
            *,
            threshold: Optional[Union[float, Tuple[float, float]]] = None,
            experiment_only: bool = True,
            distinct: bool = True,
        ) -> pd.DataFrame:
            """Query previously loaded DC summaries from BigQuery.

            Parameters
            ----------
            threshold : float | (down, up) | None
                If provided, filters rows whose downward==upward==threshold (float)
                or separately by down/up when a pair is given. If None, no threshold
                filter.
            experiment_only : bool
                If True, restrict to this object's experiment_id.
            distinct : bool
                If True, SELECT DISTINCT; else SELECT *.
            """
            select_kw = "DISTINCT" if distinct else "*"
            table_fqn = f"`{self.project_id}.{self.dataset_id}.{self.table_id}`"

            where = []
            if experiment_only:
                where.append(f"experiment_id = '{self.experiment_id}'")

            if threshold is not None:
                if isinstance(threshold, (tuple, list)):
                    down, up = float(threshold[0]), float(threshold[1])
                    where.append(f"threshold_down = {down}")
                    where.append(f"threshold_up = {up}")
                else:
                    t = float(threshold)
                    where.append(f"threshold_down = {t}")
                    where.append(f"threshold_up = {t}")

            where_sql = (" WHERE " + " AND ".join(where)) if where else ""
            query = f"""
            SELECT {select_kw} *
            FROM {table_fqn}
            {where_sql}
            ORDER BY start_time, end_time ASC
            """
            return self.client.query(query).to_dataframe()

        # ---------------------------------------------------------------------
        # Internals
        # ---------------------------------------------------------------------
        def _ensure_table(self) -> None:
            """Create destination table if it doesn't exist (idempotent)."""
            table_ref = self.client.dataset(self.dataset_id).table(self.table_id)
            try:
                self.client.get_table(table_ref)
                return
            except NotFound:
                pass
            except Exception as exc:
                message = str(exc).lower()
                if "notfound" in message or "not found" in message:
                    pass
                else:
                    raise

            schema = [
                bigquery.SchemaField("start_time", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("end_time", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("start_price", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("end_price", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("threshold_down", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("threshold_up", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("experiment_name", "STRING"),
                bigquery.SchemaField("experiment_id", "STRING"),
            ]
            table = bigquery.Table(table_ref, schema=schema)
            self.client.create_table(table)
            print(
                f"Created table {table_ref.project}."
                f"{table_ref.dataset_id}."
                f"{table_ref.table_id}"
            )

        def _process_threshold(self, down_up: Tuple[float, float]) -> None:
            """
            Process a single (down, up) threshold pair: detect DC events, store,
            plot, and optionally load a summary to BigQuery.
            """
            threshold_down, threshold_up = float(down_up[0]), float(down_up[1])

            # Copy dataframe and extract series
            # TODO: investigate if we can use the original dataframe without copying....again!
            # We copied already in the constructor
            df = self.df_original.copy()
            ask_prices = df[self.price_col]
            dates = df[self.time_col]

            # Prepare columns
            # TODO: DO WE NEED THESE COLUMNS? We write an events table not a ticks table
            df["event"] = pd.Series(pd.NA, index=df.index, dtype="object")
            df["event_type"] = pd.Series(pd.NA, index=df.index, dtype="object")
            # Date placeholders (timestamps)
            PDCC_date = pd.NaT
            max_date = pd.NaT
            osv_down_date_today = pd.NaT
            osv_up_date_today = pd.NaT
            PDCC2_date = pd.NaT
            min_date = pd.NaT

            # Initialize tracking variables for downward trends
            maxPrice = 0.5  # highest price seen before confirming a down-trend
            minPrice = 10_000_000  # placeholder large number; reset once trend starts
            PDCC = 0  # price at which down-trend is confirmed
            OSV = 0  # instantaneous overshoot during a down-trend
            max_OSV = 0  # maximum overshoot seen in a single down-trend
            filter_down = 0  # flag: 0=not in down-trend, 1=in down-trend
            filter_up = 0  # flag: prevents overlap with up-trend logic

            # Initialize tracking variables for upward trends
            maxPrice2 = 0.5  # lowest price seen before confirming an up-trend
            minPrice2 = (
                10_000_000  # placeholder large number; reset once up-trend starts
            )
            PDCC2 = 0  # price at which up-trend is confirmed
            OSV2 = 0  # instantaneous overshoot during an up-trend
            max_OSV2 = 0  # maximum overshoot seen in a single up-trend

            # Lists to record each event for later plotting

            # Initialize containers for detected DC events.
            # Each event is stored as a tuple in the form:
            # (start_time, end_time, start_price, end_price, event_type)
            # Using lists of tuples keeps the inner loop lightweight and fast,
            # avoiding the overhead of appending rows to a DataFrame during iteration.
            # These lists are later converted to a pandas DataFrame for BigQuery loading.
            downward_events: list[tuple] = []
            upward_events: list[tuple] = []

            # Main loop over each price/time
            for askPrice, date_today in zip(ask_prices, dates):

                # --- Section A: Detect start of a downward trend ---
                if filter_down == 0:
                    difference = maxPrice - askPrice
                    # If price has fallen at least Threshold% from the previous max
                    if maxPrice > askPrice and difference >= maxPrice * threshold_down:
                        # Confirm PDCC_Down event
                        minPrice = askPrice
                        PDCC = askPrice
                        filter_down = 1  # enter down-trend mode
                        max_OSV = 0  # reset overshoot counter
                        filter_up = 0  # ensure up-trend logic is off

                        # Record the start of the down-trend
                        downward_events.append(
                            (max_date, date_today, maxPrice, askPrice, "PDCC_Down")
                        )
                        PDCC_date = date_today
                        df.loc[df[self.time_col] == date_today, "event"] = "PDCC_Down"
                        df.loc[
                            df[self.time_col] == date_today, "event_type"
                        ] = "Downward Start"

                        # Also mark the last up-trend’s overshoot update
                        upward_events.append(
                            (PDCC2_date, osv_up_date_today, PDCC2, maxPrice2, "OSV_Up")
                        )
                        df.loc[
                            df[self.time_col] == osv_up_date_today, "event"
                        ] = "OSV_Up"
                        df.loc[
                            df[self.time_col] == osv_up_date_today, "event_type"
                        ] = "Upward Update"

                        # Reset up-trend trackers
                        minPrice2 = 10_000_000
                        maxPrice2 = 0.5

                    # If price goes above the previous max, just update maxPrice
                    if askPrice > maxPrice:
                        maxPrice = askPrice
                        max_date = date_today

                # --- Section B: While inside a downward trend, measure overshoot ---
                if filter_down == 1:
                    # If price has risen above the local low, compute overshoot
                    if minPrice < askPrice:
                        OSV = (PDCC - askPrice) / (PDCC * threshold_down)
                        max_OSV = max(max_OSV, OSV)
                        # Clamp negative overshoots to zero
                        if OSV < 0:
                            OSV = 0
                            # If an upward signal is already active, exit down-trend
                            if filter_up == 1:
                                filter_down = 0
                                minPrice = 10_000_000
                                maxPrice = 0.5

                    # Always update the running low and overshoot marker date
                    if minPrice >= askPrice or filter_up == 1:
                        minPrice = askPrice
                        osv_down_date_today = date_today
                        OSV = (PDCC - askPrice) / (PDCC * threshold_down)
                        max_OSV = max(max_OSV, OSV)

                # --- Section C: Detect start of an upward trend ---
                if filter_up == 0:
                    difference2 = askPrice - minPrice2
                    # If price has risen at least Threshold2% from the previous min
                    if minPrice2 < askPrice and difference2 >= minPrice2 * threshold_up:
                        # Confirm PDCC2_UP event
                        maxPrice2 = askPrice
                        PDCC2 = askPrice
                        filter_up = 1  # enter up-trend mode
                        max_OSV2 = 0  # reset overshoot counter

                        # Record the start of the up-trend
                        upward_events.append(
                            (min_date, date_today, minPrice2, PDCC2, "PDCC2_UP")
                        )
                        PDCC2_date = date_today
                        df.loc[df[self.time_col] == date_today, "event"] = "PDCC2_UP"
                        df.loc[
                            df[self.time_col] == date_today, "event_type"
                        ] = "Upward Start"

                        # Also mark the last down-trend’s overshoot update
                        downward_events.append(
                            (PDCC_date, osv_down_date_today, PDCC, minPrice, "OSV_Down")
                        )
                        df.loc[
                            df[self.time_col] == osv_down_date_today, "event"
                        ] = "OSV_Down"
                        df.loc[
                            df[self.time_col] == osv_down_date_today, "event_type"
                        ] = "Downward Update"

                        # Reset down-trend trackers
                        filter_down = 0
                        minPrice = 10_000_000
                        maxPrice = 0.5

                    # If price dips below previous low, just update minPrice2
                    if askPrice < minPrice2:
                        minPrice2 = askPrice
                        min_date = date_today

                # --- Section D: While inside an upward trend, measure overshoot ---
                if filter_up == 1:
                    # If price has fallen back below the running max, compute overshoot
                    if askPrice < maxPrice2:
                        OSV2 = (askPrice - PDCC2) / (PDCC2 * threshold_up)
                        max_OSV2 = max(max_OSV2, OSV2)
                        # Clamp negative overshoots to zero
                        if OSV2 < 0:
                            OSV2 = 0
                            # If a down-signal is already active, exit up-trend
                            if filter_down == 1:
                                filter_up = 0
                                minPrice2 = 10_000_000
                                maxPrice2 = 0.5

                    # Always update the running high and overshoot marker date
                    if askPrice >= maxPrice2 or filter_down == 1:
                        maxPrice2 = askPrice
                        osv_up_date_today = date_today
                        OSV2 = (askPrice - PDCC2) / (PDCC2 * threshold_up)
                        max_OSV2 = max(max_OSV2, OSV2)

            # Final results: the largest overshoot seen in any down- or up-trend
            print(
                f"(down={threshold_down}, up={threshold_up}) "
                f"Final maximum OSV for downward trends: {max_OSV}"
            )
            print(
                f"(down={threshold_down}, up={threshold_up}) "
                f"Final maximum OSV2 for upward trends:   {max_OSV2}"
            )

            # Store events for this threshold pair
            key = (threshold_down, threshold_up)
            self.events[key] = {
                "downward": downward_events,
                "upward": upward_events,
                "max_OSV": max_OSV,
                "max_OSV2": max_OSV2,
                "df": df,
            }

            # Build a Plotly figure marking all PDCC and OSV events
            fig = self._build_plot(
                df=df,
                downward_events=downward_events,
                upward_events=upward_events,
                threshold_pair=key,
            )
            self.figures[key] = fig

            # Load summary to BigQuery
            if self.load_to_bq:
                self._load_dc_events(
                    downward_events + upward_events, threshold_pair=key
                )

        def _build_plot(
            self,
            *,
            df: pd.DataFrame,
            downward_events: list[tuple],
            upward_events: list[tuple],
            threshold_pair: Tuple[float, float],
        ) -> go.Figure:
            """
            Build a Plotly figure marking PDCC and OSV events on prices.
            """
            dates = df[self.time_col]
            prices = df[self.price_col]
            th_down, th_up = threshold_pair
            fig = go.Figure()
            # Use WebGL for faster rendering when requested
            price_trace_cls = go.Scattergl if self.fast_plot else go.Scatter
            fig.add_trace(
                price_trace_cls(x=dates, y=prices, mode="lines", name="Ask Prices")
            )

            # Plot downward-trend starts and updates
            downward_start_added = False
            downward_update_added = False
            down_iter = (
                downward_events[-self.max_event_markers :]
                if (self.fast_plot and self.max_event_markers)
                else downward_events
            )
            for prev, today, price0, price1, event in down_iter:
                if event == "PDCC_Down":
                    fig.add_trace(
                        go.Scatter(
                            x=[today],
                            y=[price1],
                            mode="markers",
                            marker=dict(color="red", symbol="triangle-down"),
                            name="Downward Start",
                            showlegend=not downward_start_added,
                        )
                    )
                    downward_start_added = True
                elif event == "OSV_Down":
                    fig.add_trace(
                        go.Scatter(
                            x=[today],
                            y=[price1],
                            mode="markers",
                            marker=dict(color="orange", symbol="triangle-down"),
                            name="Downward Update",
                            showlegend=not downward_update_added,
                        )
                    )
                    downward_update_added = True

                if pd.notna(prev) and pd.notna(today):
                    fig.add_shape(
                        type="line",
                        x0=prev,
                        y0=price0,
                        x1=today,
                        y1=price1,
                        line=dict(
                            color="red" if event == "PDCC_Down" else "black",
                            width=2,
                            dash="dashdot",
                        ),
                        opacity=0.7,
                    )

            # Plot upward-trend starts and updates on the *same* figure
            upward_start_added = False
            upward_update_added = False
            up_iter = (
                upward_events[-self.max_event_markers :]
                if (self.fast_plot and self.max_event_markers)
                else upward_events
            )
            for prev, today, price0, price1, event in up_iter:
                if event == "PDCC2_UP":
                    fig.add_trace(
                        go.Scatter(
                            x=[today],
                            y=[price1],
                            mode="markers",
                            marker=dict(color="green", symbol="triangle-up"),
                            name="Upward Start",
                            showlegend=not upward_start_added,
                        )
                    )
                    upward_start_added = True
                elif event == "OSV_Up":
                    fig.add_trace(
                        go.Scatter(
                            x=[today],
                            y=[price1],
                            mode="markers",
                            marker=dict(color="magenta", symbol="triangle-up"),
                            name="Upward Update",
                            showlegend=not upward_update_added,
                        )
                    )
                    upward_update_added = True

                if pd.notna(prev) and pd.notna(today):
                    fig.add_shape(
                        type="line",
                        x0=prev,
                        y0=price0,
                        x1=today,
                        y1=price1,
                        line=dict(
                            color="green" if event == "PDCC2_UP" else "magenta",
                            width=2,
                            dash="dashdot",
                        ),
                        opacity=0.7,
                    )

            fig.update_layout(
                title=f"DC Events (Thresholds: down={th_down}, up={th_up})",
                xaxis_title="Date",
                yaxis_title="Price",
                legend_title="Events",
                legend=dict(itemsizing="constant"),
            )
            return fig

        def _load_dc_events(
            self, events: list[tuple], *, threshold_pair: Tuple[float, float]
        ) -> Optional[pd.DataFrame]:
            """
            Loads DC summary events to BigQuery with experiment metadata.

            - Remove the default null start/end dates.
            - Create DC summary events DataFrame and load it to BQ.
            """
            if len(events) <= 1:
                print("No DC events to load")
                return None

            # Remove the default null start/end dates
            clean_events = [ev for ev in events if pd.notna(ev[0]) and pd.notna(ev[1])]
            if not clean_events:
                print("No DC events to load")
                return None

            # Create DC summary events dataframe
            df_ev = pd.DataFrame(
                clean_events,
                columns=[
                    "start_time",
                    "end_time",
                    "start_price",
                    "end_price",
                    "event_type",
                ],
            )
            df_ev["threshold_down"], df_ev["threshold_up"] = threshold_pair
            df_ev["experiment_name"] = self.experiment_name
            df_ev["experiment_id"] = self.experiment_id

            # # Ensure pandas Timestamps become UTC tz-aware
            # df_ev["start_time"] = pd.to_datetime(df_ev["start_time"], utc=True, errors="coerce")
            # df_ev["end_time"] = pd.to_datetime(df_ev["end_time"], utc=True, errors="coerce")
            
            # Ensure timestamps are timezone-aware but do NOT drop any rows
            df_ev["start_time"] = pd.to_datetime(
                df_ev["start_time"], utc=True, format="mixed", errors="raise"
            )
            df_ev["end_time"] = pd.to_datetime(
                df_ev["end_time"], utc=True, format="mixed", errors="raise"
            )

            table_ref = self.client.dataset(self.dataset_id).table(self.table_id)
            job = self.client.load_table_from_dataframe(df_ev, table_ref)
            job.result()  # wait for load
            print("Loaded DC events to BigQuery.")
            return df_ev

        def _save_figure_to_gcs(
            self, fig: go.Figure, threshold_pair: Tuple[float, float]
        ) -> str:
            """Save a Plotly figure to GCS as HTML and return the GCS path."""
            if not self.figures_output_path:
                return ""

            # Create filename based on threshold
            down_th, up_th = threshold_pair
            filename = f"dc_events_down_{down_th}_up_{up_th}.html"

            # Parse GCS path
            if self.figures_output_path.startswith("gs://"):
                bucket_name = self.figures_output_path[5:].split("/")[0]
                prefix = "/".join(self.figures_output_path[5:].split("/")[1:])
                if prefix and not prefix.endswith("/"):
                    prefix += "/"
                gcs_path = f"{prefix}{filename}"
            else:
                # Assume it's a local path for testing
                gcs_path = os.path.join(self.figures_output_path, filename)
                os.makedirs(os.path.dirname(gcs_path), exist_ok=True)
                fig.write_html(gcs_path)
                return gcs_path

            # Save to GCS
            client = storage.Client(project=self.project_id)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(gcs_path)

            # Convert figure to HTML string
            html_content = fig.to_html(include_plotlyjs="cdn")
            blob.upload_from_string(html_content, content_type="text/html")

            full_gcs_path = f"gs://{bucket_name}/{gcs_path}"
            print(f"Saved figure to: {full_gcs_path}")
            return full_gcs_path

        def _save_all_figures(self) -> None:
            """Save all generated figures to GCS."""
            if not self.figures_output_path:
                return

            saved_paths = []
            for threshold_pair, fig in self.figures.items():
                gcs_path = self._save_figure_to_gcs(fig, threshold_pair)
                if gcs_path:
                    saved_paths.append(gcs_path)

            # Save metadata about saved figures
            if saved_paths:
                metadata = {
                    "experiment_name": self.experiment_name,
                    "experiment_id": self.experiment_id,
                    "saved_figures": saved_paths,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # Save metadata to GCS
                if self.figures_output_path.startswith("gs://"):
                    bucket_name = self.figures_output_path[5:].split("/")[0]
                    prefix = "/".join(self.figures_output_path[5:].split("/")[1:])
                    if prefix and not prefix.endswith("/"):
                        prefix += "/"
                    metadata_path = f"{prefix}figures_metadata.json"

                    client = storage.Client(project=self.project_id)
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(metadata_path)
                    blob.upload_from_string(
                        json.dumps(metadata, indent=2), content_type="application/json"
                    )
                    print(f"Saved metadata to: gs://{bucket_name}/{metadata_path}")

    # Start here
    # Create df from input dataset
    df = pd.read_csv(df.path)

    # Define threshold string e.g. 0p01
    thr_str = str(thresholds).replace('.', 'p')
    # Define experiment name e.g. exp_dc_detection_0p01_20251010173542
    experiment_name = f"exp_dc_detection_{thr_str}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    # Define BQ table id e.g. dc_events_threshold_0p01
    table_id = f"dc_events_threshold_{thr_str}"

    # Create a DirectionalChange object
    dc = DirectionalChange(
        df=df,  # must have columns PRICE and load_time_toronto
        thresholds=[thresholds],  # used for both down & up
        experiment_name=experiment_name,
        price_col=price_col,
        time_col=time_col,
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,  # optional, defaults to "dc_events"
        load_to_bq=load_to_bq,  # set False to skip BQ loads
        fast_plot=fast_plot,
        max_event_markers=max_event_markers,
        figures_output_path=figures_output.path,
    )
    dc.run_all()

    # Save events data to output
    events_data = []
    for threshold_pair in dc.threshold_sets:
        events = dc.get_events(threshold_pair)
        if events:
            # Convert events to DataFrame format for output
            down_events = events["downward"]
            up_events = events["upward"]

            for event in down_events + up_events:
                # Filter out events with missing start/end times.
                # These are typically the first PDCC events where start time is NaN.
                if len(event) >= 5 and pd.notna(event[0]) and pd.notna(event[1]):
                    events_data.append(
                        {
                            "start_time": event[0],
                            "end_time": event[1],
                            "start_price": event[2],
                            "end_price": event[3],
                            "event_type": event[4],
                            "threshold_down": threshold_pair[0],
                            "threshold_up": threshold_pair[1],
                            "experiment_name": experiment_name,
                            "experiment_id": dc.experiment_id,
                        }
                    )
    # Save events data to output
    if events_data:
        events_df = pd.DataFrame(events_data)
        events_df.to_csv(directional_change_events.path, index=False)
        print(f"Saved {len(events_data)} events to {directional_change_events.path}")

    # Figures_output.path is a directory like: /gcs/.../figures_output
    out_dir = Path(figures_output.path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # File name inside the directory
    out_file = out_dir / "figures_metadata.json"

    # Save figures metadata to output
    figures_metadata = {
        "experiment_name": experiment_name,
        "experiment_id": dc.experiment_id,
        "saved_figures": list(dc.figures.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(figures_metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved figures metadata to {out_file}")
    print(
        f"Generated {len(dc.figures)} figures for thresholds: {list(dc.figures.keys())}"
    )
