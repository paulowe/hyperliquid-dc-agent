# components/feature_engineering.py
from kfp.v2.dsl import component, Input, Output, Dataset, Metrics

# Example output we want to achieve:
# -----------------------------------------------------------------------
# | Tick | Event     | Regime | PDCC_up | PDCC_down | OSV_Up | OSV_Down |
# | ---- | --------- | ------ | ------- | --------- | ------ | -------- |
# | t0   | —         | None   | 0       | 0         | 0      | 0        |
# | t1   | PDCC_UP   | Up     | 1       | 0         | 0      | 0        |
# | t2   | —         | Up     | 1       | 0         | 0.05   | 0        |
# | t3   | —         | Up     | 1       | 0         | 0.10   | 0        |
# | t4   | PDCC_DOWN | Down   | 0       | 1         | 0      | 0        |
# | t5   | —         | Down   | 0       | 1         | 0      | 0.07     |
# | t6   | —         | Down   | 0       | 1         | 0      | 0.12     |
# | t7   | PDCC_UP   | Up     | 1       | 0         | 0      | 0        |
# -----------------------------------------------------------------------

@component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas", 
        "numpy", 
        "pyarrow",
        "google-cloud-bigquery",
    ],
)
def feature_engineering(
    dc_summary: Input[Dataset],
    raw_ticks: Input[Dataset],
    threshold: float,
    dc_ticks: Output[Dataset],
    metrics: Output[Metrics],
    columns_json: str = "",
):
    """Component wrapper: calls FeatureEngineer().run()"""
    import os
    import pandas as pd
    import numpy as np
    import json
    import logging
    from google.cloud import bigquery
      

    class FeatureEngineer:
        """
        Stateful DC Feature Engineering for full-tick coverage.

        This class augments every tick with Directional Change (DC) regime features.
        It tracks the last known PDCC event (upward or downward) and computes
        instantaneous overshoot magnitudes for all ticks within that regime.

        It is designed to be fully threshold-isolated: each instance processes
        a single threshold level independently.
        """
        def __init__(self, ticks_path, dc_path, threshold, columns_json, metrics, output_dataset):
            """
            Parameters
            ----------
            ticks_path : str
                Path to CSV of raw tick data.
            dc_path : str
                Path to CSV of DC summary data (output from DC detection component).
            threshold : float
                The DC threshold currently being processed (up/down symmetry assumed).
            columns_json : str
                Optional JSON mapping of column names to adapt to other schemas.
            metrics : kfp.v2.dsl.Metrics or None
                Vertex AI metrics object for experiment logging.
            """
            self.ticks_path = ticks_path
            self.dc_path = dc_path
            self.threshold = float(threshold)
            self.metrics = metrics
            self.output_dataset = output_dataset
            # Schema abstraction layer to make FE logic generic
            default_cols = {
                "ticks_time": "trade_ts",
                "ticks_price": "PRICE",
                "ticks_event": "event",

                "dc_start_time": "start_time",
                "dc_end_time": "end_time",
                "dc_start_price": "start_price",
                "dc_end_price": "end_price",
                "dc_threshold": "threshold_up",
                "dc_event_type": "event_type",

                # engineered outputs
                "col_pdcc_up": "PDCC2_UP",
                "col_pdcc_down": "PDCC_Down",
                "col_osv_up": "OSV_Up",
                "col_osv_down": "OSV_Down",
                # "col_pdcc_flag": "PDCC_event_ind",
                # "col_osv_up_flag": "OSV_Up_flag",
                # "col_osv_down_flag": "OSV_Down_flag",
            }
            overrides = json.loads(columns_json) if columns_json else {}
            self.C = {**default_cols, **overrides}

        # ---------- Helpers ----------
        @staticmethod
        def _to_utc_naive(series: pd.Series) -> pd.Series:
            """
            Convert timestamps to UTC and strip timezone info.
            Handles strings with trailing ' UTC' or mixed formats robustly.
            Raises on invalid datetime formats to enforce data integrity.
            """
            try:
                # Normalize potential ' UTC' suffixes and mixed formats
                series = series.astype(str).str.replace(" UTC", "", regex=False)
                s = pd.to_datetime(series, errors="raise", utc=True, format='mixed')
                return s.dt.tz_convert("UTC").dt.tz_localize(None)
            except Exception as e:
                raise ValueError(f"[FeatureEngineer] Timestamp conversion failed: {e}")
        
        def _load_inputs(self):
            ticks_df = pd.read_csv(self.ticks_path)
            dc_df = pd.read_csv(self.dc_path)
            logging.info(f"[FE] Loaded ticks={len(ticks_df)}, dc_summary={len(dc_df)}")
            return ticks_df, dc_df

        def _prepare_dc_anchors(self, dc_df):
            """
            Extract PDCC events (PDCC2_UP, PDCC_Down) as regime change anchors.

            Purpose:
            --------
            We need to know timestamps where the market switched regimes (DC confirmation events)
            (e.g., from downward → upward, or vice versa). Those events define the start
            of a new regime whose overshoot and features we'll track until the next PDCC event.

            This function extracts and cleans that "anchor timeline".

            Returns
            -------
            anchors : DataFrame
                Columns include:
                - C["dc_end_time"]: timestamp of event
                - C["dc_end_price"]: price at event
                - C["dc_event_type"]: event label ("PDCC2_UP" or "PDCC_Down")
                - regime: derived label ("up" or "down")
            """
            C = self.C
            # Work on a copy so we never mutate the input dataframe
            dc_f = dc_df.copy()
            # Filter DC summary rows to only the current threshold
            # The DC summary table MAY contain multiple thresholds (0.001, 0.005, etc.).
            # We only want anchors that correspond to *this* component's active threshold.
            if C["dc_threshold"] in dc_f.columns:
                dc_f = dc_f[np.isclose(
                    dc_f[C["dc_threshold"]].astype(float), # cast to float
                    self.threshold,  # this is the component's active threshold
                    rtol=0, atol=1e-12 # exact match within tiny tolerance
                )]
            # Keep only PDCC events (regime change anchors)
            # We don't want OSV (overshoot) rows here; those don’t start a new regime.
            # PDCC2_UP → marks the beginning of an upward regime
            # PDCC_Down → marks the beginning of a downward regime
            want = dc_f[C["dc_event_type"]].isin(["PDCC2_UP", "PDCC_Down"])
            # --- Select and rename relevant columns ---
            # Rename them to uniform names so downstream logic doesn’t depend on dataset schema.
            # 'end_time' becomes 'event_time' → the timestamp of the PDCC event
            # 'end_price' becomes 'anchor_price' → the price level at that event
            anchors = dc_f.loc[want, [
                C["dc_end_time"], C["dc_end_price"], C["dc_event_type"]
            ]]

            # Ensure end_time is parsed and timezone-normalized
            anchors[C["dc_end_time"]] = self._to_utc_naive(anchors[C["dc_end_time"]])
            
            # Sort anchors chronologically by end_time (required for merge_asof)
            anchors = (
                anchors
                .sort_values(C["dc_end_time"])
                .reset_index(drop=True)
            )

            # Label regime direction
            anchors["regime"] = np.where(
                anchors[C["dc_event_type"]].eq("PDCC2_UP"), "up", "down"
            )

            # Warn if there are duplicates (i.e. multiple PDCC events at the same time)
            num_dupes = anchors.duplicated(subset=[C["dc_end_time"]]).sum()
            if num_dupes > 0:
                logging.warning(f"[FE] Found {num_dupes} anchors sharing the same end_time.")

            return anchors

        def _merge_state(self, ticks_df, anchors):
            """Attach latest PDCC regime to each tick.
            
            Align each tick with the latest PDCC anchor (regime state).

            - For every tick timestamp, find the most recent PDCC event that
            occurred before or at that tick.
            - This assigns a regime ('up' or 'down') and anchor price per tick.

            Returns
            -------
            merged : DataFrame
                Columns include:
                - C["ticks_time"]: timestamp of tick
                - C["anchor_price"]: price at anchor event
                - regime: derived label ("up" or "down")
            """
            C = self.C

            # Normalize tick timestamps and sort for chronological merge
            ticks_df[C["ticks_time"]] = self._to_utc_naive(ticks_df[C["ticks_time"]])
            ticks_df = ticks_df.sort_values(C["ticks_time"]).reset_index(drop=True)
            
            # Handle edge case: no anchors for this threshold
            if anchors.empty:
                raise ValueError(
                    f"[FE] No PDCC anchors found for threshold={self.threshold}. "
                    "Cannot compute directional-change features without at least one PDCC event."
                )

            # Merge each tick with the last PDCC anchor before it (backward match)

            # merge_asof performs a time-aware “last known value” join
            # it finds, for each left-side timestamp, the latest right-side row whose 
            # timestamp is less than or equal to the left one.

            # i.e For each tick, what was the most recent PDCC event 
            # that happened before that tick?
            merged = pd.merge_asof(
                left=ticks_df,
                right=anchors[[C["dc_end_time"], C["dc_end_price"], "regime"]],
                left_on=C["ticks_time"], # trade_ts
                right_on=C["dc_end_time"], # PDCC event_time
                direction="backward", # Match latest event <= tick time
            )
            return merged

        @staticmethod
        def _compute_features(df, C):
            """
            Generate continuous Directional Change (DC) features for every tick.

            Each tick inherits its last known PDCC regime (up/down/none) and computes
            instantaneous overshoot magnitudes relative to the corresponding regime's
            anchor price.

            - PDCC2_UP / PDCC_Down fire only on the actual PDCC confirmation tick.
            - regime_up / regime_down added as persistent regime state trackers.

            For each tick:
            - Determine current PDCC regime ("up", "down", or "none")
            - Compute % overshoot from anchor price based on regime:
                    up   → (PRICE - anchor_price) / anchor_price * 100
                    down → (anchor_price - PRICE) / anchor_price * 100
            - Add binary regime indicators and overshoot flags

            Returns
            -------
            df : pandas.DataFrame
                Input ticks annotated with the following engineered columns:

                - C["col_pdcc_flag"]: 1 if tick belongs to any PDCC regime, else 0
                - C["col_pdcc_up"]: 1 if in upward regime, else 0
                - C["col_pdcc_down"]: 1 if in downward regime, else 0
                - C["col_osv_up"]: upward overshoot magnitude (%)
                - C["col_osv_down"]: downward overshoot magnitude (%)
                - C["col_osv_up_flag"]: 1 if overshoot magnitude > 0 in upward regime
                - C["col_osv_down_flag"]: 1 if overshoot magnitude > 0 in downward regime

            Notes
            -----
            - Overshoots are set to 0 outside active regimes.
            - NaN or infinite values are replaced with 0 for stability.
            - This method assumes df already includes columns:
                PRICE, anchor_price (C["dc_end_price"]), and regime.
            """
            
            # Normalize regime column
            reg = df["regime"].fillna("none")

            # Continuous regime indicators
            regime_up = (reg == "up").astype(int)
            regime_down = (reg == "down").astype(int)

            # PDCC event confirmation flags 
            # Mark only the tick that aligns exactly with the PDCC event itself
            tick_time = df[C["ticks_time"]]
            anchor_time = df[C["dc_end_time"]]
            pdcc2_up = ((reg == "up") & (tick_time == anchor_time)).astype(int)
            pdcc_down = ((reg == "down") & (tick_time == anchor_time)).astype(int)

            # Signed OSV magnitudes
            price = df[C["ticks_price"]].astype(float)
            # Anchors will be NaN for ticks before the first PDCC event.
            # Guard division to avoid propagating NaNs or Infs in those rows.
            anchor = df[C["dc_end_price"]].astype(float)
            safe_anchor = np.where(np.isnan(anchor) | (anchor == 0.0), np.nan, anchor)
            

            # Continuous signed OSV magnitudes (actual magnitudes, even during retracements)
            osv_up = np.where(
                regime_up.astype(bool),
                (price - safe_anchor) / safe_anchor * 100.0,
                0.0,   # outside regime
            )

            osv_down = np.where(
                regime_down.astype(bool),
                (safe_anchor - price) / safe_anchor * 100.0,
                0.0,   # outside regime
            )

            # Clamped OSV magnitudes (floor at 0)
            osv_up_v2 = np.where(
                regime_up.astype(bool),
                np.maximum(0.0, (price - safe_anchor) / safe_anchor * 100.0),
                0.0,
            )
            osv_down_v2 = np.where(
                regime_down.astype(bool),
                np.maximum(0.0, (safe_anchor - price) / safe_anchor * 100.0),
                0.0,
            )

            # Replace NaNs from safe_anchor with 0 only outside regime; inside regime we want continuity.
            # (If anchor was NaN inside regime, that indicates an upstream join issue and should be rare.)
            # Pragmatically fill any residual NaNs with 0:
            osv_up, osv_down, osv_up_v2, osv_down_v2 = [
                np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                for x in (osv_up, osv_down, osv_up_v2, osv_down_v2)
            ]


            # Detect NaN or Inf before cleaning
            nan_up, nan_dn = np.isnan(osv_up).sum(), np.isnan(osv_down).sum()
            inf_up, inf_dn = np.isinf(osv_up).sum(), np.isinf(osv_down).sum()
            if any([nan_up, nan_dn, inf_up, inf_dn]):
                logging.warning(f"[FE] NaNs: up={nan_up}, down={nan_dn}; infs: up={inf_up}, down={inf_dn}")

            # Write engineered columns
            df[C["col_pdcc_up"]] = pdcc2_up
            df[C["col_pdcc_down"]] = pdcc_down

            # Persistent regime trackers
            df["regime_up"] = regime_up
            df["regime_down"] = regime_down

            # Continuous signed OSV magnitudes
            df[C["col_osv_up"]] = osv_up
            df[C["col_osv_down"]] = osv_down

            # Clamped OSV magnitudes
            df[f"{C['col_osv_up']}_clamped"] = osv_up_v2
            df[f"{C['col_osv_down']}_clamped"] = osv_down_v2

            return df

        @staticmethod
        def _compute_regime_stats(df, C):
            """
            Compute per-regime duration statistics (in ticks).

            For each regime between consecutive PDCC events,
            count how many ticks belong to that regime and report
            mean/std per regime type (up vs down).
            """
            # Sort chronologically to ensure consistent order
            df = df.sort_values(C["ticks_time"]).reset_index(drop=True)

            # Mark PDCC boundaries and regime type at each start
            pdcc_idx = df.index[(df[C["col_pdcc_up"]] == 1) | (df[C["col_pdcc_down"]] == 1)]
            regime_type = np.where(
                df.loc[pdcc_idx, C["col_pdcc_up"]] == 1, "up", "down"
            )

            # If no PDCC events, return zeros
            if len(pdcc_idx) == 0:
                return {
                    "avg_ticks_up_regime": 0.0,
                    "avg_ticks_down_regime": 0.0,
                    "std_ticks_up_regime": 0.0,
                    "std_ticks_down_regime": 0.0,
                    "num_up_regimes": 0,
                    "num_down_regimes": 0,
                }

            # Append the last index to close final segment
            pdcc_idx = list(pdcc_idx) + [len(df)]
            lengths = np.diff(pdcc_idx)

            # Match each regime length with its regime type
            regimes = pd.DataFrame({
                "type": regime_type,
                "length": lengths,
            })

            # Compute per-type stats
            stats = {}
            for t in ["up", "down"]:
                lengths_t = regimes.loc[regimes["type"] == t, "length"]
                stats[f"avg_ticks_{t}_regime"] = float(lengths_t.mean()) if len(lengths_t) else 0.0
                stats[f"std_ticks_{t}_regime"] = float(lengths_t.std()) if len(lengths_t) else 0.0
                stats[f"num_{t}_regimes"] = int(len(lengths_t))

            return stats

        def _collect_metrics(self, ticks_df, dc_df, anchors, engineered):
            """Collect record counts, coverage, and NaN sanity checks."""
            C = self.C
            fe_metrics = {
                "num_ticks_input": len(ticks_df),
                "num_dc_events_input": len(dc_df),
                "num_pdcc_anchors": len(anchors),
                "num_ticks_output": len(engineered),
                "tick_count_match": int(len(engineered) == len(ticks_df)),
            }

            # Count regime coverage
            fe_metrics["regime_up_ticks"] = int(engineered["regime_up"].sum())
            fe_metrics["regime_down_ticks"] = int(engineered["regime_down"].sum())

            # Compute regime duration statistics
            regime_stats = self._compute_regime_stats(engineered, C)
            fe_metrics.update(regime_stats)

            # Check for NaNs in engineered DC columns
            engineered_cols = [
                C["col_pdcc_up"], C["col_pdcc_down"],
                C["col_osv_up"], C["col_osv_down"],
                f"{C['col_osv_up']}_clamped", f"{C['col_osv_down']}_clamped",
            ]
            nan_stats = engineered[engineered_cols].isna().sum().to_dict()
            for k, v in nan_stats.items():
                fe_metrics[f"nan_count_{k}"] = int(v)

            # Log none regime count
            none_count = int(engineered["regime"].isna().sum())
            fe_metrics["none_regime_count"] = none_count

            # Log metrics to Cloud Logging
            logging.info(f"[FE metrics] {fe_metrics}")

            # Log metrics to Vertex AI metrics
            if self.metrics is not None:
                for k, v in fe_metrics.items():
                    if isinstance(v, (int, float)):
                        self.metrics.log_metric(k, float(v))
            return

        def _load_to_bq(
            self,
            df_ticks: pd.DataFrame,
            project_id: str,
            dataset_id: str,
            table_id: str,
        ):
            """
            Load feature-engineered DC tick dataset to BigQuery with metadata.

            This writes every augmented tick (PRICE, PDCC flags, OSV magnitudes, etc.)
            to BigQuery under a given experiment and threshold context.

            Parameters
            ----------
            df_ticks : pd.DataFrame
                Full tick-level feature-engineered dataset.
            threshold : float
                The DC threshold value used for this processing run.
            project_id : str
                GCP project ID where the table resides.
            dataset_id : str
                BigQuery dataset ID (e.g. 'coindesk').
            table_id : str
                BigQuery table ID (e.g. 'feature_engineered_ticks').

            Returns
            -------
            df_ticks : pd.DataFrame
                The same DataFrame, returned after successful load.
            """

            if df_ticks.empty:
                logging.warning("[FE->BQ] No rows to load (empty DataFrame).")
                return None

            # Ensure timestamps are parsed and UTC-normalized
            C = self.C
            try:
                df_ticks[C["ticks_time"]] = pd.to_datetime(
                    df_ticks[C["ticks_time"]], utc=True, format="mixed", errors="raise"
                )
            except Exception as e:
                raise ValueError(f"[FE->BQ] Timestamp normalization failed: {e}")    
    
            # Initialize BigQuery client
            client = bigquery.Client(project=project_id)
            table_ref = client.dataset(dataset_id).table(table_id)

            # Define BigQuery job config
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE",
                # source_format=bigquery.SourceFormat.PARQUET,
                time_partitioning=bigquery.TimePartitioning(field=self.C["ticks_time"]),
                # clustering_fields=[
                #     C["ticks_price"],
                #     C["col_pdcc_up"], C["col_pdcc_down"],
                #     C["col_osv_up"], f"{C['col_osv_up']}_clamped",
                #     C["col_osv_down"], f"{C['col_osv_down']}_clamped",
                # ],
            )

            # Load DataFrame to BigQuery
            try:
                job = client.load_table_from_dataframe(df_ticks, table_ref, job_config=job_config)
                job.result()  # Wait for the job to finish
                logging.info(f"[FE->BQ] Loaded {len(df_ticks)} ticks into {dataset_id}.{table_id}")
            except Exception as e:
                raise RuntimeError(f"[FE->BQ] Failed to load feature-engineered ticks: {e}")

            return df_ticks

        def run(self):
            """
            Run the complete feature engineering workflow:
                1. Load data
                2. Prepare PDCC anchors
                3. Merge anchors with ticks
                4. Compute DC features
                5. Collect metrics
                6. Save engineered output
            """
            ticks_df, dc_df = self._load_inputs()
            anchors = self._prepare_dc_anchors(dc_df)
            merged = self._merge_state(ticks_df, anchors)

            engineered = self._compute_features(merged, self.C)
            self._collect_metrics(ticks_df, dc_df, anchors, engineered)
            
            thr_str = str(self.threshold).replace('.', 'p')

            os.makedirs(self.output_dataset.path, exist_ok=True)
            out_file = os.path.join(self.output_dataset.path, f"feat_eng_{thr_str}.csv")
            engineered.to_csv(out_file, index=False, encoding="utf-8")
            logging.info(f"[FE] Saved {len(engineered)} rows to {out_file}")
            
            
            self._load_to_bq(
                df_ticks=engineered,
                project_id="derivatives-417104",
                dataset_id="coindesk",
                table_id=f"feat_eng_ticks_{thr_str}",
            )
            logging.info(f"[FE] Loaded {len(engineered)} ticks to BigQuery")

            return engineered

    fe = FeatureEngineer(
        ticks_path=raw_ticks.path,
        dc_path=dc_summary.path,
        threshold=threshold,
        columns_json=columns_json,
        metrics=metrics,
        output_dataset=dc_ticks,
    )
    fe.run()




