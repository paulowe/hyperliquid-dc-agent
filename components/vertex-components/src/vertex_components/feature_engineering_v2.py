# components/feature_engineering.py
from kfp.v2.dsl import component, Input, Output, Dataset, Metrics
from typing import List, Optional


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
        "pandas",
        "numpy",
        "pyarrow",
        "python-json-logger",
        "google-cloud-logging",
    ],
)
def feature_engineering(
    dc_summary: Input[Dataset],
    raw_ticks: Input[Dataset],
    # threshold: List[float],
    threshold: float,
    dc_ticks: Output[Dataset],
    metrics: Output[Metrics],
    columns_json: str = "",
):
    """
    Augment ALL ticks with DC features using a stateful regime model:
    - The latest PDCC event (PDCC2_UP or PDCC_Down) defines the regime.
    - Regime persists until the opposite PDCC occurs.
    - Overshoot is measured continuously from the PDCC confirmation price.
    - No tick-level filtering; all ticks are retained with zeros when out-of-context.

    `columns_json` can override expected column names, e.g.:
      {
        "ticks_time": "LOAD_TIME",
        "ticks_price": "PRICE",
        "ticks_event": "event",
        "dc_start_time": "start_time",
        "dc_end_time": "end_time",
        "dc_start_price": "start_price",
        "dc_end_price": "end_price",
        "dc_threshold": "threshold"
      }
    """
    import json
    import logging
    from typing import Dict
    import numpy as np
    import pandas as pd
    

    # -----------------------------
    # Column defaults (overridable)
    # -----------------------------
    default_cols = {
        "ticks_time": "load_time_toronto",
        "ticks_price": "PRICE",
        "ticks_event": "event",
        "dc_start_time": "start_time",
        "dc_end_time": "end_time",
        "dc_start_price": "start_price",
        "dc_end_price": "end_price",
        "dc_threshold": "threshold_up",
        # engineered columns (created)
        "col_pdcc_up": "PDCC2_UP",
        "col_pdcc_down": "PDCC_Down",
        "col_osv_up": "OSV_Up",
        "col_osv_down": "OSV_Down",
        "col_pdcc_flag": "PDCC_event_ind",
        "col_osv_up_flag": "OSV_Up_flag",
        "col_osv_down_flag": "OSV_Down_flag",
    }
    # start_time,end_time,start_price,end_price,event_type,threshold_down,threshold_up,experiment_name,experiment_id

    # MARKET,INSTRUMENT,trade_id,trade_ts,load_time_toronto,PRICE,vol_base,vol_quote,cvd_base,cvd_quote,log_return,inter_trade_duration_seconds

    overrides = json.loads(columns_json) if columns_json else {}
    C = {**default_cols, **overrides}

    # -----------------------------
    # Load inputs into DataFrames
    # -----------------------------
    dc_df = pd.read_csv(dc_summary.path)
    ticks_df = pd.read_csv(raw_ticks.path)

    # Ensure native float for KFP double
    # Normalize threshold to a float
    if isinstance(threshold, list):
        thr = float(threshold[0])
    else:
        thr = float(threshold)

    print(f"Using threshold: {threshold}")

    # -----------------------------
    # Feature Engineering class
    # -----------------------------
    class FeatureEngineer:
        def __init__(self, cols: Dict[str, str]):
            self.cols = cols
            self._int_binary = [cols["col_pdcc_up"], cols["col_pdcc_down"]]
            self._float_mag = [cols["col_osv_down"], cols["col_osv_up"]]
            self._osv_flags = [cols["col_osv_down_flag"], cols["col_osv_up_flag"]]
            self._dc_feats = [
                cols["col_pdcc_down"],
                cols["col_osv_down"],
                cols["col_pdcc_up"],
                cols["col_osv_up"],
                *self._osv_flags,
            ]

        def dc_summary_features(
            self, dc: pd.DataFrame, threshold: float
        ) -> pd.DataFrame:

            # Column mapping
            c = self.cols

            # Filter DC summary by threshold
            dc = dc[dc[c["dc_threshold"]] == float(threshold)].copy()

            # init engineered cols to 0 and 0.0
            for col in self._int_binary:
                dc[col] = 0
            for col in self._float_mag:
                dc[col] = 0.0
            for col in self._osv_flags:
                dc[col] = 0

            # flags for PDCC2_UP, OSV_Up, PDCC_Down, OSV_Down
            mask_pdcc2_up = dc["event_type"].eq("PDCC2_UP")
            mask_osv_up = dc["event_type"].eq("OSV_Up")
            mask_pdcc_dn = dc["event_type"].eq("PDCC_Down")
            mask_osv_dn = dc["event_type"].eq("OSV_Down")

            dc.loc[mask_pdcc2_up, c["col_pdcc_up"]] = 1
            dc.loc[mask_pdcc_dn, c["col_pdcc_down"]] = 1
            dc.loc[mask_osv_up, c["col_osv_up_flag"]] = 1
            dc.loc[mask_osv_dn, c["col_osv_down_flag"]] = 1

            # magnitudes: |start - end| / start * 100
            if {c["dc_start_price"], c["dc_end_price"]}.issubset(dc.columns):

                def pct(a, b):
                    return np.abs((a - b) / a * 100.0)

                dc.loc[mask_osv_up, c["col_osv_up"]] = pct(
                    dc.loc[mask_osv_up, c["dc_start_price"]],
                    dc.loc[mask_osv_up, c["dc_end_price"]],
                )
                dc.loc[mask_osv_dn, c["col_osv_down"]] = pct(
                    dc.loc[mask_osv_dn, c["dc_start_price"]],
                    dc.loc[mask_osv_dn, c["dc_end_price"]],
                )
            return dc

        def ticks_features(self, ticks: pd.DataFrame, dc: pd.DataFrame) -> pd.DataFrame:
            """
            Map each tick to the most recent DC summary row, copy one-hot flags, and
            compute tick-level magnitudes (|price - start_price| / start_price * 100).
            Returns ticks with engineered overshoot features.
            """
            # Column mapping
            c = self.cols
            # --- Normalise time columns for merge_asof ---
            lt = c["ticks_time"]  # e.g., "load_time_toronto"
            rt = c["dc_start_time"]  # e.g., "start_time"

            def _to_utc_naive_datetime(series: pd.Series) -> pd.Series:
                s = pd.to_datetime(series, errors="coerce", utc=True)
                # merge_asof prefers naive or same tz on both sides; use naive
                # UTC consistently
                return s.dt.tz_convert("UTC").dt.tz_localize(None)

            # Coerce both sides to the same datetime64[ns] dtype
            ticks[lt] = _to_utc_naive_datetime(ticks[lt])
            dc[rt] = _to_utc_naive_datetime(dc[rt])

            # Identify NaT positions
            nat_mask_ticks = ticks[lt].isna()
            nat_mask_dc = dc[rt].isna()

            # Log
            n_left = int(nat_mask_ticks.sum())
            n_right = int(nat_mask_dc.sum())
            if n_left:
                print(f"[DEBUG] Dropping {n_left} tick rows with NaT in {lt}")
                print(ticks.loc[nat_mask_ticks, [lt]])
            if n_right:
                print(f"[DEBUG] Dropping {n_right} dc rows with NaT in {rt}")
                print(dc.loc[nat_mask_dc, [rt]])

            # **Drop** NaTs
            ticks = ticks.loc[~nat_mask_ticks].copy()
            dc = dc.loc[~nat_mask_dc].copy()

            # Sort after conversion (required by merge_asof)
            ticks = ticks.sort_values(lt, na_position="last").reset_index(drop=True)
            dc = dc.sort_values(rt, na_position="last").reset_index(drop=True)

            # Missing value percentages for all columns
            ticks_missing_percentages = ticks.isnull().mean() * 100
            print(ticks_missing_percentages)
            dc_missing_percentages = dc.isnull().mean() * 100
            print(dc_missing_percentages)

            # Optional: sanity checks & logging
            if not ticks[lt].is_monotonic_increasing:
                raise ValueError(
                    f"ticks[{lt}] must be sorted ascending for merge_asof."
                )
            if not dc[rt].is_monotonic_increasing:
                raise ValueError(f"dc[{rt}] must be sorted ascending for merge_asof.")

            # PDCC indicator on ticks (optional if event present)
            if c["ticks_event"] in ticks.columns:
                mask_pdcc = (
                    ticks[c["ticks_event"]]
                    .astype("string")
                    .str.contains("PDCC", na=False)
                )
                ticks[c["col_pdcc_flag"]] = mask_pdcc.astype(int)
            else:
                ticks[c["col_pdcc_flag"]] = 0

            # Merge-asof: bring dc features to ticks
            need = [
                c["dc_start_time"],
                c["dc_end_time"],
                c["dc_start_price"],
                c["dc_end_price"],
                c["dc_threshold"],
            ]
            merge_cols = [col for col in need + self._dc_feats if col in dc.columns]

            if (
                c["ticks_time"] not in ticks.columns
                or c["dc_start_time"] not in dc.columns
            ):
                raise ValueError(
                    "Missing columns for merge_asof: ticks[%s], dc[%s]"
                    % (c["ticks_time"], c["dc_start_time"])
                )

            # Constrain by instrument/venue to avoid cross-matching
            # by_cols = []
            # for k in ("MARKET", "INSTRUMENT"):
            #     if k in ticks.columns and k in dc.columns:
            #         by_cols.append(k)

            augmented = pd.merge_asof(
                left=ticks,
                right=dc[merge_cols],
                left_on=lt,
                right_on=rt,
                direction="backward",
                # by=by_cols if by_cols else None,
                # tolerance=None  # Means no tolerance.
                # If no start_time <= load_time_toronto the summary fields stay NaN.
            )

            # Normalise end_time for comparison and keep ticks in [start_time, end_time)
            if c["dc_end_time"] in augmented.columns:
                augmented[c["dc_end_time"]] = _to_utc_naive_datetime(
                    augmented[c["dc_end_time"]]
                )
                in_win = (
                    augmented[c["dc_start_time"]] <= augmented[c["ticks_time"]]
                ) & (augmented[c["ticks_time"]] < augmented[c["dc_end_time"]])
            else:
                in_win = augmented[c["dc_start_time"]].notna()

            augmented = augmented.loc[in_win].copy()

            # Forward-fill overshoot flags within each DC window so every tick in the
            # overshoot segment inherits the flag (spec expects OSV columns to grow).
            if c["dc_start_time"] in augmented.columns:
                augmented[self._osv_flags] = (
                    augmented.groupby(c["dc_start_time"])[self._osv_flags]
                    .transform("max")
                    .fillna(0)
                    .astype(int)
                )

            # Ensure engineered cols exist
            for col in self._int_binary:
                if col not in augmented.columns:
                    augmented[col] = 0
            for col in self._float_mag:
                if col not in augmented.columns:
                    augmented[col] = 0.0
            for col in self._osv_flags:
                if col not in augmented.columns:
                    augmented[col] = 0

            # Tick-level magnitudes: |start - PRICE| / start * 100 where flags==1
            if {c["ticks_price"], c["dc_start_price"]}.issubset(augmented.columns):

                def pct(a, b):
                    return np.abs((b - a) / a * 100.0)

                base = pct(augmented[c["dc_start_price"]], augmented[c["ticks_price"]])
                augmented[c["col_osv_up"]] = np.where(
                    augmented[c["col_osv_up_flag"]].eq(1), base, 0.0
                )
                augmented[c["col_osv_down"]] = np.where(
                    augmented[c["col_osv_down_flag"]].eq(1), base, 0.0
                )

            # Fill NaNs for engineered cols
            augmented[self._int_binary] = augmented[self._int_binary].fillna(0)
            augmented[self._float_mag] = augmented[self._float_mag].fillna(0.0)
            augmented[self._osv_flags] = augmented[self._osv_flags].fillna(0)
            return augmented

    fe = FeatureEngineer(C)
    new_dc = fe.dc_summary_features(dc_df, thr)
    engineered = fe.ticks_features(ticks_df, new_dc)

    # Write outputs
    engineered.to_csv(dc_ticks.path, index=False)
    print(f"Saved {len(engineered)} ticks to {dc_ticks.path}")

    # Write metrics

    fe_metrics = {}

    # Basic record counts
    fe_metrics["num_ticks_input"] = len(ticks_df)
    fe_metrics["num_dc_events_input"] = len(dc_df)
    fe_metrics["num_dc_events_filtered"] = len(new_dc)
    fe_metrics["num_ticks_output"] = len(engineered)
    # Consistency checks
    fe_metrics["tick_count_match"] = int(len(engineered) == len(ticks_df))
    # Column-level sanity: count NaNs for engineered columns
    engineered_cols = fe._int_binary + fe._float_mag + fe._osv_flags
    nan_stats = engineered[engineered_cols].isna().sum().to_dict()
    for k, v in nan_stats.items():
        fe_metrics[f"nan_count_{k}"] = int(v)

    for k, v in fe_metrics.items():
        if isinstance(v, (float, int)) and v is not None:
            metrics.log_metric(k, float(v))
        else:
            # Non-numeric values are logged to GCP logs for debugging,
            # but won’t appear in Vertex UI
            logging.info(f"Non-numeric metric {k}: {v}")
