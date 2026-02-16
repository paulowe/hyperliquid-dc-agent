# dataflow_feature_engineering.py
"""
Dataflow pipeline for feature engineering DC summary and ticks data.
Converts the pandas-based feature engineering logic to Apache Beam/Dataflow.
"""

import argparse
import json
import logging
from typing import Dict, List, Optional, Any, Iterator, Tuple
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import pandas as pd
import numpy as np


class FeatureEngineeringConfig:
    """Configuration class for feature engineering pipeline."""
    
    def __init__(self, columns_json: str = ""):
        self.default_cols = {
            "ticks_time": "load_time_toronto",
            "ticks_price": "PRICE",
            "ticks_event": "event",
            "dc_start_time": "start_time",
            "dc_end_time": "end_time",
            "dc_start_price": "start_price",
            "dc_end_price": "end_price",
            "dc_threshold": "threshold_up",
            # engineered columns
            "col_pdcc_up": "PDCC2_UP",
            "col_pdcc_down": "PDCC_Down",
            "col_osv_up": "OSV_Up",
            "col_osv_down": "OSV_Down",
            "col_pdcc_flag": "PDCC_event_ind",
        }
        
        overrides = json.loads(columns_json) if columns_json else {}
        self.cols = {**self.default_cols, **overrides}
        
        self._int_binary = [self.cols["col_pdcc_up"], self.cols["col_pdcc_down"]]
        self._float_mag = [self.cols["col_osv_down"], self.cols["col_osv_up"]]
        self._dc_feats = [
            self.cols["col_pdcc_down"], self.cols["col_osv_down"],
            self.cols["col_pdcc_up"], self.cols["col_osv_up"]
        ]


class ProcessDCSummary(beam.DoFn):
    """DoFn to process DC summary records and add engineered features."""
    
    def __init__(self, threshold: float, config: FeatureEngineeringConfig):
        self.threshold = threshold
        self.config = config
        
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Process a single DC summary record."""
        c = self.config.cols
        
        # Filter by threshold
        if float(element.get(c["dc_threshold"], 0)) != self.threshold:
            return
            
        # Initialize engineered columns
        for col in self.config._int_binary:
            element[col] = 0
        for col in self.config._float_mag:
            element[col] = 0.0
            
        event_type = element.get("event_type", "")
        
        # Set binary flags based on event type
        if event_type in ["PDCC2_UP", "OSV_Up"]:
            element[c["col_pdcc_up"]] = 1
        elif event_type in ["PDCC_Down", "OSV_Down"]:
            element[c["col_pdcc_down"]] = 1
            
        # Calculate magnitudes for OSV events
        if event_type in ["OSV_Up", "OSV_Down"]:
            start_price = float(element.get(c["dc_start_price"], 0))
            end_price = float(element.get(c["dc_end_price"], 0))
            
            if start_price != 0:
                magnitude = abs((start_price - end_price) / start_price * 100.0)
                
                if event_type == "OSV_Up":
                    element[c["col_osv_up"]] = magnitude
                else:
                    element[c["col_osv_down"]] = magnitude
                    
        yield element


class EngineerTicksFeatures(beam.DoFn):
    """DoFn to engineer features for ticks using DC summary data."""
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.dc_lookup = {}
        
    def setup(self):
        """Setup method called once per worker."""
        pass
        
    def process(self, element: Tuple[str, Dict]) -> Iterator[Dict[str, Any]]:
        """Process ticks with their corresponding DC summary data."""
        key, grouped_data = element
        ticks = grouped_data.get('ticks', [])
        dc_summaries = grouped_data.get('dc_summaries', [])
        
        if not ticks or not dc_summaries:
            return
            
        c = self.config.cols
        
        # Convert to pandas for easier processing (keeping the core logic)
        ticks_df = pd.DataFrame(ticks)
        dc_df = pd.DataFrame(dc_summaries)
        
        # Apply the original ticks feature engineering logic
        try:
            engineered_ticks = self._engineer_tick_features(ticks_df, dc_df)
            
            # Convert back to dictionaries and yield
            for _, row in engineered_ticks.iterrows():
                yield row.to_dict()
                
        except Exception as e:
            logging.error(f"Error processing ticks for key {key}: {e}")
            # Yield original ticks with default engineered features
            for tick in ticks:
                tick.update({col: 0 for col in self.config._int_binary})
                tick.update({col: 0.0 for col in self.config._float_mag})
                tick[c["col_pdcc_flag"]] = 0
                yield tick
    
    def _engineer_tick_features(self, ticks: pd.DataFrame, dc: pd.DataFrame) -> pd.DataFrame:
        """Apply the original tick feature engineering logic."""
        c = self.config.cols
        
        # Time normalization (simplified for Dataflow)
        def _to_utc_naive_datetime(series: pd.Series) -> pd.Series:
            s = pd.to_datetime(series, errors="coerce", utc=True)
            return s.dt.tz_convert("UTC").dt.tz_localize(None)
        
        lt = c["ticks_time"]
        rt = c["dc_start_time"]
        
        ticks[lt] = _to_utc_naive_datetime(ticks[lt])
        dc[rt] = _to_utc_naive_datetime(dc[rt])
        
        # Drop NaT values
        ticks = ticks.dropna(subset=[lt])
        dc = dc.dropna(subset=[rt])
        
        # Sort for merge_asof
        ticks = ticks.sort_values(lt).reset_index(drop=True)
        dc = dc.sort_values(rt).reset_index(drop=True)
        
        # PDCC indicator
        if c["ticks_event"] in ticks.columns:
            mask_pdcc = ticks[c["ticks_event"]].astype("string").str.contains("PDCC", na=False)
            ticks[c["col_pdcc_flag"]] = mask_pdcc.astype(int)
        else:
            ticks[c["col_pdcc_flag"]] = 0
        
        # Prepare merge columns
        need = [c["dc_start_time"], c["dc_end_time"], c["dc_start_price"],
                c["dc_end_price"], c["dc_threshold"]]
        merge_cols = [col for col in need + self.config._dc_feats if col in dc.columns]
        
        # Merge asof
        augmented = pd.merge_asof(
            left=ticks,
            right=dc[merge_cols],
            left_on=lt,
            right_on=rt,
            direction="backward"
        )
        
        # Filter to ticks within time windows
        if c["dc_end_time"] in augmented.columns:
            in_win = ((augmented[c["dc_start_time"]] < augmented[c["ticks_time"]]) & 
                     (augmented[c["ticks_time"]] < augmented[c["dc_end_time"]]))
            augmented = augmented.loc[in_win].copy()
        
        # Initialize engineered columns
        for col in self.config._int_binary:
            if col not in augmented.columns:
                augmented[col] = 0
        for col in self.config._float_mag:
            if col not in augmented.columns:
                augmented[col] = 0.0
        
        # Calculate tick-level magnitudes
        if {c["ticks_price"], c["dc_start_price"]}.issubset(augmented.columns):
            def pct(a, b): 
                return np.abs((a - b) / a * 100.0)
            
            mask_up = augmented[c["col_pdcc_up"]].eq(1)
            mask_down = augmented[c["col_pdcc_down"]].eq(1)
            
            if mask_up.any():
                augmented.loc[mask_up, c["col_osv_up"]] = pct(
                    augmented.loc[mask_up, c["dc_start_price"]],
                    augmented.loc[mask_up, c["ticks_price"]]
                )
            
            if mask_down.any():
                augmented.loc[mask_down, c["col_osv_down"]] = pct(
                    augmented.loc[mask_down, c["dc_start_price"]],
                    augmented.loc[mask_down, c["ticks_price"]]
                )
        
        # Fill NaNs
        augmented[self.config._int_binary] = augmented[self.config._int_binary].fillna(0)
        augmented[self.config._float_mag] = augmented[self.config._float_mag].fillna(0.0)
        
        return augmented


def create_grouping_key(element: Dict[str, Any]) -> str:
    """Create a grouping key for combining ticks and DC summaries."""
    # Group by market and instrument if available
    market = element.get('MARKET', 'default')
    instrument = element.get('INSTRUMENT', 'default')
    return f"{market}_{instrument}"


def run_pipeline(argv=None):
    """Main pipeline function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dc_summary_input', required=True, help='DC summary input file path')
    parser.add_argument('--raw_ticks_input', required=True, help='Raw ticks input file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--threshold', type=float, required=True, help='Threshold value')
    parser.add_argument('--columns_json', default='', help='Column mapping JSON')
    
    known_args, pipeline_args = parser.parse_known_args(argv)
    
    pipeline_options = PipelineOptions(pipeline_args)
    config = FeatureEngineeringConfig(known_args.columns_json)
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        
        # Read DC summaries
        dc_summaries = (
            pipeline
            | 'Read DC Summaries' >> beam.io.ReadFromText(known_args.dc_summary_input)
            | 'Parse DC CSV' >> beam.Map(lambda line: dict(zip(
                # Add your expected DC summary column names here
                ['start_time', 'end_time', 'start_price', 'end_price', 'event_type', 
                 'threshold_down', 'threshold_up', 'experiment_name', 'experiment_id'],
                line.split(',')
            )))
            | 'Process DC Features' >> beam.ParDo(ProcessDCSummary(known_args.threshold, config))
            | 'Key DC by Group' >> beam.Map(lambda x: (create_grouping_key(x), x))
        )
        
        # Read ticks
        raw_ticks = (
            pipeline
            | 'Read Raw Ticks' >> beam.io.ReadFromText(known_args.raw_ticks_input)
            | 'Parse Ticks CSV' >> beam.Map(lambda line: dict(zip(
                # Add your expected ticks column names here
                ['MARKET', 'INSTRUMENT', 'trade_id', 'trade_ts', 'load_time_toronto', 
                 'PRICE', 'vol_base', 'vol_quote', 'cvd_base', 'cvd_quote', 
                 'log_return', 'inter_trade_duration_seconds'],
                line.split(',')
            )))
            | 'Key Ticks by Group' >> beam.Map(lambda x: (create_grouping_key(x), x))
        )
        
        # Group ticks and DC summaries together
        grouped_data = (
            {'ticks': raw_ticks, 'dc_summaries': dc_summaries}
            | 'Group by Key' >> beam.CoGroupByKey()
        )
        
        # Engineer features
        engineered_ticks = (
            grouped_data
            | 'Engineer Tick Features' >> beam.ParDo(EngineerTicksFeatures(config))
        )
        
        # Write output
        (
            engineered_ticks
            | 'Convert to CSV' >> beam.Map(lambda x: ','.join([str(v) for v in x.values()]))
            | 'Write Output' >> beam.io.WriteToText(known_args.output)
        )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run_pipeline()