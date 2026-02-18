"""Configuration for the DC Forecast strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# Default feature names for midprice-only mode (no vol/CVD)
DEFAULT_FEATURE_NAMES = [
    "PRICE_std",
    "PDCC_Down",
    "OSV_Down_std",
    "PDCC2_UP",
    "OSV_Up_std",
    "regime_up",
    "regime_down",
]


@dataclass
class DCForecastConfig:
    """Configuration for the DC Forecast trading strategy."""

    symbol: str = "BTC"

    # DC detector thresholds: list of (down, up) pairs
    dc_thresholds: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.001, 0.001)]
    )

    # Model paths (GCS or local)
    model_uri: str = ""
    scaler_uri: str = ""

    # Feature configuration (must match training feature set)
    feature_names: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_NAMES))

    # Windowing
    window_size: int = 50

    # Signal generation
    signal_threshold_pct: float = 0.1
    position_size_usd: float = 50.0
    max_position_size_usd: float = 200.0

    # Modes
    inference_on_every_tick: bool = False
    log_dc_events: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DCForecastConfig:
        """Create config from a flat dict (e.g., from YAML)."""
        # Normalize thresholds
        raw_thresholds = d.get("dc_thresholds", [(0.001, 0.001)])
        thresholds = []
        for t in raw_thresholds:
            if isinstance(t, (list, tuple)) and len(t) == 2:
                thresholds.append((float(t[0]), float(t[1])))
            elif isinstance(t, (int, float)):
                thresholds.append((float(t), float(t)))
            else:
                raise ValueError(f"Invalid threshold: {t}")

        return cls(
            symbol=d.get("symbol", "BTC"),
            dc_thresholds=thresholds,
            model_uri=d.get("model_uri", ""),
            scaler_uri=d.get("scaler_uri", ""),
            feature_names=d.get("feature_names", list(DEFAULT_FEATURE_NAMES)),
            window_size=int(d.get("window_size", 50)),
            signal_threshold_pct=float(d.get("signal_threshold_pct", 0.1)),
            position_size_usd=float(d.get("position_size_usd", 50.0)),
            max_position_size_usd=float(d.get("max_position_size_usd", 200.0)),
            inference_on_every_tick=bool(d.get("inference_on_every_tick", False)),
            log_dc_events=bool(d.get("log_dc_events", True)),
        )
