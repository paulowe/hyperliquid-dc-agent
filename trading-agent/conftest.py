"""Root conftest for trading-agent tests.

Adds src/ to sys.path so tests can import modules like:
    from strategies.dc_forecast.live_dc_detector import LiveDCDetector
"""

import sys
from pathlib import Path

# Add trading-agent/src to the import path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
