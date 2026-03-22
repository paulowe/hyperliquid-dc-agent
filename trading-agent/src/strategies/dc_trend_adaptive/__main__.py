"""Module entry point — delegates to trend_bridge."""

from strategies.dc_trend_adaptive.trend_bridge import main
import asyncio

asyncio.run(main())
