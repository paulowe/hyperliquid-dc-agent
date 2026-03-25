"""Entry point for running Archon strategy as a module."""
import asyncio
from strategies.archon.bridge import main

asyncio.run(main())
