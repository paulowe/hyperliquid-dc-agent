"""Fetch user fills from Hyperliquid API (read-only, no Exchange needed).

Only requires a wallet address — does not need a private key since it
uses the Info (read-only) endpoint only.

For delegation setups, fills are recorded on the main (delegated-to) wallet,
not the API wallet. Set MAINNET_ACCOUNT_ADDRESS / TESTNET_ACCOUNT_ADDRESS
to the main wallet, or pass --wallet on the CLI.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from hyperliquid.info import Info

# Resolve trading-agent/.env (same pattern as live_bridge.py)
_SRC_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _SRC_DIR.parent / ".env"

# Network configuration (mirrors live_bridge.py)
NETWORK_CONFIG = {
    "testnet": {
        "base_url": "https://api.hyperliquid-testnet.xyz",
        "key_env": "HYPERLIQUID_TESTNET_PRIVATE_KEY",
        "wallet_env": "TESTNET_WALLET_ADDRESS",
        "account_env": "TESTNET_ACCOUNT_ADDRESS",
    },
    "mainnet": {
        "base_url": "https://api.hyperliquid.xyz",
        "key_env": "HYPERLIQUID_MAINNET_PRIVATE_KEY",
        "wallet_env": "MAINNET_WALLET_ADDRESS",
        "account_env": "MAINNET_ACCOUNT_ADDRESS",
    },
}


@dataclass
class FillFetcherConfig:
    """Configuration for fetching fills from Hyperliquid."""

    network: str = "mainnet"
    wallet_address: str | None = None

    @classmethod
    def from_env(
        cls,
        network_override: str | None = None,
        wallet_override: str | None = None,
    ) -> "FillFetcherConfig":
        """Build config from environment variables.

        Resolution order for wallet address:
        1. wallet_override (from --wallet CLI flag)
        2. MAINNET_ACCOUNT_ADDRESS / TESTNET_ACCOUNT_ADDRESS (delegation target)
        3. MAINNET_WALLET_ADDRESS / TESTNET_WALLET_ADDRESS
        4. Derived from private key (fallback)

        For delegation setups, fills are on the ACCOUNT (main) wallet, not the
        API wallet. Use MAINNET_ACCOUNT_ADDRESS or --wallet to specify it.
        """
        load_dotenv(_ENV_FILE, override=True)

        network = network_override or os.environ.get(
            "HYPERLIQUID_NETWORK", "mainnet"
        ).lower().strip()

        if network not in NETWORK_CONFIG:
            print(f"Error: HYPERLIQUID_NETWORK must be 'testnet' or 'mainnet', got '{network}'")
            sys.exit(1)

        net_cfg = NETWORK_CONFIG[network]

        # Resolve wallet: CLI override > account address (delegation) > wallet > derive from key
        wallet_address = wallet_override
        if not wallet_address:
            wallet_address = os.environ.get(net_cfg["account_env"])
        if not wallet_address:
            wallet_address = os.environ.get(net_cfg["wallet_env"])
        if not wallet_address:
            private_key = os.environ.get(net_cfg["key_env"])
            if private_key:
                from eth_account import Account
                wallet_address = Account.from_key(private_key).address
            else:
                print(
                    f"Error: Set {net_cfg['account_env']}, {net_cfg['wallet_env']}, "
                    f"or {net_cfg['key_env']} in .env (or use --wallet)"
                )
                sys.exit(1)

        return cls(network=network, wallet_address=wallet_address)


class FillFetcher:
    """Fetch user fills from Hyperliquid API (read-only)."""

    def __init__(self, config: FillFetcherConfig):
        self._config = config
        self._info: Info | None = None
        self._wallet_address: str = config.wallet_address or ""

    def connect(self) -> None:
        """Initialize Info SDK connection."""
        net_cfg = NETWORK_CONFIG[self._config.network]
        self._info = Info(net_cfg["base_url"], skip_ws=True)
        self._wallet_address = self._config.wallet_address or ""

    def fetch(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch fills for a specific coin, filtered by time range.

        Args:
            symbol: Trading pair (e.g., "HYPE", "BTC", "SOL").
            start_time_ms: Start of time range (epoch milliseconds).
            end_time_ms: Optional end of time range (epoch milliseconds).

        Returns:
            List of fill dicts sorted by time ascending, filtered to the
            requested coin.
        """
        if self._info is None:
            raise RuntimeError("Call connect() before fetch()")

        try:
            fills = self._info.user_fills_by_time(
                self._wallet_address, start_time_ms
            )
        except Exception as e:
            print(f"Warning: API error fetching fills: {e}", file=sys.stderr)
            return []
        if not fills:
            return []

        # Filter to requested coin
        fills = [f for f in fills if f.get("coin") == symbol]

        # Filter by end time if specified
        if end_time_ms is not None:
            fills = [f for f in fills if f["time"] <= end_time_ms]

        # Sort by time ascending
        fills.sort(key=lambda f: f["time"])

        return fills
