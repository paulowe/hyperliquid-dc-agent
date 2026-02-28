"""Pair raw Hyperliquid fills into round-trip TradeRecords.

Uses the ``dir`` field from Hyperliquid fills to match entry fills
with their corresponding exit fills.  Supported ``dir`` values:

- ``"Open Long"`` / ``"Open Short"`` — new position entry
- ``"Close Long"`` / ``"Close Short"`` — position exit
- ``"Long > Short"`` / ``"Short > Long"`` — atomic flip (close + open)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backtesting.engine import TradeRecord

# Default taker fee on Hyperliquid (0.035% per side)
TAKER_FEE_PCT = 0.00035

# Maximum gap (ms) between a close fill and an opposite-direction open fill
# for the close to be labelled a "reversal_close".
REVERSAL_WINDOW_MS = 2000


def pair_fills_to_trades(
    fills: list[dict[str, Any]],
    taker_fee_pct: float = TAKER_FEE_PCT,
) -> list[TradeRecord]:
    """Convenience wrapper around :class:`FillPairer`.

    Sorts fills by time, pairs them into round-trip trades, and returns
    the completed :class:`TradeRecord` list.
    """
    pairer = FillPairer(taker_fee_pct=taker_fee_pct)
    return pairer.process_fills(fills)


class FillPairer:
    """Stateful fill pairer that handles partial fills and multi-fill trades.

    Handles:
    - Multiple fills at the same timestamp (partial fills)
    - Scaling into a position (multiple opens before a close)
    - Scaling out (partial closes)
    - Reversals (close + open in opposite direction within REVERSAL_WINDOW_MS)
    """

    def __init__(self, taker_fee_pct: float = TAKER_FEE_PCT):
        self._taker_fee_pct = taker_fee_pct
        # Tracks currently open position state.
        # Keys: side, entry_price, remaining_size, entry_notional, first_fill_time
        self.open_position: dict[str, Any] | None = None

    def process_fills(self, fills: list[dict[str, Any]]) -> list[TradeRecord]:
        """Process fills chronologically and return completed trades."""
        if not fills:
            return []

        # Sort by time ascending
        sorted_fills = sorted(fills, key=lambda f: f["time"])
        trades: list[TradeRecord] = []

        for i, fill in enumerate(sorted_fills):
            dir_field = fill["dir"]

            if dir_field.startswith("Open"):
                self._handle_open(fill)

            elif dir_field.startswith("Close"):
                # Peek ahead to detect reversals
                next_fill = sorted_fills[i + 1] if i + 1 < len(sorted_fills) else None
                is_reversal = self._is_reversal(fill, next_fill)

                trade = self._handle_close(fill, is_reversal)
                if trade is not None:
                    trades.append(trade)

            elif ">" in dir_field:
                # Atomic flip: "Long > Short" or "Short > Long"
                # Closes the old position and opens a new one in opposite direction.
                flip_trades = self._handle_flip(fill)
                trades.extend(flip_trades)

        return trades

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_open(self, fill: dict[str, Any]) -> None:
        """Handle an Open fill: start a new position or scale in."""
        side = "LONG" if "Long" in fill["dir"] else "SHORT"
        px = float(fill["px"])
        sz = float(fill["sz"])
        notional = px * sz

        if self.open_position is None:
            # Start a new position
            self.open_position = {
                "side": side,
                "entry_price": px,
                "remaining_size": sz,
                "entry_notional": notional,
                "first_fill_time": fill["time"],
            }
        elif self.open_position["side"] == side:
            # Scale in: update weighted average entry price
            pos = self.open_position
            old_notional = pos["entry_notional"]
            new_notional = old_notional + notional
            old_size = pos["remaining_size"]
            new_size = old_size + sz
            pos["entry_price"] = new_notional / new_size
            pos["remaining_size"] = new_size
            pos["entry_notional"] = new_notional
        # else: open in opposite direction while position exists — ignore
        # (shouldn't happen with proper fill data)

    def _handle_close(
        self, fill: dict[str, Any], is_reversal: bool
    ) -> TradeRecord | None:
        """Handle a Close fill: emit a TradeRecord for the closed portion."""
        if self.open_position is None:
            return None  # orphan close, skip

        pos = self.open_position
        exit_px = float(fill["px"])
        close_sz = float(fill["sz"])

        # Determine actual close size (can't close more than position)
        actual_close_sz = min(close_sz, pos["remaining_size"])

        # Compute P&L
        entry_px = pos["entry_price"]
        if pos["side"] == "LONG":
            pnl_pct = (exit_px - entry_px) / entry_px
        else:
            pnl_pct = (entry_px - exit_px) / entry_px

        pnl_usd = pnl_pct * (entry_px * actual_close_sz)

        # Fee accounting: entry fee proportional to closed portion
        entry_notional = entry_px * actual_close_sz
        exit_notional = exit_px * actual_close_sz
        entry_fee = entry_notional * self._taker_fee_pct
        exit_fee = exit_notional * self._taker_fee_pct
        total_fees = entry_fee + exit_fee
        net_pnl_usd = pnl_usd - total_fees

        # Infer exit reason
        if is_reversal:
            reason = "reversal_close"
        elif pnl_pct > 0:
            reason = "take_profit"
        else:
            reason = "stop_loss"

        trade = TradeRecord(
            side=pos["side"],
            entry_price=entry_px,
            exit_price=exit_px,
            size=actual_close_sz,
            entry_time=pos["first_fill_time"] / 1000.0,
            exit_time=fill["time"] / 1000.0,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            total_fees=total_fees,
            net_pnl_usd=net_pnl_usd,
            reason=reason,
        )

        # Update or clear position
        remaining = pos["remaining_size"] - actual_close_sz
        if remaining <= 1e-12:
            self.open_position = None
        else:
            pos["remaining_size"] = remaining
            # Entry notional proportionally reduced
            pos["entry_notional"] = entry_px * remaining

        return trade

    def _handle_flip(self, fill: dict[str, Any]) -> list[TradeRecord]:
        """Handle an atomic flip fill (``"Long > Short"`` or ``"Short > Long"``).

        A flip fill closes the existing position and opens a new one in the
        opposite direction within a single exchange fill.  The ``startPosition``
        field tells us how large the old position was so we can split the fill
        size into the close portion and the new-open portion.
        """
        trades: list[TradeRecord] = []
        px = float(fill["px"])
        total_sz = float(fill["sz"])
        dir_field = fill["dir"]

        # Determine old and new sides from the dir field.
        # "Long > Short" means: close LONG, open SHORT
        # "Short > Long" means: close SHORT, open LONG
        if dir_field == "Long > Short":
            close_dir = "Close Long"
            open_dir = "Open Short"
        else:  # "Short > Long"
            close_dir = "Close Short"
            open_dir = "Open Long"

        # Use startPosition to determine close size.
        # startPosition is the signed position before this fill.
        start_pos = abs(float(fill.get("startPosition", "0")))
        close_sz = start_pos if start_pos > 0 else total_sz
        open_sz = total_sz - close_sz

        # 1. Close the old position (if we have one tracked)
        if self.open_position is not None:
            # Use the smaller of tracked remaining vs reported startPosition
            close_sz = min(close_sz, self.open_position["remaining_size"])
            close_fill = {**fill, "dir": close_dir, "sz": str(close_sz)}
            trade = self._handle_close(close_fill, is_reversal=True)
            if trade is not None:
                trades.append(trade)
            # Recalculate open size after actual close
            open_sz = total_sz - close_sz

        # 2. Open the new position
        if open_sz > 1e-12:
            open_fill = {**fill, "dir": open_dir, "sz": str(open_sz)}
            self._handle_open(open_fill)

        return trades

    @staticmethod
    def _is_reversal(close_fill: dict[str, Any], next_fill: dict[str, Any] | None) -> bool:
        """Check if the close fill is followed by an immediate opposite-direction open."""
        if next_fill is None:
            return False
        if not next_fill["dir"].startswith("Open"):
            return False

        # Check opposite direction
        close_is_long = "Long" in close_fill["dir"]
        next_is_long = "Long" in next_fill["dir"]
        if close_is_long == next_is_long:
            return False  # same direction, not a reversal

        # Check time gap
        gap_ms = next_fill["time"] - close_fill["time"]
        return gap_ms <= REVERSAL_WINDOW_MS
