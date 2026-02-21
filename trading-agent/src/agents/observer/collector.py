"""ResultCollector: parse JSON reports and compute simulated PnL."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Taker fee on Hyperliquid (0.035% per side)
TAKER_FEE_PCT = 0.00035


@dataclass
class SessionReport:
    """Parsed result from a single observe-only session."""

    session_id: str
    symbol: str
    threshold: float
    sl_pct: float
    tp_pct: float
    trail_pct: float
    position_size_usd: float
    leverage: int
    duration_seconds: float
    tick_count: int
    signal_count: int
    dc_event_count: int
    signals: list[dict]

    @property
    def signal_frequency_per_min(self) -> float:
        """Signals per minute of observation."""
        if self.duration_seconds <= 0:
            return 0.0
        return self.signal_count / (self.duration_seconds / 60.0)

    @property
    def buy_signals(self) -> int:
        return sum(1 for s in self.signals if s.get("type") == "BUY")

    @property
    def sell_signals(self) -> int:
        return sum(1 for s in self.signals if s.get("type") == "SELL")

    @property
    def close_signals(self) -> int:
        return sum(1 for s in self.signals if s.get("type") == "CLOSE")


@dataclass
class PnLSummary:
    """Simulated PnL from observed signals."""

    total_trades: int
    gross_pnl_usd: float
    total_fees_usd: float
    net_pnl_usd: float
    wins: int
    losses: int

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades


class ResultCollector:
    """Collects and parses JSON reports from completed sessions."""

    def __init__(self, report_dir: Path):
        self._report_dir = report_dir

    def collect(self, session_ids: list[str]) -> list[SessionReport]:
        """Read JSON reports for completed sessions."""
        reports = []
        for sid in session_ids:
            report_path = self._report_dir / f"report_{sid}.json"
            if not report_path.exists():
                logger.warning("Report not found for session %s: %s", sid, report_path)
                continue
            try:
                data = json.loads(report_path.read_text())
                report = SessionReport(
                    session_id=sid,
                    symbol=data["symbol"],
                    threshold=data["threshold"],
                    sl_pct=data["sl_pct"],
                    tp_pct=data["tp_pct"],
                    trail_pct=data.get("trail_pct", 0.5),
                    position_size_usd=data.get("position_size_usd", 100.0),
                    leverage=data.get("leverage", 10),
                    duration_seconds=data.get("duration_seconds", 0.0),
                    tick_count=data.get("tick_count", 0),
                    signal_count=data.get("signal_count", 0),
                    dc_event_count=data.get("dc_event_count", 0),
                    signals=data.get("signals", []),
                )
                reports.append(report)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to parse report %s: %s", sid, e)
                continue
        return reports

    def compute_simulated_pnl(self, report: SessionReport) -> PnLSummary:
        """Estimate hypothetical PnL from observe-only signals.

        Replays entry/exit signal pairs to simulate what would have happened
        if trades were executed. Uses the signal prices directly.
        """
        trades: list[dict] = []
        current_position: dict | None = None

        for sig in report.signals:
            sig_type = sig.get("type", "")

            if sig_type in ("BUY", "SELL") and current_position is None:
                # Entry
                current_position = {
                    "side": "LONG" if sig_type == "BUY" else "SHORT",
                    "entry_price": sig["price"],
                    "size": sig.get("size", report.position_size_usd / sig["price"]),
                }
            elif sig_type == "CLOSE" and current_position is not None:
                # Exit
                entry_price = current_position["entry_price"]
                exit_price = sig["price"]
                size = current_position["size"]
                side = current_position["side"]

                if side == "LONG":
                    pnl = (exit_price - entry_price) * size
                else:
                    pnl = (entry_price - exit_price) * size

                entry_fee = entry_price * size * TAKER_FEE_PCT
                exit_fee = exit_price * size * TAKER_FEE_PCT
                total_fees = entry_fee + exit_fee

                trades.append({
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pnl": pnl,
                    "fees": total_fees,
                    "net_pnl": pnl - total_fees,
                })
                current_position = None
            elif sig_type in ("BUY", "SELL") and current_position is not None:
                # Reversal: close current + open new
                entry_price = current_position["entry_price"]
                exit_price = sig["price"]
                size = current_position["size"]
                side = current_position["side"]

                if side == "LONG":
                    pnl = (exit_price - entry_price) * size
                else:
                    pnl = (entry_price - exit_price) * size

                entry_fee = entry_price * size * TAKER_FEE_PCT
                exit_fee = exit_price * size * TAKER_FEE_PCT

                trades.append({
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pnl": pnl,
                    "fees": entry_fee + exit_fee,
                    "net_pnl": pnl - (entry_fee + exit_fee),
                })

                # Open new position
                current_position = {
                    "side": "LONG" if sig_type == "BUY" else "SHORT",
                    "entry_price": sig["price"],
                    "size": sig.get("size", report.position_size_usd / sig["price"]),
                }

        if not trades:
            return PnLSummary(
                total_trades=0, gross_pnl_usd=0.0, total_fees_usd=0.0,
                net_pnl_usd=0.0, wins=0, losses=0,
            )

        gross_pnl = sum(t["gross_pnl"] for t in trades)
        total_fees = sum(t["fees"] for t in trades)
        wins = sum(1 for t in trades if t["net_pnl"] > 0)
        losses = sum(1 for t in trades if t["net_pnl"] <= 0)

        return PnLSummary(
            total_trades=len(trades),
            gross_pnl_usd=gross_pnl,
            total_fees_usd=total_fees,
            net_pnl_usd=gross_pnl - total_fees,
            wins=wins,
            losses=losses,
        )
