"""Parameter sweep: grid search over DC Overshoot strategy parameters."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.metrics import compute_metrics


# Default parameter grid (6 * 6 * 6 * 3 * 2 = 1,296 combos)
DEFAULT_THRESHOLDS = [0.002, 0.004, 0.006, 0.008, 0.01, 0.015]
DEFAULT_SL_PCTS = [0.003, 0.005, 0.008, 0.01, 0.015, 0.02]
DEFAULT_TP_PCTS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
DEFAULT_TRAIL_PCTS = [0.3, 0.5, 0.7]
DEFAULT_MIN_PROFIT_TO_TRAIL_PCTS = [0.001, 0.002]


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep."""

    symbol: str = "SOL"
    position_size_usd: float = 100.0
    leverage: int = 10
    taker_fee_pct: float = 0.00035
    cooldown_seconds: float = 10.0
    min_trades: int = 5

    # Parameter grid
    thresholds: list[float] = field(default_factory=lambda: list(DEFAULT_THRESHOLDS))
    sl_pcts: list[float] = field(default_factory=lambda: list(DEFAULT_SL_PCTS))
    tp_pcts: list[float] = field(default_factory=lambda: list(DEFAULT_TP_PCTS))
    trail_pcts: list[float] = field(default_factory=lambda: list(DEFAULT_TRAIL_PCTS))
    min_profit_to_trail_pcts: list[float] = field(
        default_factory=lambda: list(DEFAULT_MIN_PROFIT_TO_TRAIL_PCTS)
    )

    def combinations(self) -> list[BacktestConfig]:
        """Generate all BacktestConfig combinations from the grid."""
        combos = []
        for thresh, sl, tp, trail, min_trail in itertools.product(
            self.thresholds,
            self.sl_pcts,
            self.tp_pcts,
            self.trail_pcts,
            self.min_profit_to_trail_pcts,
        ):
            combos.append(
                BacktestConfig(
                    symbol=self.symbol,
                    threshold=thresh,
                    position_size_usd=self.position_size_usd,
                    initial_stop_loss_pct=sl,
                    initial_take_profit_pct=tp,
                    trail_pct=trail,
                    min_profit_to_trail_pct=min_trail,
                    cooldown_seconds=self.cooldown_seconds,
                    leverage=self.leverage,
                    taker_fee_pct=self.taker_fee_pct,
                )
            )
        return combos


@dataclass
class SweepResult:
    """Result of a single backtest config within a sweep."""

    config: BacktestConfig
    total_trades: int
    wins: int
    losses: int
    win_rate_net: float
    gross_pnl_usd: float
    total_fees_usd: float
    net_pnl_usd: float
    profit_factor: float
    max_drawdown_usd: float
    net_pnl_per_day: float
    trades_eaten_by_fees: int


class ParameterSweep:
    """Runs a grid search over DC Overshoot strategy parameters."""

    def __init__(self, sweep_config: SweepConfig):
        self._config = sweep_config

    def run(
        self,
        candles: list[dict],
        days: float = 7.0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SweepResult]:
        """Run the sweep on candle data.

        Args:
            candles: Historical candle data.
            days: Number of days of data (for per-day calculations).
            progress_callback: Called with (current, total) after each config.

        Returns:
            List of SweepResult sorted by net_pnl_usd descending.
            Only configs with >= min_trades are included.
        """
        combos = self._config.combinations()
        results: list[SweepResult] = []

        for i, config in enumerate(combos):
            engine = BacktestEngine(config)
            bt_result = engine.run(candles, quiet=True)

            if bt_result.trades and len(bt_result.trades) >= self._config.min_trades:
                metrics = compute_metrics(bt_result.trades, bt_result.total_signals, days)
                results.append(
                    SweepResult(
                        config=config,
                        total_trades=metrics.total_trades,
                        wins=metrics.wins,
                        losses=metrics.losses,
                        win_rate_net=metrics.win_rate_net,
                        gross_pnl_usd=metrics.gross_pnl_usd,
                        total_fees_usd=metrics.total_fees_usd,
                        net_pnl_usd=metrics.net_pnl_usd,
                        profit_factor=metrics.profit_factor,
                        max_drawdown_usd=metrics.max_drawdown_usd,
                        net_pnl_per_day=metrics.net_pnl_per_day,
                        trades_eaten_by_fees=metrics.trades_eaten_by_fees,
                    )
                )

            if progress_callback:
                progress_callback(i + 1, len(combos))

        # Sort by net P&L descending
        results.sort(key=lambda r: r.net_pnl_usd, reverse=True)
        return results

    @staticmethod
    def analyze_patterns(results: list[SweepResult]) -> dict[str, Any]:
        """Analyze parameter patterns in profitable vs unprofitable configs.

        Returns:
            Dict with 'profitable', 'unprofitable' sub-dicts of average params,
            plus counts.
        """
        profitable = [r for r in results if r.net_pnl_usd > 0]
        unprofitable = [r for r in results if r.net_pnl_usd <= 0]

        def avg_params(group: list[SweepResult]) -> dict[str, float]:
            if not group:
                return {}
            n = len(group)
            return {
                "avg_threshold": sum(r.config.threshold for r in group) / n,
                "avg_sl_pct": sum(r.config.initial_stop_loss_pct for r in group) / n,
                "avg_tp_pct": sum(r.config.initial_take_profit_pct for r in group) / n,
                "avg_trail_pct": sum(r.config.trail_pct for r in group) / n,
                "avg_min_profit_to_trail": sum(
                    r.config.min_profit_to_trail_pct for r in group
                ) / n,
            }

        return {
            "profitable": avg_params(profitable),
            "unprofitable": avg_params(unprofitable),
            "profitable_count": len(profitable),
            "unprofitable_count": len(unprofitable),
        }

    @staticmethod
    def format_results(results: list[SweepResult], top_n: int = 20) -> str:
        """Format sweep results as a human-readable table.

        Args:
            results: Sorted list of SweepResult.
            top_n: Number of top results to show.

        Returns:
            Formatted string with results table.
        """
        if not results:
            return "No results to display."

        lines = []
        header = (
            f"{'Rank':>4} | {'Thresh':>6} | {'SL%':>5} | {'TP%':>5} | {'Trail':>5} | "
            f"{'MinTr':>5} | {'Trades':>6} | {'WinR%':>5} | {'Gross$':>8} | "
            f"{'Fees$':>7} | {'Net$':>8} | {'PF':>6} | {'MaxDD':>7} | "
            f"{'FeeEat':>6} | {'$/day':>7}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for i, r in enumerate(results[:top_n], 1):
            pf_str = f"{r.profit_factor:6.2f}" if r.profit_factor != float("inf") else "   inf"
            lines.append(
                f"{i:4d} | {r.config.threshold:6.3f} | "
                f"{r.config.initial_stop_loss_pct * 100:5.2f} | "
                f"{r.config.initial_take_profit_pct * 100:5.1f} | "
                f"{r.config.trail_pct:5.1f} | "
                f"{r.config.min_profit_to_trail_pct:5.3f} | "
                f"{r.total_trades:6d} | {r.win_rate_net:5.1f} | "
                f"{r.gross_pnl_usd:+8.2f} | {r.total_fees_usd:7.2f} | "
                f"{r.net_pnl_usd:+8.2f} | {pf_str} | "
                f"{r.max_drawdown_usd:7.2f} | "
                f"{r.trades_eaten_by_fees:6d} | {r.net_pnl_per_day:+7.2f}"
            )

        return "\n".join(lines)

    @staticmethod
    def results_to_json(results: list[SweepResult]) -> list[dict[str, Any]]:
        """Convert sweep results to JSON-serializable dicts.

        Handles float('inf') by converting to string "Infinity".
        """
        output = []
        for r in results:
            pf = r.profit_factor
            # JSON doesn't support Infinity
            if pf == float("inf"):
                pf = "Infinity"
            output.append(
                {
                    "threshold": r.config.threshold,
                    "sl_pct": r.config.initial_stop_loss_pct,
                    "tp_pct": r.config.initial_take_profit_pct,
                    "trail_pct": r.config.trail_pct,
                    "min_profit_to_trail": r.config.min_profit_to_trail_pct,
                    "total_trades": r.total_trades,
                    "wins": r.wins,
                    "losses": r.losses,
                    "win_rate_net": round(r.win_rate_net, 2),
                    "gross_pnl_usd": round(r.gross_pnl_usd, 4),
                    "total_fees_usd": round(r.total_fees_usd, 4),
                    "net_pnl_usd": round(r.net_pnl_usd, 4),
                    "profit_factor": pf,
                    "max_drawdown_usd": round(r.max_drawdown_usd, 4),
                    "net_pnl_per_day": round(r.net_pnl_per_day, 4),
                    "trades_eaten_by_fees": r.trades_eaten_by_fees,
                }
            )
        return output
