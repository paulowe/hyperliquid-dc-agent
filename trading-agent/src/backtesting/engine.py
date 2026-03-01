"""Core backtest engine: feeds candles through strategy and records trades."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from interfaces.strategy import MarketData, SignalType, TradingSignal
from strategies.dc_overshoot.dc_overshoot_strategy import DCOvershootStrategy
from strategies.dc_overshoot.multi_scale_strategy import MultiScaleDCStrategy


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    symbol: str = "SOL"
    threshold: float = 0.004
    position_size_usd: float = 100.0
    initial_stop_loss_pct: float = 0.003
    initial_take_profit_pct: float = 0.10
    trail_pct: float = 0.5
    min_profit_to_trail_pct: float = 0.001
    cooldown_seconds: float = 10.0
    leverage: int = 10
    taker_fee_pct: float = 0.00035  # 0.035% per side


@dataclass
class TradeRecord:
    """A single completed trade with P&L accounting."""

    side: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: float
    exit_time: float
    pnl_pct: float
    pnl_usd: float
    entry_fee: float
    exit_fee: float
    total_fees: float
    net_pnl_usd: float
    reason: str
    # Intra-trade excursion from trailing RM water marks
    max_favorable_excursion_pct: float = 0.0
    max_adverse_excursion_pct: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest result: trades + metadata."""

    config: BacktestConfig
    trades: list[TradeRecord]
    total_signals: int
    candle_count: int
    first_price: float
    last_price: float
    price_change_pct: float


class BacktestEngine:
    """Feeds candles through DCOvershootStrategy and records trades.

    The engine creates a fresh strategy instance, feeds each candle's close
    price as a tick, handles BUY/SELL/CLOSE/reversal signals, and records
    completed trades with fee accounting.
    """

    def __init__(self, config: BacktestConfig):
        self._config = config

    def run(self, candles: list[dict], quiet: bool = True) -> BacktestResult:
        """Run the strategy on historical candles.

        Args:
            candles: List of candle dicts from CandleFetcher (must have "t" and "c" keys).
            quiet: Suppress strategy logging output.

        Returns:
            BacktestResult with all completed trades.
        """
        if not candles:
            return BacktestResult(
                config=self._config,
                trades=[],
                total_signals=0,
                candle_count=0,
                first_price=0.0,
                last_price=0.0,
                price_change_pct=0.0,
            )

        if quiet:
            logging.disable(logging.CRITICAL)

        try:
            result = self._run_strategy(candles)
        finally:
            if quiet:
                logging.disable(logging.NOTSET)

        return result

    def _run_strategy(self, candles: list[dict]) -> BacktestResult:
        """Internal: run strategy loop on candles."""
        strategy = DCOvershootStrategy(self._build_strategy_config())
        strategy.start()

        trades: list[TradeRecord] = []
        current_position: dict[str, Any] | None = None
        total_signals = 0

        for candle in candles:
            price = float(candle["c"])
            ts = float(candle["t"]) / 1000.0

            md = MarketData(asset=self._config.symbol, price=price, volume_24h=0.0, timestamp=ts)
            signals = strategy.generate_signals(md, [], 100_000.0)

            for signal in signals:
                total_signals += 1

                if signal.signal_type == SignalType.CLOSE:
                    if current_position:
                        trade = self._close_position(current_position, signal, price, ts)
                        trades.append(trade)
                        current_position = None

                elif signal.signal_type in (SignalType.BUY, SignalType.SELL):
                    is_reversal = signal.metadata.get("reversal", False)

                    # Close old position on reversal
                    if is_reversal and current_position:
                        trade = self._close_position(
                            current_position, signal, price, ts, reason="reversal_close"
                        )
                        trades.append(trade)
                        current_position = None

                    # Open new position
                    current_position = self._open_position(signal, price, ts)
                    strategy.on_trade_executed(signal, price, signal.size)

        strategy.stop()

        first_price = float(candles[0]["c"])
        last_price = float(candles[-1]["c"])

        return BacktestResult(
            config=self._config,
            trades=trades,
            total_signals=total_signals,
            candle_count=len(candles),
            first_price=first_price,
            last_price=last_price,
            price_change_pct=(last_price - first_price) / first_price * 100,
        )

    def _build_strategy_config(self) -> dict:
        """Convert BacktestConfig to the dict format DCOvershootStrategy expects."""
        return {
            "symbol": self._config.symbol,
            "dc_thresholds": [[self._config.threshold, self._config.threshold]],
            "position_size_usd": self._config.position_size_usd,
            "max_position_size_usd": self._config.position_size_usd * 4,
            "initial_stop_loss_pct": self._config.initial_stop_loss_pct,
            "initial_take_profit_pct": self._config.initial_take_profit_pct,
            "trail_pct": self._config.trail_pct,
            "min_profit_to_trail_pct": self._config.min_profit_to_trail_pct,
            "cooldown_seconds": self._config.cooldown_seconds,
            "max_open_positions": 1,
            "log_events": False,
        }

    def _open_position(
        self, signal: TradingSignal, price: float, ts: float
    ) -> dict[str, Any]:
        """Create position tracking dict from an entry signal."""
        is_reversal = signal.metadata.get("reversal", False)
        new_size = (
            signal.metadata.get("new_position_size", signal.size)
            if is_reversal
            else signal.size
        )
        new_side = "LONG" if signal.signal_type == SignalType.BUY else "SHORT"
        entry_notional = new_size * price
        entry_fee = entry_notional * self._config.taker_fee_pct

        return {
            "side": new_side,
            "entry_price": price,
            "size": new_size,
            "entry_time": ts,
            "entry_fee": entry_fee,
        }

    def _close_position(
        self,
        position: dict[str, Any],
        signal: TradingSignal,
        price: float,
        ts: float,
        reason: str | None = None,
    ) -> TradeRecord:
        """Close a position and create a TradeRecord with fee accounting."""
        side = position["side"]
        entry_price = position["entry_price"]
        size = position["size"]
        exit_notional = size * price

        if side == "LONG":
            pnl_pct = (price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - price) / entry_price

        pnl_usd = pnl_pct * (size * entry_price)
        exit_fee = exit_notional * self._config.taker_fee_pct
        entry_fee = position["entry_fee"]
        total_fees = entry_fee + exit_fee

        # Extract MFE/MAE from trailing RM water marks in signal metadata
        mfe, mae = _compute_excursions(side, entry_price, signal.metadata)

        return TradeRecord(
            side=side,
            entry_price=entry_price,
            exit_price=price,
            size=size,
            entry_time=position["entry_time"],
            exit_time=ts,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            total_fees=total_fees,
            net_pnl_usd=pnl_usd - total_fees,
            reason=reason or signal.reason,
            max_favorable_excursion_pct=mfe,
            max_adverse_excursion_pct=mae,
        )


def _compute_excursions(
    side: str, entry_price: float, metadata: dict[str, Any]
) -> tuple[float, float]:
    """Compute MFE/MAE from close signal water marks.

    Returns (mfe_pct, mae_pct). Defaults to (0.0, 0.0) if water marks
    are not present (e.g. reversal close signals from strategy).
    """
    hwm = metadata.get("high_water_mark")
    lwm = metadata.get("low_water_mark")
    if hwm is None or lwm is None:
        return 0.0, 0.0

    if side == "LONG":
        mfe = (hwm - entry_price) / entry_price
        mae = (entry_price - lwm) / entry_price
    else:  # SHORT
        mfe = (entry_price - lwm) / entry_price
        mae = (hwm - entry_price) / entry_price

    return max(mfe, 0.0), max(mae, 0.0)


# ===========================================================================
# Multi-Scale Backtest Engine
# ===========================================================================

@dataclass
class MultiScaleBacktestConfig:
    """Configuration for a multi-scale backtest run."""

    symbol: str = "SOL"

    # Sensor thresholds (intelligence layer)
    sensor_thresholds: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.002, 0.002), (0.004, 0.004), (0.008, 0.008)]
    )

    # Trade threshold (entry trigger)
    trade_threshold: float = 0.015

    # Momentum filter
    momentum_alpha: float = 0.3
    min_momentum_score: float = 0.3

    # Position sizing and risk
    position_size_usd: float = 100.0
    initial_stop_loss_pct: float = 0.015
    initial_take_profit_pct: float = 0.005
    trail_pct: float = 0.3
    min_profit_to_trail_pct: float = 0.002
    cooldown_seconds: float = 10.0
    leverage: int = 10
    taker_fee_pct: float = 0.00035


@dataclass
class MultiScaleBacktestResult:
    """Complete multi-scale backtest result."""

    config: MultiScaleBacktestConfig
    trades: list[TradeRecord]
    total_signals: int
    filtered_signals: int
    sensor_events: int
    trade_events: int
    candle_count: int
    first_price: float
    last_price: float
    price_change_pct: float


class MultiScaleBacktestEngine:
    """Feeds candles through MultiScaleDCStrategy and records trades.

    Follows the same pattern as BacktestEngine but uses MultiScaleDCStrategy
    with momentum filtering.
    """

    def __init__(self, config: MultiScaleBacktestConfig):
        self._config = config

    def run(self, candles: list[dict], quiet: bool = True) -> MultiScaleBacktestResult:
        """Run multi-scale strategy on historical candles."""
        if not candles:
            return MultiScaleBacktestResult(
                config=self._config,
                trades=[],
                total_signals=0,
                filtered_signals=0,
                sensor_events=0,
                trade_events=0,
                candle_count=0,
                first_price=0.0,
                last_price=0.0,
                price_change_pct=0.0,
            )

        if quiet:
            logging.disable(logging.CRITICAL)

        try:
            result = self._run_strategy(candles)
        finally:
            if quiet:
                logging.disable(logging.NOTSET)

        return result

    def _run_strategy(self, candles: list[dict]) -> MultiScaleBacktestResult:
        """Internal: build strategy config, run candle loop, record trades."""
        strategy = MultiScaleDCStrategy(self._build_strategy_config())
        strategy.start()

        trades: list[TradeRecord] = []
        current_position: dict[str, Any] | None = None
        total_signals = 0

        for candle in candles:
            price = float(candle["c"])
            ts = float(candle["t"]) / 1000.0

            md = MarketData(asset=self._config.symbol, price=price, volume_24h=0.0, timestamp=ts)
            signals = strategy.generate_signals(md, [], 100_000.0)

            for signal in signals:
                total_signals += 1

                if signal.signal_type == SignalType.CLOSE:
                    if current_position:
                        trade = self._close_position(current_position, signal, price, ts)
                        trades.append(trade)
                        current_position = None

                elif signal.signal_type in (SignalType.BUY, SignalType.SELL):
                    is_reversal = signal.metadata.get("reversal", False)

                    # Close old position on reversal
                    if is_reversal and current_position:
                        trade = self._close_position(
                            current_position, signal, price, ts, reason="reversal_close"
                        )
                        trades.append(trade)
                        current_position = None

                    # Open new position
                    current_position = self._open_position(signal, price, ts)
                    strategy.on_trade_executed(signal, price, signal.size)

        strategy.stop()

        # Extract multi-scale-specific counters from strategy status
        status = strategy.get_status()
        first_price = float(candles[0]["c"])
        last_price = float(candles[-1]["c"])

        return MultiScaleBacktestResult(
            config=self._config,
            trades=trades,
            total_signals=total_signals,
            filtered_signals=status.get("filtered_count", 0),
            sensor_events=status.get("sensor_event_count", 0),
            trade_events=status.get("trade_event_count", 0),
            candle_count=len(candles),
            first_price=first_price,
            last_price=last_price,
            price_change_pct=(last_price - first_price) / first_price * 100,
        )

    def _build_strategy_config(self) -> dict:
        """Convert MultiScaleBacktestConfig to the dict MultiScaleConfig expects."""
        return {
            "symbol": self._config.symbol,
            "sensor_thresholds": list(self._config.sensor_thresholds),
            "trade_threshold": [self._config.trade_threshold, self._config.trade_threshold],
            "momentum_alpha": self._config.momentum_alpha,
            "min_momentum_score": self._config.min_momentum_score,
            "position_size_usd": self._config.position_size_usd,
            "max_position_size_usd": self._config.position_size_usd * 4,
            "initial_stop_loss_pct": self._config.initial_stop_loss_pct,
            "initial_take_profit_pct": self._config.initial_take_profit_pct,
            "trail_pct": self._config.trail_pct,
            "min_profit_to_trail_pct": self._config.min_profit_to_trail_pct,
            "cooldown_seconds": self._config.cooldown_seconds,
            "max_open_positions": 1,
            "log_events": False,
            "log_sensor_events": False,
        }

    def _open_position(
        self, signal: TradingSignal, price: float, ts: float
    ) -> dict[str, Any]:
        """Create position tracking dict from an entry signal."""
        is_reversal = signal.metadata.get("reversal", False)
        new_size = (
            signal.metadata.get("new_position_size", signal.size)
            if is_reversal
            else signal.size
        )
        new_side = "LONG" if signal.signal_type == SignalType.BUY else "SHORT"
        entry_notional = new_size * price
        entry_fee = entry_notional * self._config.taker_fee_pct

        return {
            "side": new_side,
            "entry_price": price,
            "size": new_size,
            "entry_time": ts,
            "entry_fee": entry_fee,
        }

    def _close_position(
        self,
        position: dict[str, Any],
        signal: TradingSignal,
        price: float,
        ts: float,
        reason: str | None = None,
    ) -> TradeRecord:
        """Close a position and create a TradeRecord with fee accounting."""
        side = position["side"]
        entry_price = position["entry_price"]
        size = position["size"]
        exit_notional = size * price

        if side == "LONG":
            pnl_pct = (price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - price) / entry_price

        pnl_usd = pnl_pct * (size * entry_price)
        exit_fee = exit_notional * self._config.taker_fee_pct
        entry_fee = position["entry_fee"]
        total_fees = entry_fee + exit_fee

        # Extract MFE/MAE from trailing RM water marks in signal metadata
        mfe, mae = _compute_excursions(side, entry_price, signal.metadata)

        return TradeRecord(
            side=side,
            entry_price=entry_price,
            exit_price=price,
            size=size,
            entry_time=position["entry_time"],
            exit_time=ts,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            total_fees=total_fees,
            net_pnl_usd=pnl_usd - total_fees,
            reason=reason or signal.reason,
            max_favorable_excursion_pct=mfe,
            max_adverse_excursion_pct=mae,
        )
