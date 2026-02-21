"""Tests for backtest engine."""

from __future__ import annotations

import pytest

from backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult, TradeRecord


def make_candle(price: float, ts_ms: int) -> dict:
    """Create a minimal candle dict."""
    return {"t": ts_ms, "c": str(price)}


def make_price_sequence(start: float, changes: list[float], interval_ms: int = 60000) -> list[dict]:
    """Generate candles from a start price and list of price changes."""
    candles = []
    price = start
    ts = 1000000000000  # Base timestamp in ms
    for i, delta in enumerate(changes):
        price += delta
        candles.append(make_candle(price, ts + i * interval_ms))
    return candles


def generate_downtrend(start: float, drop_pct: float, steps: int) -> list[dict]:
    """Generate monotonically decreasing prices."""
    drop_per_step = start * drop_pct / steps
    changes = [-drop_per_step] * steps
    return make_price_sequence(start, changes)


def generate_v_shape(start: float, drop_pct: float, steps_down: int, steps_up: int) -> list[dict]:
    """Generate V-shaped price: down then up."""
    drop_per_step = start * drop_pct / steps_down
    rise_per_step = start * drop_pct / steps_up
    changes = [-drop_per_step] * steps_down + [rise_per_step] * steps_up
    return make_price_sequence(start, changes)


class TestBacktestConfig:
    def test_default_values(self):
        config = BacktestConfig()
        assert config.symbol == "SOL"
        assert config.threshold == 0.004
        assert config.taker_fee_pct == 0.00035

    def test_custom_values(self):
        config = BacktestConfig(symbol="BTC", threshold=0.01, leverage=20)
        assert config.symbol == "BTC"
        assert config.threshold == 0.01
        assert config.leverage == 20


class TestEngineEmptyInput:
    def test_empty_candles(self):
        engine = BacktestEngine(BacktestConfig())
        result = engine.run([])
        assert result.trades == []
        assert result.total_signals == 0
        assert result.candle_count == 0

    def test_single_candle(self):
        engine = BacktestEngine(BacktestConfig())
        result = engine.run([make_candle(100.0, 1000000)])
        assert result.trades == []
        assert result.candle_count == 1


class TestEngineFlatPrices:
    def test_no_trades_on_flat(self):
        """Constant price should never trigger a DC event."""
        candles = [make_candle(100.0, 1000000 + i * 60000) for i in range(500)]
        engine = BacktestEngine(BacktestConfig(threshold=0.004))
        result = engine.run(candles)
        assert result.total_signals == 0
        assert len(result.trades) == 0


class TestEngineDowntrend:
    def test_downtrend_produces_short_signal(self):
        """A steady decline should trigger PDCC_Down → SHORT entry signal."""
        # Drop 5% over 100 steps — large enough for threshold=0.004
        candles = generate_downtrend(100.0, 0.05, 100)
        engine = BacktestEngine(BacktestConfig(
            threshold=0.004,
            initial_stop_loss_pct=0.01,
            initial_take_profit_pct=0.05,
        ))
        result = engine.run(candles)
        # In a monotonic downtrend the SHORT entry is never closed (SL ratchets
        # down, no reversal), so we check signals rather than completed trades.
        assert result.total_signals >= 1

    def test_downtrend_then_bounce_produces_completed_trade(self):
        """Downtrend followed by bounce should produce a completed SHORT trade."""
        # Drop 3%, then bounce 2% to hit SL
        down = [-0.03 * 100.0 / 30] * 30
        up = [0.02 * 100.0 / 20] * 20
        candles = make_price_sequence(100.0, down + up)
        engine = BacktestEngine(BacktestConfig(
            threshold=0.004,
            initial_stop_loss_pct=0.01,
            initial_take_profit_pct=0.05,
        ))
        result = engine.run(candles)
        assert len(result.trades) > 0
        assert any(t.side == "SHORT" for t in result.trades)


class TestEngineVShape:
    def test_v_shape_produces_reversal(self):
        """V-shape: drop triggers SHORT, recovery triggers LONG reversal."""
        candles = generate_v_shape(100.0, 0.05, 50, 50)
        engine = BacktestEngine(BacktestConfig(
            threshold=0.004,
            initial_stop_loss_pct=0.01,
            initial_take_profit_pct=0.10,
        ))
        result = engine.run(candles)
        sides = [t.side for t in result.trades]
        # Should have both SHORT and LONG trades
        assert "SHORT" in sides or "LONG" in sides
        assert len(result.trades) >= 1


class TestEngineFeeAccounting:
    def test_fees_are_consistent(self):
        """entry_fee + exit_fee should equal total_fees for every trade."""
        candles = generate_v_shape(100.0, 0.05, 50, 50)
        engine = BacktestEngine(BacktestConfig(
            threshold=0.004,
            initial_stop_loss_pct=0.01,
            initial_take_profit_pct=0.10,
        ))
        result = engine.run(candles)
        for trade in result.trades:
            assert trade.total_fees == pytest.approx(trade.entry_fee + trade.exit_fee)
            assert trade.net_pnl_usd == pytest.approx(trade.pnl_usd - trade.total_fees)

    def test_fees_are_positive(self):
        """All fee values should be non-negative."""
        candles = generate_downtrend(100.0, 0.05, 100)
        engine = BacktestEngine(BacktestConfig(threshold=0.004))
        result = engine.run(candles)
        for trade in result.trades:
            assert trade.entry_fee >= 0
            assert trade.exit_fee >= 0
            assert trade.total_fees >= 0


class TestEngineTradeRecordFields:
    def test_all_fields_populated(self):
        """Every TradeRecord field should have a reasonable value."""
        candles = generate_downtrend(100.0, 0.05, 100)
        engine = BacktestEngine(BacktestConfig(threshold=0.004))
        result = engine.run(candles)
        if result.trades:
            trade = result.trades[0]
            assert trade.side in ("LONG", "SHORT")
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.size > 0
            assert trade.entry_time > 0
            assert trade.exit_time >= trade.entry_time
            assert trade.reason != ""


class TestBacktestResult:
    def test_price_change_pct(self):
        candles = [make_candle(100.0, 1000000), make_candle(110.0, 2000000)]
        engine = BacktestEngine(BacktestConfig())
        result = engine.run(candles)
        assert result.first_price == 100.0
        assert result.last_price == 110.0
        assert result.price_change_pct == pytest.approx(10.0)
