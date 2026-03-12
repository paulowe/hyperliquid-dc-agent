"""Tests for BasisTradeStrategy."""

import time
import pytest
from strategies.basis_trade.basis_strategy import (
    BasisState,
    BasisPosition,
    BasisTradeStrategy,
)
from strategies.basis_trade.config import BasisTradeConfig


class TestBasisPosition:
    """Test position tracking."""

    def _make_position(self, **kwargs):
        defaults = dict(
            spot_entry_price=36.0,
            spot_size=0.07,
            spot_notional=2.52,
            perp_entry_price=36.0,
            perp_size=0.07,
            perp_notional=2.52,
            entry_time=1000.0,
        )
        defaults.update(kwargs)
        return BasisPosition(**defaults)

    def test_notional(self):
        pos = self._make_position()
        assert pos.notional() == pytest.approx(2.52)

    def test_hold_hours(self):
        pos = self._make_position(entry_time=1000.0)
        assert pos.hold_hours(4600.0) == pytest.approx(1.0)
        assert pos.hold_hours(1000.0) == 0.0

    def test_spot_pnl_profit(self):
        pos = self._make_position(spot_entry_price=36.0, spot_size=0.07)
        # Price goes up → spot profits
        assert pos.spot_pnl(37.0) == pytest.approx(0.07)

    def test_spot_pnl_loss(self):
        pos = self._make_position(spot_entry_price=36.0, spot_size=0.07)
        # Price goes down → spot loses
        assert pos.spot_pnl(35.0) == pytest.approx(-0.07)

    def test_perp_pnl_profit(self):
        pos = self._make_position(perp_entry_price=36.0, perp_size=0.07)
        # Price goes down → short perp profits
        assert pos.perp_pnl(35.0) == pytest.approx(0.07)

    def test_perp_pnl_loss(self):
        pos = self._make_position(perp_entry_price=36.0, perp_size=0.07)
        # Price goes up → short perp loses
        assert pos.perp_pnl(37.0) == pytest.approx(-0.07)

    def test_net_price_pnl_delta_neutral(self):
        pos = self._make_position(
            spot_entry_price=36.0, spot_size=0.07,
            perp_entry_price=36.0, perp_size=0.07,
        )
        # Price moves → net P&L should be ~0
        assert pos.net_price_pnl(40.0) == pytest.approx(0.0)
        assert pos.net_price_pnl(30.0) == pytest.approx(0.0)
        assert pos.net_price_pnl(36.0) == pytest.approx(0.0)

    def test_net_price_pnl_with_basis(self):
        # Entry prices differ (basis spread)
        pos = self._make_position(
            spot_entry_price=36.0, spot_size=0.07,
            perp_entry_price=36.1, perp_size=0.07,
        )
        # At $36.0: spot P&L = 0, perp P&L = (36.1 - 36.0) * 0.07 = +0.007
        assert pos.net_price_pnl(36.0) == pytest.approx(0.007)


class TestBasisTradeStrategy:
    """Test strategy lifecycle and decision logic."""

    def _make_strategy(self, **config_overrides):
        cfg = BasisTradeConfig.from_dict({
            "symbol": "HYPE",
            "position_size_usd": 3.0,
            "min_funding_rate": 0.0001,
            "min_funding_hours": 3,
            "exit_funding_rate": -0.00005,
            "exit_funding_hours": 6,
            **config_overrides,
        })
        return BasisTradeStrategy(cfg)

    # --- State machine ---

    def test_initial_state(self):
        strat = self._make_strategy()
        assert strat.state == BasisState.IDLE
        assert strat.position is None

    def test_start_from_stopped(self):
        strat = self._make_strategy()
        strat.stop()
        assert strat.state == BasisState.STOPPED
        strat.start()
        assert strat.state == BasisState.IDLE

    # --- Entry decision ---

    def test_should_enter_false_initially(self):
        strat = self._make_strategy()
        assert strat.should_enter() is False

    def test_should_enter_after_enough_observations(self):
        strat = self._make_strategy(min_funding_hours=3)

        # Feed 3 hours of good funding
        for i in range(3):
            strat.update_funding(rate=0.0002, mark_price=36.5, timestamp=1000 + i * 3600)

        assert strat.should_enter() is True

    def test_should_enter_false_if_funding_too_low(self):
        strat = self._make_strategy(min_funding_hours=3, min_funding_rate=0.0001)

        # Feed 3 hours but funding below threshold
        for i in range(3):
            strat.update_funding(rate=0.00005, mark_price=36.5, timestamp=1000 + i * 3600)

        assert strat.should_enter() is False

    def test_should_enter_false_if_already_active(self):
        strat = self._make_strategy(min_funding_hours=1)

        strat.update_funding(rate=0.0005, mark_price=36.5, timestamp=1000)
        assert strat.should_enter() is True

        # Open position
        strat.on_entry_filled(
            spot_price=36.5, spot_size=0.07,
            perp_price=36.5, perp_size=0.07,
            timestamp=1000,
        )
        assert strat.should_enter() is False

    def test_should_enter_resets_on_funding_dip(self):
        strat = self._make_strategy(min_funding_hours=3)

        # 2 good, 1 bad, 1 good → only 1 consecutive
        strat.update_funding(rate=0.0002, mark_price=36.5, timestamp=1000)
        strat.update_funding(rate=0.0002, mark_price=36.5, timestamp=4600)
        strat.update_funding(rate=0.00001, mark_price=36.5, timestamp=8200)  # dip
        strat.update_funding(rate=0.0002, mark_price=36.5, timestamp=11800)

        assert strat.should_enter() is False
        assert strat.funding_monitor.consecutive_above == 1

    # --- Exit decision ---

    def test_should_exit_none_when_idle(self):
        strat = self._make_strategy()
        assert strat.should_exit() is None

    def test_should_exit_funding_negative(self):
        strat = self._make_strategy(exit_funding_hours=3)

        # Open position
        strat.update_funding(rate=0.0002, mark_price=36.5, timestamp=1000)
        strat.on_entry_filled(
            spot_price=36.5, spot_size=0.07,
            perp_price=36.5, perp_size=0.07,
            timestamp=1000,
        )

        # Feed 3 hours of negative funding
        for i in range(3):
            strat.update_funding(rate=-0.0001, mark_price=36.5, timestamp=4600 + i * 3600)

        assert strat.should_exit() == "funding_negative"

    def test_should_exit_max_hold_time(self):
        strat = self._make_strategy(max_hold_hours=2)

        strat.on_entry_filled(
            spot_price=36.5, spot_size=0.07,
            perp_price=36.5, perp_size=0.07,
            timestamp=time.time() - 3 * 3600,  # 3 hours ago
        )

        assert strat.should_exit() == "max_hold_time"

    def test_should_exit_target_profit(self):
        strat = self._make_strategy(target_profit_usd=0.05)

        strat.on_entry_filled(
            spot_price=36.5, spot_size=0.07,
            perp_price=36.5, perp_size=0.07,
            timestamp=1000,
        )

        # Accumulate funding above target
        # 0.001 rate * 2.555 notional = 0.002555 per payment
        # Need 20 payments to reach $0.05
        for i in range(20):
            strat.update_funding(rate=0.001, mark_price=36.5, timestamp=4600 + i * 3600)

        assert strat.should_exit() == "target_profit"

    def test_should_exit_max_loss(self):
        strat = self._make_strategy(max_loss_usd=0.01)

        strat.on_entry_filled(
            spot_price=36.5, spot_size=0.07,
            perp_price=36.5, perp_size=0.07,
            timestamp=1000,
        )

        # Accumulate negative funding past max loss
        for i in range(5):
            strat.update_funding(rate=-0.001, mark_price=36.5, timestamp=4600 + i * 3600)

        assert strat.should_exit() == "max_loss"

    # --- Lifecycle ---

    def test_full_lifecycle(self):
        strat = self._make_strategy(min_funding_hours=2, exit_funding_hours=2)
        strat.start()

        # Phase 1: Monitor funding
        strat.update_funding(rate=0.0003, mark_price=36.5, timestamp=1000)
        strat.update_funding(rate=0.0003, mark_price=36.5, timestamp=4600)
        assert strat.should_enter() is True

        # Phase 2: Open position
        strat.on_entry_filled(
            spot_price=36.5, spot_size=0.07,
            perp_price=36.5, perp_size=0.07,
            total_fees=0.003,
            timestamp=4600,
        )
        assert strat.state == BasisState.ACTIVE
        assert strat.position is not None

        # Phase 3: Collect funding (2 more hours of positive)
        strat.update_funding(rate=0.0003, mark_price=36.8, timestamp=8200)
        strat.update_funding(rate=0.0002, mark_price=37.0, timestamp=11800)
        assert strat.should_exit() is None

        # Phase 4: Funding turns negative
        strat.update_funding(rate=-0.0001, mark_price=36.5, timestamp=15400)
        strat.update_funding(rate=-0.0001, mark_price=36.5, timestamp=19000)
        assert strat.should_exit() == "funding_negative"

        # Phase 5: Close position
        summary = strat.on_exit_filled(
            spot_price=36.5, perp_price=36.5,
            total_fees=0.003, reason="funding_negative",
            timestamp=19000,
        )
        assert strat.state == BasisState.IDLE
        assert strat.position is None
        assert summary["reason"] == "funding_negative"
        assert summary["funding_pnl"] > 0  # Had positive funding initially
        assert strat._trades_closed == 1

    def test_on_entry_filled_transitions_to_active(self):
        strat = self._make_strategy()
        strat.on_entry_filled(
            spot_price=36.5, spot_size=0.07,
            perp_price=36.5, perp_size=0.07,
            timestamp=1000,
        )
        assert strat.state == BasisState.ACTIVE
        assert strat._trades_opened == 1

    def test_on_exit_filled_returns_summary(self):
        strat = self._make_strategy()
        strat.on_entry_filled(
            spot_price=36.0, spot_size=0.07,
            perp_price=36.0, perp_size=0.07,
            total_fees=0.003,
            timestamp=1000,
        )

        # Record some funding
        strat.update_funding(rate=0.0002, mark_price=36.5, timestamp=4600)

        summary = strat.on_exit_filled(
            spot_price=37.0, perp_price=37.0,
            total_fees=0.003, reason="manual",
            timestamp=8200,
        )

        assert summary["symbol"] == "HYPE"
        assert summary["reason"] == "manual"
        assert summary["spot_pnl"] == pytest.approx(0.07)   # (37-36)*0.07
        assert summary["perp_pnl"] == pytest.approx(-0.07)  # (36-37)*0.07
        assert summary["price_pnl"] == pytest.approx(0.0)   # delta neutral
        assert summary["funding_pnl"] > 0
        assert summary["total_fees"] == pytest.approx(0.006)

    def test_funding_payments_recorded_when_active(self):
        strat = self._make_strategy()
        strat.on_entry_filled(
            spot_price=36.5, spot_size=0.07,
            perp_price=36.5, perp_size=0.07,
            timestamp=1000,
        )

        # Update funding while active → should record payment
        strat.update_funding(rate=0.0002, mark_price=36.5, timestamp=4600)
        assert strat.funding_monitor.total_payments == 1

    def test_funding_payments_not_recorded_when_idle(self):
        strat = self._make_strategy()
        # Update funding while idle → should NOT record payment
        strat.update_funding(rate=0.0002, mark_price=36.5, timestamp=1000)
        assert strat.funding_monitor.total_payments == 0

    def test_get_status_idle(self):
        strat = self._make_strategy()
        status = strat.get_status()
        assert status["state"] == "idle"
        assert "position" not in status

    def test_get_status_active(self):
        strat = self._make_strategy()
        strat.on_entry_filled(
            spot_price=36.5, spot_size=0.07,
            perp_price=36.5, perp_size=0.07,
            timestamp=1000,
        )
        status = strat.get_status()
        assert status["state"] == "active"
        assert "position" in status
        assert status["position"]["notional"] == pytest.approx(2.555, abs=0.01)

    def test_multiple_trades(self):
        strat = self._make_strategy(min_funding_hours=1, exit_funding_hours=1)

        # Trade 1
        strat.update_funding(rate=0.0005, mark_price=36.5, timestamp=1000)
        strat.on_entry_filled(
            spot_price=36.5, spot_size=0.07,
            perp_price=36.5, perp_size=0.07,
            timestamp=1000,
        )
        strat.update_funding(rate=-0.0001, mark_price=36.5, timestamp=4600)
        strat.on_exit_filled(spot_price=36.5, perp_price=36.5, reason="funding_negative")

        # Trade 2 — funding monitor reset, need new observations
        strat.update_funding(rate=0.0005, mark_price=37.0, timestamp=8200)
        strat.on_entry_filled(
            spot_price=37.0, spot_size=0.07,
            perp_price=37.0, perp_size=0.07,
            timestamp=8200,
        )
        strat.update_funding(rate=-0.0001, mark_price=37.0, timestamp=11800)
        strat.on_exit_filled(spot_price=37.0, perp_price=37.0, reason="funding_negative")

        assert strat._trades_opened == 2
        assert strat._trades_closed == 2
        assert strat.state == BasisState.IDLE
