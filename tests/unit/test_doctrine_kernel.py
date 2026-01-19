"""
Tests for Doctrine Kernel — v7.X Constitutional Law

These tests verify the hard laws encoded in the doctrine kernel.
"""

import pytest
import time
from execution.doctrine_kernel import (
    doctrine_entry_verdict,
    doctrine_exit_verdict,
    DoctrineVerdict,
    ExitReason,
    ExitUrgency,
    RegimeSnapshot,
    IntentSnapshot,
    ExecutionSnapshot,
    PortfolioSnapshot,
    PositionSnapshot,
    AlphaHealthSnapshot,
    REGIME_STABILITY_CYCLES,
    REGIME_CONFIDENCE_FLOOR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stable_trend_up_regime():
    """A stable TREND_UP regime with good confidence."""
    return RegimeSnapshot(
        primary_regime="TREND_UP",
        confidence=0.70,
        cycles_stable=5,
        crisis_flag=False,
        updated_ts=time.time(),
    )


@pytest.fixture
def unstable_regime():
    """A regime that hasn't been stable long enough."""
    return RegimeSnapshot(
        primary_regime="TREND_UP",
        confidence=0.70,
        cycles_stable=1,  # Below REGIME_STABILITY_CYCLES
        crisis_flag=False,
        updated_ts=time.time(),
    )


@pytest.fixture
def crisis_regime():
    """A CRISIS regime."""
    return RegimeSnapshot(
        primary_regime="CRISIS",
        confidence=0.90,
        cycles_stable=10,
        crisis_flag=True,
        crisis_reason="Extreme volatility detected",
        updated_ts=time.time(),
    )


@pytest.fixture
def choppy_regime():
    """A CHOPPY regime that permits no directional trades."""
    return RegimeSnapshot(
        primary_regime="CHOPPY",
        confidence=0.60,
        cycles_stable=5,
        crisis_flag=False,
        updated_ts=time.time(),
    )


@pytest.fixture
def buy_intent():
    """A standard BUY intent."""
    return IntentSnapshot(
        symbol="BTCUSDT",
        direction="BUY",
        head="TREND",
        raw_size_usd=1000.0,
        alpha_router_allocation=1.0,
        conviction=0.7,
    )


@pytest.fixture
def sell_intent():
    """A standard SELL intent."""
    return IntentSnapshot(
        symbol="BTCUSDT",
        direction="SELL",
        head="TREND",
        raw_size_usd=1000.0,
        alpha_router_allocation=1.0,
        conviction=0.7,
    )


@pytest.fixture
def normal_execution():
    """Normal execution conditions."""
    return ExecutionSnapshot(
        regime="NORMAL",
        quality_score=0.85,
        avg_slippage_bps=2.0,
        throttling_active=False,
    )


@pytest.fixture
def crunch_execution():
    """Execution crunch — trading blocked."""
    return ExecutionSnapshot(
        regime="CRUNCH",
        quality_score=0.30,
        avg_slippage_bps=50.0,
        throttling_active=True,
    )


@pytest.fixture
def healthy_portfolio():
    """Healthy portfolio with budget."""
    return PortfolioSnapshot(
        head_budget_remaining={"TREND": 1.0, "CARRY": 0.5},
        total_exposure_pct=0.30,
        drawdown_pct=0.02,
        risk_mode="OK",
    )


@pytest.fixture
def exhausted_portfolio():
    """Portfolio with no budget for TREND head."""
    return PortfolioSnapshot(
        head_budget_remaining={"TREND": 0.0, "CARRY": 0.5},
        total_exposure_pct=0.80,
        drawdown_pct=0.10,
        risk_mode="CAUTION",
    )


# ---------------------------------------------------------------------------
# Entry Verdict Tests
# ---------------------------------------------------------------------------


class TestDoctrineEntryVerdict:
    """Tests for doctrine_entry_verdict — the supreme entry gate."""

    def test_allow_trend_up_with_buy(
        self, stable_trend_up_regime, buy_intent, normal_execution, healthy_portfolio
    ):
        """TREND_UP regime should allow BUY intents."""
        decision = doctrine_entry_verdict(
            regime=stable_trend_up_regime,
            intent=buy_intent,
            execution=normal_execution,
            portfolio=healthy_portfolio,
        )
        assert decision.allowed is True
        assert decision.verdict == DoctrineVerdict.ALLOW

    def test_veto_trend_up_with_sell(
        self, stable_trend_up_regime, sell_intent, normal_execution, healthy_portfolio
    ):
        """TREND_UP regime should VETO SELL intents."""
        decision = doctrine_entry_verdict(
            regime=stable_trend_up_regime,
            intent=sell_intent,
            execution=normal_execution,
            portfolio=healthy_portfolio,
        )
        assert decision.allowed is False
        assert decision.verdict == DoctrineVerdict.VETO_DIRECTION_MISMATCH

    def test_veto_unstable_regime(
        self, unstable_regime, buy_intent, normal_execution, healthy_portfolio
    ):
        """Unstable regime should VETO all entries."""
        decision = doctrine_entry_verdict(
            regime=unstable_regime,
            intent=buy_intent,
            execution=normal_execution,
            portfolio=healthy_portfolio,
        )
        assert decision.allowed is False
        assert decision.verdict == DoctrineVerdict.VETO_REGIME_UNSTABLE

    def test_veto_crisis(
        self, crisis_regime, buy_intent, normal_execution, healthy_portfolio
    ):
        """CRISIS regime should VETO all entries."""
        decision = doctrine_entry_verdict(
            regime=crisis_regime,
            intent=buy_intent,
            execution=normal_execution,
            portfolio=healthy_portfolio,
        )
        assert decision.allowed is False
        assert decision.verdict == DoctrineVerdict.VETO_CRISIS

    def test_veto_choppy_direction(
        self, choppy_regime, buy_intent, normal_execution, healthy_portfolio
    ):
        """CHOPPY regime should VETO directional trades."""
        decision = doctrine_entry_verdict(
            regime=choppy_regime,
            intent=buy_intent,
            execution=normal_execution,
            portfolio=healthy_portfolio,
        )
        assert decision.allowed is False
        assert decision.verdict == DoctrineVerdict.VETO_DIRECTION_MISMATCH

    def test_veto_execution_crunch(
        self, stable_trend_up_regime, buy_intent, crunch_execution, healthy_portfolio
    ):
        """Execution CRUNCH should VETO entries."""
        decision = doctrine_entry_verdict(
            regime=stable_trend_up_regime,
            intent=buy_intent,
            execution=crunch_execution,
            portfolio=healthy_portfolio,
        )
        assert decision.allowed is False
        assert decision.verdict == DoctrineVerdict.VETO_EXECUTION_CRUNCH

    def test_veto_no_head_budget(
        self, stable_trend_up_regime, buy_intent, normal_execution, exhausted_portfolio
    ):
        """No head budget should VETO entries."""
        decision = doctrine_entry_verdict(
            regime=stable_trend_up_regime,
            intent=buy_intent,
            execution=normal_execution,
            portfolio=exhausted_portfolio,
        )
        assert decision.allowed is False
        assert decision.verdict == DoctrineVerdict.VETO_NO_HEAD_BUDGET

    def test_veto_stale_regime(
        self, buy_intent, normal_execution, healthy_portfolio
    ):
        """Stale regime should VETO entries."""
        stale_regime = RegimeSnapshot(
            primary_regime="TREND_UP",
            confidence=0.70,
            cycles_stable=5,
            crisis_flag=False,
            updated_ts=time.time() - 1000,  # Very old
        )
        decision = doctrine_entry_verdict(
            regime=stale_regime,
            intent=buy_intent,
            execution=normal_execution,
            portfolio=healthy_portfolio,
        )
        assert decision.allowed is False
        assert decision.verdict == DoctrineVerdict.VETO_REGIME_STALE

    def test_veto_low_confidence(
        self, buy_intent, normal_execution, healthy_portfolio
    ):
        """Low confidence should VETO entries."""
        low_conf_regime = RegimeSnapshot(
            primary_regime="TREND_UP",
            confidence=0.30,  # Below floor
            cycles_stable=5,
            crisis_flag=False,
            updated_ts=time.time(),
        )
        decision = doctrine_entry_verdict(
            regime=low_conf_regime,
            intent=buy_intent,
            execution=normal_execution,
            portfolio=healthy_portfolio,
        )
        assert decision.allowed is False
        assert decision.verdict == DoctrineVerdict.VETO_REGIME_CONFIDENCE

    def test_multipliers_applied_on_allow(
        self, stable_trend_up_regime, buy_intent, normal_execution, healthy_portfolio
    ):
        """When allowed, multipliers should be computed."""
        decision = doctrine_entry_verdict(
            regime=stable_trend_up_regime,
            intent=buy_intent,
            execution=normal_execution,
            portfolio=healthy_portfolio,
        )
        assert decision.allowed is True
        assert decision.regime_multiplier > 0
        assert decision.execution_multiplier > 0
        assert decision.composite_multiplier > 0


# ---------------------------------------------------------------------------
# Exit Verdict Tests
# ---------------------------------------------------------------------------


class TestDoctrineExitVerdict:
    """Tests for doctrine_exit_verdict — the exit precedence law."""

    @pytest.fixture
    def long_position(self):
        """A long position entered in TREND_UP."""
        return PositionSnapshot(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            current_price=51000.0,
            qty=0.1,
            entry_regime="TREND_UP",
            entry_regime_confidence=0.70,
            entry_ts=time.time() - 3600,
            bars_held=24,
            unrealized_pnl_pct=0.02,
            entry_head="TREND",
        )

    def test_hold_when_thesis_valid(
        self, stable_trend_up_regime, long_position, normal_execution
    ):
        """Should HOLD when thesis remains valid."""
        decision = doctrine_exit_verdict(
            regime=stable_trend_up_regime,
            position=long_position,
            execution=normal_execution,
        )
        assert decision.should_exit is False
        assert decision.reason == ExitReason.HOLD

    def test_exit_on_crisis(self, crisis_regime, long_position, normal_execution):
        """Should exit IMMEDIATELY on CRISIS."""
        decision = doctrine_exit_verdict(
            regime=crisis_regime,
            position=long_position,
            execution=normal_execution,
        )
        assert decision.should_exit is True
        assert decision.reason == ExitReason.CRISIS_OVERRIDE
        assert decision.urgency == ExitUrgency.IMMEDIATE

    def test_exit_on_regime_flip(self, long_position, normal_execution):
        """Should exit when regime flips against position."""
        trend_down = RegimeSnapshot(
            primary_regime="TREND_DOWN",
            confidence=0.70,
            cycles_stable=5,
            crisis_flag=False,
            updated_ts=time.time(),
        )
        decision = doctrine_exit_verdict(
            regime=trend_down,
            position=long_position,
            execution=normal_execution,
        )
        assert decision.should_exit is True
        assert decision.reason == ExitReason.REGIME_FLIP
        assert decision.urgency == ExitUrgency.STEPPED

    def test_exit_on_time_stop(self, stable_trend_up_regime, normal_execution):
        """Should exit on time stop if position not progressing."""
        stale_position = PositionSnapshot(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            current_price=50000.0,  # No progress
            qty=0.1,
            entry_regime="TREND_UP",
            entry_regime_confidence=0.70,
            entry_ts=time.time() - 86400,
            bars_held=100,  # Above TIME_STOP_BARS
            unrealized_pnl_pct=0.001,  # Less than 0.5%
            entry_head="TREND",
        )
        decision = doctrine_exit_verdict(
            regime=stable_trend_up_regime,
            position=stale_position,
            execution=normal_execution,
        )
        assert decision.should_exit is True
        assert decision.reason == ExitReason.TIME_STOP

    def test_exit_on_stop_loss_seatbelt(
        self, stable_trend_up_regime, normal_execution
    ):
        """Should exit on stop-loss seatbelt (emergency)."""
        position_at_stop = PositionSnapshot(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            current_price=47000.0,  # Below stop
            qty=0.1,
            entry_regime="TREND_UP",
            entry_regime_confidence=0.70,
            entry_ts=time.time() - 3600,
            bars_held=24,
            unrealized_pnl_pct=-0.06,
            entry_head="TREND",
            sl_price=48000.0,  # Stop-loss triggered
        )
        decision = doctrine_exit_verdict(
            regime=stable_trend_up_regime,
            position=position_at_stop,
            execution=normal_execution,
        )
        assert decision.should_exit is True
        assert decision.reason == ExitReason.STOP_LOSS_SEATBELT
        assert decision.urgency == ExitUrgency.IMMEDIATE


# ---------------------------------------------------------------------------
# Regime Direction Mapping Tests
# ---------------------------------------------------------------------------


class TestRegimeDirectionMapping:
    """Tests for regime direction permissions."""

    def test_trend_up_permits_long(self):
        regime = RegimeSnapshot(
            primary_regime="TREND_UP",
            confidence=0.70,
            updated_ts=time.time(),
        )
        assert regime.permits_direction("BUY") is True
        assert regime.permits_direction("LONG") is True
        assert regime.permits_direction("SELL") is False
        assert regime.permits_direction("SHORT") is False

    def test_trend_down_permits_short(self):
        regime = RegimeSnapshot(
            primary_regime="TREND_DOWN",
            confidence=0.70,
            updated_ts=time.time(),
        )
        assert regime.permits_direction("BUY") is False
        assert regime.permits_direction("LONG") is False
        assert regime.permits_direction("SELL") is True
        assert regime.permits_direction("SHORT") is True

    def test_mean_revert_permits_both(self):
        regime = RegimeSnapshot(
            primary_regime="MEAN_REVERT",
            confidence=0.70,
            updated_ts=time.time(),
        )
        assert regime.permits_direction("BUY") is True
        assert regime.permits_direction("SELL") is True

    def test_choppy_permits_nothing(self):
        regime = RegimeSnapshot(
            primary_regime="CHOPPY",
            confidence=0.70,
            updated_ts=time.time(),
        )
        assert regime.permits_direction("BUY") is False
        assert regime.permits_direction("SELL") is False

    def test_crisis_permits_nothing(self):
        regime = RegimeSnapshot(
            primary_regime="CRISIS",
            confidence=0.90,
            updated_ts=time.time(),
        )
        assert regime.permits_direction("BUY") is False
        assert regime.permits_direction("SELL") is False
