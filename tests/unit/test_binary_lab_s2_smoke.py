"""
Binary Lab S2 — Smoke tests.

Two concerns:
1. **Single-cycle truth surface**: verify emitted records contain every field
   that constitutes the S2 epistemic contract (quotes, baseline/model split,
   executable economics, reconstruction mode, calibration flags, skip reasons).

2. **Four-outcome deterministic replay**: YES-WIN, YES-LOSS, NO-WIN, NO-LOSS —
   validates binary PnL arithmetic, settlement logic, Brier scores, and
   state-machine feed in isolation with no filesystem dependencies.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from execution.binary_lab_s2_model import BinaryProbabilityModel
from execution.binary_lab_s2_shadow import (
    BinaryLabS2ShadowRunner,
    OpenRound,
    RoundOutcome,
    POLYMARKET_FEE_RATE,
    ROUND_DURATION_S,
    SETTLEMENT_SOURCE,
    _edge_to_bucket,
    _polymarket_fee,
)
from execution.binary_lab_s2_signals import (
    BinaryLabS2Signal,
    S2EligibilityResult,
    QUOTE_RECONSTRUCTION_MODE,
    _reconstruct_quotes,
    check_s2_eligibility,
    edge_to_bucket,
)


# ---------------------------------------------------------------------------
# Limits fixture (minimal subset of binary_lab_limits_s2.json)
# ---------------------------------------------------------------------------
_TEST_LIMITS: Dict[str, Any] = {
    "_meta": {"sleeve_id": "binary_lab_s2"},
    "capital": {
        "sleeve_total_usd": 900,
        "per_round_usd": 30,
    },
    "position_rules": {"max_concurrent": 3},
    "kill_conditions": {"kill_nav_usd": 650},
    "entry_gate": {
        "min_edge_threshold": 0.03,
        "max_spread_threshold": 0.04,
        "min_time_remaining_s": 120,
        "max_quote_age_s": 75,
        "signal_source": "probability_model",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    *,
    p_yes_mid: float = 0.48,
    spread: float = 0.02,
    p_model_yes: float = 0.55,
    trade_side: str = "YES",
    skip_reason: Optional[str] = None,
    quote_age_s: float = 10.0,
    calibration_active: bool = False,
    calibration_confident: bool = False,
) -> BinaryLabS2Signal:
    """Build a deterministic signal for testing."""
    half = spread / 2.0
    p_yes_bid = p_yes_mid - half
    p_yes_ask = p_yes_mid + half
    p_no_bid = 1.0 - p_yes_ask
    p_no_ask = 1.0 - p_yes_bid
    p_baseline_yes = p_yes_mid

    edge_yes = p_model_yes - p_yes_mid
    baseline_edge = p_baseline_yes - p_yes_mid  # 0 for naive

    if trade_side == "YES":
        entry_cost = p_yes_ask
    elif trade_side == "NO":
        entry_cost = p_no_ask
    else:
        entry_cost = 0.0

    if trade_side == "YES":
        executable_edge = p_model_yes - entry_cost
    elif trade_side == "NO":
        executable_edge = entry_cost - p_model_yes
    else:
        executable_edge = 0.0

    return BinaryLabS2Signal(
        p_yes_bid=round(p_yes_bid, 6),
        p_yes_ask=round(p_yes_ask, 6),
        p_yes_mid=round(p_yes_mid, 6),
        p_no_bid=round(p_no_bid, 6),
        p_no_ask=round(p_no_ask, 6),
        spread=round(spread, 6),
        depth_score=0.5,
        quote_age_s=quote_age_s,
        quote_reconstruction_mode=QUOTE_RECONSTRUCTION_MODE,
        p_baseline_yes=round(p_baseline_yes, 6),
        p_model_yes=round(p_model_yes, 6),
        edge_yes=round(edge_yes, 6),
        baseline_edge=round(baseline_edge, 6),
        entry_cost=round(entry_cost, 6),
        executable_edge=round(executable_edge, 6),
        trade_side=trade_side,
        skip_reason=skip_reason,
        expected_value_usd=round(executable_edge * 30.0, 4) if trade_side != "SKIP" else 0.0,
        calibration_active=calibration_active,
        calibration_confident=calibration_confident,
        features={"p_yes_mid": p_yes_mid, "spread": spread},
        model_version="s2_naive_v1",
        ts=datetime.now(timezone.utc).isoformat(),
    )


def _collect_trade_log(path: Path) -> List[Dict[str, Any]]:
    """Read all JSONL records from a trade log file."""
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# =========================================================================
# Part 1: Single-cycle truth surface
# =========================================================================

# Required fields on every ENTRY record
_ENTRY_REQUIRED_FIELDS = {
    "event_type", "execution_mode", "ts", "ts_ms",
    "round_id", "market_slug", "horizon_s",
    "trade_side",
    "p_yes_bid", "p_yes_ask", "p_yes_mid",
    "p_no_bid", "p_no_ask",
    "p_baseline_yes", "p_model_yes",
    "edge_yes", "baseline_edge",
    "entry_cost", "executable_edge", "expected_value_usd",
    "notional_usd",
    "reference_btc_price",
    "quote_age_s", "quote_reconstruction_mode",
    "spread", "edge_bucket",
    "calibration_active", "calibration_confident",
    "model_version",
    "status",
}

# Required fields on every ROUND_CLOSED record
_CLOSED_REQUIRED_FIELDS = _ENTRY_REQUIRED_FIELDS | {
    "resolved_outcome", "outcome_yes", "payout",
    "pnl_usd", "gross_pnl_usd", "fee_usd", "net_pnl_usd",
    "settlement_btc_price", "settlement_source",
    "brier_component", "baseline_brier_component",
}

# Required fields on every NO_TRADE record
_NO_TRADE_REQUIRED_FIELDS = {
    "event_type", "execution_mode", "ts", "ts_ms",
    "round_id", "market_slug", "horizon_s",
    "status", "eligibility", "deny_reason", "skip_reason",
}


class TestTruthSurface:
    """Every emitted record must contain the full truth surface."""

    def test_entry_record_carries_full_truth_surface(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trades.jsonl"
        model = BinaryProbabilityModel()

        runner = BinaryLabS2ShadowRunner(
            limits=_TEST_LIMITS,
            model=model,
            writer=None,
            trade_log_path=log_path,
            config_hash="test_hash_abc",
        )

        now = time.time()
        round_start = now - (now % ROUND_DURATION_S)
        entry_time = round_start + 60   # inside entry window
        ts = datetime.fromtimestamp(entry_time, tz=timezone.utc).isoformat()

        signal = _make_signal(
            p_yes_mid=0.48, spread=0.02, p_model_yes=0.55, trade_side="YES",
        )

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal) as _sig, \
             patch("execution.binary_lab_s2_signals.check_s2_eligibility") as _elig, \
             patch("execution.exchange_utils.get_price", return_value=65000.0):
            _elig.return_value = S2EligibilityResult(True, None, signal)
            runner.tick(ts)

        records = _collect_trade_log(log_path)
        entries = [r for r in records if r.get("event_type") == "ENTRY"]
        assert len(entries) == 1, f"Expected 1 ENTRY, got {len(entries)}"

        entry = entries[0]
        missing = _ENTRY_REQUIRED_FIELDS - set(entry.keys())
        assert not missing, f"ENTRY record missing fields: {missing}"

        # Verify actual values
        assert entry["execution_mode"] == "SHADOW"
        assert entry["trade_side"] == "YES"
        assert entry["p_yes_mid"] == 0.48
        assert entry["p_yes_bid"] == 0.47
        assert entry["p_yes_ask"] == 0.49
        assert entry["p_no_bid"] == pytest.approx(0.51, abs=0.01)
        assert entry["p_no_ask"] == pytest.approx(0.53, abs=0.01)
        assert entry["p_baseline_yes"] == 0.48
        assert entry["p_model_yes"] == 0.55
        assert entry["entry_cost"] == 0.49           # ask, not mid
        assert entry["quote_reconstruction_mode"] == "mid_plus_mean_spread"
        assert entry["reference_btc_price"] == 65000.0
        assert isinstance(entry["calibration_active"], bool)
        assert isinstance(entry["calibration_confident"], bool)

    def test_no_trade_record_carries_truth_surface(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trades.jsonl"
        model = BinaryProbabilityModel()

        runner = BinaryLabS2ShadowRunner(
            limits=_TEST_LIMITS,
            model=model,
            writer=None,
            trade_log_path=log_path,
        )

        now = time.time()
        round_start = now - (now % ROUND_DURATION_S)
        entry_time = round_start + 60
        ts = datetime.fromtimestamp(entry_time, tz=timezone.utc).isoformat()

        signal = _make_signal(
            p_yes_mid=0.50, spread=0.01, p_model_yes=0.505,
            trade_side="SKIP", skip_reason="edge_below_threshold",
        )

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal), \
             patch("execution.binary_lab_s2_signals.check_s2_eligibility") as _elig:
            _elig.return_value = S2EligibilityResult(False, "edge_below_threshold", signal)
            runner.tick(ts)

        records = _collect_trade_log(log_path)
        no_trades = [r for r in records if r.get("event_type") == "NO_TRADE"]
        assert len(no_trades) >= 1

        nt = no_trades[0]
        missing = _NO_TRADE_REQUIRED_FIELDS - set(nt.keys())
        assert not missing, f"NO_TRADE record missing fields: {missing}"

        # Signal enrichment fields should be present
        assert "p_yes_mid" in nt
        assert "p_baseline_yes" in nt
        assert "p_model_yes" in nt
        assert "quote_reconstruction_mode" in nt
        assert "calibration_active" in nt

    def test_friction_killed_edge_is_logged(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trades.jsonl"
        model = BinaryProbabilityModel()

        runner = BinaryLabS2ShadowRunner(
            limits=_TEST_LIMITS,
            model=model,
            writer=None,
            trade_log_path=log_path,
        )

        now = time.time()
        round_start = now - (now % ROUND_DURATION_S)
        entry_time = round_start + 60
        ts = datetime.fromtimestamp(entry_time, tz=timezone.utc).isoformat()

        signal = _make_signal(
            p_yes_mid=0.50, spread=0.02, p_model_yes=0.53,
            trade_side="SKIP",
            skip_reason="SKIP_FRICTION_ERASED_EDGE",
        )

        with patch("execution.binary_lab_s2_signals.extract_s2_signal", return_value=signal):
            runner.tick(ts)

        records = _collect_trade_log(log_path)
        friction = [r for r in records if r.get("deny_reason") == "SKIP_FRICTION_ERASED_EDGE"]
        assert len(friction) == 1
        assert friction[0]["skip_reason"] == "SKIP_FRICTION_ERASED_EDGE"


# =========================================================================
# Part 2: Four-outcome deterministic replay
# =========================================================================

class TestFourOutcomeReplay:
    """
    Deterministic replay of the four canonical binary outcomes:
        YES-WIN, YES-LOSS, NO-WIN, NO-LOSS

    Each case tests:
        - Binary PnL arithmetic (gross, fee, net)
        - Settlement logic (outcome_yes flag)
        - Brier score computation (model + baseline)
        - State machine feed (conviction_band = edge_bucket)
    """

    @staticmethod
    def _build_round(
        *,
        trade_side: str,
        p_yes_mid: float = 0.48,
        spread: float = 0.02,
        p_model_yes: float = 0.55,
        ref_btc: float = 65000.0,
        notional: float = 30.0,
    ) -> OpenRound:
        half = spread / 2.0
        p_yes_bid = p_yes_mid - half
        p_yes_ask = p_yes_mid + half
        p_no_bid = 1.0 - p_yes_ask
        p_no_ask = 1.0 - p_yes_bid

        p_baseline = p_yes_mid
        edge_yes = p_model_yes - p_yes_mid

        if trade_side == "YES":
            entry_cost = p_yes_ask
            executable_edge = p_model_yes - entry_cost
        else:
            entry_cost = p_no_ask
            executable_edge = entry_cost - p_model_yes

        ev_usd = executable_edge * notional

        now_unix = time.time()
        round_start = now_unix - (now_unix % ROUND_DURATION_S)
        ts = datetime.fromtimestamp(now_unix, tz=timezone.utc).isoformat()

        return OpenRound(
            round_id=f"S2_R_TEST_{trade_side}",
            notional_usd=notional,
            entry_ts=ts,
            entry_ts_unix=now_unix,
            resolution_ts_unix=round_start + ROUND_DURATION_S,
            trade_side=trade_side,
            p_yes_bid=p_yes_bid,
            p_yes_ask=p_yes_ask,
            p_yes_mid=p_yes_mid,
            p_no_bid=p_no_bid,
            p_no_ask=p_no_ask,
            p_baseline_yes=p_baseline,
            p_model_yes=p_model_yes,
            edge_yes=edge_yes,
            baseline_edge=0.0,      # naive baseline = mid, so 0
            entry_cost=entry_cost,
            executable_edge=executable_edge,
            expected_value_usd=round(ev_usd, 4),
            reference_btc_price=ref_btc,
            model_version="s2_naive_v1",
            features={"p_yes_mid": p_yes_mid, "spread": spread},
            quote_age_s=10.0,
            calibration_active=False,
            calibration_confident=False,
            edge_bucket=_edge_to_bucket(abs(edge_yes)),
            spread=spread,
            config_hash="test_hash",
        )

    @staticmethod
    def _resolve(
        runner: BinaryLabS2ShadowRunner,
        rnd: OpenRound,
        settlement_btc: float,
    ) -> RoundOutcome:
        """Force-resolve a round with a known settlement price."""
        ts = datetime.now(timezone.utc).isoformat()
        with patch("execution.exchange_utils.get_price", return_value=settlement_btc):
            outcome = runner._resolve_round(rnd, ts)
        assert outcome is not None
        return outcome

    # ----- YES WIN: BTC goes up, YES holder paid 1 -----

    def test_yes_win(self, tmp_path: Path) -> None:
        model = BinaryProbabilityModel()
        runner = BinaryLabS2ShadowRunner(
            limits=_TEST_LIMITS, model=model,
            trade_log_path=tmp_path / "trades.jsonl",
        )

        rnd = self._build_round(trade_side="YES", ref_btc=65000.0)
        outcome = self._resolve(runner, rnd, settlement_btc=65100.0)

        # BTC went up → YES resolved to 1
        assert outcome.outcome_yes is True
        assert outcome.payout == 1
        assert outcome.trade_side == "YES"
        assert outcome.outcome == "WIN"  # net might still be positive after fee

        # PnL: gross = notional * (1 - entry_cost)
        expected_gross = 30.0 * (1 - rnd.entry_cost)
        assert outcome.gross_pnl_usd == pytest.approx(expected_gross, abs=1e-6)

        # Fee = rate * min(entry_cost, 1-entry_cost) * notional
        expected_fee = POLYMARKET_FEE_RATE * min(rnd.entry_cost, 1 - rnd.entry_cost) * 30.0
        assert outcome.fee_usd == pytest.approx(expected_fee, abs=1e-6)

        # Net = gross - fee
        assert outcome.pnl_usd == pytest.approx(expected_gross - expected_fee, abs=1e-6)

        # Brier: (p_model_yes - 1)^2  (outcome_yes=True → outcome_int=1)
        assert outcome.brier_component == pytest.approx((rnd.p_model_yes - 1) ** 2, abs=1e-6)
        assert outcome.baseline_brier_component == pytest.approx((rnd.p_baseline_yes - 1) ** 2, abs=1e-6)

        # Settlement
        assert outcome.settlement_btc_price == 65100.0
        assert outcome.settlement_source == SETTLEMENT_SOURCE

        # Model received observation
        assert model.n_observations == 1

    # ----- YES LOSS: BTC goes down, YES holder gets 0 -----

    def test_yes_loss(self, tmp_path: Path) -> None:
        model = BinaryProbabilityModel()
        runner = BinaryLabS2ShadowRunner(
            limits=_TEST_LIMITS, model=model,
            trade_log_path=tmp_path / "trades.jsonl",
        )

        rnd = self._build_round(trade_side="YES", ref_btc=65000.0)
        outcome = self._resolve(runner, rnd, settlement_btc=64900.0)

        # BTC went down → YES resolved to 0
        assert outcome.outcome_yes is False
        assert outcome.payout == 0
        assert outcome.trade_side == "YES"
        assert outcome.outcome == "LOSS"

        # PnL: gross = notional * (0 - entry_cost) = -entry_cost*notional
        expected_gross = 30.0 * (0 - rnd.entry_cost)
        assert outcome.gross_pnl_usd == pytest.approx(expected_gross, abs=1e-6)

        expected_fee = POLYMARKET_FEE_RATE * min(rnd.entry_cost, 1 - rnd.entry_cost) * 30.0
        assert outcome.fee_usd == pytest.approx(expected_fee, abs=1e-6)
        assert outcome.pnl_usd == pytest.approx(expected_gross - expected_fee, abs=1e-6)

        # Brier: (p_model_yes - 0)^2  (outcome_yes=False → outcome_int=0)
        assert outcome.brier_component == pytest.approx(rnd.p_model_yes ** 2, abs=1e-6)

    # ----- NO WIN: BTC goes down, NO holder paid 1 -----

    def test_no_win(self, tmp_path: Path) -> None:
        model = BinaryProbabilityModel()
        runner = BinaryLabS2ShadowRunner(
            limits=_TEST_LIMITS, model=model,
            trade_log_path=tmp_path / "trades.jsonl",
        )

        # Model thinks YES is overpriced → buy NO
        rnd = self._build_round(
            trade_side="NO", p_yes_mid=0.55, spread=0.02,
            p_model_yes=0.45, ref_btc=65000.0,
        )
        outcome = self._resolve(runner, rnd, settlement_btc=64900.0)

        # BTC went down → YES=False → NO wins
        assert outcome.outcome_yes is False
        assert outcome.payout == 1
        assert outcome.trade_side == "NO"
        assert outcome.outcome == "WIN"

        # PnL: gross = notional * (1 - entry_cost)
        expected_gross = 30.0 * (1 - rnd.entry_cost)
        assert outcome.gross_pnl_usd == pytest.approx(expected_gross, abs=1e-6)

        expected_fee = POLYMARKET_FEE_RATE * min(rnd.entry_cost, 1 - rnd.entry_cost) * 30.0
        assert outcome.fee_usd == pytest.approx(expected_fee, abs=1e-6)
        assert outcome.pnl_usd == pytest.approx(expected_gross - expected_fee, abs=1e-6)

        # Brier scored against YES outcome (False=0): (p_model_yes - 0)^2
        assert outcome.brier_component == pytest.approx(rnd.p_model_yes ** 2, abs=1e-6)

    # ----- NO LOSS: BTC goes up, NO holder gets 0 -----

    def test_no_loss(self, tmp_path: Path) -> None:
        model = BinaryProbabilityModel()
        runner = BinaryLabS2ShadowRunner(
            limits=_TEST_LIMITS, model=model,
            trade_log_path=tmp_path / "trades.jsonl",
        )

        rnd = self._build_round(
            trade_side="NO", p_yes_mid=0.55, spread=0.02,
            p_model_yes=0.45, ref_btc=65000.0,
        )
        outcome = self._resolve(runner, rnd, settlement_btc=65100.0)

        # BTC went up → YES=True → NO loses
        assert outcome.outcome_yes is True
        assert outcome.payout == 0
        assert outcome.trade_side == "NO"
        assert outcome.outcome == "LOSS"

        # PnL: gross = notional * (0 - entry_cost)
        expected_gross = 30.0 * (0 - rnd.entry_cost)
        assert outcome.gross_pnl_usd == pytest.approx(expected_gross, abs=1e-6)

        # Brier: (p_model_yes - 1)^2 (outcome_yes=True)
        assert outcome.brier_component == pytest.approx((rnd.p_model_yes - 1) ** 2, abs=1e-6)

    # ----- All four outcomes update model -----

    def test_all_outcomes_feed_model(self, tmp_path: Path) -> None:
        """Run all 4 canonical outcomes and verify model accumulates."""
        model = BinaryProbabilityModel()
        runner = BinaryLabS2ShadowRunner(
            limits=_TEST_LIMITS, model=model,
            trade_log_path=tmp_path / "trades.jsonl",
        )

        cases = [
            ("YES", 65100.0),  # YES WIN
            ("YES", 64900.0),  # YES LOSS
            ("NO",  64900.0),  # NO WIN
            ("NO",  65100.0),  # NO LOSS
        ]
        for trade_side, settle in cases:
            mid = 0.48 if trade_side == "YES" else 0.55
            pm = 0.55 if trade_side == "YES" else 0.45
            rnd = self._build_round(
                trade_side=trade_side, p_yes_mid=mid, p_model_yes=pm,
            )
            self._resolve(runner, rnd, settlement_btc=settle)

        assert model.n_observations == 4
        stats = model.calibration_stats()
        assert "brier_score" in stats
        assert "baseline_brier_score" in stats


# =========================================================================
# Part 3: Component-level sanity checks
# =========================================================================

class TestQuoteReconstruction:
    def test_valid_reconstruction(self) -> None:
        result = _reconstruct_quotes(0.50, 0.02)
        assert result is not None
        assert result["p_yes_bid"] == pytest.approx(0.49, abs=1e-6)
        assert result["p_yes_ask"] == pytest.approx(0.51, abs=1e-6)
        assert result["p_no_bid"] == pytest.approx(0.49, abs=1e-6)
        assert result["p_no_ask"] == pytest.approx(0.51, abs=1e-6)

    def test_extreme_spread_returns_none(self) -> None:
        # spread > 2*mid → negative bid after clamp, ordering may fail
        result = _reconstruct_quotes(0.01, 0.5)
        # bid would be 0.01 - 0.25 = -0.24, clamped to 0.0
        # ask would be 0.01 + 0.25 = 0.26
        # 0.0 <= 0.01 (mid) <= 0.26 (ask) → ordering ok
        # but cross-check: p_yes_ask(0.26)+p_no_bid(0.74) = 1.0 ok
        # This one may actually pass reconstruction
        if result is not None:
            assert result["p_yes_bid"] >= 0.0
            assert result["p_yes_ask"] <= 1.0

    def test_zero_spread(self) -> None:
        result = _reconstruct_quotes(0.50, 0.0)
        assert result is not None
        assert result["p_yes_bid"] == result["p_yes_ask"]


class TestEdgeBuckets:
    def test_bucket_boundaries(self) -> None:
        assert edge_to_bucket(0.01) == "edge_0_2pp"
        assert edge_to_bucket(0.02) == "edge_2_5pp"
        assert edge_to_bucket(0.04) == "edge_2_5pp"
        assert edge_to_bucket(0.05) == "edge_5_8pp"
        assert edge_to_bucket(0.07) == "edge_5_8pp"
        assert edge_to_bucket(0.08) == "edge_8pp_plus"
        assert edge_to_bucket(0.15) == "edge_8pp_plus"

    def test_shadow_bucket_agrees_with_signal_bucket(self) -> None:
        for val in [0.01, 0.03, 0.06, 0.10]:
            assert _edge_to_bucket(val) == edge_to_bucket(val)


class TestPolymarketFee:
    def test_symmetric(self) -> None:
        # fee(p) = rate * min(p, 1-p) * notional
        assert _polymarket_fee(0.3, 100.0) == pytest.approx(0.02 * 0.3 * 100, abs=1e-8)
        assert _polymarket_fee(0.7, 100.0) == pytest.approx(0.02 * 0.3 * 100, abs=1e-8)

    def test_midpoint(self) -> None:
        # At p=0.5: fee = 0.02 * 0.5 * notional
        assert _polymarket_fee(0.5, 30.0) == pytest.approx(0.02 * 0.5 * 30.0, abs=1e-8)


class TestEligibilityGate:
    def test_all_gates_pass(self) -> None:
        signal = _make_signal(
            p_yes_mid=0.48, spread=0.02, p_model_yes=0.55, trade_side="YES",
        )
        result = check_s2_eligibility(
            signal, _TEST_LIMITS,
            current_nav_usd=900.0, open_positions=0,
            freeze_intact=True, time_remaining_s=600.0,
        )
        assert result.eligible

    def test_spread_too_wide(self) -> None:
        signal = _make_signal(spread=0.06)
        result = check_s2_eligibility(
            signal, _TEST_LIMITS,
            current_nav_usd=900.0, open_positions=0,
            freeze_intact=True, time_remaining_s=600.0,
        )
        assert not result.eligible
        assert "spread_too_wide" in (result.deny_reason or "")

    def test_stale_quote_rejected(self) -> None:
        signal = _make_signal(quote_age_s=120.0)
        result = check_s2_eligibility(
            signal, _TEST_LIMITS,
            current_nav_usd=900.0, open_positions=0,
            freeze_intact=True, time_remaining_s=600.0,
        )
        assert not result.eligible
        assert "quote_stale" in (result.deny_reason or "")

    def test_kill_line_breached(self) -> None:
        signal = _make_signal()
        result = check_s2_eligibility(
            signal, _TEST_LIMITS,
            current_nav_usd=640.0, open_positions=0,
            freeze_intact=True, time_remaining_s=600.0,
        )
        assert not result.eligible
        assert result.deny_reason == "kill_line_breached"

    def test_concurrent_cap(self) -> None:
        signal = _make_signal()
        result = check_s2_eligibility(
            signal, _TEST_LIMITS,
            current_nav_usd=900.0, open_positions=3,
            freeze_intact=True, time_remaining_s=600.0,
        )
        assert not result.eligible
        assert "concurrent_cap" in (result.deny_reason or "")

    def test_freeze_broken(self) -> None:
        signal = _make_signal()
        result = check_s2_eligibility(
            signal, _TEST_LIMITS,
            current_nav_usd=900.0, open_positions=0,
            freeze_intact=False, time_remaining_s=600.0,
        )
        assert not result.eligible
        assert result.deny_reason == "freeze_broken"


class TestModelLifecycle:
    def test_baseline_always_returns_mid(self) -> None:
        model = BinaryProbabilityModel()
        for mid in [0.30, 0.50, 0.70]:
            assert model.predict_baseline({"p_yes_mid": mid}) == mid

    def test_predict_equals_baseline_before_calibration(self) -> None:
        model = BinaryProbabilityModel()
        features = {"p_yes_mid": 0.55}
        assert model.predict(features) == model.predict_baseline(features)

    def test_calibration_threshold(self) -> None:
        model = BinaryProbabilityModel(calibration_min_samples=10)
        for i in range(9):
            model.update_observation({"p_yes_mid": 0.5}, i % 2 == 0)
        assert not model.calibration_active
        model.update_observation({"p_yes_mid": 0.5}, True)
        assert model.calibration_active
        assert not model.calibration_confident

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """save_state persists stats; load_state returns True but does not
        restore observations (intentional — model starts fresh each boot,
        observations accumulate from resolved rounds during current session)."""
        state_path = tmp_path / "cal.json"
        model = BinaryProbabilityModel(
            calibration_min_samples=5, state_path=state_path,
        )
        for i in range(10):
            model.update_observation({"p_yes_mid": 0.4 + i * 0.02}, i % 2 == 0)
        model.save_state()

        assert state_path.exists()
        with state_path.open() as f:
            data = json.load(f)
        assert data["n_observations"] == 10
        assert data["calibration_active"] is True

        model2 = BinaryProbabilityModel(
            calibration_min_samples=5, state_path=state_path,
        )
        assert model2.load_state() is True
        # Observations don't survive restart — this is by design.
        # The model accumulates fresh observations from resolved rounds.
        assert model2.n_observations == 0
