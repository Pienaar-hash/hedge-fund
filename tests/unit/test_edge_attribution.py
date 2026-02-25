"""
Tests for Edge Attribution Report (v7.9-TE1).

Validates:
- Deterministic bucketing
- Stable output schema
- Handles empty sets gracefully
- Correct fee netting math
- k_atr suggestions calculation
- CSV/Markdown rendering
"""

import math
import pytest
from research.edge_attribution_report import (
    TradeRecord,
    BucketStats,
    KAtrSuggestion,
    AttributionReport,
    build_trade_records,
    generate_report,
    render_markdown,
    render_csv,
    _compute_bucket_stats,
    _bucket_by_edge_decile,
    _bucket_by_confidence,
    _compute_k_atr_suggestions,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_trade(
    *,
    episode_id: str = "EP_test",
    symbol: str = "BTCUSDT",
    strategy: str = "TREND",
    side: str = "LONG",
    regime: str = "TREND_UP",
    confidence: float = 0.65,
    pred_edge_pct: float = 0.001,
    pred_edge_usd: float = 1.0,
    atr_pct: float = 0.002,
    k_atr: float = 0.6,
    adv: float = 0.15,
    edge_source: str = "atr_conf_v1",
    notional: float = 1000.0,
    fees: float = 0.08,
    pnl: float = 0.50,
    entry_price: float = 50000.0,
    exit_price: float = 50025.0,
    holding_s: float = 900.0,
) -> TradeRecord:
    """Create a TradeRecord for testing."""
    move_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
    if side == "SHORT":
        move_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0.0
    return TradeRecord(
        episode_id=episode_id,
        symbol=symbol,
        strategy=strategy,
        side=side,
        regime_at_entry=regime,
        regime_at_exit=regime,
        entry_ts="2026-02-25T10:00:00Z",
        exit_ts="2026-02-25T10:15:00Z",
        holding_time_s=holding_s,
        notional_usd=notional,
        fees_usd=fees,
        slippage_est_usd=0.0,
        pred_expected_edge_pct=pred_edge_pct,
        pred_expected_edge_usd=pred_edge_usd,
        confidence=confidence,
        atr_pct=atr_pct,
        k_atr=k_atr,
        adv=adv,
        edge_source=edge_source,
        realized_pnl_usd=pnl,
        realized_move_pct=move_pct,
        avg_entry_price=entry_price,
        avg_exit_price=exit_price,
    )


def _make_trades_spread(n: int = 20) -> list[TradeRecord]:
    """Create a spread of trades with varying confidence and outcomes."""
    trades = []
    for i in range(n):
        conf = 0.50 + (i / n) * 0.25  # 0.50 → 0.75
        pnl = (i - n / 2) * 0.10  # some winners, some losers
        trades.append(_make_trade(
            episode_id=f"EP_{i:03d}",
            confidence=conf,
            pred_edge_pct=0.001 * (i + 1),
            pred_edge_usd=1.0 * (i + 1),
            adv=max(0, conf - 0.5),
            pnl=pnl,
            atr_pct=0.002,
        ))
    return trades


# ── Empty sets ─────────────────────────────────────────────────────────────

class TestEmptySets:
    def test_generate_report_empty(self):
        """Empty trade list produces valid report with zero counts."""
        report = generate_report([])
        assert report.total_trades == 0
        assert report.trades_with_atr_edge == 0
        assert report.trades_with_fallback == 0
        assert report.edge_decile_buckets == []
        assert report.confidence_buckets == []
        assert report.k_atr_suggestions == []

    def test_bucket_by_edge_decile_empty(self):
        assert _bucket_by_edge_decile([]) == []

    def test_bucket_by_confidence_empty(self):
        assert _bucket_by_confidence([]) == []

    def test_compute_bucket_stats_empty(self):
        b = _compute_bucket_stats("empty", [])
        assert b.count == 0
        assert b.win_rate == 0.0

    def test_k_atr_suggestions_empty(self):
        assert _compute_k_atr_suggestions([]) == []

    def test_render_markdown_empty(self):
        report = generate_report([])
        md = render_markdown(report)
        assert "Total trades: 0" in md

    def test_render_csv_empty(self):
        csv = render_csv([])
        lines = csv.strip().split("\n")
        assert len(lines) == 1  # header only


# ── Bucket stats computation ──────────────────────────────────────────────

class TestBucketStats:
    def test_single_winning_trade(self):
        t = _make_trade(pnl=1.50, fees=0.08)
        b = _compute_bucket_stats("test", [t])
        assert b.count == 1
        assert b.win_rate == 1.0
        assert b.mean_realized_pnl == pytest.approx(1.50, abs=0.01)
        assert b.mean_fees == pytest.approx(0.08, abs=0.01)
        assert b.profit_factor == 999.0  # no losses

    def test_single_losing_trade(self):
        t = _make_trade(pnl=-0.50)
        b = _compute_bucket_stats("test", [t])
        assert b.win_rate == 0.0
        assert b.profit_factor == 0.0  # no wins

    def test_mixed_trades(self):
        trades = [
            _make_trade(pnl=2.0),
            _make_trade(pnl=-1.0),
            _make_trade(pnl=0.5),
        ]
        b = _compute_bucket_stats("mixed", trades)
        assert b.count == 3
        assert b.win_rate == pytest.approx(2 / 3, abs=0.01)
        assert b.total_realized_pnl == pytest.approx(1.5, abs=0.01)
        # profit_factor = 2.5 / 1.0 = 2.5
        assert b.profit_factor == pytest.approx(2.5, abs=0.01)

    def test_fee_netting_reflected_in_pnl(self):
        """PnL should already be net of fees — verify stats use net values."""
        t_win = _make_trade(pnl=1.00, fees=0.08)  # net PnL after fees
        t_lose = _make_trade(pnl=-0.50, fees=0.08)
        b = _compute_bucket_stats("net", [t_win, t_lose])
        assert b.total_realized_pnl == pytest.approx(0.50, abs=0.01)
        assert b.mean_fees == pytest.approx(0.08, abs=0.01)


# ── Edge decile bucketing ─────────────────────────────────────────────────

class TestEdgeDecileBuckets:
    def test_deterministic_bucketing(self):
        """Same input → same bucket assignment every time."""
        trades = _make_trades_spread(20)
        b1 = _bucket_by_edge_decile(trades)
        b2 = _bucket_by_edge_decile(trades)
        assert len(b1) == len(b2)
        for i in range(len(b1)):
            assert b1[i].count == b2[i].count
            assert b1[i].mean_pred_edge_pct == b2[i].mean_pred_edge_pct

    def test_bucket_count_matches_deciles(self):
        trades = _make_trades_spread(20)
        buckets = _bucket_by_edge_decile(trades)
        assert len(buckets) == 10
        total = sum(b.count for b in buckets)
        assert total == 20

    def test_monotonic_edge_across_deciles(self):
        """Mean predicted edge should increase across deciles."""
        trades = _make_trades_spread(30)
        buckets = _bucket_by_edge_decile(trades)
        for i in range(1, len(buckets)):
            assert buckets[i].mean_pred_edge_pct >= buckets[i - 1].mean_pred_edge_pct

    def test_small_sample(self):
        """Fewer trades than deciles → fewer buckets."""
        trades = _make_trades_spread(3)
        buckets = _bucket_by_edge_decile(trades)
        assert len(buckets) <= 10
        total = sum(b.count for b in buckets)
        assert total == 3


# ── Confidence bucketing ──────────────────────────────────────────────────

class TestConfidenceBuckets:
    def test_5pp_buckets(self):
        """Trades should land in correct 5pp confidence buckets."""
        trades = [
            _make_trade(confidence=0.52),
            _make_trade(confidence=0.57),
            _make_trade(confidence=0.62),
        ]
        buckets = _bucket_by_confidence(trades)
        labels = [b.bucket_label for b in buckets]
        assert "[0.50, 0.55)" in labels
        assert "[0.55, 0.60)" in labels
        assert "[0.60, 0.65)" in labels

    def test_deterministic(self):
        trades = _make_trades_spread(20)
        b1 = _bucket_by_confidence(trades)
        b2 = _bucket_by_confidence(trades)
        assert len(b1) == len(b2)
        for i in range(len(b1)):
            assert b1[i].bucket_label == b2[i].bucket_label


# ── k_atr suggestions ─────────────────────────────────────────────────────

class TestKAtrSuggestions:
    def test_basic_computation(self):
        """Median move multiple calculated correctly."""
        trades = [
            _make_trade(atr_pct=0.002, pnl=1.0, entry_price=100.0, exit_price=100.3),  # move=0.3%
            _make_trade(atr_pct=0.002, pnl=0.5, entry_price=100.0, exit_price=100.2),  # move=0.2%
            _make_trade(atr_pct=0.002, pnl=-0.3, entry_price=100.0, exit_price=99.9),   # move=0.1%
        ]
        suggestions = _compute_k_atr_suggestions(trades, current_k_atr=0.6)
        assert len(suggestions) >= 1  # at least ALL
        overall = suggestions[0]
        assert overall.regime == "ALL"
        assert overall.sample_count == 3
        # Moves: 0.003, 0.002, 0.001 → multiples: 1.5, 1.0, 0.5 → median=1.0
        assert overall.median_move_multiple == pytest.approx(1.0, abs=0.01)

    def test_zero_atr_excluded(self):
        """Trades with zero ATR should not contribute to k_atr suggestions."""
        trades = [
            _make_trade(atr_pct=0.0, pnl=1.0),
            _make_trade(atr_pct=0.002, pnl=0.5, entry_price=100.0, exit_price=100.2),
            _make_trade(atr_pct=0.002, pnl=0.5, entry_price=100.0, exit_price=100.2),
            _make_trade(atr_pct=0.002, pnl=0.5, entry_price=100.0, exit_price=100.2),
        ]
        suggestions = _compute_k_atr_suggestions(trades)
        overall = suggestions[0]
        assert overall.sample_count == 3  # zero-ATR excluded


# ── Report schema stability ───────────────────────────────────────────────

class TestReportSchema:
    def test_report_to_dict_has_required_keys(self):
        report = generate_report(_make_trades_spread(20))
        d = report.to_dict()
        required_keys = {
            "generated_at", "total_trades", "trades_with_atr_edge",
            "trades_with_fallback", "overall_win_rate", "overall_mean_pnl",
            "overall_profit_factor", "edge_decile_buckets",
            "confidence_buckets", "k_atr_suggestions",
            "strategy_buckets", "regime_buckets",
        }
        assert required_keys.issubset(d.keys())

    def test_bucket_stats_schema(self):
        b = _compute_bucket_stats("test", [_make_trade()])
        d = b.to_dict()
        required_keys = {
            "bucket_label", "count", "mean_pred_edge_pct",
            "median_pred_edge_pct", "mean_realized_pnl",
            "median_realized_pnl", "total_realized_pnl",
            "win_rate", "mean_fees", "profit_factor", "mean_holding_s",
        }
        assert required_keys.issubset(d.keys())


# ── Build trade records from episodes ─────────────────────────────────────

class TestBuildTradeRecords:
    def test_basic_episode_conversion(self):
        """Raw episode dict converts to TradeRecord."""
        episodes = [
            {
                "episode_id": "EP_001",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "strategy": "TREND",
                "entry_ts": "2026-02-25T10:00:00Z",
                "exit_ts": "2026-02-25T10:15:00Z",
                "duration_hours": 0.25,
                "avg_entry_price": 50000.0,
                "avg_exit_price": 50100.0,
                "entry_notional": 1000.0,
                "net_pnl": 2.0,
                "fees": 0.08,
                "confidence": 0.65,
                "regime_at_entry": "TREND_UP",
                "regime_at_exit": "TREND_UP",
            },
        ]
        records = build_trade_records(episodes)
        assert len(records) == 1
        r = records[0]
        assert r.symbol == "BTCUSDT"
        assert r.confidence == 0.65
        assert r.realized_pnl_usd == 2.0
        assert r.realized_move_pct == pytest.approx(0.002, abs=0.001)

    def test_short_side_move_calculation(self):
        """SHORT move_pct is inverted vs LONG."""
        episodes = [
            {
                "episode_id": "EP_002",
                "symbol": "ETHUSDT",
                "side": "SHORT",
                "strategy": "TREND",
                "entry_ts": "2026-02-25T10:00:00Z",
                "exit_ts": "2026-02-25T10:15:00Z",
                "duration_hours": 0.25,
                "avg_entry_price": 3000.0,
                "avg_exit_price": 2970.0,  # price dropped → SHORT wins
                "entry_notional": 500.0,
                "net_pnl": 5.0,
                "fees": 0.04,
                "confidence": 0.60,
                "regime_at_entry": "TREND_DOWN",
                "regime_at_exit": "TREND_DOWN",
            },
        ]
        records = build_trade_records(episodes)
        r = records[0]
        assert r.realized_move_pct == pytest.approx(0.01, abs=0.001)  # positive for SHORT win

    def test_missing_confidence_defaults_zero(self):
        episodes = [
            {
                "episode_id": "EP_003",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "avg_entry_price": 50000.0,
                "avg_exit_price": 50100.0,
                "entry_notional": 1000.0,
                "net_pnl": 1.0,
            },
        ]
        records = build_trade_records(episodes)
        assert len(records) == 1
        assert records[0].confidence == 0.0

    def test_malformed_episode_skipped(self):
        """Malformed episode should be skipped, not crash."""
        episodes = [
            {"episode_id": "EP_ok", "symbol": "BTCUSDT", "side": "LONG",
             "avg_entry_price": 50000.0, "avg_exit_price": 50100.0,
             "entry_notional": 1000.0, "net_pnl": 1.0},
            "not_a_dict",  # malformed — should be skipped
        ]
        records = build_trade_records(episodes)
        assert len(records) == 1


# ── Join strategy ──────────────────────────────────────────────────────────

class TestJoinStrategy:
    """Fee gate event join uses intent_id (exact) then symbol+side (fallback)."""

    def test_exact_join_by_intent_id(self):
        """When intent_id matches, use that event even if symbol+side also matches."""
        episodes = [
            {
                "episode_id": "EP_100",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "intent_id": "ord_ABC",
                "avg_entry_price": 50000.0,
                "avg_exit_price": 50100.0,
                "entry_notional": 1000.0,
                "net_pnl": 2.0,
                "confidence": 0.65,
            },
        ]
        fee_gate_events = {
            "by_intent": {
                "ord_ABC": {
                    "intent_id": "ord_ABC",
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "true_edge_v1.expected_edge_pct": 0.00045,
                    "true_edge_v1.expected_edge_usd": 0.45,
                    "true_edge_v1.atr_pct": 0.002,
                    "true_edge_v1.k_atr": 0.6,
                    "true_edge_v1.adv": 0.15,
                    "true_edge_v1.confidence": 0.65,
                    "edge_source": "atr_conf_v1",
                },
            },
            "by_symbol_side": {
                "BTCUSDT_LONG": {
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "true_edge_v1.expected_edge_pct": 0.999,  # stale/wrong
                    "edge_source": "atr_conf_v1",
                },
            },
        }
        records = build_trade_records(episodes, fee_gate_events)
        assert len(records) == 1
        r = records[0]
        # Should use exact intent join, not symbol+side fallback
        assert r.pred_expected_edge_pct == pytest.approx(0.00045, abs=1e-6)
        assert r.atr_pct == pytest.approx(0.002, abs=1e-6)
        assert r.edge_source == "atr_conf_v1"

    def test_fallback_to_symbol_side_when_no_intent_id(self):
        """Without intent_id, fall back to symbol+side."""
        episodes = [
            {
                "episode_id": "EP_200",
                "symbol": "ETHUSDT",
                "side": "SHORT",
                "avg_entry_price": 3000.0,
                "avg_exit_price": 2990.0,
                "entry_notional": 500.0,
                "net_pnl": 1.0,
                "confidence": 0.60,
            },
        ]
        fee_gate_events = {
            "by_intent": {},
            "by_symbol_side": {
                "ETHUSDT_SHORT": {
                    "symbol": "ETHUSDT",
                    "side": "SHORT",
                    "true_edge_v1.expected_edge_pct": 0.0003,
                    "true_edge_v1.expected_edge_usd": 0.15,
                    "true_edge_v1.atr_pct": 0.0015,
                    "true_edge_v1.k_atr": 0.6,
                    "true_edge_v1.adv": 0.10,
                    "edge_source": "atr_conf_v1",
                },
            },
        }
        records = build_trade_records(episodes, fee_gate_events)
        r = records[0]
        assert r.pred_expected_edge_pct == pytest.approx(0.0003, abs=1e-6)
        assert r.edge_source == "atr_conf_v1"

    def test_no_matching_event_uses_episode_proxy(self):
        """With no fee gate event match, derive from episode confidence."""
        episodes = [
            {
                "episode_id": "EP_300",
                "symbol": "SOLUSDT",
                "side": "LONG",
                "avg_entry_price": 100.0,
                "avg_exit_price": 101.0,
                "entry_notional": 200.0,
                "net_pnl": 2.0,
                "confidence": 0.70,
            },
        ]
        fee_gate_events = {"by_intent": {}, "by_symbol_side": {}}
        records = build_trade_records(episodes, fee_gate_events)
        r = records[0]
        assert r.edge_source == "episode_confidence_proxy"
        assert r.pred_expected_edge_pct == pytest.approx(0.20, abs=0.01)


# ── Rendering ──────────────────────────────────────────────────────────────

class TestRendering:
    def test_markdown_contains_sections(self):
        report = generate_report(_make_trades_spread(20))
        md = render_markdown(report)
        assert "## A) Calibration by Predicted Edge Decile" in md
        assert "## B) Reliability by Confidence Bucket" in md
        assert "## C) k_atr Tuning Suggestions" in md
        assert "## D) Strategy Breakdown" in md
        assert "## E) Regime Breakdown" in md

    def test_csv_header_and_rows(self):
        trades = _make_trades_spread(5)
        csv = render_csv(trades)
        lines = csv.strip().split("\n")
        assert len(lines) == 6  # header + 5 rows
        assert "episode_id" in lines[0]
        assert "pred_expected_edge_pct" in lines[0]

    def test_csv_deterministic(self):
        trades = _make_trades_spread(5)
        csv1 = render_csv(trades)
        csv2 = render_csv(trades)
        assert csv1 == csv2
