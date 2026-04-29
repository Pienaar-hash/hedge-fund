"""Tests for shadow_validation_report.py — Candidate D operational diagnostics."""
import json
import time


from scripts.shadow_validation_report import (
    MIN_SCORES_SUFFICIENT,
    _section_daily_throughput,
    _section_decision_distribution,
    _section_drift_alerts,
    _section_mask_hits,
    _section_promotion_readiness,
    _section_score_boundary,
    _section_score_drift_metrics,
    _section_zero_score,
    generate_report,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_event(
    *,
    symbol: str = "BTCUSDT",
    d_rule: str = "D_abstain",
    schema: str = "selector_v2_shadow_v3",
    hydra_score: float = 0.45,
    ts: float = 0.0,
    d_zero_score: bool = False,
    d_mask_boundaries: list = None,
    profit_region: bool = False,
) -> dict:
    if ts == 0.0:
        ts = time.time()
    return {
        "ts": ts,
        "schema": schema,
        "symbol": symbol,
        "cycle": 1,
        "hydra_score": hydra_score,
        "legacy_score": 0.40,
        "score_delta": 0.05,
        "hydra_regime_band": "PROFIT_ZONE",
        "profit_region": profit_region,
        "profit_mask_version": "2026-04-07_v2",
        "ecs_conflict": True,
        "ecs_choice": "hydra",
        "d_choice": "hydra" if d_rule == "D_profit_region" else "none",
        "d_abstain": d_rule != "D_profit_region",
        "d_rule": d_rule,
        "d_zero_score": d_zero_score,
        "d_mask_boundaries": d_mask_boundaries or [],
        "d_regime_boundary": 0.5369,
    }


def _batch(n: int, **kwargs) -> list:
    return [_make_event(**kwargs) for _ in range(n)]


# ── Daily throughput ────────────────────────────────────────────────────────

class TestDailyThroughput:
    def test_empty(self):
        result = _section_daily_throughput([])
        assert result["total_events"] == 0
        assert result["rows"] == []

    def test_groups_by_day(self):
        now = time.time()
        yesterday = now - 86400
        events = [
            _make_event(d_rule="D_profit_region", ts=yesterday),
            _make_event(d_rule="D_abstain", ts=yesterday),
            _make_event(d_rule="D_profit_region", ts=now),
        ]
        result = _section_daily_throughput(events)
        assert result["total_days"] == 2
        assert result["total_events"] == 3

    def test_counts_d_rules(self):
        events = [
            _make_event(d_rule="D_profit_region"),
            _make_event(d_rule="D_abstain"),
            _make_event(d_rule="D_zero_score"),
            _make_event(d_rule="D_no_score"),
        ]
        result = _section_daily_throughput(events)
        row = result["rows"][0]
        assert row["D_profit_region"] == 1
        assert row["D_abstain"] == 1
        assert row["D_zero_score"] == 1
        assert row["D_no_score"] == 1


# ── ZERO_SCORE diagnostics ─────────────────────────────────────────────────

class TestZeroScore:
    def test_no_zero_scores(self):
        events = _batch(10, d_rule="D_abstain")
        result = _section_zero_score(events)
        assert result["D_zero_score"] == 0
        assert result["D_no_score"] == 0
        assert result["health"]["pipeline_ok"] is True
        assert result["health"]["model_informative"] is True

    def test_high_zero_rate_degrades(self):
        events = _batch(8, d_rule="D_abstain") + _batch(2, d_rule="D_zero_score")
        result = _section_zero_score(events)
        assert result["D_zero_score"] == 2
        assert result["D_zero_score_rate"] == 0.2
        assert result["health"]["model_informative"] is False  # > 10%

    def test_high_no_score_degrades(self):
        events = _batch(8, d_rule="D_abstain") + _batch(2, d_rule="D_no_score")
        result = _section_zero_score(events)
        assert result["D_no_score"] == 2
        assert result["health"]["pipeline_ok"] is False  # > 5%


# ── Mask hits ───────────────────────────────────────────────────────────────

class TestMaskHits:
    def test_no_d_events(self):
        # v1 events have no d_rule
        events = [{"ts": time.time(), "schema": "v1", "symbol": "BTC"}]
        result = _section_mask_hits(events)
        assert result["status"] == "no_d_events"

    def test_hit_rate_calculation(self):
        events = _batch(3, d_rule="D_profit_region") + _batch(7, d_rule="D_abstain")
        result = _section_mask_hits(events)
        assert result["profit_region_hits"] == 3
        assert result["overall_hit_rate"] == 0.3

    def test_per_symbol_breakdown(self):
        events = [
            _make_event(symbol="BTCUSDT", d_rule="D_profit_region"),
            _make_event(symbol="BTCUSDT", d_rule="D_abstain"),
            _make_event(symbol="SOLUSDT", d_rule="D_abstain"),
        ]
        result = _section_mask_hits(events)
        assert result["by_symbol"]["BTCUSDT"]["hits"] == 1
        assert result["by_symbol"]["BTCUSDT"]["rate"] == 0.5
        assert result["by_symbol"]["SOLUSDT"]["hits"] == 0


# ── Score boundary analysis ─────────────────────────────────────────────────

class TestScoreBoundary:
    def test_scores_inside_mask(self):
        events = [_make_event(hydra_score=0.45) for _ in range(10)]
        result = _section_score_boundary(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["inside_mask"] == 10
        assert btc["inside_rate"] == 1.0
        assert btc["below_mask"] == 0
        assert btc["above_mask"] == 0

    def test_scores_outside_mask(self):
        events = [_make_event(hydra_score=0.55) for _ in range(5)]
        result = _section_score_boundary(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["above_mask"] == 5
        assert btc["inside_mask"] == 0


# ── Score drift metrics ─────────────────────────────────────────────────────

class TestScoreDriftMetrics:
    def test_distance_to_midpoint_high_bias(self):
        # Mask midpoint = (0.4197 + 0.4953) / 2 = 0.4575
        events = [_make_event(hydra_score=0.49) for _ in range(10)]
        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["distance_mean"] > 0  # above midpoint
        assert btc["bias_direction"] == "high"

    def test_distance_to_midpoint_centered(self):
        events = [_make_event(hydra_score=0.4575) for _ in range(10)]
        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert abs(btc["distance_mean"]) < 0.005
        assert btc["bias_direction"] == "centered"

    def test_near_miss_rate(self):
        # Mask hi = 0.4953, near-miss band = (0.4953, 0.5253]
        events = [
            _make_event(hydra_score=0.50),  # near miss
            _make_event(hydra_score=0.51),  # near miss
            _make_event(hydra_score=0.45),  # inside mask
            _make_event(hydra_score=0.45),  # inside mask
        ]
        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["near_miss_count"] == 2
        assert btc["near_miss_rate"] == 0.5

    def test_spillover_pressure_normal(self):
        # 2 inside mask, 0 near-miss → spillover = 0
        events = [_make_event(hydra_score=0.45) for _ in range(4)]
        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["spillover_pressure"] == 0.0
        assert btc["spillover_severity"] == "normal"

    def test_spillover_pressure_warning(self):
        # 2 inside, 1 near-miss → 0.5
        events = [
            _make_event(hydra_score=0.45),
            _make_event(hydra_score=0.45),
            _make_event(hydra_score=0.50),  # near miss
        ]
        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["spillover_pressure"] == 0.5
        assert btc["spillover_severity"] == "warning"

    def test_spillover_pressure_critical(self):
        # 1 inside, 3 near-miss → 3.0
        events = [
            _make_event(hydra_score=0.45),  # inside
            _make_event(hydra_score=0.50),  # near miss
            _make_event(hydra_score=0.51),  # near miss
            _make_event(hydra_score=0.52),  # near miss
        ]
        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["spillover_pressure"] == 3.0
        assert btc["spillover_severity"] == "critical"

    def test_spillover_velocity_positive(self):
        """Velocity = latest_day_spillover - avg(prior 7 days)."""
        base = 1_700_000_000  # fixed epoch
        DAY = 86400
        events = []
        # Days 0-6: each day has 2 inside, 0 near-miss → spillover = 0.0
        for d in range(7):
            ts = base + d * DAY
            events.append(_make_event(hydra_score=0.45, ts=ts))
            events.append(_make_event(hydra_score=0.46, ts=ts))
        # Day 7 (latest): 1 inside, 1 near-miss → spillover = 1.0
        ts7 = base + 7 * DAY
        events.append(_make_event(hydra_score=0.45, ts=ts7))
        events.append(_make_event(hydra_score=0.50, ts=ts7))  # near-miss

        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        # velocity = 1.0 - avg(0,0,0,0,0,0,0) = 1.0
        assert btc["spillover_velocity"] == 1.0
        assert btc["spillover_momentum"] is not None

    def test_spillover_momentum_3d(self):
        """Momentum = latest_day_spillover - avg(prior 3 days)."""
        base = 1_700_000_000
        DAY = 86400
        events = []
        # Days 0-2: each day has 2 inside, 1 near-miss → spillover = 0.5
        for d in range(3):
            ts = base + d * DAY
            events.append(_make_event(hydra_score=0.45, ts=ts))
            events.append(_make_event(hydra_score=0.46, ts=ts))
            events.append(_make_event(hydra_score=0.50, ts=ts))  # near-miss
        # Day 3 (latest): 1 inside, 2 near-miss → spillover = 2.0
        ts3 = base + 3 * DAY
        events.append(_make_event(hydra_score=0.45, ts=ts3))
        events.append(_make_event(hydra_score=0.50, ts=ts3))
        events.append(_make_event(hydra_score=0.51, ts=ts3))

        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        # momentum = 2.0 - avg(0.5, 0.5, 0.5) = 1.5
        assert btc["spillover_momentum"] == 1.5

    def test_spillover_velocity_none_single_day(self):
        """With only 1 day of data, velocity/momentum should be None."""
        events = [_make_event(hydra_score=0.45), _make_event(hydra_score=0.50)]
        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["spillover_velocity"] is None
        assert btc["spillover_momentum"] is None

    def test_daily_spillover_in_output(self):
        """daily_spillover should appear and be capped to last 7 days."""
        base = 1_700_000_000
        DAY = 86400
        events = []
        # 10 days of data
        for d in range(10):
            ts = base + d * DAY
            events.append(_make_event(hydra_score=0.45, ts=ts))
            events.append(_make_event(hydra_score=0.46, ts=ts))
        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert "daily_spillover" in btc
        assert len(btc["daily_spillover"]) <= 7

    def test_histogram_bins(self):
        events = [
            _make_event(hydra_score=0.40),  # <0.42
            _make_event(hydra_score=0.43),  # 0.42-0.45
            _make_event(hydra_score=0.46),  # 0.45-0.4953 (mask)
            _make_event(hydra_score=0.50),  # 0.4953-0.53
            _make_event(hydra_score=0.55),  # >0.53
        ]
        result = _section_score_drift_metrics(events)
        hist = result["by_symbol"]["BTCUSDT"]["histogram"]
        assert hist["<0.42"] == 1
        assert hist["0.42\u20130.45"] == 1
        assert hist["0.45\u20130.4953"] == 1
        assert hist["0.4953\u20130.53"] == 1
        assert hist[">0.53"] == 1

    def test_no_data(self):
        events = [_make_event(symbol="SOLUSDT")]  # no REFERENCE_MASK entry
        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["status"] == "no_data"

    def test_data_sufficiency_insufficient(self):
        """Below MIN_SCORES_SUFFICIENT → INSUFFICIENT_DATA."""
        events = [_make_event(hydra_score=0.45) for _ in range(MIN_SCORES_SUFFICIENT - 1)]
        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["data_sufficiency"] == "INSUFFICIENT_DATA"

    def test_data_sufficiency_ok(self):
        """At MIN_SCORES_SUFFICIENT → OK."""
        events = [_make_event(hydra_score=0.45) for _ in range(MIN_SCORES_SUFFICIENT)]
        result = _section_score_drift_metrics(events)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["data_sufficiency"] == "OK"


# ── Drift alerts ────────────────────────────────────────────────────────────

class TestDriftAlerts:
    def test_no_alerts_on_healthy_data(self):
        now = time.time()
        events = [
            _make_event(
                symbol="BTCUSDT",
                d_rule="D_profit_region",
                ts=now - i * 3600,
                d_mask_boundaries=[[0.4197, 0.4953]],
            )
            for i in range(40)
        ]
        result = _section_drift_alerts(events)
        assert result["kill_conditions_active"] is False

    def test_low_region_count_kills(self):
        now = time.time()
        events = [
            _make_event(
                symbol="BTCUSDT",
                d_rule="D_abstain",
                ts=now - i * 3600,
            )
            for i in range(50)
        ]
        result = _section_drift_alerts(events)
        kill_alerts = [a for a in result["alerts"] if a["severity"] == "KILL"]
        assert len(kill_alerts) >= 1
        assert any(a["type"] == "LOW_REGION_COUNT" for a in kill_alerts)

    def test_boundary_drift_detected(self):
        events = [
            _make_event(
                symbol="BTCUSDT",
                d_mask_boundaries=[[0.38, 0.48]],  # lo drifted by 0.04
            )
        ]
        result = _section_drift_alerts(events)
        assert result["boundary_drift_detected"] is True
        drift_alerts = [a for a in result["alerts"] if a["type"] == "REGION_DRIFT"]
        assert len(drift_alerts) == 1


# ── Promotion readiness ────────────────────────────────────────────────────

class TestPromotionReadiness:
    def test_not_ready_on_fresh_start(self):
        events = _batch(5, schema="selector_v2_shadow_v3")
        result = _section_promotion_readiness(events)
        assert result["promote_ready"] is False
        assert result["checks"]["soak_days"]["pass"] is False
        assert result["checks"]["v3_episode_count"]["pass"] is False

    def test_only_v3_counts(self):
        v1 = _batch(200, schema="selector_v2_shadow_v1")
        v3 = _batch(3, schema="selector_v2_shadow_v3")
        result = _section_promotion_readiness(v1 + v3)
        assert result["checks"]["v3_episode_count"]["value"] == 3
        assert result["checks"]["schema_consistency"]["v1_count"] == 200
        assert result["checks"]["schema_consistency"]["v3_count"] == 3


# ── Decision distribution ──────────────────────────────────────────────────

class TestDecisionDistribution:
    def test_distribution_percentages(self):
        events = _batch(7, d_rule="D_abstain") + _batch(3, d_rule="D_profit_region")
        result = _section_decision_distribution(events)
        assert result["total"] == 10
        assert result["distribution"]["D_abstain"]["count"] == 7
        assert result["distribution"]["D_abstain"]["pct"] == 70.0

    def test_per_symbol(self):
        events = [
            _make_event(symbol="BTCUSDT", d_rule="D_profit_region"),
            _make_event(symbol="SOLUSDT", d_rule="D_abstain"),
        ]
        result = _section_decision_distribution(events)
        assert result["by_symbol"]["BTCUSDT"]["D_profit_region"] == 1
        assert result["by_symbol"]["SOLUSDT"]["D_abstain"] == 1


# ── Integration: generate_report ────────────────────────────────────────────

class TestGenerateReport:
    def test_report_structure(self, monkeypatch, tmp_path):
        log_file = tmp_path / "shadow.jsonl"
        events = _batch(5, d_rule="D_profit_region") + _batch(5, d_rule="D_abstain")
        with open(log_file, "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        monkeypatch.setattr(
            "scripts.shadow_validation_report.V2_SHADOW_LOG", log_file
        )
        report = generate_report(json_output=False)
        assert "sections" in report
        assert "generated_at" in report
        assert report["event_count"] == 10
        assert len(report["sections"]) == 8

    def test_days_filter(self, monkeypatch, tmp_path):
        log_file = tmp_path / "shadow.jsonl"
        now = time.time()
        old = [_make_event(ts=now - 86400 * 10)]  # 10 days ago
        recent = [_make_event(ts=now - 3600)]  # 1 hour ago
        with open(log_file, "w") as f:
            for ev in old + recent:
                f.write(json.dumps(ev) + "\n")

        monkeypatch.setattr(
            "scripts.shadow_validation_report.V2_SHADOW_LOG", log_file
        )
        report = generate_report(days=2, json_output=False)
        assert report["event_count"] == 1
