"""Tests for shadow_pnl_join.py — Candidate D PnL join and counterfactual analysis."""
import json
import time


from scripts.shadow_pnl_join import (
    MIN_EPISODES_SUFFICIENT,
    NEAR_MISS_BAND,
    REFERENCE_MASK,
    WIDEN_STEPS,
    build_scored_episodes,
    classify_region,
    enrich_with_shadow,
    generate_report,
    section_counterfactual_widening,
    section_join_quality,
    section_lost_ev,
    section_near_miss_comparison,
    section_region_pnl,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

BTC_LO = REFERENCE_MASK["BTCUSDT"]["lo"]  # 0.4197
BTC_HI = REFERENCE_MASK["BTCUSDT"]["hi"]  # 0.4953


def _make_episode(
    *,
    symbol: str = "BTCUSDT",
    hybrid_score: float = 0.45,
    net_pnl: float = 1.0,
    entry_ts: str = "2026-03-20T10:00:00+00:00",
    exit_ts: str = "2026-03-20T12:00:00+00:00",
    episode_id: str = "EP_TEST",
    side: str = "LONG",
    intent_id: str = "ord_test",
) -> dict:
    return {
        "episode_id": episode_id,
        "symbol": symbol,
        "side": side,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "hybrid_score": hybrid_score,
        "net_pnl": net_pnl,
        "intent_id": intent_id,
    }


def _make_shadow(
    *,
    symbol: str = "BTCUSDT",
    ts: float = 0.0,
    d_rule: str = "D_profit_region",
    d_choice: str = "hydra",
    schema: str = "selector_v2_shadow_v3",
) -> dict:
    if ts == 0.0:
        ts = time.time()
    return {
        "ts": ts,
        "schema": schema,
        "symbol": symbol,
        "d_rule": d_rule,
        "d_choice": d_choice,
        "hydra_score": 0.45,
    }


# ── classify_region ─────────────────────────────────────────────────────────

class TestClassifyRegion:
    def test_inside_mask(self):
        assert classify_region("BTCUSDT", 0.45) == "mask_interior"

    def test_at_lower_boundary(self):
        assert classify_region("BTCUSDT", BTC_LO) == "mask_interior"

    def test_at_upper_boundary(self):
        assert classify_region("BTCUSDT", BTC_HI) == "mask_interior"

    def test_near_miss_above(self):
        assert classify_region("BTCUSDT", BTC_HI + 0.01) == "near_miss"

    def test_near_miss_upper_bound(self):
        assert classify_region("BTCUSDT", BTC_HI + NEAR_MISS_BAND) == "near_miss"

    def test_outside_above(self):
        assert classify_region("BTCUSDT", BTC_HI + NEAR_MISS_BAND + 0.001) == "outside"

    def test_outside_below(self):
        assert classify_region("BTCUSDT", BTC_LO - 0.01) == "outside"

    def test_no_mask_symbol(self):
        assert classify_region("SOLUSDT", 0.45) == "no_mask"


# ── build_scored_episodes ───────────────────────────────────────────────────

class TestBuildScoredEpisodes:
    def test_filters_zero_score(self):
        episodes = [_make_episode(hybrid_score=0.0)]
        assert build_scored_episodes(episodes) == []

    def test_filters_wrong_symbol(self):
        episodes = [_make_episode(symbol="DOGEUSDT")]
        assert build_scored_episodes(episodes) == []

    def test_classifies_region(self):
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=1.0),
            _make_episode(hybrid_score=0.50, net_pnl=-0.5),
            _make_episode(hybrid_score=0.55, net_pnl=0.2),
        ]
        scored = build_scored_episodes(episodes)
        assert len(scored) == 3
        assert scored[0]["region"] == "mask_interior"
        assert scored[1]["region"] == "near_miss"
        assert scored[2]["region"] == "outside"

    def test_symbol_filter(self):
        episodes = [
            _make_episode(symbol="BTCUSDT"),
            _make_episode(symbol="ETHUSDT"),
        ]
        scored = build_scored_episodes(episodes, symbols={"ETHUSDT"})
        # ETH has no REFERENCE_MASK entry but symbol filter still applies
        assert all(ep["symbol"] == "ETHUSDT" for ep in scored)


# ── enrich_with_shadow ──────────────────────────────────────────────────────

class TestEnrichWithShadow:
    def test_matches_within_window(self):
        ep_ts = 1700000000.0
        scored = [{
            "symbol": "BTCUSDT", "entry_ts": ep_ts,
            "hybrid_score": 0.45, "net_pnl": 1.0, "region": "mask_interior",
        }]
        shadows = [_make_shadow(ts=ep_ts + 5)]
        matched, unmatched = enrich_with_shadow(scored, shadows)
        assert matched == 1
        assert unmatched == 0
        assert scored[0]["shadow_linked"] is True
        assert scored[0]["link_distance_s"] == 5.0

    def test_no_match_outside_window(self):
        ep_ts = 1700000000.0
        scored = [{
            "symbol": "BTCUSDT", "entry_ts": ep_ts,
            "hybrid_score": 0.45, "net_pnl": 1.0, "region": "mask_interior",
        }]
        shadows = [_make_shadow(ts=ep_ts + 200)]
        matched, unmatched = enrich_with_shadow(scored, shadows)
        assert matched == 0
        assert unmatched == 1
        assert scored[0]["shadow_linked"] is False

    def test_empty_shadows(self):
        scored = [{
            "symbol": "BTCUSDT", "entry_ts": 1700000000.0,
            "hybrid_score": 0.45, "net_pnl": 1.0, "region": "mask_interior",
        }]
        matched, unmatched = enrich_with_shadow(scored, [])
        assert matched == 0
        assert unmatched == 1


# ── section_region_pnl ─────────────────────────────────────────────────────

class TestSectionRegionPnl:
    def test_basic_stats(self):
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=2.0),
            _make_episode(hybrid_score=0.45, net_pnl=-1.0),
            _make_episode(hybrid_score=0.50, net_pnl=0.5),
        ]
        scored = build_scored_episodes(episodes)
        result = section_region_pnl(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        mask = btc["regions"]["mask_interior"]
        assert mask["count"] == 2
        assert mask["mean"] == 0.5  # (2 + -1) / 2
        nm = btc["regions"]["near_miss"]
        assert nm["count"] == 1

    def test_data_sufficiency_insufficient(self):
        episodes = [_make_episode(hybrid_score=0.45, net_pnl=1.0)]
        scored = build_scored_episodes(episodes)
        result = section_region_pnl(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["data_sufficiency"] == "INSUFFICIENT_DATA"

    def test_data_sufficiency_ok(self):
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=float(i))
            for i in range(MIN_EPISODES_SUFFICIENT)
        ]
        scored = build_scored_episodes(episodes)
        result = section_region_pnl(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["data_sufficiency"] == "OK"


# ── section_near_miss_comparison ────────────────────────────────────────────

class TestSectionNearMissComparison:
    def test_mask_correct(self):
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=5.0),   # mask, positive
            _make_episode(hybrid_score=0.45, net_pnl=3.0),   # mask, positive
            _make_episode(hybrid_score=0.50, net_pnl=0.5),   # near-miss
            _make_episode(hybrid_score=0.51, net_pnl=-0.5),  # near-miss
        ]
        scored = build_scored_episodes(episodes)
        result = section_near_miss_comparison(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["verdict"] == "mask_correct"
        assert btc["ev_delta"] is not None
        assert btc["ev_delta"] < 0  # near_miss EV < mask EV

    def test_mask_too_tight(self):
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=-1.0),  # mask, negative
            _make_episode(hybrid_score=0.50, net_pnl=5.0),   # near-miss, positive
            _make_episode(hybrid_score=0.51, net_pnl=3.0),   # near-miss, positive
        ]
        scored = build_scored_episodes(episodes)
        result = section_near_miss_comparison(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["verdict"] == "mask_too_tight"

    def test_thesis_degrading(self):
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=-2.0),  # mask, negative
            _make_episode(hybrid_score=0.50, net_pnl=-1.0),  # near-miss, negative
        ]
        scored = build_scored_episodes(episodes)
        result = section_near_miss_comparison(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["verdict"] == "thesis_degrading"

    def test_insufficient_data(self):
        episodes = [_make_episode(hybrid_score=0.45, net_pnl=1.0)]
        scored = build_scored_episodes(episodes)
        result = section_near_miss_comparison(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["verdict"] == "insufficient_data"


# ── section_lost_ev ─────────────────────────────────────────────────────────

class TestSectionLostEv:
    def test_beneficial_abstention(self):
        """Near-miss trades were net negative → abstention saves money."""
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=1.0),  # mask (needed for scored)
            _make_episode(hybrid_score=0.50, net_pnl=-3.0),  # near-miss losing
            _make_episode(hybrid_score=0.51, net_pnl=-1.0),  # near-miss losing
        ]
        scored = build_scored_episodes(episodes)
        result = section_lost_ev(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["abstention_verdict"] == "beneficial"
        assert btc["net_lost_ev"] < 0

    def test_costly_abstention(self):
        """Near-miss trades were net positive → abstention costs money."""
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=1.0),
            _make_episode(hybrid_score=0.50, net_pnl=5.0),
            _make_episode(hybrid_score=0.51, net_pnl=3.0),
        ]
        scored = build_scored_episodes(episodes)
        result = section_lost_ev(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["abstention_verdict"] == "costly"
        assert btc["net_lost_ev"] > 0
        assert btc["lost_gains"] == 8.0

    def test_no_near_miss(self):
        episodes = [_make_episode(hybrid_score=0.45, net_pnl=1.0)]
        scored = build_scored_episodes(episodes)
        result = section_lost_ev(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["status"] == "no_near_miss_episodes"

    def test_mixed_near_miss(self):
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=1.0),
            _make_episode(hybrid_score=0.50, net_pnl=2.0),   # near-miss win
            _make_episode(hybrid_score=0.51, net_pnl=-5.0),  # near-miss loss
        ]
        scored = build_scored_episodes(episodes)
        result = section_lost_ev(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["profitable_abstained"] == 1
        assert btc["losing_abstained"] == 1
        assert btc["lost_gains"] == 2.0
        assert btc["avoided_losses"] == 5.0


# ── section_counterfactual_widening ─────────────────────────────────────────

class TestSectionCounterfactualWidening:
    def test_widening_captures_near_miss(self):
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=1.0),   # mask
            _make_episode(hybrid_score=0.50, net_pnl=2.0),   # near-miss (hi+0.005)
        ]
        scored = build_scored_episodes(episodes)
        result = section_counterfactual_widening(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        # +0.01 widen should capture score=0.50 (new_hi=0.5053)
        first = btc["scenarios"][0]
        assert first["widen_by"] == WIDEN_STEPS[0]
        assert first["episodes_added"] == 1

    def test_current_stats_correct(self):
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=3.0),
            _make_episode(hybrid_score=0.46, net_pnl=1.0),
        ]
        scored = build_scored_episodes(episodes)
        result = section_counterfactual_widening(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        assert btc["current_stats"]["count"] == 2
        assert btc["current_stats"]["mean"] == 2.0

    def test_widen_steps_count(self):
        episodes = [_make_episode(hybrid_score=0.45, net_pnl=1.0)]
        scored = build_scored_episodes(episodes)
        result = section_counterfactual_widening(scored)
        btc = result["by_symbol"]["BTCUSDT"]
        assert len(btc["scenarios"]) == len(WIDEN_STEPS)


# ── section_join_quality ────────────────────────────────────────────────────

class TestSectionJoinQuality:
    def test_high_linkage(self):
        scored = [
            {"symbol": "BTCUSDT", "shadow_linked": True} for _ in range(10)
        ]
        result = section_join_quality(scored, matched=10, unmatched=0)
        assert result["linkage_tier"] == "HIGH"
        assert result["match_rate"] == 1.0

    def test_none_linkage(self):
        scored = [
            {"symbol": "BTCUSDT", "shadow_linked": False} for _ in range(5)
        ]
        result = section_join_quality(scored, matched=0, unmatched=5)
        assert result["linkage_tier"] == "NONE"

    def test_moderate_linkage(self):
        scored = [
            {"symbol": "BTCUSDT", "shadow_linked": True} for _ in range(6)
        ] + [
            {"symbol": "BTCUSDT", "shadow_linked": False} for _ in range(4)
        ]
        result = section_join_quality(scored, matched=6, unmatched=4)
        assert result["linkage_tier"] == "MODERATE"

    def test_per_symbol_sufficiency(self):
        scored = [{"symbol": "BTCUSDT", "shadow_linked": False} for _ in range(3)]
        result = section_join_quality(scored, matched=0, unmatched=3)
        assert result["by_symbol"]["BTCUSDT"]["data_sufficiency"] == "INSUFFICIENT_DATA"


# ── generate_report ─────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_empty_ledger(self, tmp_path):
        ep_file = tmp_path / "episodes.json"
        ep_file.write_text(json.dumps({"episodes": []}))
        shadow_file = tmp_path / "shadow.jsonl"
        shadow_file.write_text("")
        result = generate_report(
            episode_path=ep_file,
            shadow_path=shadow_file,
        )
        assert result["total_scored_episodes"] == 0
        assert "Region PnL Table" in result["sections"]

    def test_full_report_structure(self, tmp_path):
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=2.0),
            _make_episode(hybrid_score=0.50, net_pnl=-0.5),
            _make_episode(hybrid_score=0.55, net_pnl=0.1),
        ]
        ep_file = tmp_path / "episodes.json"
        ep_file.write_text(json.dumps({"episodes": episodes}))
        shadow_file = tmp_path / "shadow.jsonl"
        shadow_file.write_text("")
        result = generate_report(
            episode_path=ep_file,
            shadow_path=shadow_file,
        )
        assert result["total_scored_episodes"] == 3
        titles = set(result["sections"].keys())
        assert titles == {
            "Region PnL Table",
            "Near-Miss vs Mask Comparison",
            "Lost EV Estimate",
            "Counterfactual Widening",
            "Join Quality",
        }
