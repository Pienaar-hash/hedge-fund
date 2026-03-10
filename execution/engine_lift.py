"""
Engine Lift — Outcome-based Hydra vs Legacy selector comparison.

Answers: "When Hydra and Legacy disagree, who picks better trades?"

Computes:
    Outcome CEL = mean(return | hydra_only) - mean(return | legacy_only)
    Participation Lift = hydra_trade_count / legacy_trade_count

Data sources:
    - episode_ledger.json  (episodes with strategy field)
    - orders_attempted.jsonl (intent_id join for engine attribution)

Engine classification:
    - "hydra" source/strategy → hydra episode
    - "vol_target", "legacy", or legacy-pattern strategies → legacy episode
    - Exits, unknowns → excluded from CEL

State output: logs/state/engine_lift.json
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

_STATE_PATH = os.path.join("logs", "state", "engine_lift.json")
_ATTEMPTS_PATH = os.path.join("logs", "execution", "orders_attempted.jsonl")

# Strategy names that map to "legacy" engine
_LEGACY_STRATEGIES = frozenset({"vol_target", "legacy", "btc_m15", "sol_m15", "eth_m15"})
# Strategy names that map to "hydra" engine
_HYDRA_STRATEGIES = frozenset({"hydra"})
# Excluded from CEL (not entry trades)
_EXCLUDED_STRATEGIES = frozenset({"doctrine_exit", "auto_reduce", "vol_target_exit", "unknown", ""})


def _safe_float(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        v = float(val)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _classify_engine(episode: Dict[str, Any], attempt_lookup: Dict[str, Dict]) -> Optional[str]:
    """Classify episode as 'hydra' or 'legacy' or None (unclassifiable)."""
    # 0. Direct engine_source from fill metadata (v7.9 forward)
    src = str(episode.get("engine_source", "") or "").lower()
    if src in _HYDRA_STRATEGIES:
        return "hydra"
    if src in _LEGACY_STRATEGIES:
        return "legacy"

    # 1. Try join via intent_id to get authoritative source
    intent_id = episode.get("intent_id", "")
    if intent_id and intent_id in attempt_lookup:
        attempt = attempt_lookup[intent_id]
        strat = str(attempt.get("strategy", "") or "").lower()
        if strat in _HYDRA_STRATEGIES:
            return "hydra"
        if strat in _LEGACY_STRATEGIES:
            return "legacy"

    # 2. Fall back to episode strategy field
    strat = str(episode.get("strategy", "") or "").lower()
    if strat in _HYDRA_STRATEGIES:
        return "hydra"
    if strat in _LEGACY_STRATEGIES:
        return "legacy"
    if strat in _EXCLUDED_STRATEGIES:
        return None
    return None


def _realized_return(episode: Dict[str, Any]) -> Optional[float]:
    """Compute side-adjusted realized return for an episode."""
    entry = _safe_float(episode.get("avg_entry_price"))
    exit_ = _safe_float(episode.get("avg_exit_price"))
    if entry <= 0 or exit_ <= 0:
        return None
    side = str(episode.get("side", "")).upper()
    if side == "LONG":
        return (exit_ - entry) / entry
    elif side == "SHORT":
        return (entry - exit_) / entry
    return None


def _load_attempt_lookup() -> Dict[str, Dict]:
    """Build intent_id → attempt dict for engine join."""
    lookup: Dict[str, Dict] = {}
    try:
        with open(_ATTEMPTS_PATH) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    iid = d.get("intent_id", "")
                    if iid:
                        lookup[iid] = d
                except (json.JSONDecodeError, TypeError):
                    continue
    except OSError:
        pass
    return lookup


def compute_engine_lift(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute outcome-based CEL and participation metrics.

    Returns dict with:
        outcome_cel, hydra_mean, legacy_mean, hydra_count, legacy_count,
        participation_lift, hydra_pnl_sum, legacy_pnl_sum, ts
    """
    attempt_lookup = _load_attempt_lookup()

    hydra_returns: List[float] = []
    legacy_returns: List[float] = []
    hydra_pnl_sum = 0.0
    legacy_pnl_sum = 0.0

    for ep in episodes:
        engine = _classify_engine(ep, attempt_lookup)
        if engine is None:
            continue

        ret = _realized_return(ep)
        if ret is None:
            continue

        net_pnl = _safe_float(ep.get("net_pnl"))

        if engine == "hydra":
            hydra_returns.append(ret)
            hydra_pnl_sum += net_pnl
        elif engine == "legacy":
            legacy_returns.append(ret)
            legacy_pnl_sum += net_pnl

    h_mean = sum(hydra_returns) / len(hydra_returns) if hydra_returns else 0.0
    l_mean = sum(legacy_returns) / len(legacy_returns) if legacy_returns else 0.0
    outcome_cel = h_mean - l_mean if (hydra_returns and legacy_returns) else None
    participation = (
        len(hydra_returns) / len(legacy_returns)
        if legacy_returns
        else None
    )

    return {
        "outcome_cel": round(outcome_cel, 6) if outcome_cel is not None else None,
        "hydra_mean_return": round(h_mean, 6),
        "legacy_mean_return": round(l_mean, 6),
        "hydra_count": len(hydra_returns),
        "legacy_count": len(legacy_returns),
        "participation_lift": round(participation, 4) if participation is not None else None,
        "hydra_pnl_sum": round(hydra_pnl_sum, 4),
        "legacy_pnl_sum": round(legacy_pnl_sum, 4),
        "ts": time.time(),
    }


def persist_snapshot(
    episodes: List[Dict[str, Any]],
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute engine lift and write to state file for dashboard."""
    dest = path or _STATE_PATH
    snap = compute_engine_lift(episodes)

    try:
        tmp = dest + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(snap, fh, indent=2)
        os.replace(tmp, dest)
    except OSError as exc:
        LOG.debug("[engine_lift] persist failed: %s", exc)

    return snap
