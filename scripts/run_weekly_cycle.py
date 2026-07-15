#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from execution.weekly_audit_ledger import append_decision_record, verify_decision_chain
from execution.weekly_execution_planner import build_weekly_plan
from execution.weekly_market_data import (
    build_weekly_input_manifest,
    load_completed_weekly_closes,
    validate_weekly_history,
)
from execution.weekly_momentum_engine import (
    classify_conviction,
    compute_weekly_features,
    rank_weekly_features,
    select_candidates,
)
from execution.weekly_position_sizer import compute_target_positions
from execution.weekly_risk_gate import classify_weekly_risk_mode, validate_portfolio_targets


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DEFAULT_LEDGER = ROOT / "logs" / "weekly_audit_ledger.jsonl"
LEGACY_FLAG_TOKENS = {
    "hydra_execution",
    "ml",
    "sentinel",
    "sentinel_x",
    "mean_reversion",
    "vol_target",
    "crossfire",
    "alpha_miner",
    "intraday",
    "signals",
    "strategies",
    "dynamic_leverage",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _reject_legacy_flags(extra_paths: list[Path] | None = None) -> None:
    for path in extra_paths or []:
        if not path.exists():
            continue
        payload = _load_json(path)
        if any(token in payload for token in LEGACY_FLAG_TOKENS):
            raise SystemExit(f"legacy strategy flags present in {path}")


def _build_cycle_id(decision_ts: datetime) -> str:
    iso_year, iso_week, _ = decision_ts.isocalendar()
    return f"WM_{iso_year}_{iso_week:02d}"


def _load_price_history(price_path: Path) -> dict[str, Any]:
    payload = _load_json(price_path)
    if not isinstance(payload, dict):
        raise SystemExit("price history file must be a symbol map")
    return payload


def _make_decision(
    *,
    nav_usd: float,
    price_history: dict[str, Any],
    weekly_config: dict[str, Any],
    risk_config: dict[str, Any],
    universe_config: dict[str, Any],
    decision_ts: datetime,
    current_positions: list[dict[str, Any]],
    weekly_loss_pct: float,
    drawdown_pct: float,
) -> dict[str, Any]:
    completed_week_end = decision_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    weekly_closes = load_completed_weekly_closes(price_history, as_of=completed_week_end)
    validated = validate_weekly_history(weekly_closes)
    manifest = build_weekly_input_manifest(
        validated,
        price_source=str(universe_config["price_source"]),
        completed_week_end=completed_week_end.isoformat(),
    )
    features = compute_weekly_features(validated)
    classified = classify_conviction(rank_weekly_features(features))
    groups = {
        item["symbol"]: item["correlation_group"]
        for item in universe_config["symbols"]
    }
    for item in classified:
        item["correlation_group"] = groups.get(item["symbol"])
    selected = select_candidates(classified, groups)
    precision = {
        item["symbol"]: {
            "notional_precision": item.get("notional_precision", 2),
            "min_notional": item.get("min_notional_usd", 0.0),
        }
        for item in universe_config["symbols"]
    }
    target_positions = compute_target_positions(
        selected,
        nav_usd=nav_usd,
        minimum_position_pct=float(weekly_config["sizing"]["minimum_position_pct"]),
        absolute_position_cap_pct=float(weekly_config["sizing"]["maximum_position_pct"]),
        maximum_gross_exposure_pct=float(weekly_config["sizing"]["maximum_gross_exposure_pct"]),
        max_loss_pct_per_position=float(weekly_config["sizing"]["max_loss_pct_per_position"]),
        hard_stop_pct=float(weekly_config["sizing"]["hard_stop_pct"]),
        symbol_precision=precision,
    )
    risk_mode = classify_weekly_risk_mode(
        weekly_loss_pct=weekly_loss_pct,
        drawdown_pct=drawdown_pct,
        data_valid=bool(validated),
        audit_state_valid=True,
        maximum_weekly_loss_pct=float(risk_config["maximum_weekly_loss_pct"]),
        maximum_drawdown_pct=float(risk_config["maximum_drawdown_pct"]),
    )
    portfolio_ok, portfolio_reasons = validate_portfolio_targets(
        target_positions,
        maximum_gross_exposure_pct=float(weekly_config["sizing"]["maximum_gross_exposure_pct"]),
        maximum_concurrent_position_risk_pct=float(risk_config["maximum_concurrent_position_risk_pct"]),
    )
    plan = build_weekly_plan(
        cycle_id=_build_cycle_id(decision_ts),
        risk_mode=risk_mode,
        current_positions=current_positions,
        target_positions=target_positions if portfolio_ok and risk_mode == "ACTIVE" else [],
        rebalance_threshold_pct=float(weekly_config["sizing"]["rebalance_threshold_pct"]),
    )
    rejected = [
        item for item in classified
        if item["symbol"] not in {candidate["symbol"] for candidate in selected}
    ]
    return {
        "schema_version": "weekly_momentum_v1",
        "cycle_id": _build_cycle_id(decision_ts),
        "decision_ts": decision_ts.isoformat(),
        "completed_week_end": completed_week_end.isoformat(),
        "engine_commit": "workspace",
        "module_sha256": "workspace",
        "config_sha256": "workspace",
        "universe_sha256": "workspace",
        "price_source": universe_config["price_source"],
        "input_manifest_sha256": manifest["input_manifest_sha256"],
        "nav_usd": nav_usd,
        "risk_mode": risk_mode,
        "candidates": classified,
        "selected": selected,
        "rejected": rejected,
        "existing_positions": current_positions,
        "target_positions": plan["target_positions"],
        "orders": plan["orders"],
        "risk_checks": [
            {"name": "portfolio_targets", "ok": portfolio_ok, "reasons": portfolio_reasons},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-authorized", action="store_true")
    parser.add_argument("--cycle-id")
    parser.add_argument("--ledger-path", default=str(DEFAULT_LEDGER))
    parser.add_argument("--prices-path", required=True)
    parser.add_argument("--nav-usd", type=float, default=10000.0)
    parser.add_argument("--weekly-loss-pct", type=float, default=0.0)
    parser.add_argument("--drawdown-pct", type=float, default=0.0)
    parser.add_argument("--current-positions-path")
    parser.add_argument(
        "--legacy-config-path",
        action="append",
        default=[],
        help="Optional JSON config path that must not contain legacy strategy flags.",
    )
    args = parser.parse_args()

    if not (args.validate_only or args.plan_only or args.execute):
        args.validate_only = True
    if args.execute and not args.execution_authorized:
        raise SystemExit("--execute requires --execution-authorized")

    decision_ts = datetime.now(UTC).replace(microsecond=0)
    cycle_id = args.cycle_id or _build_cycle_id(decision_ts)
    _reject_legacy_flags([Path(path) for path in args.legacy_config_path])
    ledger_ok, ledger_errors = verify_decision_chain(args.ledger_path)
    if not ledger_ok:
        raise SystemExit("; ".join(ledger_errors))

    weekly_config = _load_json(CONFIG_DIR / "weekly_momentum.json")
    risk_config = _load_json(CONFIG_DIR / "weekly_risk_limits.json")["risk"]
    universe_config = _load_json(CONFIG_DIR / "weekly_universe.json")
    current_positions = []
    if args.current_positions_path:
        current_positions = list(_load_json(Path(args.current_positions_path)))
    decision = _make_decision(
        nav_usd=args.nav_usd,
        price_history=_load_price_history(Path(args.prices_path)),
        weekly_config=weekly_config,
        risk_config=risk_config,
        universe_config=universe_config,
        decision_ts=decision_ts,
        current_positions=current_positions,
        weekly_loss_pct=args.weekly_loss_pct,
        drawdown_pct=args.drawdown_pct,
    )
    if decision["cycle_id"] != cycle_id:
        raise SystemExit(f"cycle id mismatch: expected {cycle_id}, computed {decision['cycle_id']}")

    if args.validate_only or args.plan_only:
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0

    recorded = append_decision_record(args.ledger_path, decision)
    print(json.dumps(recorded, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
