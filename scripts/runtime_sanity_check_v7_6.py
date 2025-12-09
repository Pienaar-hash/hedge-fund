#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from execution.preflight import read_engine_metadata


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _fmt(value: Any) -> str:
    try:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:,.4f}" if abs(value) < 1000 else f"{value:,.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)
    except Exception:
        return "n/a"


def _nav_summary(state_dir: Path) -> str:
    nav_state = _load_json(state_dir / "nav_state.json")
    nav_detail = _load_json(state_dir / "nav.json")
    nav_total = nav_state.get("nav_total") or nav_state.get("nav") or nav_state.get("nav_usd") or nav_detail.get("nav")
    dd_state = nav_state.get("dd_state") or (nav_state.get("drawdown") or {}).get("state")
    return f"NAV={_fmt(nav_total)} dd_state={dd_state or 'unknown'}"


def _risk_summary(state_dir: Path) -> str:
    risk = _load_json(state_dir / "risk_snapshot.json")
    dd_state = risk.get("dd_state") or (risk.get("drawdown") or {}).get("state")
    risk_mode = risk.get("risk_mode") or "unknown"
    anomalies = list((risk.get("anomalies") or {}).keys())
    var_block = risk.get("var") or {}
    cvar_block = risk.get("cvar") or {}
    var_pct = var_block.get("portfolio_var_nav_pct")
    cvar_flags = [k for k, v in (cvar_block.get("per_symbol") or {}).items() if isinstance(v, dict) and not v.get("within_limit", True)]
    return f"risk_mode={risk_mode} dd_state={dd_state or 'normal'} var_nav_pct={_fmt(var_pct)} anomalies={anomalies or 'none'} cvar_flags={len(cvar_flags)}"


def _router_summary(state_dir: Path) -> str:
    router = _load_json(state_dir / "router_health.json")
    summary = router.get("summary") or {}
    quality = summary.get("quality_counts") or {}
    score = router.get("router_health_score") or (router.get("global") or {}).get("quality_score")
    return f"router_score={_fmt(score)} quality_counts={quality or {}}"


def _diagnostics_summary(state_dir: Path) -> str:
    diag_root = _load_json(state_dir / "diagnostics.json")
    diag = diag_root.get("runtime_diagnostics") if isinstance(diag_root.get("runtime_diagnostics"), dict) else diag_root
    exit_pipeline = diag.get("exit_pipeline") if isinstance(diag, dict) else {}
    coverage = exit_pipeline.get("tp_sl_coverage_pct")
    mismatch = exit_pipeline.get("ledger_registry_mismatch")
    veto = diag.get("veto_counters") if isinstance(diag, dict) else {}
    return (
        f"tp_sl_coverage={_fmt(coverage)} "
        f"ledger_mismatch={bool(mismatch)} "
        f"veto_total={_fmt(veto.get('total_vetoes') if isinstance(veto, dict) else None)}"
    )


def _factor_summary(state_dir: Path) -> str:
    factor = _load_json(state_dir / "factor_diagnostics.json")
    weights = (factor.get("factor_weights") or {}).get("weights") if isinstance(factor, dict) else {}
    top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:3] if isinstance(weights, dict) else []
    return "factor_weights=" + ", ".join(f"{k}:{_fmt(v)}" for k, v in top) if top else "factor_weights=n/a"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Runtime sanity check for v7.6 surfaces")
    parser.add_argument("--state-dir", default="logs/state", help="State directory to inspect")
    args = parser.parse_args(argv)
    state_dir = Path(args.state_dir)

    meta = read_engine_metadata(state_dir)
    engine_version = meta.get("engine_version") or "unknown"
    updated_ts = meta.get("updated_ts") or meta.get("ts")

    print(f"engine_version={engine_version} updated_ts={updated_ts}")
    print(_nav_summary(state_dir))
    print(_risk_summary(state_dir))
    print(_router_summary(state_dir))
    print(_diagnostics_summary(state_dir))
    print(_factor_summary(state_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
