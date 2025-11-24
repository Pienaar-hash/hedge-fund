#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


STATE_DIR_DEFAULT = Path("logs/state")


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execution health debug helper")
    parser.add_argument("--state-dir", default=str(STATE_DIR_DEFAULT), help="Path to state directory (default: logs/state)")
    parser.add_argument("--show-json", action="store_true", help="Print raw JSON payloads after summary")
    args = parser.parse_args(argv)

    state_dir = Path(args.state_dir)
    nav_path = state_dir / "nav.json"
    router_path = state_dir / "router_health.json"
    exec_health_path = state_dir / "execution_health.json"

    nav_state = _load_json(nav_path)
    router_state = _load_json(router_path)
    exec_health = _load_json(exec_health_path)

    missing = []
    if nav_state is None:
        missing.append(str(nav_path))
    if router_state is None:
        missing.append(str(router_path))
    if exec_health is None:
        missing.append(str(exec_health_path))

    if missing:
        sys.stderr.write(f"[exec_debug] missing state files: {', '.join(missing)}\n")
        return 1

    nav_val = nav_state.get("nav_usd") or nav_state.get("nav")
    router_summary = (router_state.get("summary") or {}) if isinstance(router_state, Mapping) else {}
    quality_counts = router_summary.get("quality_counts") or {}
    exec_updated = exec_health.get("updated_ts") or exec_health.get("ts")
    print("== Execution Debug ==")
    print(f"nav_usd={nav_val} updated_ts={nav_state.get('updated_ts')}")
    print(f"router_health symbols={len(router_state.get('symbols') or [])} quality_counts={quality_counts}")
    errors = (exec_health.get("errors") or {}) if isinstance(exec_health, Mapping) else {}
    by_component = {k: v.get("count") for k, v in errors.items()} if isinstance(errors, Mapping) else {}
    print(f"execution_health snapshot_ts={exec_updated} component_errors={by_component}")
    print(f"raw_nav={nav_path} raw_router={router_path} raw_exec_health={exec_health_path}")

    if args.show_json:
        print("--- nav_state ---")
        print(json.dumps(nav_state, indent=2, sort_keys=True))
        print("--- router_health ---")
        print(json.dumps(router_state, indent=2, sort_keys=True))
        print("--- execution_health ---")
        print(json.dumps(exec_health, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
