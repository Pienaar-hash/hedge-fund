#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANNED_IMPORT_TOKENS = {
    "hydra",
    "sentinel",
    "mean_revert",
    "vol_harvest",
    "crossfire",
    "alpha_miner",
    "machine_learning",
    "sentiment",
    "intraday",
}
WEEKLY_FILES = [
    ROOT / "execution" / "weekly_market_data.py",
    ROOT / "execution" / "weekly_momentum_engine.py",
    ROOT / "execution" / "weekly_position_sizer.py",
    ROOT / "execution" / "weekly_risk_gate.py",
    ROOT / "execution" / "weekly_execution_planner.py",
    ROOT / "execution" / "weekly_audit_ledger.py",
    ROOT / "scripts" / "run_weekly_cycle.py",
]


def _collect_imports(path: Path) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def main() -> int:
    errors: list[str] = []
    for path in WEEKLY_FILES:
        for imported in _collect_imports(path):
            lowered = imported.lower()
            if any(token in lowered for token in BANNED_IMPORT_TOKENS):
                errors.append(f"{path.relative_to(ROOT)} imports banned module {imported}")
    payload = {
        "validated_files": [str(path.relative_to(ROOT)) for path in WEEKLY_FILES],
        "errors": errors,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
