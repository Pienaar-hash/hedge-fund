#!/usr/bin/env python3
"""
Doc invariants check — fast CI guard against high-risk doc drift.

Each rule is either:
  MUST_CONTAIN  — the pattern must appear in the target file(s)
  MUST_NOT_CONTAIN — the pattern must NOT appear in any target file

Exit 0 = all rules pass.  Exit 1 = one or more failures (details printed to stdout).

Usage:
    python scripts/doc_invariants_check.py
    python scripts/doc_invariants_check.py --docs-dir /path/to/docs
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, List


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    kind: str          # "MUST_CONTAIN" | "MUST_NOT_CONTAIN"
    pattern: str       # literal string or regex (use regex=True)
    files: List[str]   # glob patterns relative to docs_dir
    description: str   # human label shown in output
    regex: bool = False
    why: str = ""      # root-cause note shown on failure


@dataclass
class DerivedRule:
    description: str
    derive: Callable[[Path], List[str]]
    why: str = ""


RULES: List[Rule] = [

    # -------------------------------------------------------------------------
    # Log file paths — frequently confused
    # -------------------------------------------------------------------------
    Rule(
        kind="MUST_NOT_CONTAIN",
        pattern="logs/execution/doctrine_events",
        files=["*.md"],
        description="doctrine_events path must NOT be under logs/execution/",
        why="doctrine_kernel.py writes to logs/doctrine_events.jsonl (repo root logs/)",
    ),
    Rule(
        kind="MUST_CONTAIN",
        pattern="logs/doctrine_events.jsonl",
        files=["STRATEGY_REGISTRY.md", "RISK_POLICY.md", "EXECUTION_ARCHITECTURE.md"],
        description="doctrine_events.jsonl correct path present in key docs",
        why="Correct path is logs/doctrine_events.jsonl per doctrine_kernel.py:655",
    ),
    Rule(
        kind="MUST_NOT_CONTAIN",
        pattern="logs/state/nav_log",
        files=["*.md"],
        description="nav_log must NOT be referenced under logs/state/",
        why="nav.py writes to logs/nav_log.json (top-level logs/, not logs/state/)",
    ),
    Rule(
        kind="MUST_CONTAIN",
        pattern="logs/nav_log.json",
        files=["EXECUTION_ARCHITECTURE.md", "RISK_POLICY.md"],
        description="nav_log.json correct path present in key docs",
        why="Writer path: nav.py _NAV_LOG_PATH = 'logs/nav_log.json'",
    ),
    Rule(
        kind="MUST_NOT_CONTAIN",
        pattern="logs/router_health.jsonl",
        files=["RISK_POLICY.md"],
        description="router_health log not mis-referenced in RISK_POLICY",
        why="RISK_POLICY doesn't document router_health; EXECUTION_ARCHITECTURE does",
    ),

    # -------------------------------------------------------------------------
    # Binance testnet URL
    # -------------------------------------------------------------------------
    Rule(
        kind="MUST_NOT_CONTAIN",
        pattern="testnet.binancefutures.com",
        files=["*.md"],
        description="Wrong testnet URL (extra 's') must not appear",
        why="exchange_utils.py uses https://testnet.binancefuture.com (no trailing s)",
    ),
    Rule(
        kind="MUST_CONTAIN",
        pattern="testnet.binancefuture.com",
        files=["EXECUTION_ARCHITECTURE.md"],
        description="Correct Binance testnet URL present",
        why="exchange_utils.py _base_url() returns 'https://testnet.binancefuture.com'",
    ),

    # -------------------------------------------------------------------------
    # Telegram environment variables
    # -------------------------------------------------------------------------
    Rule(
        kind="MUST_NOT_CONTAIN",
        pattern="EXEC_TELEGRAM_ENABLED",
        files=["*.md"],
        description="Wrong Telegram enable var (EXEC_ prefix) must not appear",
        why="telegram_alerts_v7.py docstring and telegram_v7.json use TELEGRAM_ENABLED",
    ),
    Rule(
        kind="MUST_NOT_CONTAIN",
        pattern="EXEC_TELEGRAM_BOT_TOKEN",
        files=["*.md"],
        description="Wrong Telegram bot token var must not appear",
        why="config/telegram_v7.json bot_token_env = 'TELEGRAM_BOT_TOKEN'",
    ),
    Rule(
        kind="MUST_NOT_CONTAIN",
        pattern="EXEC_TELEGRAM_CHAT_ID",
        files=["*.md"],
        description="Wrong Telegram chat ID var must not appear",
        why="config/telegram_v7.json chat_id_env = 'TELEGRAM_CHAT_ID'",
    ),
    Rule(
        kind="MUST_CONTAIN",
        pattern="TELEGRAM_ENABLED",
        files=["EXECUTION_ARCHITECTURE.md"],
        description="Correct TELEGRAM_ENABLED var present in arch doc",
        why="Top-level disable flag per telegram_alerts_v7.py docstring",
    ),

    # -------------------------------------------------------------------------
    # Doctrine regime labels
    # -------------------------------------------------------------------------
    Rule(
        kind="MUST_NOT_CONTAIN",
        pattern=r"\bRANGE\b",
        files=["*.md"],
        description="Regime label RANGE must not appear (not in code)",
        why="REGIME_DIRECTION_MAP has TREND_UP/TREND_DOWN/MEAN_REVERT/BREAKOUT/CHOPPY/CRISIS",
        regex=True,
    ),
    Rule(
        kind="MUST_CONTAIN",
        pattern="MEAN_REVERT",
        files=["STRATEGY_REGISTRY.md", "INVESTOR_REPORT_TEMPLATE.md"],
        description="Correct regime label MEAN_REVERT present",
        why="doctrine_kernel.py REGIME_DIRECTION_MAP keys",
    ),

    # -------------------------------------------------------------------------
    # nav_log.json entry schema
    # -------------------------------------------------------------------------
    Rule(
        kind="MUST_NOT_CONTAIN",
        pattern='e.get("ts"',
        files=["*.md"],
        description='nav_log filter must not use key "ts"',
        why='nav.py writes entries as {"t": ts, "nav": nav_float} — key is "t" not "ts"',
    ),
    Rule(
        kind="MUST_CONTAIN",
        pattern='e.get("t"',
        files=["INVESTOR_REPORT_TEMPLATE.md"],
        description='nav_log filter uses correct key "t"',
        why='nav.py entry = {"t": ts, "nav": nav_float}',
    ),

    # -------------------------------------------------------------------------
    # Environment label accuracy
    # -------------------------------------------------------------------------
    Rule(
        kind="MUST_NOT_CONTAIN",
        pattern="ENV=testnet",
        files=["*.md"],
        description="ENV=testnet must not appear (deploy uses production/prod)",
        why="config/runtime.yaml env=production; supervisor ENV=prod; only exchange routing is testnet",
    ),

    # -------------------------------------------------------------------------
    # Doctrine exit correctness
    # -------------------------------------------------------------------------
    Rule(
        kind="MUST_NOT_CONTAIN",
        pattern="90-day circular buffer",
        files=["*.md"],
        description="nav_log 90-day circular buffer claim must not appear",
        why="nav.py appends without truncation; no size cap in current code",
    ),

    # -------------------------------------------------------------------------
    # Per-symbol min notional
    # -------------------------------------------------------------------------
    Rule(
        kind="MUST_NOT_CONTAIN",
        pattern="Min notional per order: $500",
        files=["*.md"],
        description="$500 min notional claim must not appear",
        why="risk_limits.json min_notional_usdt=25; $500 is TWAP trigger, not order minimum",
    ),
]


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _pairs_universe_enabled_symbols(docs_dir: Path) -> List[str]:
    payload = _read_json(docs_dir / "config" / "pairs_universe.json")
    if not isinstance(payload, dict):
        return []
    raw_universe = payload.get("universe")
    if not isinstance(raw_universe, list):
        return []
    symbols: List[str] = []
    for entry in raw_universe:
        if not isinstance(entry, dict):
            continue
        if not entry.get("enabled"):
            continue
        symbol = entry.get("symbol")
        if isinstance(symbol, str) and symbol:
            symbols.append(symbol)
    return symbols


def _doctrine_enum_values(docs_dir: Path) -> List[str]:
    repo_root = str(docs_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    module = import_module("execution.doctrine_kernel")
    values: List[str] = []
    for enum_name in ("DoctrineVerdict", "ExitReason"):
        enum_cls = getattr(module, enum_name)
        for member in enum_cls:
            values.append(str(member.value))
    return values


def _risk_engine_threshold_constants(docs_dir: Path) -> dict[str, float]:
    source = _read_text(docs_dir / "execution" / "risk_engine_v6.py")
    tree = ast.parse(source, filename="execution/risk_engine_v6.py")
    wanted = {
        "_NAV_STALE_THRESHOLD_S",
        "_DD_DEFENSIVE_THRESHOLD",
        "_DAILY_LOSS_DEFENSIVE_THRESHOLD",
    }
    found: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in wanted:
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        if isinstance(value, (int, float)):
            found[name] = float(value)
    return found


def _supervisor_program_names(docs_dir: Path) -> List[str]:
    text = _read_text(docs_dir / "deploy" / "supervisor" / "hedge.conf")
    return re.findall(r"^\[program:([^\]]+)\]$", text, flags=re.MULTILINE)


def _doc_contains(path: Path, token: str) -> bool:
    return token in _read_text(path)


def _derive_enabled_universe_symbols_present(docs_dir: Path) -> List[str]:
    target = docs_dir / "STRATEGY_REGISTRY.md"
    failures: List[str] = []
    for symbol in _pairs_universe_enabled_symbols(docs_dir):
        if not _doc_contains(target, symbol):
            failures.append(f"  MISSING in {target.name}: enabled universe symbol {symbol!r}")
    return failures


def _derive_doctrine_enums_present(docs_dir: Path) -> List[str]:
    target = docs_dir / "STRATEGY_REGISTRY.md"
    failures: List[str] = []
    for value in _doctrine_enum_values(docs_dir):
        if not _doc_contains(target, value):
            failures.append(f"  MISSING in {target.name}: doctrine enum value {value!r}")
    return failures


def _derive_risk_thresholds_present(docs_dir: Path) -> List[str]:
    target = docs_dir / "RISK_POLICY.md"
    failures: List[str] = []
    constants = _risk_engine_threshold_constants(docs_dir)
    expected = {
        "_NAV_STALE_THRESHOLD_S": ["90", "90s"],
        "_DD_DEFENSIVE_THRESHOLD": ["0.30", "30%"],
        "_DAILY_LOSS_DEFENSIVE_THRESHOLD": ["0.10", "10%"],
    }
    for name, value in constants.items():
        if not any(token in _read_text(target) for token in expected.get(name, [str(value)])):
            failures.append(
                f"  MISSING in {target.name}: threshold {name}={value:g} "
                f"(expected one of {expected.get(name, [str(value)])})"
            )
    missing_constants = set(expected) - set(constants)
    for name in sorted(missing_constants):
        failures.append(f"  CANNOT DERIVE constant from execution/risk_engine_v6.py: {name}")
    return failures


def _derive_supervisor_programs_present(docs_dir: Path) -> List[str]:
    target = docs_dir / "EXECUTION_ARCHITECTURE.md"
    failures: List[str] = []
    for program in _supervisor_program_names(docs_dir):
        if not _doc_contains(target, program):
            failures.append(f"  MISSING in {target.name}: supervisor program {program!r}")
    return failures


DERIVED_RULES: List[DerivedRule] = [
    DerivedRule(
        description="Enabled pairs_universe symbols are all documented in STRATEGY_REGISTRY",
        derive=_derive_enabled_universe_symbols_present,
        why="pairs_universe.json is authoritative for executor/risk-engine runtime universe",
    ),
    DerivedRule(
        description="Doctrine enum values are all documented in STRATEGY_REGISTRY",
        derive=_derive_doctrine_enums_present,
        why="DoctrineVerdict and ExitReason are the code-level contract for entry/exit outcomes",
    ),
    DerivedRule(
        description="Risk engine classification thresholds are documented in RISK_POLICY",
        derive=_derive_risk_thresholds_present,
        why="Risk mode docs should track the exact hard-coded thresholds in risk_engine_v6.py",
    ),
    DerivedRule(
        description="Supervisor program names are documented in EXECUTION_ARCHITECTURE",
        derive=_derive_supervisor_programs_present,
        why="The architecture doc should enumerate the actual managed services from hedge.conf",
    ),
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def collect_files(docs_dir: Path, glob: str) -> List[Path]:
    return sorted(docs_dir.glob(glob))


def check_rule(rule: Rule, docs_dir: Path) -> List[str]:
    """Return list of failure messages (empty = pass)."""
    failures: List[str] = []
    matched_files: List[Path] = []
    for g in rule.files:
        matched_files.extend(collect_files(docs_dir, g))
    matched_files = list(dict.fromkeys(matched_files))  # dedupe, preserve order

    for path in matched_files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            failures.append(f"  CANNOT READ {path}: {exc}")
            continue

        if rule.regex:
            found = bool(re.search(rule.pattern, text))
        else:
            found = rule.pattern in text

        if rule.kind == "MUST_CONTAIN" and not found:
            failures.append(f"  MISSING in {path.name}: {rule.pattern!r}")
        elif rule.kind == "MUST_NOT_CONTAIN" and found:
            failures.append(f"  PRESENT in {path.name}: {rule.pattern!r}")

    return failures


def run(docs_dir: Path) -> int:
    passed = 0
    failed = 0
    all_failures: List[str] = []

    for rule in RULES:
        failures = check_rule(rule, docs_dir)
        tag = "PASS" if not failures else "FAIL"
        status = f"[{tag}] {rule.description}"
        if failures:
            failed += 1
            all_failures.append(status)
            for line in failures:
                all_failures.append(line)
            if rule.why:
                all_failures.append(f"       Why: {rule.why}")
        else:
            passed += 1
            print(status)

    for rule in DERIVED_RULES:
        failures = rule.derive(docs_dir)
        tag = "PASS" if not failures else "FAIL"
        status = f"[{tag}] {rule.description}"
        if failures:
            failed += 1
            all_failures.append(status)
            for line in failures:
                all_failures.append(line)
            if rule.why:
                all_failures.append(f"       Why: {rule.why}")
        else:
            passed += 1
            print(status)

    if all_failures:
        print()
        for line in all_failures:
            print(line)

    total_rules = len(RULES) + len(DERIVED_RULES)
    print(f"\n{passed} passed, {failed} failed ({total_rules} rules total)")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir",
        default=".",
        help="Directory containing the *.md doc files (default: repo root)",
    )
    args = parser.parse_args()
    docs_dir = Path(args.docs_dir).resolve()
    if not docs_dir.is_dir():
        print(f"ERROR: --docs-dir {docs_dir} is not a directory")
        sys.exit(2)
    sys.exit(run(docs_dir))


if __name__ == "__main__":
    main()
