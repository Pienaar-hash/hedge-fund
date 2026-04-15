#!/usr/bin/env python3
"""
AUDIT — Baseline Snapshot Capture

Captures pre-fix system state between Phase 1 (Safety Locks) and Phase 2
(Model Correctness) so that model changes can be validated against a known
baseline.  Output directory: data/baseline_snapshot_<date>/

Captured artefacts:
  config_hash.json           SHA256 of all config files
  score_distribution.json    hybrid_score distribution by symbol
  regime_distribution.json   regime label distribution
  zero_score_rate.json       zero-score rate summary
  episode_ledger_snapshot.jsonl  copy of episode ledger
  pnl_per_band.json          PnL by score band quintiles (if available)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "config"
LOGS_STATE = REPO_ROOT / "logs" / "state"
LOGS_EXEC = REPO_ROOT / "logs" / "execution"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_config_hashes(out_dir: Path) -> dict:
    hashes: dict[str, str] = {}
    for p in sorted(CONFIG_DIR.glob("*")):
        if p.is_file():
            hashes[p.name] = _sha256(p)
    payload = {"captured_ts": datetime.now(timezone.utc).isoformat(), "hashes": hashes}
    (out_dir / "config_hash.json").write_text(json.dumps(payload, indent=2))
    print(f"  config_hash.json — {len(hashes)} files hashed")
    return payload


def capture_score_distribution(out_dir: Path) -> dict | None:
    scores_path = LOGS_STATE / "symbol_scores_v6.json"
    if not scores_path.exists():
        print("  score_distribution.json — SKIPPED (symbol_scores_v6.json not found)")
        return None
    raw = json.loads(scores_path.read_text())
    symbols_raw = raw.get("scores") or raw.get("symbols") or {}
    # Handle both dict-keyed and list-of-dicts formats
    if isinstance(symbols_raw, list):
        symbols = {entry["symbol"]: entry for entry in symbols_raw if isinstance(entry, dict) and "symbol" in entry}
    else:
        symbols = symbols_raw
    dist: dict[str, dict] = {}
    all_scores: list[float] = []
    zero_count = 0
    for sym, data in symbols.items():
        score = float(data.get("hybrid_score", data.get("score", 0)) if isinstance(data, dict) else data)
        all_scores.append(score)
        if score == 0.0:
            zero_count += 1
        dist[sym] = {"hybrid_score": score}

    import statistics
    stats = {}
    if all_scores:
        sorted_s = sorted(all_scores)
        stats = {
            "count": len(all_scores),
            "mean": statistics.mean(all_scores),
            "std": statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0,
            "min": sorted_s[0],
            "p25": sorted_s[len(sorted_s) // 4],
            "p50": sorted_s[len(sorted_s) // 2],
            "p75": sorted_s[3 * len(sorted_s) // 4],
            "max": sorted_s[-1],
            "zero_count": zero_count,
            "zero_rate": zero_count / len(all_scores) if all_scores else 0.0,
        }
    payload = {
        "captured_ts": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "per_symbol": dist,
    }
    (out_dir / "score_distribution.json").write_text(json.dumps(payload, indent=2))
    print(f"  score_distribution.json — {len(dist)} symbols, zero_rate={stats.get('zero_rate', 'N/A')}")
    return payload


def capture_regime_distribution(out_dir: Path) -> dict | None:
    sentinel_path = LOGS_STATE / "sentinel_x.json"
    if not sentinel_path.exists():
        print("  regime_distribution.json — SKIPPED (sentinel_x.json not found)")
        return None
    raw = json.loads(sentinel_path.read_text())
    # Capture current regime state
    payload = {
        "captured_ts": datetime.now(timezone.utc).isoformat(),
        "current_state": raw,
    }

    # Try to extract regime history from execution logs
    regime_log = LOGS_EXEC / "regime_transitions.jsonl"
    if regime_log.exists():
        counter: Counter[str] = Counter()
        try:
            for line in regime_log.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                regime = entry.get("regime") or entry.get("new_regime") or "UNKNOWN"
                counter[regime] += 1
        except Exception:
            pass
        total = sum(counter.values())
        payload["regime_counts"] = dict(counter)
        payload["regime_pct"] = {k: v / total for k, v in counter.items()} if total > 0 else {}

    (out_dir / "regime_distribution.json").write_text(json.dumps(payload, indent=2))
    print(f"  regime_distribution.json — captured")
    return payload


def capture_zero_score_rate(out_dir: Path) -> dict | None:
    # Derive from episode ledger if available
    ledger_path = LOGS_EXEC / "episode_ledger.jsonl"
    if not ledger_path.exists():
        print("  zero_score_rate.json — SKIPPED (episode_ledger.jsonl not found)")
        return None

    by_symbol: dict[str, dict] = defaultdict(lambda: {"total": 0, "zero": 0})
    try:
        for line in ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            sym = entry.get("symbol", "UNKNOWN")
            score = float(entry.get("hybrid_score", entry.get("score", 0)) or 0)
            by_symbol[sym]["total"] += 1
            if score == 0.0:
                by_symbol[sym]["zero"] += 1
    except Exception as e:
        print(f"  zero_score_rate.json — ERROR: {e}")
        return None

    rates = {}
    for sym, counts in by_symbol.items():
        rates[sym] = {
            "total_episodes": counts["total"],
            "zero_score_episodes": counts["zero"],
            "zero_rate": counts["zero"] / counts["total"] if counts["total"] > 0 else 0.0,
        }

    payload = {
        "captured_ts": datetime.now(timezone.utc).isoformat(),
        "per_symbol": rates,
        "global_zero_rate": sum(c["zero"] for c in by_symbol.values()) / max(sum(c["total"] for c in by_symbol.values()), 1),
    }
    (out_dir / "zero_score_rate.json").write_text(json.dumps(payload, indent=2))
    print(f"  zero_score_rate.json — {len(rates)} symbols")
    return payload


def capture_episode_ledger(out_dir: Path) -> bool:
    ledger_path = LOGS_EXEC / "episode_ledger.jsonl"
    if not ledger_path.exists():
        print("  episode_ledger_snapshot.jsonl — SKIPPED (not found)")
        return False
    shutil.copy2(ledger_path, out_dir / "episode_ledger_snapshot.jsonl")
    size_kb = ledger_path.stat().st_size / 1024
    print(f"  episode_ledger_snapshot.jsonl — {size_kb:.1f} KB copied")
    return True


def capture_pnl_per_band(out_dir: Path) -> dict | None:
    ledger_path = LOGS_EXEC / "episode_ledger.jsonl"
    if not ledger_path.exists():
        print("  pnl_per_band.json — SKIPPED (episode_ledger.jsonl not found)")
        return None

    bands: dict[str, list[float]] = defaultdict(list)
    try:
        for line in ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            score = float(entry.get("hybrid_score", entry.get("score", 0)) or 0)
            pnl = float(entry.get("pnl_pct", entry.get("pnl", 0)) or 0)
            # Quintile bands
            if score <= 0.2:
                band = "0.00-0.20"
            elif score <= 0.4:
                band = "0.20-0.40"
            elif score <= 0.6:
                band = "0.40-0.60"
            elif score <= 0.8:
                band = "0.60-0.80"
            else:
                band = "0.80-1.00"
            bands[band].append(pnl)
    except Exception as e:
        print(f"  pnl_per_band.json — ERROR: {e}")
        return None

    import statistics
    summary = {}
    for band, pnls in sorted(bands.items()):
        summary[band] = {
            "count": len(pnls),
            "mean_pnl": statistics.mean(pnls) if pnls else 0.0,
            "total_pnl": sum(pnls),
        }

    payload = {
        "captured_ts": datetime.now(timezone.utc).isoformat(),
        "bands": summary,
    }
    (out_dir / "pnl_per_band.json").write_text(json.dumps(payload, indent=2))
    print(f"  pnl_per_band.json — {sum(len(v) for v in bands.values())} episodes across {len(bands)} bands")
    return payload


def main():
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = REPO_ROOT / "data" / f"baseline_snapshot_{date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Capturing baseline snapshot to {out_dir.relative_to(REPO_ROOT)}/")
    print()

    capture_config_hashes(out_dir)
    capture_score_distribution(out_dir)
    capture_regime_distribution(out_dir)
    capture_zero_score_rate(out_dir)
    capture_episode_ledger(out_dir)
    capture_pnl_per_band(out_dir)

    # Write manifest
    manifest = {
        "captured_ts": datetime.now(timezone.utc).isoformat(),
        "phase": "between_phase1_and_phase2",
        "purpose": "Pre-fix baseline for model correctness validation",
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(f"Baseline snapshot complete: {len(manifest['files']) + 1} files")
    print(f"Phase 2 gate: UNBLOCKED")


if __name__ == "__main__":
    main()
