#!/usr/bin/env python3
"""
Phase A.3: Doctrine Veto Attribution Analysis

Mirrors risk veto attribution but for the doctrine layer.
Shows which heads are "fighting doctrine" during blocking regimes.

Usage:
    python scripts/doctrine_attribution.py [--hours N] [--since TIMESTAMP]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Any, Optional


def load_doctrine_events(
    log_path: Path,
    since_ts: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Load doctrine events from JSONL log."""
    events = []
    if not log_path.exists():
        return events
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                # Filter by timestamp if provided
                if since_ts:
                    ts_str = event.get("ts", "")
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts < since_ts:
                            continue
                    except Exception:
                        continue
                events.append(event)
            except json.JSONDecodeError:
                continue
    return events


def analyze_doctrine_vetoes(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze doctrine veto patterns."""
    
    # Only look at ENTRY_VERDICT events that were vetoed
    vetoes = [e for e in events if e.get("type") == "ENTRY_VERDICT" and not e.get("allowed", True)]
    allows = [e for e in events if e.get("type") == "ENTRY_VERDICT" and e.get("allowed", False)]
    
    if not vetoes:
        return {
            "total_verdicts": len([e for e in events if e.get("type") == "ENTRY_VERDICT"]),
            "total_vetoes": 0,
            "total_allows": len(allows),
            "veto_rate": 0.0,
            "by_verdict": {},
            "by_regime": {},
            "by_head": {},
            "by_symbol": {},
            "head_regime_matrix": {},
        }
    
    total_verdicts = len(vetoes) + len(allows)
    
    # Aggregate by verdict type
    by_verdict: Dict[str, int] = defaultdict(int)
    for v in vetoes:
        verdict = v.get("verdict", "UNKNOWN")
        by_verdict[verdict] += 1
    
    # Aggregate by regime
    by_regime: Dict[str, int] = defaultdict(int)
    for v in vetoes:
        regime = v.get("regime", "UNKNOWN")
        by_regime[regime] += 1
    
    # Aggregate by head (source_head)
    by_head: Dict[str, int] = defaultdict(int)
    head_with_attribution = 0
    for v in vetoes:
        head = v.get("source_head") or "UNKNOWN"
        by_head[head] += 1
        if head != "UNKNOWN":
            head_with_attribution += 1
    
    # Aggregate by symbol
    by_symbol: Dict[str, int] = defaultdict(int)
    for v in vetoes:
        sym = v.get("symbol", "UNKNOWN")
        by_symbol[sym] += 1
    
    # Head × Regime matrix
    head_regime: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for v in vetoes:
        head = v.get("source_head") or "UNKNOWN"
        regime = v.get("regime", "UNKNOWN")
        head_regime[head][regime] += 1
    
    # Head × Verdict matrix
    head_verdict: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for v in vetoes:
        head = v.get("source_head") or "UNKNOWN"
        verdict = v.get("verdict", "UNKNOWN")
        head_verdict[head][verdict] += 1
    
    return {
        "total_verdicts": total_verdicts,
        "total_vetoes": len(vetoes),
        "total_allows": len(allows),
        "veto_rate": len(vetoes) / total_verdicts if total_verdicts > 0 else 0.0,
        "head_attribution_rate": head_with_attribution / len(vetoes) if vetoes else 0.0,
        "by_verdict": dict(by_verdict),
        "by_regime": dict(by_regime),
        "by_head": dict(by_head),
        "by_symbol": dict(by_symbol),
        "head_regime_matrix": {h: dict(r) for h, r in head_regime.items()},
        "head_verdict_matrix": {h: dict(v) for h, v in head_verdict.items()},
    }


def format_report(analysis: Dict[str, Any], hours: float) -> str:
    """Format analysis as markdown report."""
    lines = [
        f"# Doctrine Veto Attribution Report",
        f"",
        f"**Window:** Last {hours:.1f} hours",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Verdicts | {analysis['total_verdicts']} |",
        f"| Total Vetoes | {analysis['total_vetoes']} |",
        f"| Total Allows | {analysis['total_allows']} |",
        f"| Veto Rate | {analysis['veto_rate']*100:.1f}% |",
        f"| Head Attribution Rate | {analysis.get('head_attribution_rate', 0)*100:.1f}% |",
        f"",
    ]
    
    if analysis['total_vetoes'] == 0:
        lines.append("*No doctrine vetoes in this window.*")
        return "\n".join(lines)
    
    # By verdict type
    lines.extend([
        f"## Vetoes by Verdict Type",
        f"",
        f"| Verdict | Count | % |",
        f"|---------|-------|---|",
    ])
    for verdict, count in sorted(analysis['by_verdict'].items(), key=lambda x: -x[1]):
        pct = count / analysis['total_vetoes'] * 100
        lines.append(f"| {verdict} | {count} | {pct:.1f}% |")
    lines.append("")
    
    # By regime
    lines.extend([
        f"## Vetoes by Regime",
        f"",
        f"| Regime | Count | % |",
        f"|--------|-------|---|",
    ])
    for regime, count in sorted(analysis['by_regime'].items(), key=lambda x: -x[1]):
        pct = count / analysis['total_vetoes'] * 100
        lines.append(f"| {regime} | {count} | {pct:.1f}% |")
    lines.append("")
    
    # By head
    lines.extend([
        f"## Vetoes by Source Head",
        f"",
        f"| Head | Count | % |",
        f"|------|-------|---|",
    ])
    for head, count in sorted(analysis['by_head'].items(), key=lambda x: -x[1]):
        pct = count / analysis['total_vetoes'] * 100
        lines.append(f"| {head} | {count} | {pct:.1f}% |")
    lines.append("")
    
    # By symbol
    lines.extend([
        f"## Vetoes by Symbol",
        f"",
        f"| Symbol | Count | % |",
        f"|--------|-------|---|",
    ])
    for sym, count in sorted(analysis['by_symbol'].items(), key=lambda x: -x[1]):
        pct = count / analysis['total_vetoes'] * 100
        lines.append(f"| {sym} | {count} | {pct:.1f}% |")
    lines.append("")
    
    # Head × Regime matrix
    if analysis.get('head_regime_matrix'):
        regimes = sorted(set(r for hr in analysis['head_regime_matrix'].values() for r in hr.keys()))
        lines.extend([
            f"## Head × Regime Matrix",
            f"",
            f"| Head | " + " | ".join(regimes) + " |",
            f"|------|" + "|".join(["---"] * len(regimes)) + "|",
        ])
        for head, regime_counts in sorted(analysis['head_regime_matrix'].items()):
            row = [str(regime_counts.get(r, 0)) for r in regimes]
            lines.append(f"| {head} | " + " | ".join(row) + " |")
        lines.append("")
    
    # Head × Verdict matrix
    if analysis.get('head_verdict_matrix'):
        verdicts = sorted(set(v for hv in analysis['head_verdict_matrix'].values() for v in hv.keys()))
        # Truncate long verdict names for display
        short_verdicts = [v.replace("VETO_", "") for v in verdicts]
        lines.extend([
            f"## Head × Verdict Matrix",
            f"",
            f"| Head | " + " | ".join(short_verdicts) + " |",
            f"|------|" + "|".join(["---"] * len(verdicts)) + "|",
        ])
        for head, verdict_counts in sorted(analysis['head_verdict_matrix'].items()):
            row = [str(verdict_counts.get(v, 0)) for v in verdicts]
            lines.append(f"| {head} | " + " | ".join(row) + " |")
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Doctrine veto attribution analysis")
    parser.add_argument("--hours", type=float, default=24.0, help="Hours to look back")
    parser.add_argument("--since", type=str, help="ISO timestamp to start from")
    parser.add_argument("--output", type=str, default="docs/PHASE_A3_DOCTRINE_ATTRIBUTION.md")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of markdown")
    args = parser.parse_args()
    
    log_path = Path("logs/doctrine_events.jsonl")
    
    # Determine time window
    if args.since:
        since_ts = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
    else:
        since_ts = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    
    print(f"Loading doctrine events since {since_ts.isoformat()}...")
    events = load_doctrine_events(log_path, since_ts)
    print(f"Loaded {len(events)} events")
    
    analysis = analyze_doctrine_vetoes(events)
    
    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        report = format_report(analysis, args.hours)
        print(report)
        
        # Write to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
