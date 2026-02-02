#!/usr/bin/env python3
"""
Veto Attribution Analysis — Phase A.3 Geometry Probe

Analyzes risk_vetoes.jsonl to understand constraint pressure:
- Where does intent collide with feasibility?
- Is that collision systematic or incidental?
- Are some heads structurally misaligned with the feasible region?

Output: Markdown report + CSV summaries
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

VETO_LOG = Path("logs/execution/risk_vetoes.jsonl")
OUTPUT_DIR = Path("docs")
CSV_DIR = Path("analysis")


def load_vetoes() -> list[dict[str, Any]]:
    """Load and parse all veto records."""
    records = []
    with open(VETO_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized fields from a veto record."""
    # Strategy/head extraction (nested in intent metadata)
    strategy = None
    regime = None
    regime_conf = None
    
    # Phase A.3: Check for new source_head field first
    source_head = record.get("source_head")
    
    intent = record.get("veto_detail", {}).get("intent", {})
    metadata = intent.get("metadata", {})
    
    if metadata:
        strategy = metadata.get("strategy")
        regime = metadata.get("entry_regime")
        regime_conf = metadata.get("entry_regime_confidence")
    
    # Fallback to top-level strategy field, then source_head
    if not strategy:
        strategy = record.get("strategy") or source_head
    
    # Context may be a string ("executor") or a dict
    ctx = record.get("context", {})
    if isinstance(ctx, str):
        ctx = {}
    
    # Phase A.3: Extract constraint geometry if present
    constraint_geometry = record.get("veto_detail", {}).get("constraint_geometry", {})
    
    return {
        "symbol": record.get("symbol"),
        "veto_reason": record.get("veto_reason"),
        "strategy": strategy or "unknown",
        "source_head": source_head or strategy or "unknown",
        "regime": regime or "unknown",
        "regime_conf": regime_conf,
        "ts": record.get("ts"),
        "date": record.get("ts", "")[:10] if record.get("ts") else None,
        "notional": record.get("veto_detail", {}).get("intent", {}).get("gross_usd") 
                   or ctx.get("requested_notional"),
        "nav": ctx.get("nav"),
        "tier": ctx.get("tier"),
        # Constraint geometry (distance-to-wall metrics)
        "requested_notional": constraint_geometry.get("requested_notional"),
        "available_budget": constraint_geometry.get("available_budget"),
        "excess_notional": constraint_geometry.get("excess_notional"),
        "shadow_feasible_size": constraint_geometry.get("shadow_feasible_size"),
        "overshoot_pct": constraint_geometry.get("overshoot_pct"),
    }


def aggregate_by(records: list[dict], *keys) -> dict[tuple, int]:
    """Aggregate record counts by specified keys."""
    counts = defaultdict(int)
    for r in records:
        key = tuple(r.get(k) for k in keys)
        counts[key] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def format_table(headers: list[str], rows: list[list], align: list[str] = None) -> str:
    """Format data as markdown table."""
    if not rows:
        return "*No data*\n"
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Build table
    lines = []
    
    # Header
    header_line = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    lines.append(header_line)
    
    # Separator
    sep_line = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    lines.append(sep_line)
    
    # Rows
    for row in rows:
        row_line = "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        lines.append(row_line)
    
    return "\n".join(lines) + "\n"


def generate_report(records: list[dict]) -> str:
    """Generate full attribution report."""
    normalized = [extract_fields(r) for r in records]
    total = len(normalized)
    
    lines = [
        "# Veto Attribution Analysis — Phase A.3",
        "",
        f"**Generated:** {datetime.utcnow().isoformat()[:19]}Z",
        f"**Total Vetoes:** {total:,}",
        f"**Date Range:** {min(r['date'] for r in normalized if r['date'])} → {max(r['date'] for r in normalized if r['date'])}",
        "",
        "---",
        "",
        "## 1. Veto Counts by Reason",
        "",
        "This shows *which constraint dimension* is doing the most work.",
        "",
    ]
    
    # By reason
    by_reason = aggregate_by(normalized, "veto_reason")
    rows = [[reason, count, f"{100*count/total:.1f}%"] for (reason,), count in by_reason.items()]
    lines.append(format_table(["Veto Reason", "Count", "%"], rows))
    
    lines.extend([
        "",
        "## 2. Veto Counts by Symbol",
        "",
        "Identifies symbols that distort the feasible region.",
        "",
    ])
    
    # By symbol
    by_symbol = aggregate_by(normalized, "symbol")
    rows = [[symbol, count, f"{100*count/total:.1f}%"] for (symbol,), count in by_symbol.items()]
    lines.append(format_table(["Symbol", "Count", "%"], rows))
    
    lines.extend([
        "",
        "## 3. Veto Counts by Strategy Head",
        "",
        "Shows which heads generate the most rejected intent.",
        "",
    ])
    
    # By strategy
    by_strategy = aggregate_by(normalized, "strategy")
    rows = [[strategy, count, f"{100*count/total:.1f}%"] for (strategy,), count in by_strategy.items()]
    lines.append(format_table(["Strategy", "Count", "%"], rows))
    
    lines.extend([
        "",
        "## 4. Veto Counts by Entry Regime",
        "",
        "Shows regime-conditioned veto density.",
        "",
    ])
    
    # By regime
    by_regime = aggregate_by(normalized, "regime")
    rows = [[regime, count, f"{100*count/total:.1f}%"] for (regime,), count in by_regime.items()]
    lines.append(format_table(["Regime", "Count", "%"], rows))
    
    lines.extend([
        "",
        "## 5. Symbol × Reason Matrix",
        "",
        "Cross-tabulation showing which symbols hit which constraints.",
        "",
    ])
    
    # Symbol × Reason
    by_sym_reason = aggregate_by(normalized, "symbol", "veto_reason")
    rows = [[sym, reason, count] for (sym, reason), count in by_sym_reason.items()]
    lines.append(format_table(["Symbol", "Reason", "Count"], rows))
    
    lines.extend([
        "",
        "## 6. Strategy × Reason Matrix",
        "",
        "Shows which constraint dimensions each head pushes against.",
        "",
    ])
    
    # Strategy × Reason
    by_strat_reason = aggregate_by(normalized, "strategy", "veto_reason")
    rows = [[strat, reason, count] for (strat, reason), count in by_strat_reason.items()]
    lines.append(format_table(["Strategy", "Reason", "Count"], rows))
    
    lines.extend([
        "",
        "## 7. Regime × Reason Matrix",
        "",
        "Shows how feasible region size varies by regime.",
        "",
    ])
    
    # Regime × Reason
    by_regime_reason = aggregate_by(normalized, "regime", "veto_reason")
    rows = [[regime, reason, count] for (regime, reason), count in by_regime_reason.items()]
    lines.append(format_table(["Regime", "Reason", "Count"], rows))
    
    lines.extend([
        "",
        "## 8. Daily Veto Trend",
        "",
    ])
    
    # By date
    by_date = aggregate_by(normalized, "date")
    rows = [[date, count] for (date,), count in sorted(by_date.items(), key=lambda x: x[0])]
    lines.append(format_table(["Date", "Count"], rows))
    
    lines.extend([
        "",
        "## 9. Tier Distribution",
        "",
    ])
    
    # By tier
    by_tier = aggregate_by(normalized, "tier")
    rows = [[tier or "unknown", count, f"{100*count/total:.1f}%"] for (tier,), count in by_tier.items()]
    lines.append(format_table(["Tier", "Count", "%"], rows))
    
    lines.extend([
        "",
        "---",
        "",
        "## Key Observations",
        "",
        "### Constraint Pressure Analysis",
        "",
    ])
    
    # Calculate key metrics
    top_reason = list(by_reason.items())[0] if by_reason else (("none",), 0)
    top_symbol = list(by_symbol.items())[0] if by_symbol else (("none",), 0)
    
    unknown_strategy = by_strategy.get(("unknown",), 0)
    known_strategy = total - unknown_strategy
    
    # Phase A.3: Constraint geometry analysis (distance-to-wall)
    symbol_cap_records = [r for r in normalized if r["veto_reason"] == "symbol_cap" and r.get("excess_notional") is not None]
    if symbol_cap_records:
        excess_values = [r["excess_notional"] for r in symbol_cap_records if r["excess_notional"] is not None]
        shadow_values = [r["shadow_feasible_size"] for r in symbol_cap_records if r.get("shadow_feasible_size") is not None]
        overshoot_values = [r["overshoot_pct"] for r in symbol_cap_records if r.get("overshoot_pct") is not None and r["overshoot_pct"] != float("inf")]
        
        if excess_values:
            avg_excess = sum(excess_values) / len(excess_values)
            max_excess = max(excess_values)
            min_excess = min(excess_values)
        else:
            avg_excess = max_excess = min_excess = 0
            
        if shadow_values:
            total_shadow = sum(shadow_values)
            avg_shadow = total_shadow / len(shadow_values)
        else:
            total_shadow = avg_shadow = 0
            
        if overshoot_values:
            avg_overshoot = sum(overshoot_values) / len(overshoot_values)
            median_overshoot = sorted(overshoot_values)[len(overshoot_values) // 2]
        else:
            avg_overshoot = median_overshoot = 0
        
        lines.extend([
            "### Constraint Geometry (Distance-to-Wall)",
            "",
            f"**Records with geometry data:** {len(symbol_cap_records)} / {total}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Avg Excess Notional | ${avg_excess:,.2f} |",
            f"| Max Excess Notional | ${max_excess:,.2f} |",
            f"| Min Excess Notional | ${min_excess:,.2f} |",
            f"| Total Shadow Feasible | ${total_shadow:,.2f} |",
            f"| Avg Shadow Feasible | ${avg_shadow:,.2f} |",
            f"| Avg Overshoot % | {avg_overshoot:.1f}% |",
            f"| Median Overshoot % | {median_overshoot:.1f}% |",
            "",
            "**Interpretation:**",
            "",
        ])
        
        if avg_overshoot < 50:
            lines.append("- Small overshoots → **pre-projection would recover most volume**")
        elif avg_overshoot < 200:
            lines.append("- Moderate overshoots → **sizing logic partially misaligned**")
        else:
            lines.append("- Large overshoots → **signal design issue, not just sizing**")
        
        if total_shadow > 0:
            lines.append(f"- Shadow feasible volume: **${total_shadow:,.2f}** could have been executed with clipping")
        
        lines.append("")
    
    lines.extend([
        f"1. **Dominant Constraint:** `{top_reason[0][0]}` accounts for {100*top_reason[1]/total:.1f}% of all vetoes",
        f"2. **Highest Pressure Symbol:** `{top_symbol[0][0]}` with {top_symbol[1]:,} vetoes ({100*top_symbol[1]/total:.1f}%)",
        f"3. **Strategy Attribution Coverage:** {100*known_strategy/total:.1f}% of vetoes have strategy metadata",
        "",
        "### Interpretation",
        "",
    ])
    
    # Interpretation based on data
    if top_reason[0][0] == "symbol_cap":
        lines.extend([
            "- `symbol_cap` dominance indicates **position concentration pressure**",
            "- Heads are trying to add to existing positions rather than diversify",
            "- Potential upstream fix: incorporate open position notional into sizing before signal generation",
            "",
        ])
    elif top_reason[0][0] == "min_notional":
        lines.extend([
            "- `min_notional` dominance indicates **size-down pressure** from risk scaling",
            "- Signals are being generated but sized below exchange minimum",
            "- This is defensive mode working correctly — not a bug",
            "",
        ])
    
    if unknown_strategy / total > 0.5:
        lines.extend([
            "⚠️ **Data Gap:** >50% of vetoes lack strategy attribution.",
            "   Consider enriching veto logging to include source head.",
            "",
        ])
    
    lines.extend([
        "---",
        "",
        "## Phase A.3 Status",
        "",
        "This analysis is **observational only**. No execution behavior changes.",
        "",
        "Next steps (if warranted):",
        "- [ ] Investigate whether symbol_cap can be pre-projected in sizing",
        "- [ ] Enrich veto logging with strategy head for unattributed records",
        "- [ ] Build regime-sliced veto rate dashboard panel",
        "",
    ])
    
    return "\n".join(lines)


def save_csvs(records: list[dict]) -> None:
    """Save CSV summaries for downstream analysis."""
    CSV_DIR.mkdir(exist_ok=True)
    normalized = [extract_fields(r) for r in records]
    
    # Full normalized dump
    with open(CSV_DIR / "veto_attribution_full.csv", "w") as f:
        headers = ["date", "symbol", "veto_reason", "strategy", "regime", "tier", "notional", "nav"]
        f.write(",".join(headers) + "\n")
        for r in normalized:
            row = [str(r.get(h, "")) for h in headers]
            f.write(",".join(row) + "\n")
    
    print(f"Saved: {CSV_DIR / 'veto_attribution_full.csv'}")


def main():
    if not VETO_LOG.exists():
        print(f"Error: {VETO_LOG} not found")
        sys.exit(1)
    
    print(f"Loading vetoes from {VETO_LOG}...")
    records = load_vetoes()
    print(f"Loaded {len(records):,} veto records")
    
    print("Generating report...")
    report = generate_report(records)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "PHASE_A3_VETO_ATTRIBUTION.md"
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Saved: {output_path}")
    
    print("Saving CSVs...")
    save_csvs(records)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
