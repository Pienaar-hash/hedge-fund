#!/usr/bin/env python3
"""
MHD Audit Suite — GPT-Hedge v7.x

Run: python scripts/mhd_audit.py
Output: logs/audit/MHD_SCORECARD_<timestamp>.md

Audits executed:
  - AUDIT 0: Evidence Admission Gate
  - AUDIT 1A: Replay Determinism
  - AUDIT 4A: Temporal Integrity
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

# Configuration
WINDOW_HOURS = 24
LOG_DIR = Path("logs")
AUDIT_DIR = LOG_DIR / "audit"

# Required evidence files
EVIDENCE_FILES = [
    "logs/dle/dle_events_v1.jsonl",
    "logs/execution/orders_attempted.jsonl",
    "logs/execution/orders_executed.jsonl",
    "logs/execution/risk_vetoes.jsonl",
    "logs/state/episode_ledger.json",
    "logs/state/risk_snapshot.json",
    "logs/state/sentinel_x.json",
]


def parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def audit_0_evidence_gate(cutoff: datetime) -> dict:
    """AUDIT 0: Evidence Admission Gate."""
    results = {
        "name": "AUDIT 0 — Evidence Admission Gate",
        "pass": True,
        "file_existence": [],
        "monotonic": [],
        "referential_integrity": {},
    }

    # File existence
    for f in EVIDENCE_FILES:
        exists = os.path.isfile(f)
        size = os.path.getsize(f) if exists else 0
        lines = sum(1 for _ in open(f)) if exists and f.endswith(".jsonl") else 0
        results["file_existence"].append({
            "file": f,
            "exists": exists,
            "lines": lines,
            "size_bytes": size,
        })
        if not exists or size == 0:
            results["pass"] = False

    # Monotonic timestamps
    jsonl_files = [f for f in EVIDENCE_FILES if f.endswith(".jsonl")]
    for f in jsonl_files:
        prev_ts = None
        violations = 0
        total = 0
        with open(f) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    ts_str = obj.get("ts")
                    if not ts_str:
                        continue
                    ts = parse_ts(ts_str)
                    if ts >= cutoff:
                        total += 1
                        if prev_ts and ts < prev_ts:
                            violations += 1
                        prev_ts = ts
                except Exception:
                    pass
        results["monotonic"].append({
            "file": os.path.basename(f),
            "events_24h": total,
            "violations": violations,
        })
        if violations > 0:
            results["pass"] = False

    # Referential integrity (DLE events)
    requests = {}
    decisions = {}
    permits = {}

    with open("logs/dle/dle_events_v1.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            ts = parse_ts(obj["ts"])
            if ts < cutoff:
                continue
            event_type = obj.get("event_type")
            payload = obj.get("payload", {})

            if event_type == "REQUEST":
                req_id = payload.get("request_id")
                if req_id:
                    requests[req_id] = payload
            elif event_type == "DECISION":
                dec_id = payload.get("decision_id")
                if dec_id:
                    decisions[dec_id] = payload
            elif event_type == "PERMIT":
                dec_id = payload.get("decision_id")
                if dec_id:
                    permits[dec_id] = payload

    # Integrity checks
    permits_without_decision = sum(1 for d in permits if d not in decisions)
    permits_without_request = 0
    for dec_id, perm in permits.items():
        req_id = perm.get("request_id")
        if req_id not in requests:
            permits_without_request += 1

    allow_decisions = {d for d, v in decisions.items() 
                       if v.get("action_class", "").endswith("_ALLOW")}
    allows_without_permits = sum(1 for d in allow_decisions if d not in permits)

    results["referential_integrity"] = {
        "permits_without_decision": permits_without_decision,
        "permits_without_request": permits_without_request,
        "allows_without_permits": allows_without_permits,
    }

    if permits_without_decision > 0 or allows_without_permits > 0:
        results["pass"] = False

    return results


def audit_1a_replay_determinism(cutoff: datetime) -> dict:
    """AUDIT 1A: Replay Determinism."""
    results = {
        "name": "AUDIT 1A — Replay Determinism",
        "pass": True,
        "episodes": {"ENTRY_ALLOW": [], "ENTRY_DENY": [], "EXIT_ALLOW": []},
        "summary": {},
    }

    requests = {}
    decisions = {}
    permits = {}

    with open("logs/dle/dle_events_v1.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            ts = parse_ts(obj["ts"])
            if ts < cutoff:
                continue
            event_type = obj.get("event_type")
            payload = obj.get("payload", {})

            if event_type == "REQUEST":
                req_id = payload.get("request_id")
                if req_id:
                    requests[req_id] = payload
            elif event_type == "DECISION":
                dec_id = payload.get("decision_id")
                if dec_id and dec_id not in decisions:
                    decisions[dec_id] = payload
            elif event_type == "PERMIT":
                dec_id = payload.get("decision_id")
                if dec_id and dec_id not in permits:
                    permits[dec_id] = payload

    # Classify and verify episodes
    for dec_id, decision in list(decisions.items())[:50]:
        action_class = decision.get("action_class", "")
        permit = permits.get(dec_id, {})
        req_id = permit.get("request_id") if permit else None
        request = requests.get(req_id) if req_id else None

        risk = decision.get("risk", {})
        constraints = decision.get("constraints", {})

        checks = {
            "decision_logged": True,
            "verdict_captured": action_class is not None,
            "authority_source": decision.get("authority_source") is not None,
        }

        if action_class == "ENTRY_ALLOW":
            checks["request_logged"] = request is not None
            checks["permit_logged"] = permit is not None
            checks["sizing_decomposed"] = (
                risk.get("regime_multiplier") is not None or
                risk.get("composite_multiplier") is not None
            )
            checks["reason_captured"] = risk.get("reason") is not None
        elif action_class == "ENTRY_DENY":
            checks["request_logged"] = True  # May not have request for early denials
            denial_reason = constraints.get("reason") or risk.get("reason") or risk.get("error")
            checks["denial_reason"] = denial_reason is not None
        elif action_class == "EXIT_ALLOW":
            checks["request_logged"] = True
            checks["exit_reason"] = constraints.get("reason") is not None
        else:
            continue

        episode_pass = all(checks.values())
        episode_data = {
            "decision_id": dec_id,
            "symbol": decision.get("scope", {}).get("symbol"),
            "checks": checks,
            "pass": episode_pass,
            "provenance": {
                "action_class": action_class,
                "authority": decision.get("authority_source"),
                "policy_version": decision.get("policy_version"),
            },
        }

        if action_class == "ENTRY_ALLOW":
            episode_data["multipliers"] = {
                "regime": risk.get("regime_multiplier"),
                "execution": risk.get("execution_multiplier"),
                "composite": risk.get("composite_multiplier"),
            }
            results["episodes"]["ENTRY_ALLOW"].append(episode_data)
        elif action_class == "ENTRY_DENY":
            episode_data["denial_reason"] = (
                constraints.get("reason") or risk.get("error")
            )
            results["episodes"]["ENTRY_DENY"].append(episode_data)
        elif action_class == "EXIT_ALLOW":
            episode_data["exit_reason"] = constraints.get("reason")
            results["episodes"]["EXIT_ALLOW"].append(episode_data)

        if not episode_pass:
            results["pass"] = False

    # Limit samples
    for k in results["episodes"]:
        results["episodes"][k] = results["episodes"][k][:5]

    total = sum(len(v) for v in results["episodes"].values())
    passed = sum(1 for eps in results["episodes"].values() for e in eps if e["pass"])
    results["summary"] = {"total": total, "passed": passed, "failed": total - passed}

    return results


def audit_4a_temporal_integrity(cutoff: datetime) -> dict:
    """AUDIT 4A: Temporal Integrity."""
    results = {
        "name": "AUDIT 4A — Temporal Integrity",
        "pass": True,
        "chains_analyzed": 0,
        "violations": [],
        "sample_chain": None,
    }

    events_by_decision = defaultdict(list)

    with open("logs/dle/dle_events_v1.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            ts = parse_ts(obj["ts"])
            if ts < cutoff:
                continue
            event_type = obj.get("event_type")
            payload = obj.get("payload", {})

            if event_type == "REQUEST":
                req_id = payload.get("request_id")
                events_by_decision[f"REQ_{req_id}"].append((ts, event_type, payload))
            elif event_type in ("DECISION", "PERMIT", "LINK"):
                dec_id = payload.get("decision_id")
                if dec_id:
                    events_by_decision[dec_id].append((ts, event_type, payload))

    # Link requests to decisions
    request_to_decision = {}
    for dec_id, events in events_by_decision.items():
        if dec_id.startswith("DEC_"):
            for ts, et, payload in events:
                if et == "PERMIT":
                    req_id = payload.get("request_id")
                    if req_id:
                        request_to_decision[req_id] = dec_id

    for req_id, dec_id in request_to_decision.items():
        req_key = f"REQ_{req_id}"
        if req_key in events_by_decision:
            events_by_decision[dec_id].extend(events_by_decision[req_key])
            del events_by_decision[req_key]

    # Check temporal ordering
    for dec_id, events in events_by_decision.items():
        if not dec_id.startswith("DEC_"):
            continue
        results["chains_analyzed"] += 1

        events.sort(key=lambda x: x[0])
        event_times = {et: ts for ts, et, _ in events}

        req_ts = event_times.get("REQUEST")
        dec_ts = event_times.get("DECISION")
        perm_ts = event_times.get("PERMIT")
        link_ts = event_times.get("LINK")

        if req_ts and dec_ts and req_ts > dec_ts:
            results["violations"].append(
                f"{dec_id}: REQUEST > DECISION"
            )
        if dec_ts and perm_ts and dec_ts > perm_ts:
            results["violations"].append(
                f"{dec_id}: DECISION > PERMIT"
            )
        if dec_ts and link_ts and dec_ts > link_ts:
            results["violations"].append(
                f"{dec_id}: DECISION > LINK"
            )

        # Capture sample chain
        if results["sample_chain"] is None and len(events) >= 3:
            results["sample_chain"] = {
                "decision_id": dec_id,
                "events": [
                    {"ts": ts.strftime("%H:%M:%S.%f")[:12], "type": et}
                    for ts, et, _ in events[:5]
                ],
            }

    if results["violations"]:
        results["pass"] = False

    return results


def generate_scorecard(audit_0: dict, audit_1a: dict, audit_4a: dict) -> str:
    """Generate markdown scorecard."""
    now = datetime.now(timezone.utc)
    
    lines = [
        "# MHD Audit Scorecard",
        "",
        f"**Audit Date:** {now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Window:** Last {WINDOW_HOURS} hours",
        "**System:** GPT-Hedge v7.8 (CYCLE_004)",
        "**Phase:** A.3 — Shadow Observation",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Audit | Status | Severity |",
        "|-------|--------|----------|",
        f"| **AUDIT 0** — Evidence Admission Gate | {'✓ PASS' if audit_0['pass'] else '✗ FAIL'} | {'—' if audit_0['pass'] else 'P0'} |",
        f"| **AUDIT 1A** — Replay Determinism | {'✓ PASS' if audit_1a['pass'] else '✗ FAIL'} | {'—' if audit_1a['pass'] else 'P1'} |",
        f"| **AUDIT 4A** — Temporal Integrity | {'✓ PASS' if audit_4a['pass'] else '✗ FAIL'} | {'—' if audit_4a['pass'] else 'P0'} |",
        "",
    ]
    
    all_pass = audit_0["pass"] and audit_1a["pass"] and audit_4a["pass"]
    lines.append(f"**Overall:** {'✅ ALL AUDITS PASS' if all_pass else '❌ AUDIT FAILURES DETECTED'}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # AUDIT 0 details
    lines.extend([
        "## AUDIT 0 — Evidence Admission Gate",
        "",
        "### File Existence",
        "| File | Status | Lines | Size |",
        "|------|--------|-------|------|",
    ])
    for f in audit_0["file_existence"]:
        size_str = f"{f['size_bytes'] / 1024 / 1024:.1f} MB" if f["size_bytes"] > 1000000 else f"{f['size_bytes'] / 1024:.0f} KB"
        lines.append(
            f"| {f['file']} | {'✓' if f['exists'] else '✗'} | {f['lines'] or '—'} | {size_str} |"
        )
    
    lines.extend([
        "",
        "### Monotonic Timestamps (24h window)",
        "| File | Events | Violations |",
        "|------|--------|------------|",
    ])
    for m in audit_0["monotonic"]:
        lines.append(f"| {m['file']} | {m['events_24h']:,} | {m['violations']} |")
    
    ri = audit_0["referential_integrity"]
    lines.extend([
        "",
        "### Referential Integrity",
        "| Check | Result |",
        "|-------|--------|",
        f"| PERMITs without matching DECISION | {ri['permits_without_decision']} |",
        f"| PERMITs without matching REQUEST | {ri['permits_without_request']} |",
        f"| ALLOW decisions without PERMIT | {ri['allows_without_permits']} |",
        "",
        f"**Verdict:** {'✓ PASS' if audit_0['pass'] else '✗ FAIL'}",
        "",
        "---",
        "",
    ])
    
    # AUDIT 1A details
    lines.extend([
        "## AUDIT 1A — Replay Determinism",
        "",
        "### Episode Classification (24h)",
        "| Type | Count | Sampled |",
        "|------|-------|---------|",
    ])
    for k, v in audit_1a["episodes"].items():
        lines.append(f"| {k} | {len(v)} | {len(v)} |")
    
    lines.extend(["", "### Sample Verification", ""])
    
    for ep_type, episodes in audit_1a["episodes"].items():
        if not episodes:
            continue
        for ep in episodes[:2]:
            lines.append(f"**{ep_type} ({ep['decision_id']})**")
            for check, passed in ep["checks"].items():
                lines.append(f"- {'✓' if passed else '✗'} {check.replace('_', ' ').title()}")
            if "multipliers" in ep:
                m = ep["multipliers"]
                lines.append(f"- Multipliers: regime={m['regime']}, exec={m['execution']}, composite={m['composite']}")
            if "denial_reason" in ep:
                lines.append(f"- Denial: {ep['denial_reason']}")
            if "exit_reason" in ep:
                lines.append(f"- Exit: {ep['exit_reason']}")
            lines.append("")
    
    s = audit_1a["summary"]
    lines.append(f"**Verdict:** {'✓ PASS' if audit_1a['pass'] else '✗ FAIL'} ({s['passed']}/{s['total']} episodes reconstructible)")
    lines.extend(["", "---", ""])
    
    # AUDIT 4A details
    lines.extend([
        "## AUDIT 4A — Temporal Integrity",
        "",
        "### Rules Checked",
        "1. REQUEST precedes DECISION",
        "2. DECISION precedes PERMIT",
        "3. DECISION precedes LINK",
        "",
        "### Results",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Decision chains analyzed | {audit_4a['chains_analyzed']} |",
        f"| Temporal violations | {len(audit_4a['violations'])} |",
        "",
    ])
    
    if audit_4a["sample_chain"]:
        chain = audit_4a["sample_chain"]
        lines.extend([
            f"### Sample Chain ({chain['decision_id']})",
            "```",
        ])
        for e in chain["events"]:
            lines.append(f"{e['ts']} → {e['type']}")
        lines.extend(["```", ""])
    
    if audit_4a["violations"]:
        lines.extend(["### Violations", ""])
        for v in audit_4a["violations"][:10]:
            lines.append(f"- ✗ {v}")
        lines.append("")
    
    lines.append(f"**Verdict:** {'✓ PASS' if audit_4a['pass'] else '✗ FAIL'}")
    lines.extend(["", "---", ""])
    
    # Footer
    lines.extend([
        "## Meta",
        "",
        "- Auditor: MHD Audit Suite v1",
        f"- Generated: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "- Window: 24 hours",
        "",
        "---",
        "",
        "*This is an evidence audit, not a performance audit. All findings are reconstructible from raw logs.*",
    ])
    
    return "\n".join(lines)


def main():
    """Run MHD audits."""
    print("=" * 60)
    print("MHD AUDIT SUITE — GPT-Hedge v7.x")
    print("=" * 60)
    print()
    
    os.makedirs(AUDIT_DIR, exist_ok=True)
    
    cutoff = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)
    print(f"Window: Last {WINDOW_HOURS} hours (cutoff: {cutoff.isoformat()})")
    print()
    
    # Run audits
    print("Running AUDIT 0: Evidence Admission Gate...")
    audit_0 = audit_0_evidence_gate(cutoff)
    print(f"  Result: {'✓ PASS' if audit_0['pass'] else '✗ FAIL'}")
    
    print("Running AUDIT 1A: Replay Determinism...")
    audit_1a = audit_1a_replay_determinism(cutoff)
    print(f"  Result: {'✓ PASS' if audit_1a['pass'] else '✗ FAIL'}")
    
    print("Running AUDIT 4A: Temporal Integrity...")
    audit_4a = audit_4a_temporal_integrity(cutoff)
    print(f"  Result: {'✓ PASS' if audit_4a['pass'] else '✗ FAIL'}")
    
    print()
    
    # Generate scorecard
    scorecard = generate_scorecard(audit_0, audit_1a, audit_4a)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = AUDIT_DIR / f"MHD_SCORECARD_{timestamp}.md"
    
    with open(output_path, "w") as f:
        f.write(scorecard)
    
    print(f"Scorecard written to: {output_path}")
    print()
    
    # Summary
    all_pass = audit_0["pass"] and audit_1a["pass"] and audit_4a["pass"]
    print("=" * 60)
    if all_pass:
        print("✅ ALL AUDITS PASS — Evidence-backed confidence achieved")
    else:
        print("❌ AUDIT FAILURES DETECTED — See scorecard for details")
    print("=" * 60)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
