#!/usr/bin/env python3
"""
CYCLE_003 Bootstrap — Initialize fresh observation cycle.

This script:
1. Validates CYCLE_002 archive integrity
2. Resets working counters (episode IDs continue from 444)
3. Creates fresh regime_pressure state
4. Logs cycle transition event
5. Does NOT touch doctrine or execution logic

Usage:
    python scripts/bootstrap_cycle_003.py [--dry-run]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def validate_cycle_002_archive() -> bool:
    """Verify CYCLE_002 archive exists and is complete."""
    archive_path = Path("archive/cycle_002")
    manifest_path = archive_path / "MANIFEST.json"
    
    if not manifest_path.exists():
        print("❌ CYCLE_002 archive manifest not found")
        return False
    
    manifest = json.loads(manifest_path.read_text())
    
    # Check required files exist
    required = [
        "logs/doctrine_events.jsonl",
        "logs/orders_executed.jsonl",
        "state/episode_ledger.json",
        "reports/CYCLE_002_Postmortem.md",
    ]
    
    archived = manifest.get("archived_files", [])
    missing = [f for f in required if f not in archived]
    
    if missing:
        print(f"❌ Missing required archive files: {missing}")
        return False
    
    print(f"✓ CYCLE_002 archive validated ({len(archived)} files, {manifest.get('total_archive_size_mb', 0)} MB)")
    return True


def reset_regime_pressure(dry_run: bool = False) -> None:
    """Reset regime pressure to fresh state."""
    pressure_path = Path("logs/state/regime_pressure.json")
    
    fresh_state = {
        "confidence": 0.0,
        "regime": "unknown",
        "dwell_hours": 0.0,
        "near_flips_24h": 0,
        "churn_24h": 0,
        "distance_to_stable": 1.0,
        "updated_ts": datetime.now(timezone.utc).isoformat(),
        "cycle": "CYCLE_003",
        "note": "Fresh state for CYCLE_003 bootstrap"
    }
    
    if dry_run:
        print(f"[DRY RUN] Would reset regime_pressure.json")
    else:
        pressure_path.write_text(json.dumps(fresh_state, indent=2))
        print("✓ Reset regime_pressure.json")


def log_cycle_transition(dry_run: bool = False) -> None:
    """Log the cycle transition event."""
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "CYCLE_TRANSITION",
        "from_cycle": "CYCLE_002",
        "to_cycle": "CYCLE_003",
        "doctrine_version": "v7.8",
        "episode_counter_start": 444,
        "notes": [
            "CYCLE_002 archived with 443 episodes",
            "Doctrine frozen - no parameter changes",
            "Episode ledger continues from EP_0444"
        ]
    }
    
    events_path = Path("logs/cycle_transitions.jsonl")
    
    if dry_run:
        print(f"[DRY RUN] Would log transition event")
    else:
        with open(events_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        print("✓ Logged cycle transition event")


def update_cycle_manifest(dry_run: bool = False) -> None:
    """Update CYCLE_003 manifest with bootstrap timestamp."""
    manifest_path = Path("logs/CYCLE_003_MANIFEST.json")
    
    if not manifest_path.exists():
        print("❌ CYCLE_003 manifest not found")
        return
    
    manifest = json.loads(manifest_path.read_text())
    manifest["bootstrapped_at"] = datetime.now(timezone.utc).isoformat() + "Z"
    manifest["bootstrap_state"]["validated"] = True
    
    if dry_run:
        print(f"[DRY RUN] Would update CYCLE_003 manifest")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print("✓ Updated CYCLE_003 manifest")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CYCLE_003")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CYCLE_003 BOOTSTRAP")
    print("=" * 60)
    print()
    
    # Step 1: Validate archive
    print("Step 1: Validate CYCLE_002 archive")
    if not validate_cycle_002_archive():
        print("\n❌ Bootstrap aborted: archive validation failed")
        sys.exit(1)
    print()
    
    # Step 2: Reset working state
    print("Step 2: Reset working state")
    reset_regime_pressure(args.dry_run)
    print()
    
    # Step 3: Log transition
    print("Step 3: Log cycle transition")
    log_cycle_transition(args.dry_run)
    print()
    
    # Step 4: Update manifest
    print("Step 4: Update CYCLE_003 manifest")
    update_cycle_manifest(args.dry_run)
    print()
    
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN COMPLETE — no changes made")
    else:
        print("✓ CYCLE_003 BOOTSTRAP COMPLETE")
        print()
        print("Next steps:")
        print("  1. Restart executor: sudo supervisorctl restart hedge:hedge-executor")
        print("  2. Monitor regime_pressure.json for first readings")
        print("  3. Episode ledger will continue from EP_0444")
    print("=" * 60)


if __name__ == "__main__":
    main()
