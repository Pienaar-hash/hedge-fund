#!/usr/bin/env python3
"""
Disk-pressure watchdog — PreToolUse hook for Claude Code Bash calls.

Checks the partition that contains CLAUDE_CODE_TMPDIR (default: /tmp) before
every Bash tool use. Blocks with a fix message when free space is critical;
warns (but allows) when space is low.

Exit codes (Claude Code PreToolUse convention):
  0  — disk OK, proceed
  2  — disk critical, Bash blocked; stdout message is shown to the agent

Thresholds:
  WARNING:  < 500 MB free  OR > 80% used
  CRITICAL: < 100 MB free  OR > 95% used

Install (once per machine):
    Add to ~/.claude/settings.json under hooks.PreToolUse — see repo README or
    the comment block at the bottom of this file.
"""
from __future__ import annotations

import os
import shutil
import sys

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_WARN_FREE_MB: float = 500.0
_CRITICAL_FREE_MB: float = 100.0
_WARN_PCT: float = 80.0
_CRITICAL_PCT: float = 95.0


def _check(path: str) -> tuple[float, float, float]:
    """Return (free_mb, total_mb, used_pct) for the partition containing path."""
    u = shutil.disk_usage(path)
    free_mb = u.free / 1_048_576
    total_mb = u.total / 1_048_576
    used_pct = 100.0 * u.used / u.total
    return free_mb, total_mb, used_pct


def main() -> int:
    tmpdir = os.environ.get("CLAUDE_CODE_TMPDIR") or "/tmp"

    try:
        free_mb, total_mb, used_pct = _check(tmpdir)
    except OSError as exc:
        # Cannot stat the partition — don't block, just log.
        print(f"[disk_watchdog] WARNING: cannot stat {tmpdir}: {exc}", file=sys.stderr)
        return 0

    is_critical = free_mb < _CRITICAL_FREE_MB or used_pct >= _CRITICAL_PCT
    is_warn = free_mb < _WARN_FREE_MB or used_pct >= _WARN_PCT

    if is_critical:
        # stdout is relayed to the agent by Claude Code on exit 2.
        print(
            f"[disk_watchdog] CRITICAL — {tmpdir} partition at {used_pct:.0f}% used "
            f"({free_mb:.0f} MB free / {total_mb:.0f} MB total). "
            f"Bash is blocked to prevent silent failures.\n"
            f"\n"
            f"Fix options:\n"
            f"  1. Redirect the tmp dir and restart Claude Code:\n"
            f"       mkdir -p /root/claude_tmp\n"
            f"       CLAUDE_CODE_TMPDIR=/root/claude_tmp claude\n"
            f"\n"
            f"  2. Free space on the partition (check first, then delete):\n"
            f"       du -sh {tmpdir}/* 2>/dev/null | sort -rh | head -20\n"
            f"       rm -rf {tmpdir}/claude-*   # only if no other sessions active\n"
        )
        return 2

    if is_warn:
        print(
            f"[disk_watchdog] WARNING — {tmpdir} partition at {used_pct:.0f}% used "
            f"({free_mb:.0f} MB free). Consider freeing space soon.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Hook installation snippet (add to ~/.claude/settings.json):
#
#   "PreToolUse": [
#     {
#       "matcher": "Bash",
#       "hooks": [
#         {
#           "type": "command",
#           "command": "python3 /home/user/hedge-fund/scripts/disk_watchdog.py"
#         }
#       ]
#     }
#   ]
# ---------------------------------------------------------------------------
