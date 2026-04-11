#!/usr/bin/env bash
# exec-determinism-guard.sh — Continuous executor determinism monitor
# Flags when executor_live enters non-deterministic territory:
#   - Process swap > threshold
#   - System swap-in/out activity
#   - Memory PSI stalls
#   - Involuntary context switch rate spike
#
# Usage:
#   ./bin/exec-determinism-guard.sh              # run once
#   watch -n 10 ./bin/exec-determinism-guard.sh  # continuous (every 10s)
#   ./bin/exec-determinism-guard.sh --loop 10    # built-in loop, 10s interval
#
# Exit codes:
#   0 = execution-grade
#   1 = degraded (non-deterministic conditions detected)
#   2 = executor not running

set -euo pipefail

# --- Thresholds (tune to your tolerance) ---
SWAP_PROC_KB=10240        # 10 MB process swap → flag
SWAP_SYS_MB=400           # 400 MB system swap used → flag
MEM_PSI_THRESH="1.00"     # memory PSI some avg10 > 1% → flag
CPU_PSI_THRESH="10.00"    # cpu PSI some avg10 > 10% → flag
INVOL_CS_RATE=500         # involuntary ctx switches/interval → flag

RED='\033[0;31m'
YEL='\033[0;33m'
GRN='\033[0;32m'
RST='\033[0m'

VIOLATIONS=0

flag() {
    local sev="$1" msg="$2"
    if [[ "$sev" == "WARN" ]]; then
        printf "${YEL}⚠ %-12s${RST} %s\n" "$sev" "$msg"
    else
        printf "${RED}✖ %-12s${RST} %s\n" "$sev" "$msg"
    fi
    VIOLATIONS=$((VIOLATIONS + 1))
}

ok() {
    printf "${GRN}✓ %-12s${RST} %s\n" "OK" "$1"
}

# --- Find executor PID ---
EXEC_PID=$(pgrep -f "execution.executor_live" 2>/dev/null | head -1 || true)
if [[ -z "$EXEC_PID" ]]; then
    echo "executor_live not running"
    exit 2
fi

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "=== exec-determinism-guard  ${TS}  PID=${EXEC_PID} ==="
echo ""

# --- 1. Executor process swap ---
PROC_SWAP_KB=$(awk '/^VmSwap:/{print $2}' /proc/"$EXEC_PID"/status 2>/dev/null || echo 0)
PROC_RSS_KB=$(awk '/^VmRSS:/{print $2}' /proc/"$EXEC_PID"/status 2>/dev/null || echo 0)
PROC_SWAP_MB=$((PROC_SWAP_KB / 1024))
PROC_RSS_MB=$((PROC_RSS_KB / 1024))

if [[ "$PROC_SWAP_KB" -gt "$SWAP_PROC_KB" ]]; then
    flag "CRITICAL" "executor VmSwap=${PROC_SWAP_MB}MB (threshold: $((SWAP_PROC_KB/1024))MB) — pages paged out"
else
    ok "executor VmSwap=${PROC_SWAP_MB}MB  RSS=${PROC_RSS_MB}MB"
fi

# --- 2. System swap usage ---
SYS_SWAP_USED_KB=$(awk '/^SwapTotal:|^SwapFree:/{a[NR]=$2} END{print a[1]-a[2]}' /proc/meminfo 2>/dev/null || echo 0)
# Handle potential negative from race
SYS_SWAP_USED_KB=$((SYS_SWAP_USED_KB < 0 ? 0 : SYS_SWAP_USED_KB))
SYS_SWAP_USED_MB=$((SYS_SWAP_USED_KB / 1024))

if [[ "$SYS_SWAP_USED_MB" -gt "$SWAP_SYS_MB" ]]; then
    flag "WARN" "system swap=${SYS_SWAP_USED_MB}MB (threshold: ${SWAP_SYS_MB}MB)"
else
    ok "system swap=${SYS_SWAP_USED_MB}MB"
fi

# --- 3. vmstat swap-in/out (single sample) ---
VMSTAT_LINE=$(vmstat 1 2 2>/dev/null | tail -1)
SI=$(echo "$VMSTAT_LINE" | awk '{print $7}')
SO=$(echo "$VMSTAT_LINE" | awk '{print $8}')
if [[ "$SI" -gt 0 || "$SO" -gt 0 ]]; then
    flag "WARN" "active swap I/O: si=${SI} so=${SO} pages/s"
else
    ok "swap I/O silent: si=0 so=0"
fi

# --- 4. Memory PSI ---
if [[ -f /proc/pressure/memory ]]; then
    MEM_PSI=$(awk '/^some/{print $2}' /proc/pressure/memory | sed 's/avg10=//')
    EXCEEDED=$(awk "BEGIN{print ($MEM_PSI > $MEM_PSI_THRESH) ? 1 : 0}")
    if [[ "$EXCEEDED" -eq 1 ]]; then
        flag "CRITICAL" "memory PSI some avg10=${MEM_PSI}% (threshold: ${MEM_PSI_THRESH}%)"
    else
        ok "memory PSI avg10=${MEM_PSI}%"
    fi
fi

# --- 5. CPU PSI ---
if [[ -f /proc/pressure/cpu ]]; then
    CPU_PSI=$(awk '/^some/{print $2}' /proc/pressure/cpu | sed 's/avg10=//')
    EXCEEDED=$(awk "BEGIN{print ($CPU_PSI > $CPU_PSI_THRESH) ? 1 : 0}")
    if [[ "$EXCEEDED" -eq 1 ]]; then
        flag "WARN" "CPU PSI some avg10=${CPU_PSI}% (threshold: ${CPU_PSI_THRESH}%)"
    else
        ok "CPU PSI avg10=${CPU_PSI}%"
    fi
fi

# --- 6. Involuntary context switches (rate) ---
INVOL_CS=$(awk '/^nonvoluntary_ctxt_switches:/{print $2}' /proc/"$EXEC_PID"/status 2>/dev/null || echo 0)
# Store last value for rate calculation
STATE_FILE="/tmp/.exec_determinism_guard_state"
NOW_S=$(date +%s)
if [[ -f "$STATE_FILE" ]]; then
    read -r PREV_CS PREV_TS < "$STATE_FILE" 2>/dev/null || { PREV_CS=0; PREV_TS=0; }
    DELTA_CS=$((INVOL_CS - PREV_CS))
    DELTA_T=$((NOW_S - PREV_TS))
    if [[ "$DELTA_T" -gt 0 && "$DELTA_T" -lt 300 ]]; then
        RATE=$((DELTA_CS / DELTA_T))
        if [[ "$RATE" -gt "$INVOL_CS_RATE" ]]; then
            flag "WARN" "involuntary ctx switch rate=${RATE}/s (threshold: ${INVOL_CS_RATE}/s)"
        else
            ok "involuntary ctx switch rate=${RATE}/s"
        fi
    else
        ok "involuntary ctx switches=${INVOL_CS} (rate: first sample)"
    fi
else
    ok "involuntary ctx switches=${INVOL_CS} (rate: first sample)"
fi
echo "$INVOL_CS $NOW_S" > "$STATE_FILE"

# --- 7. Available memory headroom ---
AVAIL_KB=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
AVAIL_MB=$((AVAIL_KB / 1024))
TOTAL_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
AVAIL_PCT=$((AVAIL_KB * 100 / TOTAL_KB))
if [[ "$AVAIL_PCT" -lt 15 ]]; then
    flag "WARN" "available memory=${AVAIL_MB}MB (${AVAIL_PCT}% of total)"
else
    ok "available memory=${AVAIL_MB}MB (${AVAIL_PCT}%)"
fi

# --- Machine-readable output (for executor integration) ---
# Emitted AFTER human-readable so parsers can read last lines
echo ""
if [[ "$VIOLATIONS" -eq 0 ]]; then
    printf "${GRN}VERDICT: EXECUTION-GRADE${RST}  (all checks passed)\n"
    echo "DETERMINISM_STATUS=OK"
    echo "DETERMINISM_VIOLATIONS=0"
    exit 0
else
    printf "${RED}VERDICT: NON-DETERMINISTIC${RST}  (%d violation(s) detected)\n" "$VIOLATIONS"
    echo "DETERMINISM_STATUS=DEGRADED"
    echo "DETERMINISM_VIOLATIONS=${VIOLATIONS}"
    echo "DETERMINISM_EXECUTOR_SWAP_KB=${PROC_SWAP_KB}"
    echo "DETERMINISM_SYSTEM_SWAP_MB=${SYS_SWAP_USED_MB}"
    echo "DETERMINISM_AVAIL_PCT=${AVAIL_PCT}"
    exit 1
fi
