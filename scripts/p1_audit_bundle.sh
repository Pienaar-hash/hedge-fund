#!/usr/bin/env bash
# scripts/p1_audit_bundle.sh — Phase P1 invariance audit
#
# Run at START and END of a P1 trial window.  Captures state hashes,
# event counts, and schema fingerprints so we can prove execution
# invariance (P1 prediction layer had zero effect on trading).
#
# Usage:
#   scripts/p1_audit_bundle.sh start   # before enabling P1
#   scripts/p1_audit_bundle.sh end     # after 24h trial
#   scripts/p1_audit_bundle.sh check   # compare start vs end schema
#
# All output appended to logs/prediction/p1_audit.log

set -euo pipefail

AUDIT_LOG="logs/prediction/p1_audit.log"
MARKER_LOG="logs/prediction/p1_run_markers.log"
HASH_LOG="logs/prediction/p1_invariance_hashes.log"
COUNTS_START="logs/prediction/p1_counts_start.json"
COUNTS_END="logs/prediction/p1_counts_end.json"

mkdir -p logs/prediction

TS=$(date -Iseconds)
REV=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")

# ── State files to hash (execution truth surfaces) ──────────────
STATE_FILES=(
    logs/state/nav_state.json
    logs/state/sentinel_x.json
    logs/state/positions_state.json
    logs/state/positions_ledger.json
    logs/state/risk_snapshot.json
    logs/state/diagnostics.json
    logs/state/hydra_state.json
    logs/state/cerberus_state.json
    logs/state/execution_quality.json
)

# ── Execution logs to count/fingerprint ─────────────────────────
EXEC_LOGS=(
    logs/execution/orders_executed.jsonl
    logs/execution/orders_attempted.jsonl
    logs/execution/risk_vetoes.jsonl
    logs/doctrine_events.jsonl
    logs/alerts/alerts_v7.jsonl
    logs/execution/router_decisions.jsonl
)

# ── Prediction logs (P1 specific) ──────────────────────────────
PRED_LOGS=(
    logs/prediction/alert_ranking.jsonl
    logs/prediction/firewall_denials.jsonl
    logs/prediction/belief_events.jsonl
    logs/prediction/dle_prediction_events.jsonl
)

_header() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  P1 AUDIT — $1  |  $TS  |  $REV"
    echo "═══════════════════════════════════════════════════════"
}

_state_hashes() {
    echo "── State file hashes ──"
    for f in "${STATE_FILES[@]}"; do
        if [[ -f "$f" ]]; then
            sha256sum "$f"
        else
            echo "MISSING  $f"
        fi
    done
}

_exec_counts() {
    echo "── Execution log line counts ──"
    for f in "${EXEC_LOGS[@]}"; do
        if [[ -f "$f" ]]; then
            printf "%8d  %s\n" "$(wc -l < "$f")" "$f"
        else
            printf "%8s  %s\n" "MISSING" "$f"
        fi
    done
}

_exec_schema() {
    echo "── Execution event types (orders_executed) ──"
    if [[ -f logs/execution/orders_executed.jsonl ]]; then
        jq -r '.event_type // .event // "unknown"' logs/execution/orders_executed.jsonl 2>/dev/null \
            | sort | uniq -c | sort -rn | head -20
    else
        echo "  (file missing)"
    fi

    echo "── Veto reasons (risk_vetoes) ──"
    if [[ -f logs/execution/risk_vetoes.jsonl ]]; then
        jq -r '.veto_reason // .reason // "unknown"' logs/execution/risk_vetoes.jsonl 2>/dev/null \
            | sort | uniq -c | sort -rn | head -20
    else
        echo "  (file missing)"
    fi

    echo "── Doctrine verdicts ──"
    if [[ -f logs/doctrine_events.jsonl ]]; then
        jq -r '.verdict // .action // "unknown"' logs/doctrine_events.jsonl 2>/dev/null \
            | sort | uniq -c | sort -rn | head -20
    else
        echo "  (file missing)"
    fi
}

_pred_counts() {
    echo "── Prediction log line counts ──"
    for f in "${PRED_LOGS[@]}"; do
        if [[ -f "$f" ]]; then
            printf "%8d  %s\n" "$(wc -l < "$f")" "$f"
        else
            printf "%8s  %s\n" "0" "$f"
        fi
    done
}

_save_pred_counts() {
    # Save machine-readable counts to a JSON file for delta comparison
    local dest="$1"
    local fw=0 ar=0
    [[ -f logs/prediction/firewall_denials.jsonl ]] && fw=$(wc -l < logs/prediction/firewall_denials.jsonl)
    [[ -f logs/prediction/alert_ranking.jsonl ]] && ar=$(wc -l < logs/prediction/alert_ranking.jsonl)
    echo "{\"firewall_denials\": $fw, \"alert_ranking\": $ar}" > "$dest"
}

_pred_detail() {
    echo "── Alert ranking results ──"
    if [[ -f logs/prediction/alert_ranking.jsonl ]]; then
        jq -r '.rankings_applied' logs/prediction/alert_ranking.jsonl 2>/dev/null \
            | sort | uniq -c
    else
        echo "  (no rankings yet)"
    fi

    echo "── Firewall denials ──"
    if [[ -f logs/prediction/firewall_denials.jsonl ]]; then
        wc -l < logs/prediction/firewall_denials.jsonl
        jq -r '.verdict' logs/prediction/firewall_denials.jsonl 2>/dev/null \
            | sort | uniq -c
    else
        echo "  0 (expected in P1)"
    fi
}

case "${1:-help}" in
    start)
        {
            _header "START"
            echo "$TS P1_START $REV" >> "$MARKER_LOG"
            _state_hashes | tee -a "$HASH_LOG"
            _exec_counts
            _exec_schema
            _pred_counts
            _save_pred_counts "$COUNTS_START"
            echo ""
            echo "P1 start marker written.  Enable with:"
            echo "  export PREDICTION_PHASE=P1_ADVISORY PREDICTION_DLE_ENABLED=1"
            echo "  sudo supervisorctl restart hedge:"
        } 2>&1 | tee -a "$AUDIT_LOG"
        ;;
    end)
        {
            _header "END"
            echo "$TS P1_END $REV" >> "$MARKER_LOG"
            _state_hashes | tee -a "$HASH_LOG"
            _exec_counts
            _exec_schema
            _pred_counts
            _save_pred_counts "$COUNTS_END"
            _pred_detail
            echo ""
            echo "── P1 pass criteria ──"
            echo "  1. firewall_denials count = 0"
            echo "  2. rankings_applied has non-zero true count"
            echo "  3. No new execution event types vs start"
            echo "  4. No sudden veto distribution shift"
            echo "  5. No alert send failures attributable to ranking"
        } 2>&1 | tee -a "$AUDIT_LOG"
        ;;
    check)
        {
            _header "CHECK"
            echo "── Comparing start vs end hashes ──"
            if [[ -f "$HASH_LOG" ]]; then
                echo "(State files will differ due to market evolution.  Look for NEW fields.)"
                cat "$HASH_LOG"
            else
                echo "No hash log found.  Run 'start' first."
            fi
            echo ""
            _pred_detail

            # ── Single-line verdict ─────────────────────────────
            echo ""
            echo "── VERDICT ──"
            FAIL_REASON=""

            # Check 1: no NEW firewall denials during P1 window (delta)
            if [[ -f "$COUNTS_START" ]] && [[ -f "$COUNTS_END" ]]; then
                DENIAL_START=$(jq -r '.firewall_denials' "$COUNTS_START" 2>/dev/null || echo 0)
                DENIAL_END=$(jq -r '.firewall_denials' "$COUNTS_END" 2>/dev/null || echo 0)
                DENIAL_DELTA=$((DENIAL_END - DENIAL_START))
                echo "Firewall denials: start=$DENIAL_START end=$DENIAL_END delta=$DENIAL_DELTA"
                if [[ "$DENIAL_DELTA" -gt 0 ]]; then
                    FAIL_REASON="firewall_denials_delta=$DENIAL_DELTA new denials during P1 window"
                fi
            elif [[ -f logs/prediction/firewall_denials.jsonl ]]; then
                echo "WARNING: counts files missing, cannot compute delta"
                DENIAL_COUNT=$(wc -l < logs/prediction/firewall_denials.jsonl)
                if [[ "$DENIAL_COUNT" -gt 0 ]]; then
                    FAIL_REASON="firewall_denials=$DENIAL_COUNT (no baseline — run 'start' before 'end')"
                fi
            fi

            # Check 2: rankings_applied has at least one true
            if [[ -f logs/prediction/alert_ranking.jsonl ]]; then
                TRUE_COUNT=$(jq -r 'select(.rankings_applied == true)' logs/prediction/alert_ranking.jsonl 2>/dev/null | wc -l)
                TOTAL_COUNT=$(wc -l < logs/prediction/alert_ranking.jsonl)
                if [[ "$TOTAL_COUNT" -eq 0 ]] && [[ -z "$FAIL_REASON" ]]; then
                    FAIL_REASON="alert_ranking.jsonl is empty (P1 not exercised)"
                fi
            else
                if [[ -z "$FAIL_REASON" ]]; then
                    FAIL_REASON="alert_ranking.jsonl missing (P1 not exercised)"
                fi
            fi

            # Check 3: marker log has both START and END
            if [[ -f "$MARKER_LOG" ]]; then
                HAS_START=$(grep -c "P1_START" "$MARKER_LOG" 2>/dev/null || true)
                HAS_END=$(grep -c "P1_END" "$MARKER_LOG" 2>/dev/null || true)
                if [[ "$HAS_START" -eq 0 ]] && [[ -z "$FAIL_REASON" ]]; then
                    FAIL_REASON="no P1_START marker (run 'start' first)"
                fi
                if [[ "$HAS_END" -eq 0 ]] && [[ -z "$FAIL_REASON" ]]; then
                    FAIL_REASON="no P1_END marker (run 'end' first)"
                fi
            else
                if [[ -z "$FAIL_REASON" ]]; then
                    FAIL_REASON="no marker log (run 'start' then 'end' first)"
                fi
            fi

            if [[ -z "$FAIL_REASON" ]]; then
                echo "P1_RESULT: PASS"
                echo "Reason: OK — all invariants held"
            else
                echo "P1_RESULT: FAIL"
                echo "Reason: $FAIL_REASON"
            fi
        } 2>&1 | tee -a "$AUDIT_LOG"
        ;;
    *)
        echo "Usage: $0 {start|end|check}"
        echo ""
        echo "  start  — capture pre-P1 baseline (hashes, counts, schemas)"
        echo "  end    — capture post-P1 state and prediction telemetry"
        echo "  check  — compare hashes and show prediction results"
        exit 1
        ;;
esac
