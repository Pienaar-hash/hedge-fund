#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# ops/tailscale-harden.sh — Authority boundary hardening for hedge box
#
# Usage:
#   ./ops/tailscale-harden.sh phase0     # Disable password auth + install Tailscale
#   ./ops/tailscale-harden.sh phase1     # Rebind SSH+NGINX to Tailscale IP (deadman armed)
#   ./ops/tailscale-harden.sh phase2     # Enable UFW, cut public surface (deadman armed)
#   ./ops/tailscale-harden.sh confirm    # Cancel deadman — you proved connectivity works
#   ./ops/tailscale-harden.sh rollback   # Manual rollback to pre-hardening state
#   ./ops/tailscale-harden.sh status     # Show current state
#
# Deadman switch:
#   phase1 and phase2 arm a 5-minute revert timer.  If you don't run
#   'confirm' within 5 minutes, everything reverts automatically.
#   This eliminates lockout risk.
#
# Prerequisite: run as root on the hedge VPS.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

BACKUP_DIR="/root/.harden-backups"
DEADMAN_PID_FILE="/tmp/harden-deadman.pid"
DEADMAN_TIMEOUT=300  # 5 minutes
NGINX_CONF="/etc/nginx/sites-available/hedge"
NGINX_CONF_ALT="/etc/nginx/sites-available/hedge-dashboard.conf"
SSHD_CONF="/etc/ssh/sshd_config"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[HARDEN]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*" >&2; exit 1; }

ensure_root() {
    [[ $EUID -eq 0 ]] || fail "Must run as root"
}

ensure_backups_dir() {
    mkdir -p "$BACKUP_DIR"
}

# ── Deadman switch ──────────────────────────────────────────────────

arm_deadman() {
    kill_deadman 2>/dev/null || true
    log "Arming deadman revert (${DEADMAN_TIMEOUT}s timeout)..."
    log "Run './ops/tailscale-harden.sh confirm' within 5 minutes to keep changes."
    (
        sleep "$DEADMAN_TIMEOUT"
        echo "[DEADMAN] Timer expired — reverting changes"
        do_rollback
    ) &
    echo $! > "$DEADMAN_PID_FILE"
    log "Deadman PID: $(cat "$DEADMAN_PID_FILE")"
}

kill_deadman() {
    if [[ -f "$DEADMAN_PID_FILE" ]]; then
        local pid
        pid=$(cat "$DEADMAN_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            log "Deadman (PID $pid) cancelled"
        fi
        rm -f "$DEADMAN_PID_FILE"
    fi
}

# ── Backup / Restore ───────────────────────────────────────────────

backup_file() {
    local src="$1"
    local tag="$2"
    if [[ -f "$src" ]]; then
        cp "$src" "${BACKUP_DIR}/$(basename "$src").${tag}.bak"
        log "Backed up $src → ${BACKUP_DIR}/$(basename "$src").${tag}.bak"
    fi
}

restore_file() {
    local src="$1"
    local tag="$2"
    local bak="${BACKUP_DIR}/$(basename "$src").${tag}.bak"
    if [[ -f "$bak" ]]; then
        cp "$bak" "$src"
        echo "[ROLLBACK] Restored $src from $bak"
    fi
}

# ── Detect active nginx config ─────────────────────────────────────

detect_nginx_conf() {
    if [[ -f "$NGINX_CONF" ]]; then
        echo "$NGINX_CONF"
    elif [[ -f "$NGINX_CONF_ALT" ]]; then
        echo "$NGINX_CONF_ALT"
    else
        echo ""
    fi
}

# ── Phase 0: Immediate — disable password auth + install Tailscale ─

do_phase0() {
    ensure_root
    ensure_backups_dir

    log "═══ Phase 0: Immediate hardening ═══"

    # 0a: Disable SSH password auth
    backup_file "$SSHD_CONF" "phase0"

    if grep -qE '^\s*PasswordAuthentication\s+yes' "$SSHD_CONF"; then
        sed -i 's/^\s*PasswordAuthentication\s\+yes/PasswordAuthentication no/' "$SSHD_CONF"
        log "PasswordAuthentication set to 'no'"
    elif grep -qE '^\s*#\s*PasswordAuthentication' "$SSHD_CONF"; then
        sed -i 's/^\s*#\s*PasswordAuthentication.*/PasswordAuthentication no/' "$SSHD_CONF"
        log "PasswordAuthentication uncommented and set to 'no'"
    else
        echo "PasswordAuthentication no" >> "$SSHD_CONF"
        log "PasswordAuthentication appended as 'no'"
    fi

    # Validate sshd config before restarting
    if sshd -t 2>/dev/null; then
        systemctl restart sshd
        log "sshd restarted (existing connections unaffected)"
    else
        warn "sshd config test failed — restoring backup"
        restore_file "$SSHD_CONF" "phase0"
        fail "sshd config invalid after edit. Aborting."
    fi

    # 0b: Install Tailscale
    if command -v tailscale &>/dev/null; then
        log "Tailscale already installed: $(tailscale version 2>/dev/null | head -1)"
    else
        log "Installing Tailscale..."
        curl -fsSL https://tailscale.com/install.sh | sh
        log "Tailscale installed"
    fi

    # 0c: Bring Tailscale up
    if tailscale status &>/dev/null; then
        log "Tailscale already connected"
        tailscale ip -4
    else
        log "Starting Tailscale auth..."
        echo ""
        warn "═══════════════════════════════════════════════════════"
        warn " A browser auth URL will appear below."
        warn " Open it, authenticate, then return here."
        warn "═══════════════════════════════════════════════════════"
        echo ""
        tailscale up
        echo ""
        log "Tailscale connected. Your Tailscale IP:"
        tailscale ip -4
    fi

    echo ""
    log "Phase 0 complete."
    log ""
    log "Next steps:"
    log "  1. Install Tailscale on your laptop/phone"
    log "  2. Verify you can SSH to $(tailscale ip -4) from your Tailscale device"
    log "  3. Once confirmed, run: ./ops/tailscale-harden.sh phase1"
}

# ── Phase 1: Rebind SSH + NGINX to Tailscale IP ────────────────────

do_phase1() {
    ensure_root
    ensure_backups_dir

    local ts_ip
    ts_ip=$(tailscale ip -4 2>/dev/null) || fail "Tailscale not connected. Run phase0 first."

    log "═══ Phase 1: Rebind services to Tailscale IP ($ts_ip) ═══"

    # Backup
    backup_file "$SSHD_CONF" "phase1"
    local nginx_active
    nginx_active=$(detect_nginx_conf)
    if [[ -n "$nginx_active" ]]; then
        backup_file "$nginx_active" "phase1"
    fi

    # 1a: Rebind sshd
    # Remove any existing ListenAddress lines, then add Tailscale + loopback
    sed -i '/^\s*ListenAddress\s/d' "$SSHD_CONF"
    # Insert after the Port line (or at end)
    if grep -q '^\s*Port\s' "$SSHD_CONF"; then
        sed -i "/^\s*Port\s/a ListenAddress ${ts_ip}\nListenAddress 127.0.0.1" "$SSHD_CONF"
    else
        echo -e "\nListenAddress ${ts_ip}\nListenAddress 127.0.0.1" >> "$SSHD_CONF"
    fi
    log "sshd bound to ${ts_ip} + 127.0.0.1"

    # 1b: Rebind NGINX
    if [[ -n "$nginx_active" ]]; then
        # Replace listen directives
        sed -i "s/listen\s\+80;/listen ${ts_ip}:80;/" "$nginx_active"
        sed -i "s/listen\s\+\[::\]:80;/# listen [::]:80; # disabled - Tailscale only/" "$nginx_active"
        # Remove public server_name, bind to Tailscale IP
        sed -i "s/server_name\s\+[0-9.]\+\s\+_;/server_name ${ts_ip};/" "$nginx_active"
        log "NGINX bound to ${ts_ip}:80"
    else
        warn "No NGINX site config found — skip NGINX rebind"
    fi

    # Validate before applying
    if ! sshd -t 2>/dev/null; then
        warn "sshd config invalid — rolling back"
        restore_file "$SSHD_CONF" "phase1"
        fail "sshd config test failed"
    fi

    if [[ -n "$nginx_active" ]] && ! nginx -t 2>/dev/null; then
        warn "NGINX config invalid — rolling back"
        restore_file "$nginx_active" "phase1"
        restore_file "$SSHD_CONF" "phase1"
        fail "NGINX config test failed"
    fi

    # Arm deadman BEFORE restarting services
    arm_deadman

    # Apply
    systemctl restart sshd
    log "sshd restarted on Tailscale IP"

    if [[ -n "$nginx_active" ]]; then
        systemctl reload nginx
        log "NGINX reloaded on Tailscale IP"
    fi

    echo ""
    warn "═══════════════════════════════════════════════════════════"
    warn " DEADMAN ARMED — 5 MINUTE REVERT TIMER RUNNING"
    warn ""
    warn " Test NOW from your Tailscale device:"
    warn "   ssh root@${ts_ip}"
    warn ""
    warn " If it works, confirm immediately:"
    warn "   ./ops/tailscale-harden.sh confirm"
    warn ""
    warn " If you lose access — do nothing."
    warn " Changes revert automatically in 5 minutes."
    warn "═══════════════════════════════════════════════════════════"
}

# ── Phase 2: Enable UFW ────────────────────────────────────────────

do_phase2() {
    ensure_root
    ensure_backups_dir

    local ts_ip
    ts_ip=$(tailscale ip -4 2>/dev/null) || fail "Tailscale not connected."

    log "═══ Phase 2: Enable UFW — cut public surface ═══"

    # Arm deadman
    arm_deadman

    # Configure UFW
    ufw --force reset
    ufw default deny incoming
    ufw default allow outgoing

    # Allow Tailscale WireGuard tunnel
    ufw allow 41641/udp comment "Tailscale WireGuard"

    # Allow all traffic on Tailscale interface
    ufw allow in on tailscale0 comment "Tailscale mesh"

    # Enable
    ufw --force enable
    log "UFW enabled"
    ufw status verbose

    echo ""
    warn "═══════════════════════════════════════════════════════════"
    warn " DEADMAN ARMED — 5 MINUTE REVERT TIMER RUNNING"
    warn ""
    warn " Test connectivity from your Tailscale device NOW."
    warn " If it works: ./ops/tailscale-harden.sh confirm"
    warn " If locked out: UFW disables in 5 minutes automatically."
    warn "═══════════════════════════════════════════════════════════"
}

# ── Confirm (cancel deadman) ───────────────────────────────────────

do_confirm() {
    kill_deadman
    log "Changes confirmed and locked in."
    echo ""
    do_status
}

# ── Rollback ───────────────────────────────────────────────────────

do_rollback() {
    echo "[ROLLBACK] Reverting all hardening changes..."

    # Restore sshd — try phase1 backup first, then phase0
    if [[ -f "${BACKUP_DIR}/sshd_config.phase1.bak" ]]; then
        restore_file "$SSHD_CONF" "phase1"
    elif [[ -f "${BACKUP_DIR}/sshd_config.phase0.bak" ]]; then
        restore_file "$SSHD_CONF" "phase0"
    fi

    # Restart sshd
    if sshd -t 2>/dev/null; then
        systemctl restart sshd 2>/dev/null || true
        echo "[ROLLBACK] sshd restarted"
    fi

    # Restore NGINX
    local nginx_active
    nginx_active=$(detect_nginx_conf)
    if [[ -n "$nginx_active" ]]; then
        local basename_conf
        basename_conf=$(basename "$nginx_active")
        if [[ -f "${BACKUP_DIR}/${basename_conf}.phase1.bak" ]]; then
            cp "${BACKUP_DIR}/${basename_conf}.phase1.bak" "$nginx_active"
            echo "[ROLLBACK] Restored $nginx_active"
        fi
        nginx -t 2>/dev/null && systemctl reload nginx 2>/dev/null || true
        echo "[ROLLBACK] NGINX reloaded"
    fi

    # Disable UFW
    ufw --force disable 2>/dev/null || true
    echo "[ROLLBACK] UFW disabled"

    # Kill deadman if running
    kill_deadman 2>/dev/null || true

    echo "[ROLLBACK] Complete. System restored to pre-hardening state."
}

# ── Status ─────────────────────────────────────────────────────────

do_status() {
    log "═══ Hardening Status ═══"
    echo ""

    # SSH password auth
    local pw_auth
    pw_auth=$(sshd -T 2>/dev/null | grep -i passwordauthentication | awk '{print $2}')
    if [[ "$pw_auth" == "no" ]]; then
        echo -e "  SSH password auth:  ${GREEN}DISABLED${NC}"
    else
        echo -e "  SSH password auth:  ${RED}ENABLED${NC}"
    fi

    # SSH listen address
    local listen_addrs
    listen_addrs=$(sshd -T 2>/dev/null | grep 'listenaddress' | awk '{print $2}' | tr '\n' ' ')
    echo "  SSH listen:         ${listen_addrs:-0.0.0.0 (public)}"

    # Tailscale
    if command -v tailscale &>/dev/null && tailscale status &>/dev/null; then
        echo -e "  Tailscale:          ${GREEN}CONNECTED${NC} ($(tailscale ip -4))"
    else
        echo -e "  Tailscale:          ${RED}NOT CONNECTED${NC}"
    fi

    # NGINX
    local nginx_active
    nginx_active=$(detect_nginx_conf)
    if [[ -n "$nginx_active" ]]; then
        local listen_line
        listen_line=$(grep -m1 'listen\s' "$nginx_active" 2>/dev/null | xargs)
        echo "  NGINX listen:       ${listen_line:-unknown}"
    fi

    # UFW
    local ufw_status
    ufw_status=$(ufw status 2>/dev/null | head -1)
    if echo "$ufw_status" | grep -q "active"; then
        echo -e "  UFW:                ${GREEN}ACTIVE${NC}"
    else
        echo -e "  UFW:                ${YELLOW}INACTIVE${NC}"
    fi

    # Deadman
    if [[ -f "$DEADMAN_PID_FILE" ]] && kill -0 "$(cat "$DEADMAN_PID_FILE")" 2>/dev/null; then
        echo -e "  Deadman revert:     ${YELLOW}ARMED${NC} (PID $(cat "$DEADMAN_PID_FILE"))"
    else
        echo "  Deadman revert:     not armed"
    fi

    # Executor
    if supervisorctl status hedge-executor 2>/dev/null | grep -q RUNNING; then
        echo -e "  Executor:           ${GREEN}RUNNING${NC}"
    else
        echo -e "  Executor:           ${YELLOW}NOT RUNNING${NC}"
    fi

    echo ""
}

# ── Main dispatcher ────────────────────────────────────────────────

case "${1:-}" in
    phase0)   do_phase0 ;;
    phase1)   do_phase1 ;;
    phase2)   do_phase2 ;;
    confirm)  do_confirm ;;
    rollback) do_rollback ;;
    status)   do_status ;;
    *)
        echo "Usage: $0 {phase0|phase1|phase2|confirm|rollback|status}"
        echo ""
        echo "  phase0   — Disable SSH password auth + install Tailscale"
        echo "  phase1   — Rebind SSH+NGINX to Tailscale IP (deadman armed)"
        echo "  phase2   — Enable UFW, cut public surface (deadman armed)"
        echo "  confirm  — Cancel deadman timer, lock in changes"
        echo "  rollback — Revert everything to pre-hardening state"
        echo "  status   — Show current hardening state"
        exit 1
        ;;
esac
