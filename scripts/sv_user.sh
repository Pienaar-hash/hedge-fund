#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SUPERVISOR_CONF="$ROOT_DIR/deploy/supervisor-user/supervisord.conf"
LOG_DIR="$ROOT_DIR/deploy/supervisor-user/logs"
mkdir -p "$LOG_DIR"

do_help() {
  cat <<USAGE
Usage: $(basename "$0") <boot|start|stop|restart|status> [program]
  boot               Start user-level supervisord daemon
  start <program>    Start program (default: all)
  stop <program>     Stop program (default: all)
  restart <program>  Restart program (default: all)
  status             Show status
USAGE
}

if [[ $# -lt 1 ]]; then
  do_help
  exit 1
fi

CMD=$1
shift || true
PROG=${1:-all}

case "$CMD" in
  boot)
    supervisord -c "$SUPERVISOR_CONF"
    echo "Hint: if \"status\" fails with connection refused, give port 9002 a second to come up." >&2
    ;;
  start)
    supervisorctl -c "$SUPERVISOR_CONF" start "$PROG"
    ;;
  stop)
    supervisorctl -c "$SUPERVISOR_CONF" stop "$PROG"
    ;;
  restart)
    supervisorctl -c "$SUPERVISOR_CONF" restart "$PROG"
    ;;
  status)
    supervisorctl -c "$SUPERVISOR_CONF" status
    ;;
  *)
    do_help
    exit 1
    ;;
esac
