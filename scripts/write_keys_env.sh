#!/usr/bin/env bash
# Usage:
#   bash scripts/write_keys_env.sh "<API_KEY>" "<API_SECRET>"
# or (safer for shell history):
#   BINANCE_API_KEY="..." BINANCE_API_SECRET="..." bash scripts/write_keys_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

KEY="${1:-${BINANCE_API_KEY:-}}"
SEC="${2:-${BINANCE_API_SECRET:-}}"

if [[ -z "${KEY}" || -z "${SEC}" ]]; then
  echo "ERROR: Missing key/secret. Provide args or BINANCE_API_KEY/BINANCE_API_SECRET env vars." >&2
  exit 2
fi

touch .env

tmp=".env.tmp.$$"
trap 'rm -f "${tmp}"' EXIT

# Strip existing sensitive lines
grep -v '^BINANCE_API_KEY=' .env 2>/dev/null | \
  grep -v '^BINANCE_API_SECRET=' > "${tmp}" || true

# Ensure baseline flags exist
if ! grep -q '^ENV=' "${tmp}" 2>/dev/null; then
  echo 'ENV=prod' >> "${tmp}"
fi
if ! grep -q '^BINANCE_TESTNET=' "${tmp}" 2>/dev/null; then
  echo 'BINANCE_TESTNET=0' >> "${tmp}"
fi
if ! grep -q '^DRY_RUN=' "${tmp}" 2>/dev/null; then
  echo 'DRY_RUN=1' >> "${tmp}"
fi

{
  echo "BINANCE_API_KEY=${KEY}"
  echo "BINANCE_API_SECRET=${SEC}"
} >> "${tmp}"

mv "${tmp}" .env
trap - EXIT
rm -f "${tmp}" 2>/dev/null || true
chmod 600 .env

printf '[ok] .env updated (key_len=%s, secret_len=%s)\n' "${#KEY}" "${#SEC}"
echo "NOTE: Secrets not printed. To use now in this shell: run 'set -a; source ./.env; set +a'"
