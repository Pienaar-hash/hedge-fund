#!/usr/bin/env bash
set -euo pipefail

cd /root/hedge-fund

RUN_ID="v8_phase5_shadow_soak_$(date -u +%Y%m%dT%H%M%SZ)"

PYTHONPATH=/root/hedge-fund \
python -m research.shadow_soak_v8 \
  --run-id "$RUN_ID" \
  --logs-root /root/hedge-fund/logs \
  --certification-dir /root/hedge-fund/data/replay_certifications/v8_phase4_fps_v2_cert_003