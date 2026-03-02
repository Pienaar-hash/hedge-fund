#!/bin/bash
# Regime boundary telemetry — watch MEAN_REVERT ↔ CHOPPY oscillation
# Usage: bash scripts/regime_boundary_watch.sh
# Refreshes every 10 seconds. Ctrl-C to exit.

cd "$(dirname "$0")/.." || exit 1

watch -n 10 'python3 -c "
import json, sys
try:
    with open(\"logs/state/sentinel_x.json\") as f:
        s = json.load(f)
except Exception as e:
    print(f\"ERROR reading sentinel state: {e}\")
    sys.exit(1)

feat = s.get(\"features\", {})
hist = s.get(\"history_meta\", {})
probs = s.get(\"regime_probs\", {})
smoothed = s.get(\"smoothed_probs\", {})

mr_score = feat.get(\"mean_reversion_score\", 0)
mr_threshold = 0.40
mr_delta = mr_score - mr_threshold

trend_slope = feat.get(\"trend_slope\", 0)
trend_threshold = 0.0003
trend_delta = abs(trend_slope) - trend_threshold

regime = s.get(\"primary_regime\", \"?\")
cycles = hist.get(\"consecutive_count\", 0)
pending = hist.get(\"pending_regime\", None)
labels = hist.get(\"last_n_labels\", [])

print(\"=== REGIME BOUNDARY TELEMETRY ===\")
print(f\"Regime:   {regime}  (stable {cycles} cycles)\")
if pending:
    print(f\"Pending:  {pending}  ← transition brewing\")
print(f\"Labels:   {\" → \".join(labels[-5:])}\")
print()
print(\"--- Key Boundaries ---\")
print(f\"MR score:     {mr_score:.4f}  threshold: {mr_threshold}  delta: {mr_delta:+.4f}  {'▲ ABOVE' if mr_delta > 0 else '▼ BELOW'}\")
print(f\"trend_slope:  {trend_slope:.6f}  threshold: ±{trend_threshold}  |delta|: {trend_delta:+.6f}\")
print(f\"vol_z:        {feat.get('vol_regime_z', 0):.4f}\")
print(f\"volume_z:     {feat.get('volume_z', 0):.4f}\")
print(f\"breakout:     {feat.get('breakout_score', 0):.4f}  (threshold: 0.02)\")
print()
print(\"--- Probability Surface ---\")
for name in ['CHOPPY', 'MEAN_REVERT', 'TREND_UP', 'TREND_DOWN', 'BREAKOUT', 'CRISIS']:
    raw = probs.get(name, 0)
    sm = smoothed.get(name, 0)
    bar = '█' * int(raw * 40)
    marker = ' ◄ PRIMARY' if name == regime else ''
    print(f\"  {name:15s} raw={raw:.3f} smooth={sm:.3f} {bar}{marker}\")
print()
print(f\"Updated: {s.get('updated_ts', '?')}\")
"'
