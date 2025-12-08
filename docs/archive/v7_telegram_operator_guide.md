# Telegram Alerts v7 — Operator Guide

## Overview

Telegram alerts v7 implements a **state-driven, low-noise** alert system. Instead of sending multiple noisy messages throughout the day, it sends a **single deterministic JSON state summary** only on 4-hour candle close.

## Operator Environment Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_ENABLED` | `0` | Top-level enable/disable. Set to `1` to enable Telegram. |
| `EXEC_TELEGRAM_4H_ONLY` | `0` | Strict mode. When `1`, only 4h JSON payloads with `atr_regime` or `last_4h_close_ts` keys are allowed. All other messages are suppressed. |
| `EXEC_TELEGRAM_MAX_PER_MIN` | `6` | Rate cap. Maximum Telegram sends per minute. Set to `0` to block all sends immediately (mitigation mode). |
| `EXEC_TELEGRAM_MIN_IDENTICAL_S` | `60` | Suppress identical messages within this many seconds. |

## 4h JSON Payload Format

The canonical state payload sent every 4 hours contains exactly these keys:

```json
{
  "atr_regime": "low",
  "drawdown_state": "none",
  "router_quality": "good",
  "aum_total": 11173.87,
  "last_4h_close_ts": 1764177600
}
```

| Key | Type | Description |
|-----|------|-------------|
| `atr_regime` | string | Current ATR regime (e.g., "low", "medium", "high") |
| `drawdown_state` | string or null | Current drawdown state (e.g., "none", "mild", "severe") |
| `router_quality` | string or null | Router quality metric |
| `aum_total` | float or null | Total AUM in USD |
| `last_4h_close_ts` | integer | Epoch seconds of the 4h bar close |

## State Persistence

Canonical state is persisted atomically to:

```
logs/state/telegram_state.json
```

The file uses atomic writes (tmp file + `os.replace`) to prevent corruption.

## Mitigation Commands

### Immediate: Block All Sends

```bash
# Edit Supervisor environment or .env
export EXEC_TELEGRAM_MAX_PER_MIN=0

# Restart the executor
sudo supervisorctl restart hedge:hedge-executor
```

### Emergency: Disable Telegram Completely

```bash
export TELEGRAM_ENABLED=0
sudo supervisorctl restart hedge:hedge-executor
```

### Enable Strict 4h-Only Mode

```bash
export EXEC_TELEGRAM_4H_ONLY=1
export EXEC_TELEGRAM_MAX_PER_MIN=1
sudo supervisorctl restart hedge:hedge-executor
```

### Verify Quietness

```bash
sudo tail -n 200 /var/log/hedge-executor.out.log | grep -E "Telegram|suppressed"
```

Expected log patterns:

- `⏳ Telegram suppressed (4h-only mode)` — Non-4h message blocked
- `⏳ Telegram suppressed (4h-only mode, not-json)` — Non-JSON message blocked
- `❌ Telegram suppressed (rate limit reached)` — Rate limit hit
- `⏳ Telegram suppressed (identical recent)` — Duplicate message blocked
- `✅ Telegram 4h-state sent.` — 4h state successfully sent

## Rollout Checklist

1. **Stage first** — Test in staging environment if available
2. **Set mitigation flags** — `EXEC_TELEGRAM_MAX_PER_MIN=0` to block during deployment
3. **Deploy code changes**
4. **Restart executor** — `sudo supervisorctl restart hedge:hedge-executor`
5. **Verify logs** — Check for suppression messages
6. **Enable 4h mode** — Set `EXEC_TELEGRAM_4H_ONLY=1` and `EXEC_TELEGRAM_MAX_PER_MIN=1`
7. **Monitor** — Watch for next 4h close alert

## Files Modified

- `execution/telegram_utils.py` — State persistence helpers, strict 4h-only filter
- `execution/telegram_alerts_v7.py` — 4h-close routine, canonical payload builder
- `tests/test_telegram_v7.py` — Unit tests for suppression and send behavior
- `logs/state/telegram_state.json` — Persisted canonical state
- `logs/alerts/alerts_v7.jsonl` — JSONL audit log of sent alerts

## Safety Notes

- **Tokens in environment only** — Never commit tokens to repo
- **Telemetry only** — Alerts do not alter strategy/risk logic
- **Fallback** — `TELEGRAM_ENABLED=0` + restart to immediately cut all sends
