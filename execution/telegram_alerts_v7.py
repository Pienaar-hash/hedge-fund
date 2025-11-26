"""
v7 Telegram alerts for investor-facing signals (ATR regime, drawdown state, risk mode, 4h close).

State-driven, low-noise: only sends one JSON payload per 4h candle close.
Reads config/telegram_v7.json and persists alert state under logs/state/telegram_state.json.
Relies on existing telegram_utils.send_telegram helper (env-driven).

Operator flags (env):
  - EXEC_TELEGRAM_4H_ONLY=1: strict mode; only 4h JSON payloads allowed
  - EXEC_TELEGRAM_MAX_PER_MIN=0: block all sends immediately (mitigation)
  - TELEGRAM_ENABLED=0: top-level disable
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from execution import telegram_utils
from execution.telegram_utils import (
    build_investor_alert_payload,
    load_telegram_state,
    save_telegram_state,
    write_alert_jsonl,
)

LOG = logging.getLogger("telegram_alerts_v7")

CONFIG_PATH = Path(os.getenv("TELEGRAM_V7_CONFIG") or "config/telegram_v7.json")
# Aligned v7 telemetry state filename
STATE_PATH = Path(os.getenv("ALERTS_V7_STATE_PATH") or "logs/state/telegram_state.json")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_state() -> Dict[str, Any]:
    """Load persisted telegram alert state using the shared helper.

    Returns the state dict. If the state file does not exist, a default
    structure is created and written atomically. Logs once on creation.
    """
    state = load_telegram_state(str(STATE_PATH))
    if state and state.get("last_4h_close_ts") is not None:
        return state

    # If no valid state found, create a fresh default state and persist it.
    default_state: Dict[str, Any] = {
        "atr_regime": None,
        "drawdown_state": None,
        "router_quality": None,
        "aum_total": None,
        "last_4h_close_ts": 0,
        "last_sent": {},
    }
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_state(default_state)
        LOG.info("[alerts-v7] created new %s (fresh state)", STATE_PATH)
    except Exception as exc:  # pragma: no cover - logging only
        LOG.warning("[alerts-v7] failed creating state file %s err=%s", STATE_PATH, exc)
    return default_state


def save_state(state: Dict[str, Any]) -> None:
    """Persist state atomically using the shared helper."""
    save_telegram_state(state, str(STATE_PATH))


def _load_config() -> Dict[str, Any]:
    cfg = _load_json(CONFIG_PATH)
    if not isinstance(cfg, dict):
        cfg = {}
    alerts = cfg.get("alerts")
    cfg["alerts"] = alerts if isinstance(alerts, dict) else {}
    cfg.setdefault("enabled", False)
    cfg.setdefault("bot_token_env", "TELEGRAM_BOT_TOKEN")
    cfg.setdefault("chat_id_env", "TELEGRAM_CHAT_ID")
    cfg.setdefault("min_interval_seconds", 60)
    # Default: enable only the 4h close summary for low-noise operation
    cfg.setdefault("close_4h", True)
    return cfg


def _resolve_env(cfg: Mapping[str, Any]) -> tuple[str, str]:
    token_env = str(cfg.get("bot_token_env") or "TELEGRAM_BOT_TOKEN")
    chat_env = str(cfg.get("chat_id_env") or "TELEGRAM_CHAT_ID")
    token = os.getenv(token_env, "").strip()
    chat = os.getenv(chat_env, "").strip()
    return token, chat


def _min_interval(cfg: Mapping[str, Any], key: str) -> int:
    try:
        per_alert = (cfg.get("alerts") or {}).get(key) or {}
        val = int(per_alert.get("min_interval_seconds", cfg.get("min_interval_seconds", 60)))
        return max(val, 30)
    except Exception:
        return 60


def _enabled(cfg: Mapping[str, Any], key: str) -> bool:
    if not cfg.get("enabled"):
        return False
    alerts = cfg.get("alerts") or {}
    entry = alerts.get(key) if isinstance(alerts, Mapping) else {}
    if isinstance(entry, Mapping):
        return bool(entry.get("enabled", True))
    return True


def _should_send(last_ts: Optional[float], min_interval: int, now_ts: float) -> bool:
    if last_ts is None:
        return True
    try:
        return (now_ts - float(last_ts)) >= float(min_interval)
    except Exception:
        return False


def _send(message: str, cfg: Mapping[str, Any], silent: bool = False) -> bool:
    token, chat = _resolve_env(cfg)
    if not token or not chat:
        LOG.debug("[telegram] missing creds token_set=%s chat_set=%s", bool(token), bool(chat))
        return False
    # temporarily override env expected by telegram_utils
    orig_env = {
        "TELEGRAM_ENABLED": os.getenv("TELEGRAM_ENABLED"),
        "BOT_TOKEN": os.getenv("BOT_TOKEN"),
        "CHAT_ID": os.getenv("CHAT_ID"),
    }
    os.environ["TELEGRAM_ENABLED"] = "1"
    os.environ["BOT_TOKEN"] = token
    os.environ["CHAT_ID"] = chat
    try:
        return bool(telegram_utils.send_telegram(message, silent=silent))
    finally:
        for key, val in orig_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def _maybe_send_atr_regime_alert(kpis: Mapping[str, Any], state: Dict[str, Any], cfg: Mapping[str, Any], now_ts: float) -> None:
    if not _enabled(cfg, "atr_regime"):
        return
    atr_regime = (kpis.get("atr") or {}).get("atr_regime") if isinstance(kpis, Mapping) else None
    if atr_regime is None:
        atr_regime = kpis.get("atr_regime") if isinstance(kpis, Mapping) else None
    if not atr_regime:
        return
    last_sent = (state.get("last_sent") or {}).get("atr_regime") or {}
    last_val = last_sent.get("value")
    last_ts = last_sent.get("ts")
    if atr_regime == last_val and not _should_send(last_ts, _min_interval(cfg, "atr_regime"), now_ts):
        return
    msg = f"ATR regime change: {last_val or 'unknown'} → {atr_regime}."
    sent = _send(msg, cfg, silent=False)
    if sent:
        payload = build_investor_alert_payload(
            "atr",
            msg,
            metadata={
                "from": last_val,
                "to": atr_regime,
            },
        )
        write_alert_jsonl(payload)
        state.setdefault("last_sent", {})["atr_regime"] = {"value": atr_regime, "ts": now_ts}


def _maybe_send_dd_state_alert(kpis: Mapping[str, Any], state: Dict[str, Any], cfg: Mapping[str, Any], now_ts: float) -> None:
    if not _enabled(cfg, "dd_state"):
        return
    dd_state = None
    dd_pct = None
    if isinstance(kpis, Mapping):
        dd_state = kpis.get("dd_state") or (kpis.get("drawdown") or {}).get("dd_state")
        try:
            dd_pct = (kpis.get("drawdown") or {}).get("dd_pct")
        except Exception:
            dd_pct = None
    if not dd_state:
        return
    last_sent = (state.get("last_sent") or {}).get("dd_state") or {}
    last_val = last_sent.get("value")
    last_ts = last_sent.get("ts")
    if dd_state == last_val and not _should_send(last_ts, _min_interval(cfg, "dd_state"), now_ts):
        return
    dd_str = f"{dd_pct:.2f}%" if isinstance(dd_pct, (int, float)) else "n/a"
    msg = f"Drawdown state change: {last_val or 'unknown'} → {dd_state} (dd={dd_str})."
    sent = _send(msg, cfg, silent=False)
    if sent:
        metadata: Dict[str, Any] = {"from": last_val, "to": dd_state}
        if dd_pct is not None:
            try:
                metadata["dd_pct"] = float(dd_pct)
            except Exception:
                metadata["dd_pct"] = dd_pct
        payload = build_investor_alert_payload("drawdown", msg, metadata=metadata)
        write_alert_jsonl(payload)
        state.setdefault("last_sent", {})["dd_state"] = {"value": dd_state, "ts": now_ts}


def _derive_risk_mode(risk_snapshot: Mapping[str, Any]) -> str:
    if not isinstance(risk_snapshot, Mapping):
        return "unknown"
    summary = risk_snapshot.get("summary")
    if isinstance(summary, Mapping):
        mode = summary.get("risk_mode") or summary.get("mode") or summary.get("state")
        if mode:
            return str(mode)
    drawdown = (risk_snapshot.get("drawdown") or {}) if isinstance(risk_snapshot, Mapping) else {}
    dd_pct = drawdown.get("dd_pct") or drawdown.get("pct")
    try:
        dd_val = float(dd_pct)
    except Exception:
        dd_val = None
    guards = risk_snapshot.get("risk_config_meta") if isinstance(risk_snapshot, Mapping) else {}
    max_dd = None
    daily_loss = None
    if isinstance(guards, Mapping):
        try:
            max_dd = float(guards.get("max_nav_drawdown_pct") or 0.0)
        except Exception:
            max_dd = None
        try:
            daily_loss = float(guards.get("daily_loss_limit_pct") or 0.0)
        except Exception:
            daily_loss = None
    if dd_val is not None:
        if max_dd and dd_val >= max_dd:
            return "dd_guard"
        if daily_loss and dd_val >= daily_loss:
            return "daily_loss_guard"
    return "normal"


def _maybe_send_risk_mode_alert(risk_snapshot: Mapping[str, Any], state: Dict[str, Any], cfg: Mapping[str, Any], now_ts: float) -> None:
    if not _enabled(cfg, "risk_mode"):
        return
    mode = _derive_risk_mode(risk_snapshot)
    last_sent = (state.get("last_sent") or {}).get("risk_mode") or {}
    last_val = last_sent.get("value")
    last_ts = last_sent.get("ts")
    if mode == last_val and not _should_send(last_ts, _min_interval(cfg, "risk_mode"), now_ts):
        return
    msg = f"Risk mode change: {last_val or 'unknown'} → {mode}."
    sent = _send(msg, cfg, silent=False)
    if sent:
        payload = build_investor_alert_payload(
            "router",
            msg,
            metadata={
                "from": last_val,
                "to": mode,
            },
        )
        write_alert_jsonl(payload)
        state.setdefault("last_sent", {})["risk_mode"] = {"value": mode, "ts": now_ts}


def _maybe_send_4h_close_alert(nav_snapshot: Mapping[str, Any], kpis: Mapping[str, Any], state: Dict[str, Any], cfg: Mapping[str, Any], now_ts: float) -> None:
    """Send the canonical 4h-close JSON payload if a new 4h bar has closed.

    Payload contains exactly:
      - atr_regime: string
      - drawdown_state: string or null
      - router_quality: string or null
      - aum_total: number (float)
      - last_4h_close_ts: integer (epoch seconds)
    """
    if not _enabled(cfg, "close_4h"):
        return

    # Determine current timestamp for bar calculation
    ts_val = None
    for candidate in (nav_snapshot.get("ts"), nav_snapshot.get("timestamp"), nav_snapshot.get("updated_at"), kpis.get("ts") if isinstance(kpis, Mapping) else None, now_ts):
        if candidate is None:
            continue
        try:
            ts_val = float(candidate)
            break
        except Exception:
            continue
    if ts_val is None:
        return

    # Calculate 4h bar close timestamp (floor to 4h boundary)
    bar_ts = int(math.floor(ts_val / (4 * 3600)) * (4 * 3600))

    # Check against state's last_4h_close_ts (not last_sent)
    last_sent_ts = state.get("last_4h_close_ts") or 0
    if bar_ts <= int(last_sent_ts):
        LOG.debug("[alerts-v7] 4h bar not new: bar_ts=%d <= last_4h_close_ts=%d", bar_ts, last_sent_ts)
        return

    # Extract values from snapshots
    nav_total = nav_snapshot.get("nav") or nav_snapshot.get("nav_usd") or (nav_snapshot.get("aum") or {}).get("futures")
    atr_regime = kpis.get("atr_regime") if isinstance(kpis, Mapping) else None
    dd_state = kpis.get("dd_state") or kpis.get("drawdown_state") if isinstance(kpis, Mapping) else None
    router_quality = kpis.get("router_quality") if isinstance(kpis, Mapping) else None

    # Build the canonical JSON payload (exact keys required by spec)
    payload_msg = {
        "atr_regime": str(atr_regime or state.get("atr_regime") or "unknown"),
        "drawdown_state": dd_state or state.get("drawdown_state"),
        "router_quality": router_quality or state.get("router_quality"),
        "aum_total": float(nav_total) if nav_total is not None else state.get("aum_total"),
        "last_4h_close_ts": bar_ts,
    }

    try:
        msg = json.dumps(payload_msg, separators=(',', ':'))
    except Exception:
        msg = str(payload_msg)

    sent = _send(msg, cfg, silent=True)
    if sent:
        # Write to JSONL audit log
        payload = build_investor_alert_payload(
            "4h",
            msg,
            metadata={
                "bar_ts": bar_ts,
                "nav": nav_total,
                "atr_regime": atr_regime,
                "dd_state": dd_state,
            },
        )
        write_alert_jsonl(payload)

        # Update internal last_sent bookkeeping
        state.setdefault("last_sent", {})["close_4h"] = {"bar_ts": bar_ts, "ts": now_ts}

        # Persist canonical state fields so future runs have a baseline
        state["atr_regime"] = payload_msg.get("atr_regime")
        state["drawdown_state"] = payload_msg.get("drawdown_state")
        state["router_quality"] = payload_msg.get("router_quality")
        state["aum_total"] = payload_msg.get("aum_total")
        state["last_4h_close_ts"] = payload_msg.get("last_4h_close_ts")

        LOG.info("[alerts-v7] 4h-state sent: bar_ts=%d", bar_ts)


def run_alerts(context: Mapping[str, Any]) -> None:
    """Run v7 telegram alerts.

    Low-noise policy: only sends the 4h-close JSON state summary.
    Per-tick alerts (atr_regime, dd_state, risk_mode) are disabled.
    """
    cfg = _load_config()
    if not cfg.get("enabled"):
        return
    token, chat = _resolve_env(cfg)
    if not token or not chat:
        return
    now_ts = float(context.get("now_ts", time.time()) or time.time())
    nav_snapshot = context.get("nav_snapshot") if isinstance(context, Mapping) else {}
    kpis_snapshot = context.get("kpis_snapshot") if isinstance(context, Mapping) else {}

    state = load_state()
    before = json.dumps(state, sort_keys=True)

    # Low-noise policy: only send the 4h-close JSON state summary.
    # Per-tick alerts (atr_regime, dd_state, risk_mode) are intentionally skipped.
    _maybe_send_4h_close_alert(nav_snapshot or {}, kpis_snapshot or {}, state, cfg, now_ts)

    after = json.dumps(state, sort_keys=True)
    if after != before:
        save_state(state)
