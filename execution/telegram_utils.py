from __future__ import annotations

# execution/telegram_utils.py — Phase 4.1 → v7 (State-driven, Low-noise)
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import requests
except Exception:
    requests = None  # handled below


# --- State persistence helpers (v7) ---
STATE_PATH = Path(os.getenv("TELEGRAM_STATE_PATH") or "logs/state/telegram_state.json")


def load_telegram_state(path: Optional[str] = None) -> Dict[str, Any]:
    """Load persisted telegram alert state from disk.

    Returns dict with canonical keys:
      - atr_regime, drawdown_state, router_quality, aum_total, last_4h_close_ts
      - last_sent: internal bookkeeping map
    """
    state_path = Path(path) if path else STATE_PATH
    try:
        if not state_path.exists():
            return _default_telegram_state()
        content = state_path.read_text(encoding="utf-8")
        state = json.loads(content)
        if not isinstance(state, dict):
            return _default_telegram_state()
        return state
    except Exception:
        return _default_telegram_state()


def _default_telegram_state() -> Dict[str, Any]:
    """Return a fresh default state structure."""
    return {
        "atr_regime": None,
        "drawdown_state": None,
        "router_quality": None,
        "aum_total": None,
        "last_4h_close_ts": 0,
        "last_sent": {},
    }


def save_telegram_state(state: Dict[str, Any], path: Optional[str] = None) -> bool:
    """Persist telegram state atomically (tmp file + os.replace).

    Returns True on success, False on failure.
    """
    state_path = Path(path) if path else STATE_PATH
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = state_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)
        os.replace(str(tmp_path), str(state_path))
        return True
    except Exception as exc:
        print(f"❌ Telegram state save error: {exc}", flush=True)
        return False


# --- Env helpers ---
def _b(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "on")


def _env():
    # Prefer explicit BOT_TOKEN/CHAT_ID, but fall back to TELEGRAM_* names
    token = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or ""
    chat = os.getenv("CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID") or ""
    return {
        "enabled": _b(os.getenv("TELEGRAM_ENABLED", "0")),
        "token": str(token).strip(),
        "chat": str(chat).strip(),
    }


def _utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def write_alert_jsonl(alert_dict: Dict[str, Any], path: str = "logs/alerts/alerts_v7.jsonl") -> bool:
    """
    Append a structured alert entry to the investor-grade JSONL log.
    Each line is a standalone JSON object.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    if "ts" not in alert_dict:
        alert_dict["ts"] = int(time.time())
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(alert_dict, ensure_ascii=False))
        handle.write("\n")
    return True


def build_investor_alert_payload(
    msg_type: str, msg_text: str, metadata: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Build a structured investor alert for JSONL logs.
    msg_type ∈ ["atr", "drawdown", "router", "aum", "4h", "custom"]
    """
    base: Dict[str, Any] = {"ts": int(time.time()), "type": msg_type, "message": msg_text}
    if metadata:
        base["meta"] = metadata
    return base


# --- Core send ---
def send_telegram(message: str, silent: bool = False, parse_mode: str | None = None) -> bool:
    env = _env()
    if not env["enabled"]:
        print("❌ Telegram disabled (TELEGRAM_ENABLED!=1).", flush=True)
        return False
    if not env["token"] or not env["chat"]:
        print(
            f"❌ Telegram missing creds (BOT_TOKEN len={len(env['token'])}, CHAT_ID set={bool(env['chat'])}).",
            flush=True,
        )
        return False
    if requests is None:
        print("❌ Telegram cannot import requests.", flush=True)
        return False

    # Check rate cap first: EXEC_TELEGRAM_MAX_PER_MIN=0 blocks immediately (mitigation mode)
    current_rate_cap = int(os.getenv("EXEC_TELEGRAM_MAX_PER_MIN", "6") or 6)
    if current_rate_cap == 0:
        print("❌ Telegram suppressed (rate limit reached).", flush=True)
        return False

    # If operator requested 4h-only mode, suppress non-4h payloads.
    is_4h_only = _b(os.getenv("EXEC_TELEGRAM_4H_ONLY", "0"))
    if is_4h_only:
        # allow only JSON payloads that match the v7 4h summary shape
        try:
            j = json.loads(str(message or ""))
            if not isinstance(j, dict):
                print("⏳ Telegram suppressed (4h-only mode, not-json).", flush=True)
                return False
            # Must have at least one of the required 4h keys
            if not any(k in j for k in _4H_REQUIRED_KEYS):
                print("⏳ Telegram suppressed (4h-only mode).", flush=True)
                return False
        except json.JSONDecodeError:
            print("⏳ Telegram suppressed (4h-only mode, not-json).", flush=True)
            return False
        except Exception:
            print("⏳ Telegram suppressed (4h-only mode).", flush=True)
            return False

    # identical-message suppression (avoid spamming the same text repeatedly)
    try:
        now = time.time()
        key = str(message or "").strip()
        # identical-message suppression
        last = _recent_msgs.get(key)
        if last and (now - float(last)) < float(MIN_IDENTICAL_S):
            print("⏳ Telegram suppressed (identical recent).", flush=True)
            return False
        # global rate cap (max current_rate_cap messages per 60s)
        # prune old timestamps
        cutoff = now - 60
        while _send_timestamps and _send_timestamps[0] < cutoff:
            _send_timestamps.pop(0)
        if len(_send_timestamps) >= current_rate_cap:
            print("❌ Telegram suppressed (rate limit reached).", flush=True)
            return False
    except Exception:
        # Best-effort only — don't block sending on suppression errors
        pass
    try:
        url = f"https://api.telegram.org/bot{env['token']}/sendMessage"
        payload = {
            "chat_id": env["chat"],
            "text": f"{_utc()}\n{message}",
            "disable_notification": bool(silent),
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        r = requests.post(url, json=payload, timeout=15)
        if r.ok:
            # record last-sent time for this exact message
            try:
                _recent_msgs[key] = time.time()
                # record global send timestamp
                try:
                    _send_timestamps.append(time.time())
                except Exception:
                    pass
            except Exception:
                pass
            # Distinguish 4h state sends in logs
            if is_4h_only:
                print("✅ Telegram 4h-state sent.", flush=True)
            else:
                print("✅ Telegram message sent.", flush=True)
            return True
        print(f"❌ Telegram send failed [{r.status_code}]: {r.text}", flush=True)
        return False
    except Exception as e:
        print(f"❌ Telegram send error: {e}", flush=True)
        return False


# --- Cadence / rate limiting ---
_last_summary_ts: float | None = None
_last_dd_ts: float | None = None
ALERT_COOLDOWN_S = int(os.getenv("EXEC_ALERT_COOLDOWN_S", "60") or 60)
_last_alert_ts: Dict[str, float] = {}
# Suppress identical messages sent within this many seconds (default 60s)
MIN_IDENTICAL_S = int(os.getenv("EXEC_TELEGRAM_MIN_IDENTICAL_S", "60") or 60)
_recent_msgs: Dict[str, float] = {}
# Global rate cap: maximum Telegram sends per minute across all messages
# EXEC_TELEGRAM_MAX_PER_MIN=0 blocks all sends immediately (mitigation mode)
SEND_CAP_PER_MIN = int(os.getenv("EXEC_TELEGRAM_MAX_PER_MIN", "6") or 6)
_send_timestamps: List[float] = []
# v7 4h-only mode: required keys in JSON payload for it to be allowed
_4H_REQUIRED_KEYS = {"atr_regime", "last_4h_close_ts"}
_MD_ESCAPE = str.maketrans(
    {
        "_": r"\_",
        "*": r"\*",
        "[": r"\[",
        "]": r"\]",
        "(": r"\(",
        ")": r"\)",
        "~": r"\~",
        "`": r"\`",
        ">": r"\>",
        "#": r"\#",
        "+": r"\+",
        "-": r"\-",
        "=": r"\=",
        "|": r"\|",
        "{": r"\{",
        "}": r"\}",
        ".": r"\.",
        "!": r"\!",
    }
)


def should_send_summary(last_sent_ts: float | None, minutes: int) -> bool:
    now = time.time()
    if not last_sent_ts:
        return True
    return (now - float(last_sent_ts)) >= max(60, minutes * 60)


# --- Message helpers used by executor_live.py ---
def send_heartbeat(
    equity: float,
    peak: float,
    dd_pct: float,
    realized: float,
    unrealized: float,
    positions_top: List[str],
):
    msg = (
        f"Heartbeat\n"
        f"Equity: {equity:,.2f} | Peak: {peak:,.2f} | DD: {dd_pct * 100:+.2f}%\n"
        f"PnL — R: {realized:,.2f} | U: {unrealized:,.2f}\n"
        f"Top: {', '.join(positions_top) if positions_top else '—'}"
    )
    send_telegram(msg, silent=True)


def send_trade_alert(
    symbol: str,
    side: str,
    qty: float,
    fill_price: float,
    realized: float,
    unrealized: float,
):
    msg = (
        f"🔔 {symbol} {side} {qty:g} @ {fill_price:,.2f}\n"
        f"R: {realized:,.2f} | U: {unrealized:,.2f}"
    )
    send_telegram(msg, silent=False)


def send_drawdown_alert(
    drawdown_pct: float, threshold_pct: float, peak_equity: float, equity: float
):
    global _last_dd_ts
    now = time.time()
    # rate‑limit to once per 15 minutes
    if _last_dd_ts and (now - _last_dd_ts) < 15 * 60:
        return
    _last_dd_ts = now
    msg = (
        f"⚠️ Drawdown Alert\n"
        f"DD: {drawdown_pct * 100:.2f}% (thr={threshold_pct * 100:.2f}%)\n"
        f"Equity: {equity:,.2f} | Peak: {peak_equity:,.2f}"
    )
    send_telegram(msg, silent=False)


def _escape_markdown(text: str) -> str:
    return str(text or "").translate(_MD_ESCAPE)


def send_execution_alerts(symbol: str, alerts: Sequence[Dict[str, Any]]) -> None:
    """
    Sends aggregated execution alerts per symbol using Markdown formatting.
    """
    if not alerts:
        return

    sym = str(symbol or "ALL").upper()
    now = time.time()
    last = _last_alert_ts.get(sym, 0.0)
    if now - last < ALERT_COOLDOWN_S:
        return

    severity_order = {"critical": 3, "warning": 2, "info": 1}
    emoji_map = {"critical": "🛑", "warning": "⚠️", "info": "ℹ️"}

    def _severity(alert: Dict[str, Any]) -> str:
        return str(alert.get("severity") or "info").lower()

    sorted_alerts = sorted(
        alerts,
        key=lambda alert: (
            -severity_order.get(_severity(alert), 0),
            str(alert.get("type") or ""),
        ),
    )
    top_severity = _severity(sorted_alerts[0])
    header = f"*Execution Alerts — {_escape_markdown(sym)}*"
    lines = []
    for alert in sorted_alerts:
        severity = _severity(alert)
        icon = emoji_map.get(severity, "•")
        msg = _escape_markdown(alert.get("msg") or f"{sym}: {alert.get('type', 'alert')}")
        severity_label = _escape_markdown(severity.upper())
        lines.append(f"{icon} *{severity_label}* — {msg}")

    body = "\n".join([header, "", *lines])
    silent = severity_order.get(top_severity, 0) < severity_order.get("critical", 3)
    try:
        send_telegram(body, silent=silent, parse_mode="MarkdownV2")
    except Exception:
        pass
    _last_alert_ts[sym] = now


# --- CLI smoke test ---
if __name__ == "__main__":
    ok = send_telegram("🚀 executor/telegram_utils.py smoke: hello.")
    print("send_ok:", ok)
