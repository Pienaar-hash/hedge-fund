import os
import sys
import json
import time
import subprocess
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path and PROJECT_ROOT.is_dir():
    sys.path.insert(0, str(PROJECT_ROOT))

# Local helpers
from dashboard.dashboard_utils import (  # noqa: E402
    get_firestore_connection,
    fetch_state_document,
    fetch_telemetry_health,
    parse_nav_to_df_and_kpis,
    positions_sorted,
    fetch_mark_price_usdt,
    get_env_float,
    get_env_badge,
)
from dashboard.nav_helpers import signal_attempts_summary  # noqa: E402
from dashboard.router_health import load_router_health, RouterHealthData  # noqa: E402
from scripts.doctor import collect_doctor_snapshot  # noqa: E402
try:
    import altair as alt  # type: ignore
except Exception:  # pragma: no cover
    alt = None  # type: ignore
try:
    from execution.utils import get_coingecko_prices
except Exception:  # pragma: no cover
    get_coingecko_prices = None
try:
    from execution.utils import get_usd_to_zar
except Exception:  # pragma: no cover
    get_usd_to_zar = None
try:
    from execution.nav import get_confirmed_nav as nav_get_confirmed_nav
    from execution.nav import get_nav_age as nav_get_age
    from execution.nav import is_nav_fresh as nav_is_fresh
except Exception:  # pragma: no cover
    nav_get_confirmed_nav = None
    nav_get_age = None
    nav_is_fresh = None
try:
    from execution.risk_limits import get_nav_age as risk_nav_get_age
except Exception:  # pragma: no cover
    risk_nav_get_age = None
try:
    from execution.reserves import load_reserves, value_reserves_usd
except Exception:  # pragma: no cover
    load_reserves = None  # type: ignore[assignment]
    value_reserves_usd = None  # type: ignore[assignment]

# Read-only exchange helpers

# Doctor helper
# removed: doctor runs via subprocess now

LOG = logging.getLogger("dash.app")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG.setLevel(logging.INFO)

def main():

    st.set_page_config(page_title="Hedge — Portfolio Dashboard", layout="wide")
    ENV = os.getenv("ENV", "dev")
    TESTNET = str(os.getenv("BINANCE_TESTNET", "0")).strip().lower() in ("1", "true", "yes", "on")
    REFRESH_SEC = int(os.getenv("DASHBOARD_REFRESH_SEC", "60"))
    # log-tail behavior
    TAIL_BYTES = int(os.getenv("DASHBOARD_LOG_TAIL_BYTES", "200000"))
    TAIL_LINES = int(os.getenv("DASHBOARD_SIGNAL_LINES", "80"))
    WANT_TAGS = tuple((os.getenv("DASHBOARD_SIGNAL_TAGS") or "[screener],[screener->executor],[decision]").split(","))

    LATENCY_CACHE_PATH = Path(os.getenv("EXEC_LATENCY_CACHE", "logs/execution/replay_cache.json"))
    LOG_PATH = Path(os.getenv("EXECUTOR_LOG", "logs/screener_tail.log"))
    HEARTBEAT_LOG_PATH = Path(os.getenv("SYNC_HEARTBEAT_LOG", "logs/execution/sync_heartbeats.jsonl"))

    # Optional auto-refresh
    try:
        from streamlit_extras.st_autorefresh import st_autorefresh
        st_autorefresh(interval=REFRESH_SEC * 1000, key="auto_refresh")
    except Exception:
        pass  # manual refresh only

    st.title("📊 Hedge — Portfolio Dashboard")
    st.caption(f"ENV = {ENV} · refresh ≈ {REFRESH_SEC}s")

    # --------------------------- Small utilities ---------------------------------
    def load_json(path: str, default=None):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {} if default is None else default

    def parse_iso_ts(ts: str):
        if not ts:
            return None
        try:
            cleaned = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None

    def format_latency(value):
        try:
            if value is None or (isinstance(value, float) and value != value):
                return "—"
            return f"{float(value):.0f} ms"
        except Exception:
            return "—"

    def human_age(ts) -> str:
        if not ts:
            return "–"
        try:
            now = int(time.time())
            d = now - int(ts)
            if d < 60:
                return f"{d}s"
            if d < 3600:
                return f"{d//60}m"
            return f"{d//3600}h"
        except Exception:
            return "–"

    def format_age_seconds(age_seconds: Optional[float]) -> str:
        if age_seconds is None:
            return "n/a"
        try:
            age_val = float(age_seconds)
        except Exception:
            return "n/a"
        if age_val != age_val or age_val < 0:
            return "n/a"
        if age_val < 60:
            return f"{age_val:.1f}s"
        age_int = int(age_val)
        if age_int < 3600:
            return f"{age_int // 60}m"
        if age_int < 86400:
            return f"{age_int // 3600}h"
        return f"{age_int // 86400}d"

    def tail_text(path: str, max_bytes: int = TAIL_BYTES) -> str:
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes), os.SEEK_SET)
                return f.read().decode(errors="ignore")
        except Exception as e:
            return f"(cannot read {path}: {e})"

    def rle_compact(lines: List[str], min_run: int = 3) -> List[str]:
        if not lines:
            return lines
        out = []
        prev = lines[0]
        count = 1
        for ln in lines[1:]:
            if ln == prev:
                count += 1
            else:
                out.append(prev if count < min_run else f"{prev}  × {count}")
                prev = ln
                count = 1
        out.append(prev if count < min_run else f"{prev}  × {count}")
        return out

    @st.cache_data(ttl=120, show_spinner=False)
    def cached_coingecko_prices() -> Dict[str, float]:
        if get_coingecko_prices is None:
            return {}
        try:
            return get_coingecko_prices()
        except Exception as exc:
            LOG.warning("[dash] coingecko unavailable: %s", exc)
            return {}

    @st.cache_data(ttl=120, show_spinner=False)
    def cached_usd_to_zar() -> Optional[float]:
        if get_usd_to_zar is None:
            return None
        try:
            return get_usd_to_zar()
        except Exception as exc:
            LOG.warning("[dash] usd→zar unavailable: %s", exc)
            return None

    def format_currency(value: Optional[float], symbol: str = "$") -> str:
        if value is None:
            return "—"
        try:
            formatted = f"{float(value):,.0f}".replace(",", " ")
            return f"{symbol}{formatted}"
        except Exception:
            return "—"

    def compare_nav_zar(prev: Optional[float], current: Optional[float], threshold: float = 0.01) -> str:
        if prev in (None, 0) or current is None:
            return ""
        try:
            prev_val = float(prev)
            curr_val = float(current)
            if prev_val <= 0:
                return ""
            pct = (curr_val - prev_val) / prev_val
            if abs(pct) < threshold:
                return ""
            arrow = "▲" if pct > 0 else "▼"
            color = "#16a34a" if pct > 0 else "#dc2626"
            pct_text = f"{pct*100:+.1f}%"
            return f"<span class='dash-nav-delta' style='color:{color};'>{arrow} {pct_text}</span>"
        except Exception:
            return ""

    def _to_epoch_seconds(value) -> Optional[float]:
        if value in (None, "", "null"):
            return None
        if isinstance(value, (int, float)):
            val = float(value)
            if val > 1e12:
                val /= 1000.0
            return val
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return None
            try:
                if txt.isdigit():
                    return _to_epoch_seconds(float(txt))
                return pd.to_datetime(txt, utc=True).timestamp()
            except Exception:
                return None
        try:
            return float(value)
        except Exception:
            return None

    def is_recent(ts: Optional[float], window_sec: int) -> bool:
        if ts is None:
            return False
        try:
            return (time.time() - float(ts)) <= window_sec
        except Exception:
            return False

    def load_recent_heartbeats(limit: int = 2) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        entries: deque[Dict[str, Any]] = deque(maxlen=limit)
        if not HEARTBEAT_LOG_PATH.exists():
            return []
        try:
            with HEARTBEAT_LOG_PATH.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        entries.append(dict(payload))
        except Exception as exc:
            LOG.debug("[dash] heartbeat log read failed: %s", exc, exc_info=True)
            return []
        return list(entries)

    def telemetry_health_card(env: str = ENV, firestore_available: bool = True) -> Tuple[Dict[str, Any], Optional[str], Optional[float]]:
        if not firestore_available:
            st.metric("Firestore", "🔴 Down", delta="unknown")
            return {}, "firestore_unavailable", None
        try:
            meta = fetch_telemetry_health(env)
            if not isinstance(meta, dict):
                meta = {}
            error: Optional[str] = None
        except Exception as exc:
            meta = {}
            error = str(exc)
        else:
            error = None

        ts_val = _to_epoch_seconds(meta.get("ts") or meta.get("updated_at"))
        age_seconds: Optional[float] = None
        if ts_val is not None:
            try:
                age_seconds = max(0.0, time.time() - float(ts_val))
            except Exception:
                age_seconds = None

        status_icon = "🔴"
        status_label = "Down"
        if error:
            status_icon = "🔴"
            status_label = "Down"
        else:
            ok = bool(meta.get("firestore_ok"))
            if age_seconds is not None and age_seconds <= 60 and ok:
                status_icon = "🟢"
                status_label = "Fresh"
            elif ok:
                status_icon = "🟡"
                status_label = "Stale"
            elif not meta:
                status_icon = "⚪"
                status_label = "Unknown"
            else:
                status_icon = "🔴"
                status_label = "Down"

        delta_text = f"{int(age_seconds)}s ago" if age_seconds is not None else "unknown"
        st.metric("Firestore", f"{status_icon} {status_label}", delta=delta_text)
        return meta, error, age_seconds

    def nav_freshness_info() -> Dict[str, Any]:
        record: Dict[str, Any] = {}
        if callable(nav_get_confirmed_nav):
            try:
                data = nav_get_confirmed_nav()
                if isinstance(data, dict):
                    record = data
            except Exception as exc:
                LOG.debug("[dash] get_confirmed_nav failed: %s", exc, exc_info=True)
        if not record:
            record = load_json("logs/cache/nav_confirmed.json", default={}) or {}
        nav_age: Optional[float] = None
        if callable(nav_get_age):
            try:
                nav_age = nav_get_age()
            except Exception as exc:
                LOG.debug("[dash] get_nav_age failed: %s", exc, exc_info=True)
        if nav_age is None:
            ts_val = record.get("ts")
            if isinstance(ts_val, (int, float)):
                try:
                    nav_age = max(0.0, time.time() - float(ts_val))
                except Exception:
                    nav_age = None
        risk_cfg = load_json("config/risk_limits.json", default={})
        threshold = None
        if isinstance(risk_cfg, dict):
            candidates = []
            global_cfg = risk_cfg.get("global") if isinstance(risk_cfg.get("global"), dict) else None
            if isinstance(global_cfg, dict):
                candidates.append(global_cfg.get("nav_freshness_seconds"))
            candidates.append(risk_cfg.get("nav_freshness_seconds"))
            for candidate in candidates:
                try:
                    val = float(candidate)
                    if val > 0:
                        threshold = val
                        break
                except Exception:
                    continue
        if threshold is not None and callable(nav_is_fresh):
            try:
                fresh = bool(nav_is_fresh(threshold))
            except Exception:
                fresh = nav_age is not None and nav_age <= threshold
        elif threshold is not None:
            fresh = nav_age is not None and nav_age <= threshold
        else:
            fresh = nav_age is not None
        return {"age": nav_age, "threshold": threshold, "fresh": bool(fresh)}

    def read_jsonl_tail(path: Path, limit: int) -> List[Dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()[-limit:]
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for line in lines:
            try:
                obj = json.loads(line.strip())
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    @st.cache_data(ttl=30, show_spinner=False)
    def load_local_nav_doc() -> Dict[str, Any]:
        path = PROJECT_ROOT / "logs" / "nav_log.json"
        data = load_json(str(path), [])
        points: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                ts_val = _to_epoch_seconds(entry.get("ts") or entry.get("t") or entry.get("time"))
                eq_raw = entry.get("equity") or entry.get("nav") or entry.get("value")
                if ts_val is None or eq_raw is None:
                    continue
                try:
                    eq_val = float(eq_raw)
                except Exception:
                    continue
                points.append({"ts": int(ts_val), "equity": eq_val})
        if not points:
            return {}
        points.sort(key=lambda item: item["ts"])
        latest_ts = points[-1]["ts"]
        return {
            "points": points,
            "total_equity": points[-1]["equity"],
            "updated_at": datetime.fromtimestamp(latest_ts, tz=timezone.utc).isoformat(),
        }

    def load_local_positions() -> Dict[str, Any]:
        path = PROJECT_ROOT / "logs" / "execution" / "position_state.jsonl"
        entries = read_jsonl_tail(path, 2000)
        latest: Dict[str, Dict[str, Any]] = {}
        for entry in reversed(entries):
            if not isinstance(entry, dict):
                continue
            sym = str(entry.get("symbol") or "").upper()
            if not sym or sym in latest:
                continue
            latest[sym] = entry

        items: List[Dict[str, Any]] = []
        for sym, data in latest.items():
            qty_raw = data.get("pos_qty") if data.get("pos_qty") is not None else data.get("qty")
            try:
                qty = float(qty_raw or 0.0)
            except Exception:
                qty = 0.0
            if abs(qty) <= 0:
                continue
            try:
                entry_px = float(data.get("entry_px") or data.get("entryPrice") or data.get("avgEntryPrice") or 0.0)
            except Exception:
                entry_px = 0.0
            try:
                mark_px_raw = data.get("mark_px") or data.get("markPrice")
                mark_px = float(mark_px_raw) if mark_px_raw not in (None, "") else None
            except Exception:
                mark_px = None
            upnl_raw = data.get("unrealized_pnl") or data.get("unrealizedPnl")
            try:
                upnl = float(upnl_raw if upnl_raw is not None else 0.0)
            except Exception:
                upnl = 0.0
            if mark_px is None and entry_px and qty:
                try:
                    side = str(data.get("mode") or data.get("side") or "LONG").upper()
                    sign = 1.0 if side in ("LONG", "BUY") else -1.0
                    mark_px = entry_px + (upnl / (abs(qty) * sign))
                except Exception:
                    mark_px = entry_px
            ts_val = _to_epoch_seconds(data.get("ts") or data.get("time") or data.get("timestamp"))
            notional = abs(qty) * float(mark_px if mark_px is not None else entry_px)
            items.append(
                {
                    "symbol": sym,
                    "positionAmt": qty,
                    "qty": qty,
                    "entryPrice": entry_px,
                    "markPrice": mark_px,
                    "unrealizedPnl": upnl,
                    "leverage": data.get("leverage") or data.get("lev"),
                    "updatedAt": ts_val,
                    "notional": notional,
                }
            )
        return {"items": items}

    def load_local_leaderboard() -> Dict[str, Any]:
        path = PROJECT_ROOT / "logs" / "leaderboard.json"
        data = load_json(str(path), {})
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return {"items": data.get("items")}
        if isinstance(data, list):
            return {"items": data}
        return {"items": []}

    def load_local_trade_log(limit: int = 200) -> pd.DataFrame:
        path = PROJECT_ROOT / "logs" / "execution" / "orders_executed.jsonl"
        entries = read_jsonl_tail(path, 5000)
        rows: List[Dict[str, Any]] = []
        for entry in reversed(entries):
            if not isinstance(entry, dict):
                continue
            etype = str(entry.get("event_type") or "").lower()
            if "executed" not in etype and "fill" not in etype:
                continue
            ts_val = _to_epoch_seconds(entry.get("ts") or entry.get("time"))
            if not is_recent(ts_val, 24 * 3600):
                continue
            try:
                qty_val = entry.get("qty") or entry.get("quantity")
                qty = float(qty_val) if qty_val not in (None, "") else None
            except Exception:
                qty = None
            try:
                price_val = entry.get("price")
                price = float(price_val) if price_val not in (None, "") else None
            except Exception:
                price = None
            try:
                pnl_val = entry.get("pnl") or entry.get("realized_pnl")
                pnl = float(pnl_val) if pnl_val not in (None, "") else None
            except Exception:
                pnl = None
            rows.append(
                {
                    "ts": ts_val,
                    "symbol": entry.get("symbol"),
                    "side": entry.get("side") or entry.get("position_side"),
                    "qty": qty,
                    "price": price,
                    "pnl": pnl,
                }
            )
        rows.sort(key=lambda row: row.get("ts") or 0.0, reverse=True)
        return pd.DataFrame(rows[:limit])

    def load_local_risk_blocks(limit: int = 200) -> pd.DataFrame:
        path = PROJECT_ROOT / "logs" / "execution" / "risk_vetoes.jsonl"
        entries = read_jsonl_tail(path, 5000)
        rows: List[Dict[str, Any]] = []
        for entry in reversed(entries):
            if not isinstance(entry, dict):
                continue
            ts_val = _to_epoch_seconds(entry.get("ts") or entry.get("time"))
            if not is_recent(ts_val, 24 * 3600):
                continue
            if str(entry.get("phase") or "").lower() not in ("blocked", "", "risk"):
                continue
            rows.append(
                {
                    "ts": ts_val,
                    "symbol": entry.get("symbol"),
                    "side": entry.get("side"),
                    "reason": entry.get("reason"),
                    "notional": entry.get("notional"),
                    "open_qty": entry.get("open_qty"),
                    "gross": entry.get("gross"),
                    "nav": entry.get("nav"),
                }
            )
        rows.sort(key=lambda row: row.get("ts") or 0.0, reverse=True)
        return pd.DataFrame(rows[:limit])

    def load_nav_state() -> Tuple[Dict[str, Any], str]:
        try:
            doc = fetch_state_document("nav", env=ENV)
            if isinstance(doc, dict) and doc:
                return doc, "firestore"
        except Exception as exc:
            LOG.warning("[dash] nav Firestore fetch failed: %s", exc)
        return load_local_nav_doc(), "local"

    def load_positions_state() -> Tuple[Dict[str, Any], str]:
        try:
            doc = fetch_state_document("positions", env=ENV)
            if isinstance(doc, dict) and doc:
                return doc, "firestore"
        except Exception as exc:
            LOG.warning("[dash] positions Firestore fetch failed: %s", exc)
        return load_local_positions(), "local"

    def load_leaderboard_state() -> Tuple[Dict[str, Any], str]:
        try:
            doc = fetch_state_document("leaderboard", env=ENV)
            if isinstance(doc, dict) and doc:
                return doc, "firestore"
        except Exception as exc:
            LOG.warning("[dash] leaderboard Firestore fetch failed: %s", exc)
        return load_local_leaderboard(), "local"

    DOCTOR_CACHE_PATH = PROJECT_ROOT / "logs" / "cache" / "doctor.json"
    SIGNAL_METRICS_PATH = PROJECT_ROOT / "logs" / "execution" / "signal_metrics.jsonl"
    ML_CACHE_PATH = PROJECT_ROOT / "logs" / "cache" / "ml_predictions.json"

    @st.cache_data(ttl=30, show_spinner=False)
    def cached_doctor_snapshot() -> Dict[str, Any]:
        if DOCTOR_CACHE_PATH.exists():
            try:
                with DOCTOR_CACHE_PATH.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:
                LOG.debug("[dash] doctor cache read failed: %s", exc)
        try:
            snapshot = collect_doctor_snapshot() or {}
        except Exception as exc:
            LOG.warning("[dash] doctor snapshot fallback failed: %s", exc)
            snapshot = {}
        try:
            DOCTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with DOCTOR_CACHE_PATH.open("w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            pass
        return snapshot

    @st.cache_data(ttl=20, show_spinner=False)
    def cached_signal_metrics(limit: int = 400) -> List[Dict[str, Any]]:
        return read_jsonl_tail(SIGNAL_METRICS_PATH, limit)

    @st.cache_data(ttl=60, show_spinner=False)
    def cached_ml_predictions() -> Dict[str, Any]:
        if ML_CACHE_PATH.exists():
            try:
                with ML_CACHE_PATH.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:
                LOG.debug("[dash] ml cache read failed: %s", exc)
        return {}

    def load_trades(limit: int, db=None) -> pd.DataFrame:
        if db is None:
            return load_local_trade_log(limit)
        rows: List[Dict[str, Any]] = []
        try:
            docs = (
                db.collection("hedge")
                .document(ENV)
                .collection("trades")
                .order_by("ts", direction="DESCENDING")
                .limit(1000)
                .stream()
            )
            for doc in docs:
                payload = doc.to_dict() or {}
                if (payload.get("env") is not None and str(payload.get("env")) != ENV) or (
                    payload.get("testnet") is not None and bool(payload.get("testnet")) != bool(os.getenv("BINANCE_TESTNET"))
                ):
                    continue
                ts_val = _to_epoch_seconds(payload.get("ts") or payload.get("time"))
                if not is_recent(ts_val, 24 * 3600):
                    continue
                rows.append(
                    {
                        "ts": ts_val,
                        "symbol": payload.get("symbol"),
                        "side": payload.get("side"),
                        "qty": payload.get("qty"),
                        "price": payload.get("price"),
                        "pnl": payload.get("pnl"),
                    }
                )
        except Exception as exc:
            LOG.warning("[dash] trades Firestore fetch failed: %s", exc)
            return load_local_trade_log(limit)
        rows.sort(key=lambda row: row.get("ts") or 0.0, reverse=True)
        return pd.DataFrame(rows[:limit])

    def load_risk_blocks(limit: int, db=None) -> pd.DataFrame:
        if db is None:
            return load_local_risk_blocks(limit)
        rows: List[Dict[str, Any]] = []
        try:
            docs = (
                db.collection("hedge")
                .document(ENV)
                .collection("risk")
                .order_by("ts", direction="DESCENDING")
                .limit(2000)
                .stream()
            )
            for doc in docs:
                payload = doc.to_dict() or {}
                if (payload.get("env") is not None and str(payload.get("env")) != ENV) or (
                    payload.get("testnet") is not None and bool(payload.get("testnet")) != bool(os.getenv("BINANCE_TESTNET"))
                ):
                    continue
                ts_val = _to_epoch_seconds(payload.get("ts") or payload.get("time"))
                if not is_recent(ts_val, 24 * 3600):
                    continue
                if "phase" in payload and str(payload.get("phase")).lower() != "blocked":
                    continue
                rows.append(
                    {
                        "ts": ts_val,
                        "symbol": payload.get("symbol"),
                        "side": payload.get("side"),
                        "reason": payload.get("reason"),
                        "notional": payload.get("notional"),
                        "open_qty": payload.get("open_qty"),
                        "gross": payload.get("gross"),
                        "nav": payload.get("nav"),
                    }
                )
        except Exception as exc:
            LOG.warning("[dash] risk Firestore fetch failed: %s", exc)
            return load_local_risk_blocks(limit)
        rows.sort(key=lambda row: row.get("ts") or 0.0, reverse=True)
        return pd.DataFrame(rows[:limit])

    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def load_signals_table(limit: int = 100) -> pd.DataFrame:
        metrics_rows = cached_signal_metrics(800)
        cutoff = time.time() - 24 * 3600
        table_rows: List[Dict[str, Any]] = []
        for row in reversed(metrics_rows):
            if not isinstance(row, dict):
                continue
            ts_val = _to_epoch_seconds(row.get("ts") or row.get("time"))
            if ts_val is None or ts_val < cutoff:
                continue
            doctor = row.get("doctor") if isinstance(row.get("doctor"), dict) else {}
            reasons_raw = doctor.get("reasons") if isinstance(doctor, dict) else None
            if isinstance(reasons_raw, (list, tuple, set)):
                reasons = ", ".join(str(item) for item in reasons_raw if item)
            elif isinstance(reasons_raw, str):
                reasons = reasons_raw
            else:
                reasons = ""
            confidence = _safe_float(doctor.get("confidence")) if isinstance(doctor, dict) else None
            status = "Emitted" if isinstance(doctor, dict) and doctor.get("ok") else "Vetoed"
            table_rows.append(
                {
                    "ts": ts_val,
                    "symbol": row.get("symbol"),
                    "signal": row.get("signal"),
                    "queue_depth": row.get("queue_depth"),
                    "latency_ms": _safe_float(row.get("latency_ms")),
                    "doctor_confidence": confidence,
                    "doctor_status": status,
                    "doctor_reasons": reasons,
                }
            )
        if table_rows:
            df = pd.DataFrame(table_rows)
            df.sort_values("ts", ascending=False, inplace=True, ignore_index=True)
            return df.head(limit)

        # Fallback: parse screener tail log
        lines: List[str] = []
        if LOG_PATH.exists():
            text = tail_text(str(LOG_PATH), max_bytes=TAIL_BYTES)
            for line in text.splitlines():
                if any(tag in line for tag in WANT_TAGS):
                    lines.append(line)
        path = PROJECT_ROOT / "logs" / "screener_tail.log"
        if not lines and path.exists():
            text = tail_text(str(path), max_bytes=TAIL_BYTES)
            for line in text.splitlines():
                if any(tag in line for tag in WANT_TAGS):
                    lines.append(line)
        import ast

        for line in lines[-200:]:
            payload = None
            if "{" in line:
                try:
                    json_part = line.split("{", 1)[1]
                    json_part = "{" + json_part.split("}", 1)[0] + "}"
                    payload = json.loads(json_part)
                except Exception:
                    try:
                        payload = ast.literal_eval(line.split(":", 1)[-1].strip())
                    except Exception:
                        payload = None
            if not isinstance(payload, dict):
                continue
            ts_val = _to_epoch_seconds(payload.get("timestamp") or payload.get("t") or payload.get("time"))
            if not is_recent(ts_val, 24 * 3600):
                continue
            veto_display = payload.get("veto") or payload.get("reason") or ""
            if isinstance(veto_display, (list, tuple, set)):
                veto_display = ", ".join(str(item) for item in veto_display if item)
            table_rows.append(
                {
                    "ts": ts_val,
                    "symbol": payload.get("symbol"),
                    "signal": payload.get("signal"),
                    "queue_depth": payload.get("queue_depth"),
                    "latency_ms": _safe_float(payload.get("latency_ms")),
                    "doctor_confidence": None,
                    "doctor_status": "Emitted" if not veto_display else "Vetoed",
                    "doctor_reasons": veto_display,
                }
            )
        if table_rows:
            df = pd.DataFrame(table_rows)
            df.sort_values("ts", ascending=False, inplace=True, ignore_index=True)
            return df.head(limit)
        return pd.DataFrame(columns=["ts", "symbol", "signal", "doctor_status"])

    # --------------------------- Load Firestore ----------------------------------
    status = st.empty()
    status.info("Loading data…")

    firestore_error: Optional[str] = None
    try:
        db = get_firestore_connection()
    except Exception as exc:
        firestore_error = str(exc)
        LOG.warning("[dash] firestore connection failed: %s", exc)
        db = None

    nav_doc, nav_source = load_nav_state()
    pos_doc, pos_source = load_positions_state()
    lb_doc, lb_source = load_leaderboard_state()

    doctor_snapshot = cached_doctor_snapshot() or {}
    if not isinstance(doctor_snapshot, dict):
        doctor_snapshot = {}

    positions_info = doctor_snapshot.get("positions") or {}
    if not isinstance(positions_info, dict):
        positions_info = {}
    firestore_info = doctor_snapshot.get("firestore") or {}
    if not isinstance(firestore_info, dict):
        firestore_info = {}
    reserves_snapshot = doctor_snapshot.get("reserves") or {}
    if not isinstance(reserves_snapshot, dict):
        reserves_snapshot = {}

    nav_df, kpis = parse_nav_to_df_and_kpis(nav_doc or {})
    kpis = dict(kpis or {})
    try:
        positions_fs = positions_sorted((pos_doc or {}).get("items") or [])
    except Exception as exc:
        LOG.warning("[dash] positions parse failed: %s", exc)
        positions_fs = []
    if not isinstance(positions_info.get("count"), (int, float)):
        positions_info = dict(positions_info)
        positions_info["count"] = len(positions_fs)
    leaderboard = (lb_doc or {}).get("items") or []

    status.success(
        f"Loaded · nav_source={nav_source} · positions_source={pos_source} · leaderboard_source={lb_source}"
    )
    if not nav_doc:
        st.warning("NAV data unavailable; showing empty series.")

    exec_stats = (pos_doc or {}).get("exec_stats") or (nav_doc or {}).get("exec_stats") or {}
    latency_cache = load_json(str(LATENCY_CACHE_PATH), default={})
    latency_summary = (
        (latency_cache.get("summary") or {}).get("latency")
        or latency_cache.get("latency")
        or {}
    )
    df_tr = load_trades(200, db)
    df_rb = load_risk_blocks(200, db)
    df_sig = load_signals_table(150)
    # ---- Top navigation header ---------------------------------------------------
    env_label, env_color = get_env_badge(TESTNET)
    badge_icon = "🟠" if TESTNET else "🟢"

    nav_value: Optional[float] = None
    raw_equity = kpis.get("total_equity")
    if isinstance(raw_equity, (int, float)):
        nav_value = float(raw_equity)
    elif not nav_df.empty:
        nav_value = float(nav_df["equity"].iloc[-1])
    elif isinstance((nav_doc or {}).get("total_equity"), (int, float)):
        nav_value = float((nav_doc or {}).get("total_equity"))

    nav_ts: Optional[float] = None
    if not nav_df.empty:
        latest_idx = pd.Timestamp(nav_df.index[-1])
        if latest_idx.tzinfo is None:
            latest_idx = latest_idx.tz_localize("UTC")
        else:
            latest_idx = latest_idx.tz_convert("UTC")
        nav_ts = float(latest_idx.timestamp())
    if nav_ts is None:
        nav_ts = _to_epoch_seconds((nav_doc or {}).get("updated_at"))

    trade_ts = float(df_tr["ts"].max()) if not df_tr.empty and "ts" in df_tr else None
    risk_ts = float(df_rb["ts"].max()) if not df_rb.empty and "ts" in df_rb else None

    last_sync_candidates = [ts for ts in (nav_ts, trade_ts, risk_ts) if ts]
    last_sync_ts = max(last_sync_candidates) if last_sync_candidates else None
    if last_sync_ts is not None:
        last_sync_label = datetime.fromtimestamp(last_sync_ts, tz=timezone.utc).strftime("%H:%M:%S")
    else:
        last_sync_label = "—"

    exchange_nav_value = nav_value if isinstance(nav_value, (int, float)) else None
    reserves_total_usd = 0.0

    if callable(load_reserves) and callable(value_reserves_usd):
        try:
            reserves_raw = load_reserves()
            if reserves_raw:
                reserves_total_calc, reserves_detail_calc = value_reserves_usd(reserves_raw)
                reserves_total_usd = float(reserves_total_calc)
                reserves_snapshot = {
                    "total": reserves_total_usd,
                    "reserves": reserves_detail_calc,
                    "raw": reserves_raw,
                }
        except Exception as exc:
            LOG.warning("[dash] reserves valuation failed: %s", exc)

    if not reserves_total_usd:
        candidate_total = reserves_snapshot.get("total")
        if isinstance(candidate_total, (int, float)):
            reserves_total_usd = float(candidate_total)

    if (
        reserves_total_usd
        and isinstance(nav_df, pd.DataFrame)
        and not nav_df.empty
        and "equity" in nav_df.columns
    ):
        nav_df = nav_df.copy()
        try:
            nav_df["equity"] = pd.to_numeric(nav_df["equity"], errors="coerce").fillna(0.0) + reserves_total_usd
        except Exception:
            nav_df["equity"] = nav_df["equity"].apply(
                lambda val: (float(val) if isinstance(val, (int, float)) else 0.0) + reserves_total_usd
            )

    total_nav_value = None
    if exchange_nav_value is not None or reserves_total_usd:
        total_nav_value = (exchange_nav_value or 0.0) + reserves_total_usd
        nav_value = total_nav_value
    else:
        nav_value = None

    if isinstance(total_nav_value, (int, float)):
        kpis["total_equity"] = total_nav_value

    prev_nav_value = None
    if isinstance(nav_df, pd.DataFrame) and nav_df.shape[0] >= 2:
        try:
            prev_nav_value = float(nav_df["equity"].iloc[-2])
        except Exception:
            prev_nav_value = None

    nav_freshness = doctor_snapshot.get("nav_freshness", {}) or {}
    if not isinstance(nav_freshness, dict):
        nav_freshness = {}
    nav_status = "FRESH" if nav_freshness.get("fresh") else "STALE"

    nav_info = doctor_snapshot.get("nav", {}) or {}
    if not isinstance(nav_info, dict):
        nav_info = {}

    nav_age_seconds: Optional[float]
    raw_age = nav_freshness.get("age")
    if isinstance(raw_age, (int, float)):
        nav_age_seconds = float(raw_age)
    else:
        nav_age_candidate = nav_info.get("age")
        if isinstance(nav_age_candidate, (int, float)):
            nav_age_seconds = float(nav_age_candidate)
        elif risk_nav_get_age is not None:
            try:
                nav_age_seconds = float(risk_nav_get_age())
            except Exception:
                nav_age_seconds = None
        elif nav_get_age is not None:
            try:
                nav_age_seconds = float(nav_get_age())
            except Exception:
                nav_age_seconds = None
        else:
            nav_age_seconds = None

    nav_usd_text = format_currency(nav_value, "$")
    exchange_nav_display = format_currency(exchange_nav_value, "$")
    nav_zar_value = None
    nav_zar_text = "≈ R—"
    zar_rate = nav_freshness.get("zar_rate") or cached_usd_to_zar()
    zar_source = nav_freshness.get("zar_source") or ("cached" if zar_rate else "none")
    if zar_rate and isinstance(nav_value, (int, float)):
        nav_zar_value = float(nav_value) * float(zar_rate)
        nav_zar_text = f"≈ {format_currency(nav_zar_value, 'R')}"

    prev_nav_zar = None
    if zar_rate and isinstance(prev_nav_value, (int, float)):
        prev_nav_zar = float(prev_nav_value) * float(zar_rate)

    nav_delta_html = compare_nav_zar(prev_nav_zar, nav_zar_value)

    nav_display = f"{nav_value:,.2f} USD" if isinstance(nav_value, (int, float)) else "n/a"

    header_css = """
    <style>
    .dash-top-nav {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: rgba(248, 250, 252, 0.88);
        color: #111827;
        padding: 0.65rem 1.2rem;
        border-radius: 0 0 14px 14px;
        border: 1px solid rgba(15, 23, 42, 0.06);
        backdrop-filter: blur(8px);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.85rem;
    }
    .dash-top-nav .dash-nav-left {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
        font-weight: 600;
    }
    .dash-top-nav .dash-nav-main-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        row-gap: 0.2rem;
    }
    .dash-top-nav .dash-nav-main {
        font-size: 1.08rem;
        font-weight: 700;
        color: #0f172a;
    }
    .dash-top-nav .dash-nav-right {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-weight: 600;
    }
    .dash-top-nav .dash-nav-rate {
        font-size: 0.85rem;
        color: rgba(30, 41, 59, 0.75);
    }
    .dash-top-nav .dash-env-badge {
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        color: #ffffff;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.12);
    }
    .dash-top-nav .dash-nav-ts {
        font-size: 0.85rem;
        color: rgba(30, 41, 59, 0.85);
    }
    .dash-top-nav .dash-nav-delta {
        font-size: 0.95rem;
        font-weight: 600;
    }
    .zar-note {
        color: rgba(71, 85, 105, 0.9);
        font-size: 0.85rem;
    }
    .dash-top-nav .dash-nav-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.1rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-left: 0.6rem;
    }
    .dash-top-nav .dash-nav-chip.fresh {
        background: rgba(34, 197, 94, 0.15);
        color: #15803d;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .dash-top-nav .dash-nav-chip.stale {
        background: rgba(248, 113, 113, 0.18);
        color: #b91c1c;
        border: 1px solid rgba(248, 113, 113, 0.4);
    }
    .dash-top-nav .dash-nav-chip.dd-ok {
        background: rgba(34, 197, 94, 0.12);
        color: #166534;
        border: 1px solid rgba(34, 197, 94, 0.25);
    }
    .dash-top-nav .dash-nav-chip.dd-warn {
        background: rgba(248, 113, 113, 0.18);
        color: #b91c1c;
        border: 1px solid rgba(248, 113, 113, 0.35);
    }
    .dash-top-nav .dash-nav-heartbeat {
        font-size: 0.78rem;
        color: rgba(30, 41, 59, 0.78);
        background: rgba(148, 163, 184, 0.18);
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.35);
    }
    </style>
    """

    nav_main_text = f"⚡ NAV {nav_usd_text} | {nav_zar_text} | {env_label}"
    rate_display = f"R{zar_rate:.2f}/USD" if zar_rate else "R—/USD"

    drawdown_snapshot = nav_freshness.get("drawdown", {})
    heartbeat_snapshot = doctor_snapshot.get("heartbeats", {})

    nav_age_display = format_age_seconds(nav_age_seconds)
    nav_is_fresh = bool(nav_freshness.get("fresh"))
    nav_chip_class = "fresh" if nav_is_fresh else "stale"
    nav_chip_label = "Fresh" if nav_is_fresh else "STALE"
    nav_chip_label = f"{nav_chip_label} · {nav_age_display}"
    nav_chip_title = f"NAV age {nav_age_display}"
    nav_chip_html = f"<span class='dash-nav-chip {nav_chip_class}' title=\"{nav_chip_title}\">{nav_chip_label}</span>"

    dd_pct = drawdown_snapshot.get("dd_pct") or 0.0
    dd_abs = drawdown_snapshot.get("dd_abs") or 0.0
    dd_chip_class = "dd-ok" if not dd_pct or dd_pct <= 0 else "dd-warn"
    dd_chip_label = f"Drawdown · {dd_pct:.2f}%"
    dd_chip_title = f"Daily drawdown {dd_pct:.2f}%"
    dd_chip_html = f"<span class='dash-nav-chip {dd_chip_class}' title=\"{dd_chip_title}\">{dd_chip_label}</span>"

    treasury_total = reserves_total_usd
    if not isinstance(treasury_total, (int, float)):
        candidate_total = reserves_snapshot.get("total")
        if isinstance(candidate_total, (int, float)):
            treasury_total = float(candidate_total)
    treasury_display = format_currency(treasury_total, "$") if treasury_total is not None else "—"
    nav_tooltip = (
        f"Total NAV (USD)\nExchange: {exchange_nav_display} (source {nav_info.get('source', 'n/a')})\nReserves: {treasury_display}"
        if nav_display != "n/a"
        else "Total NAV (USD)"
    )
    nav_tooltip = nav_tooltip.replace("\n", "&#10;")
    nav_span = f"<span class='dash-nav-main' title=\"{nav_tooltip}\">{nav_main_text}</span>"
    nav_main_block = f"<div class='dash-nav-main-row'>{nav_span}{nav_chip_html}{dd_chip_html}</div>"

    hb_services = (heartbeat_snapshot or {}).get("services", {})
    hb_status = heartbeat_snapshot.get("status", "n/a")
    hb_exec_age = hb_services.get("executor", {}).get("age_display", "n/a")
    hb_sync_age = hb_services.get("sync_state", {}).get("age_display", "n/a")
    heartbeat_brief = f"Exec {hb_exec_age} · Sync {hb_sync_age}"

    header_html = f"""
    <div class="dash-top-nav">
        <div class="dash-nav-left">
            {nav_main_block}
            {nav_delta_html}
        </div>
        <div class="dash-nav-right">
            <span class="dash-nav-rate">{rate_display}</span>
            <span class="dash-nav-ts">Last Sync: {last_sync_label} UTC</span>
            <span class="dash-nav-heartbeat">{heartbeat_brief}</span>
            <span class="dash-env-badge" style="background:{env_color};">{badge_icon} {env_label}</span>
        </div>
    </div>
    """

    st.markdown(header_css + header_html, unsafe_allow_html=True)

    log_nav_value = f"{nav_value:.2f}" if isinstance(nav_value, (int, float)) else "n/a"
    LOG.info("[dash] header updated NAV=%s source=%s env=%s", log_nav_value, nav_source, env_label)

    # ---- Read-only Reserve KPI ---------------------------------------------------
    RESERVE_BTC = get_env_float("DASHBOARD_RESERVE_BTC", 0.13)
    price_cache = cached_coingecko_prices()
    btc_price = float(price_cache.get("BTC") or 0.0)
    if not btc_price:
        btc_price = fetch_mark_price_usdt("BTCUSDT")
    reserve_usdt = RESERVE_BTC * btc_price if btc_price and btc_price > 0 else 0.0
    reserve_zar = reserve_usdt * float(zar_rate) if zar_rate and reserve_usdt else None

    # ---- Tabs layout --------------------------------------------------------------
    (
        tab_overview,
        tab_positions,
        tab_execution,
        tab_router,
        tab_leader,
        tab_signals,
        tab_ml_insights,
        tab_ml_models,
        tab_doctor,
    ) = st.tabs(
        [
            "Overview",
            "Positions",
            "Execution",
            "Router Health",
            "Leaderboard",
            "Signals",
            "ML Insights",
            "ML Models",
            "Doctor",
        ]
    )

    # --------------------------- Overview Tab ------------------------------------
    with tab_overview:
        st.subheader("Doctor Summary")
        doc_col1, doc_col2, doc_col3, doc_col4 = st.columns(4)
        doc_col1.metric("NAV", nav_display, None)
        doc_col2.metric("Freshness", nav_status, nav_age_display)
        doc_col3.metric("Drawdown", f"{dd_pct:.2f}%", format_currency(dd_abs, "$"))
        doc_col4.metric("Heartbeat", hb_status, heartbeat_brief)

        doc_row2 = st.columns(4)
        doc_row2[0].metric("Positions", positions_info.get("count", 0), None)
        doc_row2[1].metric(
            "Firestore",
            firestore_info.get("status", "n/a").upper() if firestore_info.get("enabled") else "DISABLED",
            firestore_info.get("age", "n/a"),
        )
        doc_row2[2].metric(
            "ZAR Rate",
            f"{zar_rate:.2f}" if zar_rate else "n/a",
            zar_source if zar_rate else "none",
        )
        doc_row2[3].metric(
            "Reserves",
            treasury_display,
            None,
        )

        st.markdown("---")
        st.subheader("Portfolio KPIs")
        equity = kpis.get("total_equity")
        peak_equity = kpis.get("peak_equity")
        drawdown_pct = kpis.get("drawdown")
        realized_pnl = kpis.get("realized_pnl")
        unrealized_pnl = kpis.get("unrealized_pnl")

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Equity (USDT)", f"{equity:,.0f}" if equity is not None else "—")
        k2.metric("Peak (USDT)", f"{peak_equity:,.0f}" if peak_equity is not None else "—")
        k3.metric("Drawdown (%)", f"{drawdown_pct:.2f}%" if drawdown_pct is not None else "—")
        k4.metric("Realized PnL", f"{realized_pnl:,.0f}" if realized_pnl is not None else "—")
        k5.metric("Unrealized PnL", f"{unrealized_pnl:,.0f}" if unrealized_pnl is not None else "—")

        reserve_caption = f"~{format_currency(reserve_usdt, '$')}" if reserve_usdt and reserve_usdt > 0 else "—"
        k6.metric("Reserve (BTC)", f"{RESERVE_BTC:.3f} BTC", reserve_caption)

        if zar_rate and nav_zar_value:
            st.markdown(
                f"<div class='zar-note'>{format_currency(nav_value, '$')} ≈ {format_currency(nav_zar_value, 'R')} @ R{zar_rate:.2f}/USD</div>",
                unsafe_allow_html=True,
            )
        if zar_rate and isinstance(reserve_zar, (int, float)) and reserve_zar:
            st.markdown(
                f"<div class='zar-note'>Reserves ≈ {format_currency(reserve_zar, 'R')}</div>",
                unsafe_allow_html=True,
            )

        if nav_df.empty:
            st.info("No NAV points yet. Run executor + sync_state to populate.")
        else:
            plot_df = nav_df.copy()
            equity_series = pd.to_numeric(plot_df.get("equity"), errors="coerce")
            equity_series = equity_series.fillna(method="ffill")
            peak = equity_series.cummax()
            denom = peak.replace(0, pd.NA)
            drawdown_pct = ((equity_series - denom) / denom).fillna(0.0) * 100.0
            plot_df = plot_df.assign(equity=equity_series, drawdown_pct=drawdown_pct)
            plot_df_reset = plot_df.reset_index(names="time")
            if alt is not None:
                nav_chart = (
                    alt.layer(
                        alt.Chart(plot_df_reset)
                        .mark_line(color="#1f77b4")
                        .encode(
                            x=alt.X("time:T", title="Time"),
                            y=alt.Y("equity:Q", title="Equity (USDT)", axis=alt.Axis(titleColor="#1f77b4")),
                        ),
                        alt.Chart(plot_df_reset)
                        .mark_line(color="#d62728")
                        .encode(
                            x=alt.X("time:T", title="Time"),
                            y=alt.Y(
                                "drawdown_pct:Q",
                                title="Drawdown (%)",
                                axis=alt.Axis(titleColor="#d62728", orient="right"),
                            ),
                        ),
                    ).resolve_scale(y="independent")
                )
                st.altair_chart(nav_chart, use_container_width=True)
            else:
                st.line_chart(plot_df["equity"], use_container_width=True)
                st.line_chart(plot_df[["drawdown_pct"]], use_container_width=True)

    # --------------------------- Positions Tab -----------------------------------
    with tab_positions:
        st.subheader("Open Positions")
        positions_df = pd.DataFrame(positions_fs)
        if positions_df is None or positions_df.empty:
            st.info("No open positions.")
        else:
            for col in ("entryPrice", "markPrice"):
                if col in positions_df.columns:
                    positions_df[col] = pd.to_numeric(positions_df[col], errors="coerce")
            if "entryPrice" in positions_df.columns and "markPrice" in positions_df.columns:
                entry = positions_df["entryPrice"].replace({0: pd.NA})
                pnl_pct = ((positions_df["markPrice"] - entry) / entry) * 100
                positions_df["PnL%"] = pnl_pct.round(2)
            cols_order = [
                c for c in [
                    "symbol",
                    "positionAmt",
                    "entryPrice",
                    "markPrice",
                    "PnL%",
                    "unrealizedPnl",
                    "leverage",
                    "updatedAt",
                ]
                if c in positions_df.columns
            ]
            remaining_cols = [c for c in positions_df.columns if c not in cols_order]
            positions_df = positions_df[cols_order + remaining_cols]
            st.dataframe(positions_df, use_container_width=True, height=420)

    # --------------------------- Execution Tab ----------------------------------
    with tab_execution:
        st.subheader("Execution Health")

        telemetry_health, telemetry_fetch_error, telemetry_age = telemetry_health_card(
            ENV, firestore_available=(db is not None)
        )
        telemetry_error = telemetry_fetch_error or firestore_error
        telemetry_fallback: List[Dict[str, Any]] = []
        if telemetry_error or not telemetry_health or (telemetry_age is not None and telemetry_age > 60):
            telemetry_fallback = load_recent_heartbeats()
        if not telemetry_fallback and not telemetry_health:
            telemetry_fallback = load_recent_heartbeats()

        heartbeats = exec_stats.get("last_heartbeats") or {}
        now_ts = time.time()
        hb_rows: List[Tuple[str, Optional[float]]] = []
        stale = False
        for svc in ("executor_live", "sync_daemon"):
            ts_iso = heartbeats.get(svc)
            ts_val = parse_iso_ts(ts_iso) if ts_iso else None
            if ts_val is not None:
                age = now_ts - ts_val
                hb_rows.append((svc, age))
                if age > 180:
                    stale = True
            else:
                hb_rows.append((svc, None))
                stale = True

        hb_text = []
        for svc, age in hb_rows:
            if age is None:
                hb_text.append(f"{svc}: n/a")
            elif age < 60:
                hb_text.append(f"{svc}: {int(age)}s")
            elif age < 3600:
                hb_text.append(f"{svc}: {int(age // 60)}m")
            else:
                hb_text.append(f"{svc}: {int(age // 3600)}h")

        banner_message = " · ".join(hb_text) if hb_text else "No heartbeat data"
        if stale:
            st.error(f"Heartbeat stale · {banner_message}")
        else:
            st.success(f"Heartbeats healthy · {banner_message}")

        if not exec_stats:
            st.info("Execution stats unavailable. Ensure executor and sync daemon telemetry is publishing.")
        else:
            col_attempts, col_exec, col_veto, col_fill, col_p50, col_p90 = st.columns(6)
            attempted = exec_stats.get("attempted_24h")
            executed = exec_stats.get("executed_24h")
            vetoes = exec_stats.get("vetoes_24h")
            fill_rate = exec_stats.get("fill_rate")

            col_attempts.metric("Attempted (24h)", f"{attempted:,}" if attempted is not None else "—")
            col_exec.metric("Executed (24h)", f"{executed:,}" if executed is not None else "—")
            col_veto.metric("Vetoes (24h)", f"{vetoes:,}" if vetoes is not None else "—")

            if fill_rate is None:
                fill_display = "—"
            else:
                try:
                    fr = float(fill_rate)
                    fill_display = f"{fr*100:.1f}%" if fr <= 1 else f"{fr:.1f}%"
                except Exception:
                    fill_display = "—"
            col_fill.metric("Fill Rate", fill_display)

            col_p50.metric("Latency p50", format_latency(latency_summary.get("p50_ms")))
            col_p90.metric("Latency p90", format_latency(latency_summary.get("p90_ms")))

            top_vetoes = exec_stats.get("top_vetoes") or []
            if top_vetoes:
                st.markdown("#### Top Veto Reasons (24h)")
                df_veto = pd.DataFrame(top_vetoes)
                if not df_veto.empty:
                    df_veto = df_veto.rename(columns={"reason": "Reason", "count": "Count"})
                    st.dataframe(df_veto, use_container_width=True, height=280)
                else:
                    st.info("No veto data to display.")
            else:
                st.info("No veto data in the last 24 hours.")

            if not latency_summary:
                st.caption(
                    "Latency metrics unavailable. Generate a cache with "
                    "`python scripts/replay_logs.py --since <iso> --json logs/execution/replay_cache.json`."
                )

        st.markdown("#### Trade Log (last 24h)")
        if df_tr.empty:
            st.info("No recent trades.")
        else:
            newest_tr = float(df_tr["ts"].max()) if "ts" in df_tr else None
            if newest_tr and not is_recent(newest_tr, 1800):
                st.warning("Trades data may be stale (>30m)")
            st.dataframe(df_tr, use_container_width=True, height=220)

        st.markdown("#### Risk Blocks (last 24h)")
        if df_rb.empty:
            st.info("No recent risk blocks.")
        else:
            newest_rb = float(df_rb["ts"].max()) if "ts" in df_rb else None
            if newest_rb and not is_recent(newest_rb, 1800):
                st.warning("Risk blocks may be stale (>30m)")
            st.dataframe(df_rb, use_container_width=True, height=220)

        telemetry_section = st.container()
        telemetry_section.markdown("#### Backend Telemetry")
        fallback_entries = telemetry_fallback or []
        latest_ts = _to_epoch_seconds(telemetry_health.get("ts") or telemetry_health.get("updated_at"))
        fallback_ts_values = [
            _to_epoch_seconds(entry.get("ts") or entry.get("time")) for entry in fallback_entries
        ]
        fallback_ts_values = [ts for ts in fallback_ts_values if ts is not None]
        if fallback_ts_values:
            fallback_latest = max(fallback_ts_values)
            if latest_ts is None or fallback_latest > latest_ts:
                latest_ts = fallback_latest
        age_seconds: Optional[float] = None
        if latest_ts is not None:
            try:
                age_seconds = max(0.0, time.time() - float(latest_ts))
            except Exception:
                age_seconds = None
        status_icon = "🔴"
        status_text = "DOWN"
        if age_seconds is not None:
            age_text = f"{int(age_seconds)} s"
            if age_seconds <= 60:
                status_icon = "🟢"
                status_text = f"Fresh (age {age_text})"
            else:
                status_icon = "🟡"
                status_text = f"STALE (age {age_text})"
        else:
            detail = telemetry_error or firestore_error
            if detail:
                clean_detail = str(detail).splitlines()[0]
                if len(clean_detail) > 80:
                    clean_detail = clean_detail[:77] + "..."
                status_text = f"DOWN ({clean_detail})"
        telemetry_section.markdown(f"**Status:** {status_icon} {status_text}")

        if telemetry_health:
            health_ok = bool(telemetry_health.get("firestore_ok"))
            ts_val = _to_epoch_seconds(telemetry_health.get("ts") or telemetry_health.get("updated_at"))
            updated_str = human_age(ts_val) if ts_val else "unknown"
            uptime = telemetry_health.get("uptime") or telemetry_health.get("rolling_uptime_pct") or telemetry_health.get("uptime_pct")
            try:
                uptime_value = float(uptime)
                uptime_display = f"{uptime_value:.2f}%"
            except Exception:
                uptime_display = "n/a"
            telemetry_section.write(
                {
                    "firestore_ok": health_ok,
                    "uptime": uptime_display,
                    "updated": updated_str,
                    "last_error": telemetry_health.get("last_error") or "none",
                }
            )

        if fallback_entries:
            now_ts = time.time()
            fallback_rows: List[Dict[str, Any]] = []
            for entry in reversed(fallback_entries):
                entry_ts = _to_epoch_seconds(entry.get("ts") or entry.get("time"))
                age_display = "n/a"
                if entry_ts is not None:
                    try:
                        age_val = max(0.0, now_ts - float(entry_ts))
                        age_display = f"{int(age_val)} s"
                    except Exception:
                        age_display = "n/a"
                lag = entry.get("lag_secs")
                try:
                    lag_display = f"{float(lag):.1f}"
                except Exception:
                    lag_display = "n/a"
                fallback_rows.append(
                    {
                        "service": entry.get("service"),
                        "timestamp": entry.get("ts") or entry.get("time"),
                        "age": age_display,
                        "lag_s": lag_display,
                        "host": entry.get("hostname"),
                    }
                )
            telemetry_section.caption("Heartbeat fallback (sync_heartbeats.jsonl)")
            telemetry_section.table(pd.DataFrame(fallback_rows))

        if not telemetry_health and not fallback_entries:
            telemetry_section.warning("Telemetry unavailable; no Firestore data and heartbeat log empty.")

    # --------------------------- Router Health Tab --------------------------------
    with tab_router:
        st.subheader("Router Health")
        try:
            router_health = load_router_health(window=300)
        except Exception as exc:
            LOG.warning("[dash] router health load failed: %s", exc)
            st.error(f"Router health unavailable: {exc}")
            router_health = RouterHealthData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
                "count": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "cum_pnl": 0.0,
            })

        summary = router_health.summary
        metrics = st.columns(4)
        metrics[0].metric("Trades", summary.get("count", 0))
        metrics[1].metric("Win %", f"{summary.get('win_rate', 0.0):.1f}%")
        metrics[2].metric("Avg PnL", f"{summary.get('avg_pnl', 0.0):.2f} USDT")
        metrics[3].metric("Cum PnL", f"{summary.get('cum_pnl', 0.0):.2f} USDT")

        st.markdown("### PnL & Hit Rate")
        if router_health.pnl_curve.empty:
            st.info("No router executions recorded yet.")
        else:
            curve_df = router_health.pnl_curve.copy()
            curve_df["hit_pct"] = curve_df["hit_rate"] * 100.0
            if alt is not None:
                chart = (
                    alt.layer(
                        alt.Chart(curve_df)
                        .mark_line(color="#1f77b4")
                        .encode(
                            x=alt.X("time:T", title="Time"),
                            y=alt.Y("cum_pnl:Q", title="Cumulative PnL (USDT)", axis=alt.Axis(titleColor="#1f77b4")),
                        ),
                        alt.Chart(curve_df)
                        .mark_line(color="#ff7f0e")
                        .encode(
                            x=alt.X("time:T", title="Time"),
                            y=alt.Y(
                                "hit_pct:Q",
                                title="Hit Rate (%)",
                                axis=alt.Axis(titleColor="#ff7f0e", orient="right"),
                            ),
                        ),
                    ).resolve_scale(y="independent")
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                plot_df = curve_df.set_index("time")[ ["cum_pnl", "hit_pct"] ]
                st.line_chart(plot_df, use_container_width=True)

        st.markdown("### Per-Symbol Performance")
        if router_health.per_symbol.empty:
            st.info("No per-symbol performance data yet.")
        else:
            display_df = router_health.per_symbol.copy()
            display_df["win_rate"] = display_df["win_rate"].apply(lambda v: f"{v:.1f}%")
            display_df["avg_pnl"] = display_df["avg_pnl"].apply(lambda v: f"{v:.2f}")
            display_df["cum_pnl"] = display_df["cum_pnl"].apply(lambda v: f"{v:.2f}")
            display_df["sharpe"] = display_df["sharpe"].apply(lambda v: f"{v:.2f}")
            st.dataframe(display_df, use_container_width=True, height=420)

    # --------------------------- Leaderboard Tab ---------------------------------
    with tab_leader:
        st.subheader("Leaderboard")
        leader_df = pd.DataFrame(leaderboard)
        if leader_df is None or leader_df.empty:
            st.info("No leaderboard entries yet.")
        else:
            if zar_rate:
                def _extract_return_usd(row: pd.Series) -> Optional[float]:
                    for key in ("return_usd", "return", "pnl_usd", "pnl", "usd_return"):
                        if key in row:
                            val = row.get(key)
                            if isinstance(val, (int, float)):
                                return float(val)
                    return None

                def _compute_return_zar(row: pd.Series) -> str:
                    usd_val = _extract_return_usd(row)
                    if usd_val is None or not zar_rate:
                        return "≈ R—"
                    try:
                        return f"≈ {format_currency(float(usd_val) * float(zar_rate), 'R')}"
                    except Exception:
                        return "≈ R—"

                leader_df = leader_df.copy()
                leader_df["Return (ZAR)"] = leader_df.apply(_compute_return_zar, axis=1)
            st.dataframe(leader_df, use_container_width=True, height=420)

    # --------------------------- Signals Tab -------------------------------------
    with tab_signals:
        st.subheader("Signals (last 24h)")
        if df_sig.empty:
            st.info("No recent signals.")
        else:
            newest_sig = float(df_sig["ts"].max()) if "ts" in df_sig else None
            if newest_sig and not is_recent(newest_sig, 1800):
                st.warning("Signals data may be stale (>30m)")
            display_df = df_sig.copy()
            if "ts" in display_df.columns:
                display_df["Age"] = display_df["ts"].apply(
                    lambda ts: human_age(_to_epoch_seconds(ts)) if ts else "—"
                )
            if "reduceOnly" in display_df.columns:
                display_df["reduceOnly"] = display_df["reduceOnly"].apply(lambda v: "yes" if v else "no")
            if "status" in display_df.columns:
                display_df["Status"] = display_df["status"].apply(
                    lambda s: str(s).title() if s not in (None, "") else "Pending"
                )
            display_df = display_df.drop(columns=[col for col in ("ts", "status") if col in display_df.columns])
            st.dataframe(display_df, use_container_width=True, height=320)

        st.subheader("Screener Tail")
        tail_lines: List[str] = []
        if LOG_PATH.exists():
            text = tail_text(str(LOG_PATH), max_bytes=TAIL_BYTES)
            for line in text.splitlines():
                if any(tag in line for tag in WANT_TAGS):
                    tail_lines.append(line.rstrip())
        tail_lines = tail_lines[-TAIL_LINES:]
        if not tail_lines:
            st.caption("No screener breadcrumbs captured.")
        else:
            compact = rle_compact(tail_lines, min_run=3)
            st.caption(signal_attempts_summary(tail_lines))
            st.code("\n".join(compact), language="text")

    # --------------------------- ML Insights Tab --------------------------------
    with tab_ml_insights:
        st.subheader("Live ML Insights")
        ml_payload = cached_ml_predictions() or {}
        predictions = ml_payload.get("predictions") if isinstance(ml_payload, dict) else None
        if not predictions:
            st.info("No ML predictions cached yet. Ensure the ML screener is running.")
        else:
            df_preds = pd.DataFrame(predictions)
            if not df_preds.empty:
                if "score" in df_preds.columns:
                    df_preds = df_preds.sort_values("score", ascending=False)
                if "score" in df_preds.columns:
                    df_preds["score"] = df_preds["score"].apply(lambda val: f"{float(val):.3f}")
                if "updated_at" in df_preds.columns:
                    df_preds["updated_at"] = df_preds["updated_at"].apply(
                        lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                        if isinstance(ts, (int, float))
                        else ts
                    )
                st.dataframe(df_preds, use_container_width=True, height=360)
            else:
                st.info("ML predictions file exists but contains no data.")
        generated_at = ml_payload.get("generated_at") if isinstance(ml_payload, dict) else None
        if isinstance(generated_at, (int, float)):
            ts_str = datetime.fromtimestamp(generated_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"Last updated: {ts_str} UTC")

    # --------------------------- ML Models Tab ----------------------------------
    with tab_ml_models:
        st.header("ML — Models & Evaluation")
        try:
            meta_dir = Path("models")
            meta_rows = []
            if meta_dir.exists():
                for meta_path in meta_dir.glob("*_model_metadata.json"):
                    try:
                        data = json.load(open(meta_path, "r", encoding="utf-8"))
                    except Exception:
                        continue
                    if not isinstance(data, dict):
                        continue
                    meta_rows.append(
                        {
                            "symbol": data.get("symbol"),
                            "version": data.get("version"),
                            "trained_at": data.get("trained_at"),
                            "date_start": data.get("date_start"),
                            "date_end": data.get("date_end"),
                            "oos_roc_auc": data.get("oos_roc_auc"),
                            "oos_brier": data.get("oos_brier"),
                        }
                    )
            if meta_rows:
                meta_df = pd.DataFrame(meta_rows).set_index("symbol")
                st.subheader("Model Metadata")
                st.dataframe(meta_df)

            live_metrics_path = Path("logs/ml/live_metrics.json")
            if live_metrics_path.exists():
                try:
                    live_payload = json.load(open(live_metrics_path, "r", encoding="utf-8"))
                except Exception as exc:
                    st.warning(f"Failed to load live metrics: {exc}")
                    live_payload = {}
                symbols_live = (live_payload or {}).get("symbols") or {}
                if symbols_live:
                    st.subheader("Live Hit-Rate Monitor")
                    for sym, info in symbols_live.items():
                        history = info.get("history") or []
                        if history:
                            hist_df = pd.DataFrame(history)
                            try:
                                hist_df["ts"] = pd.to_datetime(hist_df["ts"], unit="s")
                            except Exception:
                                hist_df["ts"] = pd.to_datetime(hist_df["ts"])
                            hist_df = hist_df.set_index("ts")
                            st.caption(f"{sym} — rolling hit-rate vs avg prob")
                            st.line_chart(hist_df[["hit_rate", "avg_prob"]], height=140)
                        windows = info.get("windows") or {}
                        if windows:
                            st.write({sym: windows})

            reg_path = "models/registry.json"
            if os.path.exists(reg_path):
                registry = json.load(open(reg_path, "r"))
                if registry:
                    st.subheader("Registry")
                    st.dataframe(pd.DataFrame(registry).T)
                else:
                    st.info("Registry is empty. Run ML retrain to populate.")
            else:
                st.info("No registry yet. Run ML retrain to populate.")

            rpt_path = "models/last_train_report.json"
            if os.path.exists(rpt_path):
                st.subheader("Last Retrain Report")
                st.json(json.load(open(rpt_path, "r")))

            eval_path = "models/signal_eval.json"
            if os.path.exists(eval_path):
                st.subheader("Signal Evaluation (ML vs RULE)")
                report = json.load(open(eval_path, "r"))
                st.json(report.get("aggregate", {}))

                symbols = report.get("symbols") or []
                rows = []
                for entry in symbols:
                    if "error" in entry:
                        continue
                    rows.append(
                        {
                            "symbol": entry.get("symbol"),
                            "ml_f1": entry.get("ml", {}).get("f1"),
                            "ml_cov": entry.get("ml", {}).get("coverage"),
                            "ml_hit": entry.get("ml", {}).get("hit_rate"),
                            "rule_f1": entry.get("rule", {}).get("f1"),
                            "rule_cov": entry.get("rule", {}).get("coverage"),
                            "rule_hit": entry.get("rule", {}).get("hit_rate"),
                            "n": entry.get("n"),
                        }
                    )
                if rows:
                    st.dataframe(pd.DataFrame(rows))

                errors = report.get("errors") or []
                if errors:
                    st.warning(f"Evaluation skipped for {len(errors)} symbol(s):")
                    for entry in errors[:20]:
                        st.write(f"- {entry.get('symbol')}: {entry.get('error')}")
        except Exception as exc:
            st.error(f"ML tab error: {exc}")

    # --------------------------- Doctor Tab --------------------------------------
    with tab_doctor:
        st.subheader("Doctor Snapshot")

        # Run scripts/doctor.py on demand (read-only, signed request inside the script)
        run = st.button("Run doctor.py", help="Gathers hedge/one-way, crosses, gates, and blocked_by reasons.")
        doctor_data = None
        if run:
            try:
                out = subprocess.check_output(
                    ["python3", "scripts/doctor.py"], stderr=subprocess.STDOUT, timeout=20
                ).decode()
                try:
                    doctor_data = json.loads(out)
                except Exception:
                    st.code(out[:4000], language="json")
            except Exception as e:
                st.error(f"doctor.py failed: {e}")

        # Derive flags from env and doctor output (dualSide comes from doctor if available)
        def _b(name: str) -> bool:
            return str(os.getenv(name, "")).strip().lower() in ("1","true","yes","on")

        flags = {
            "use_futures": _b("USE_FUTURES"),
            "testnet": _b("BINANCE_TESTNET"),
            "dry_run": _b("DRY_RUN"),
            "dualSide": bool(((doctor_data or {}).get("env") or {}).get("dualSide", False)),
        }

        nav_age_metric_value = f"{nav_age_seconds:.1f}" if isinstance(nav_age_seconds, (int, float)) else "n/a"

        cols = st.columns(5)
        cols[0].metric("Futures", "Yes" if flags.get("use_futures") else "No")
        cols[1].metric("Testnet", "Yes" if flags.get("testnet") else "No")
        cols[2].metric("Dry-run", "Yes" if flags.get("dry_run") else "No")
        cols[3].metric("Hedge Mode", "Yes" if flags.get("dualSide") else "No")
        cols[4].metric("NAV Freshness (s)", nav_age_metric_value)

        if doctor_data:
            with st.expander("Raw doctor output", expanded=False):
                st.json(doctor_data, expanded=False)

        # Tier diagnostics and veto stats
        with st.expander("Doctor — Universe & Risk", expanded=False):
            tpath = "logs/nav_trading.json"
            rpath = "logs/nav_reporting.json"
            zpath = "logs/nav_treasury.json"
            spath = "logs/nav_snapshot.json"
            if any(os.path.exists(p) for p in (tpath, rpath, zpath, spath)):
                try:
                    if os.path.exists(tpath):
                        tnav = json.load(open(tpath, "r"))
                        tval = float(tnav.get("nav_usdt", 0.0) or 0.0)
                        st.markdown(f"**Trading NAV (USDT, used for risk):** {tval:.2f}")
                        tbr = tnav.get("breakdown", {})
                        tfw = tbr.get("futures_wallet_usdt")
                        if tfw is not None:
                            st.caption(f"Futures wallet: {float(tfw):.2f} USDT")
                        else:
                            st.caption("Futures wallet: n/a")
                    if os.path.exists(rpath):
                        rnav = json.load(open(rpath, "r"))
                        rval = float(rnav.get("nav_usdt", 0.0) or 0.0)
                        st.markdown(f"**Reporting NAV (USDT):** {rval:.2f}")
                    if os.path.exists(zpath):
                        znav = json.load(open(zpath, "r"))
                        zval = float(znav.get("treasury_usdt", 0.0) or 0.0)
                        st.markdown(
                            "**Treasury (off-exchange, excluded from NAV):** "
                            f"{zval:.2f} USDT"
                        )
                        if zar_rate and isinstance(zval, (int, float)):
                            st.markdown(
                                f"<div class='zar-note'>≈ R{zval * zar_rate:,.0f} @ R{zar_rate:.2f}/USD</div>",
                                unsafe_allow_html=True,
                            )
                        zbr = znav.get("breakdown", {})
                        tre = zbr.get("treasury", {})
                        miss = zbr.get("missing_prices", {})
                        if tre:
                            st.write("Holdings (manual-seeded):")
                            rows = []
                            for asset, data in tre.items():
                                qty = data.get("qty")
                                try:
                                    qty_val = float(qty) if qty is not None else None
                                except Exception:
                                    qty_val = None
                                val = data.get("val_usdt")
                                usd_val = None
                                try:
                                    usd_val = float(val) if val is not None else None
                                except Exception:
                                    usd_val = None
                                cg_price = price_cache.get(str(asset).upper()) if isinstance(price_cache, dict) else None
                                if usd_val is None and cg_price and qty_val is not None:
                                    usd_val = qty_val * float(cg_price)
                                zar_val = usd_val * float(zar_rate) if zar_rate and usd_val is not None else None
                                rows.append(
                                    {
                                        "Asset": asset,
                                        "Qty": qty_val if qty_val is not None else "—",
                                        "USD": format_currency(usd_val, "$") if usd_val is not None else "—",
                                        "ZAR": f"≈ {format_currency(zar_val, 'R')}" if zar_val is not None else "≈ R—",
                                    }
                                )
                            if rows:
                                df_treasury = pd.DataFrame(rows)
                                st.dataframe(df_treasury, use_container_width=True, hide_index=True)
                        if miss:
                            st.warning(
                                "Missing prices for treasury symbols (skipped): "
                                + ", ".join(sorted(miss.keys()))
                            )
                    if (not os.path.exists(tpath)) and (not os.path.exists(rpath)) and os.path.exists(spath):
                        nav = json.load(open(spath, "r"))
                        sval = float(nav.get("nav_usdt", 0.0) or 0.0)
                        st.markdown(f"**NAV (legacy single):** {sval:.2f}")
                except Exception as exc:
                    st.caption(f"nav snapshot unavailable: {exc}")
            # Tier counts from config/symbol_tiers.json
            tier_counts = {}
            try:
                tiers_cfg = load_json(os.getenv("SYMBOL_TIERS_CONFIG", "config/symbol_tiers.json"), {})
                for t, arr in (tiers_cfg.items() if isinstance(tiers_cfg, dict) else []):
                    if isinstance(arr, list):
                        tier_counts[str(t)] = len(arr)
            except Exception:
                tier_counts = {}
            if tier_counts:
                st.write({"tier_counts": tier_counts})

            # Open positions by tier
            by_tier = {}
            try:
                # Build tier map
                tmap = {s: t for t, arr in (tiers_cfg.items() if isinstance(tiers_cfg, dict) else []) for s in (arr or [])}
            except Exception:
                tmap = {}
            try:
                # positions_fs already parsed above
                for r in positions_fs:
                    sym = str(r.get("symbol") or "").upper()
                    t = tmap.get(sym, "?")
                    by_tier[t] = by_tier.get(t, 0) + 1
            except Exception:
                by_tier = {}
            if by_tier:
                st.write({"open_positions_by_tier": by_tier})

            # Veto reasons dominated in last 24h (from risk collection)
            veto_counts = {}
            try:
                db = get_firestore_connection()
                docs = list(
                    db.collection("hedge")
                    .document(ENV)
                    .collection("risk")
                    .order_by("ts", direction="DESCENDING")
                    .limit(1000)
                    .stream()
                )
                import time as _t

                now = _t.time()
                for d in docs:
                    x = d.to_dict() or {}
                    if x.get("env") is not None and str(x.get("env")) != ENV:
                        continue
                    ts = x.get("ts") or x.get("time")
                    tnum = float(ts) if isinstance(ts, (int, float)) else 0.0
                    if tnum > 1e12:
                        tnum /= 1000.0
                    if (now - tnum) > 24 * 3600:
                        continue
                    # reason (single) or reasons (list)
                    if isinstance(x.get("reasons"), list):
                        for r in x.get("reasons"):
                            veto_counts[str(r)] = veto_counts.get(str(r), 0) + 1
                    elif x.get("reason") is not None:
                        veto_counts[str(x.get("reason"))] = veto_counts.get(str(x.get("reason")), 0) + 1
            except Exception:
                veto_counts = {}
            if veto_counts:
                # Show top reasons
                top = sorted(veto_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
                st.write({"veto_top_24h": dict(top)})


    st.caption(
        "Flags: "
        f"{'FUTURES ' if flags.get('use_futures') else ''}"
        f"{'TESTNET ' if flags.get('testnet') else ''}"
        f"{'DRY_RUN ' if flags.get('dry_run') else ''}"
        f"{'HEDGE_MODE ' if flags.get('dualSide') else 'ONE-WAY'}"
    )

if __name__ == '__main__':
    main()
