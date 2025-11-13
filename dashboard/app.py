# mypy: ignore-errors

import asyncio
import html
import os
import sys
import json
import time
import logging
import math
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    fetch_treasury_latest,
    parse_nav_to_df_and_kpis,
    positions_sorted,
    fetch_mark_price_usdt,
    get_env_float,
    get_env_badge,
    load_exec_snapshot,
    compute_total_nav_cached,
)
from dashboard.async_cache import gather_once  # noqa: E402
from dashboard.nav_helpers import signal_attempts_summary  # noqa: E402
from dashboard.router_health import load_router_health, RouterHealthData  # noqa: E402
from dashboard.live_helpers import get_nav_snapshot, get_treasury, execution_kpis, execution_health  # noqa: E402
from scripts.doctor import collect_doctor_snapshot, run_doctor_subprocess  # noqa: E402
from research.correlation_matrix import DEFAULT_OUTPUT_PATH as CORRELATION_CACHE_PATH  # noqa: E402
from execution.capital_allocator import DEFAULT_OUTPUT_PATH as CAPITAL_ALLOC_CACHE_PATH  # noqa: E402
from research.factor_fusion import (  # noqa: E402
    FactorFusion,
    FactorFusionConfig,
    prepare_factor_frame,
)
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
try:
    from ml.telemetry import aggregate_history as telemetry_aggregate, load_history as telemetry_load
except Exception:  # pragma: no cover
    telemetry_aggregate = None  # type: ignore[assignment]
    telemetry_load = None  # type: ignore[assignment]
try:
    from research.rl_sizer import LOG_DIR as RL_LOG_DIR
except Exception:  # pragma: no cover
    RL_LOG_DIR = Path("logs/research/rl_runs")

# Read-only exchange helpers

# Doctor helper
# removed: doctor runs via subprocess now

LOG = logging.getLogger("dash.app")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG.setLevel(logging.INFO)


def _dashboard_env() -> str:
    return os.getenv("ENV", "prod")


try:
    import inspect

    _ENV_LINES, _ENV_START = inspect.getsourcelines(_dashboard_env)
    LOG.info(
        "[runbook] dashboard ENV source: %s:%d-%d",
        __file__,
        _ENV_START,
        _ENV_START + len(_ENV_LINES) - 1,
    )
except Exception:
    pass


def main():

    st.set_page_config(page_title="Hedge — Portfolio Dashboard", layout="wide")
    ENV = _dashboard_env()
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

    def format_percent(value):
        if value is None:
            return "—"
        try:
            return f"{float(value) * 100:.1f}%"
        except Exception:
            return "—"

    def format_bps(value):
        if value is None:
            return "—"
        try:
            return f"{float(value):.2f} bps"
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
    def cached_usd_to_zar() -> Tuple[Optional[float], Optional[str], Optional[float]]:
        if get_usd_to_zar is None:
            return None, None, None
        try:
            rate, meta = get_usd_to_zar(with_meta=True)
            source = (meta or {}).get("source") if isinstance(meta, dict) else None
            age = (meta or {}).get("age") if isinstance(meta, dict) else None
            return rate, source, age  # type: ignore[arg-type]
        except Exception as exc:
            LOG.warning("[dash] usd→zar unavailable: %s", exc)
            return None, "error", None

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
    CORRELATION_JSON_PATH = PROJECT_ROOT / CORRELATION_CACHE_PATH
    CAPITAL_ALLOC_JSON_PATH = PROJECT_ROOT / CAPITAL_ALLOC_CACHE_PATH

    def _persist_payload(path: Path, payload: Dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            pass

    def _doctor_fetcher() -> Dict[str, Any]:
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
        if isinstance(snapshot, dict):
            _persist_payload(DOCTOR_CACHE_PATH, snapshot)
        return snapshot if isinstance(snapshot, dict) else {}

    def _router_fetcher() -> Dict[str, Any]:
        try:
            health = load_router_health(window=180)
        except Exception as exc:
            LOG.debug("[dash] router health fetch failed: %s", exc)
            return {}
        summary = dict(health.summary)
        per_symbol = health.per_symbol.head(25).to_dict(orient="records")
        pnl_curve = health.pnl_curve.tail(120).to_dict(orient="records")
        return {
            "summary": summary,
            "per_symbol": per_symbol,
            "pnl_curve": pnl_curve,
        }

    def _telemetry_fetcher(limit: int = 240) -> Dict[str, Any]:
        if telemetry_load is None:
            return {}
        try:
            history = telemetry_load(limit=limit)
        except Exception as exc:
            LOG.debug("[dash] telemetry load failed: %s", exc)
            return {}
        mapped = [point.to_mapping() for point in history]
        aggregate = telemetry_aggregate(history) if telemetry_aggregate else {}
        return {"history": mapped, "aggregate": aggregate}

    def _correlation_fetcher() -> Dict[str, Any]:
        if CORRELATION_JSON_PATH.exists():
            try:
                with CORRELATION_JSON_PATH.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:
                LOG.debug("[dash] correlation cache read failed: %s", exc)
        return {}

    def _capital_allocation_fetcher() -> Dict[str, Any]:
        if CAPITAL_ALLOC_JSON_PATH.exists():
            try:
                with CAPITAL_ALLOC_JSON_PATH.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:
                LOG.debug("[dash] capital allocation cache read failed: %s", exc)
        return {}

    ASYNC_FETCHERS: Dict[str, Callable[[], Any]] = {
        "doctor": _doctor_fetcher,
        "router_health": _router_fetcher,
        "telemetry": _telemetry_fetcher,
        "correlation": _correlation_fetcher,
        "capital_allocation": _capital_allocation_fetcher,
    }

    @st.cache_data(ttl=20, show_spinner=False)
    def load_async_entries() -> Dict[str, Dict[str, Any]]:
        async def _runner() -> Dict[str, Any]:
            return await gather_once(ASYNC_FETCHERS)

        def _run_in_thread() -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            error: Optional[BaseException] = None

            def _target() -> None:
                nonlocal result, error
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(_runner())
                except Exception as err:  # pragma: no cover - defensive
                    error = err
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()

            thread = threading.Thread(target=_target, name="dash-async-cache", daemon=True)
            thread.start()
            thread.join()
            if error is not None:
                raise error
            return result

        try:
            entries = asyncio.run(_runner())
        except RuntimeError as exc:
            if "asyncio.run()" in str(exc) and "event loop" in str(exc).lower():
                LOG.warning("[dash] falling back to threaded async cache gather: %s", exc)
                entries = _run_in_thread()
            else:
                raise
        except Exception as exc:
            LOG.warning("[dash] async cache gather failed: %s", exc, exc_info=True)
            entries = {}
        return {
            name: {
                "payload": entry.payload,
                "latency_ms": entry.latency_ms,
                "ok": entry.ok,
                "refreshed_at": entry.refreshed_at,
            }
            for name, entry in entries.items()
        }

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

    def _maybe_float(value: Any) -> Optional[float]:
        try:
            if value in (None, "", "null"):
                return None
            return float(value)
        except Exception:
            return None

    def _to_optional_float(value: Any) -> Optional[float]:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(num):
            return None
        return num

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

    nav_value: Optional[float] = None

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

    async_entries = load_async_entries()
    doctor_snapshot = async_entries.get("doctor", {}).get("payload", {}) or {}
    telemetry_async = async_entries.get("telemetry", {})
    correlation_async = async_entries.get("correlation", {})
    capital_async = async_entries.get("capital_allocation", {})
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

    router_snapshot = load_exec_snapshot("router", ENV)
    trades_snapshot = load_exec_snapshot("trades", ENV)
    signals_snapshot = load_exec_snapshot("signals", ENV)
    nav_snapshot_live = get_nav_snapshot()
    treasury_snapshot_live = get_treasury()

    # --- ZAR guard: declare upfront so early reads never crash ---
    zar_rate: Optional[float] = None
    zar_source: Optional[str] = None
    nav_zar_value: Optional[float] = None

    treasury_latest_doc = fetch_treasury_latest(ENV)
    if not isinstance(treasury_latest_doc, dict):
        treasury_latest_doc = {}
    treasury_payload = treasury_latest_doc.get("treasury") if isinstance(treasury_latest_doc, dict) else {}
    if not isinstance(treasury_payload, dict) or not treasury_payload:
        treasury_payload = treasury_snapshot_live if isinstance(treasury_snapshot_live, dict) else {}

    treasury_assets_raw = treasury_payload.get("assets") if isinstance(treasury_payload.get("assets"), list) else []
    treasury_assets_rows: List[Dict[str, Any]] = []
    for entry in treasury_assets_raw:
        if not isinstance(entry, dict):
            continue
        asset_name = str(
            entry.get("asset")
            or entry.get("Asset")
            or entry.get("symbol")
            or entry.get("code")
            or ""
        ).upper()
        if not asset_name:
            continue
        balance_val = _maybe_float(
            entry.get("balance")
            or entry.get("qty")
            or entry.get("Units")
            or entry.get("units")
            or entry.get("amount")
        )
        price_val = _maybe_float(entry.get("price_usdt") or entry.get("price") or entry.get("px"))
        usd_val = _maybe_float(
            entry.get("usd_value")
            or entry.get("USD Value")
            or entry.get("value_usd")
            or entry.get("val_usdt")
        )
        if usd_val is None and balance_val is not None and price_val is not None:
            usd_val = balance_val * price_val
        treasury_assets_rows.append(
            {
                "Asset": asset_name,
                "Balance": balance_val,
                "Price (USDT)": price_val,
                "USD Value": usd_val,
            }
        )

    def _extract_total_usd(payload: Dict[str, Any]) -> Optional[float]:
        if not isinstance(payload, dict):
            return None
        total = payload.get("total_usd") or payload.get("nav") or payload.get("treasury_usdt")
        try:
            if isinstance(total, (int, float)):
                return float(total)
            if isinstance(total, str) and total.strip():
                return float(total)
        except Exception:
            return None
        return None

    treasury_total_usd: Optional[float] = _extract_total_usd(treasury_payload)
    if treasury_total_usd is None:
        treasury_total_usd = _extract_total_usd(treasury_snapshot_live)

    treasury_updated_at = treasury_latest_doc.get("updated_at") or treasury_payload.get("updated_at")
    treasury_updated_ts = _to_epoch_seconds(treasury_updated_at)
    treasury_age_seconds: Optional[float] = None
    if treasury_updated_ts is not None:
        try:
            treasury_age_seconds = max(0.0, time.time() - float(treasury_updated_ts))
        except Exception:
            treasury_age_seconds = None
    treasury_source = treasury_latest_doc.get("source") or treasury_payload.get("source") or "firestore"

    nav_trading_usd = _to_optional_float(
        nav_snapshot_live.get("nav")
        or nav_snapshot_live.get("equity")
        or (nav_doc or {}).get("nav")
        or (nav_doc or {}).get("equity")
    )
    if nav_trading_usd is None and nav_value is not None:
        nav_trading_usd = _to_optional_float(nav_value)

    total_equity_usd: Optional[float] = None

    total_equity_zar: Optional[float] = None

    status.success(f"Loaded · nav_source={nav_source} · positions_source={pos_source}")
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
    reserves_total_usd: Optional[float] = None

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

    if reserves_total_usd is None:
        candidate_total = reserves_snapshot.get("total")
        if isinstance(candidate_total, (int, float)):
            reserves_total_usd = float(candidate_total)

    nav_value = exchange_nav_value
    if nav_value is None and nav_trading_usd is not None:
        nav_value = float(nav_trading_usd)

    if total_equity_usd is None and any(
        val is not None for val in (nav_trading_usd, treasury_total_usd, reserves_total_usd)
    ):
        total_equity_usd = (
            float(nav_trading_usd or 0.0)
            + float(treasury_total_usd or 0.0)
            + float(reserves_total_usd or 0.0)
        )

    aum_total_usd = total_equity_usd
    if isinstance(aum_total_usd, (int, float)):
        kpis["total_equity"] = aum_total_usd

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
    zar_rate = None
    zar_source: Optional[str] = None
    zar_age_seconds: Optional[float] = None
    zar_status = nav_freshness.get("zar_status")
    # Resolve zar_rate late, but variables already exist so upstream guards won't crash
    if nav_freshness.get("zar_rate") is not None:
        try:
            zar_rate = float(nav_freshness.get("zar_rate") or 0.0)
        except Exception:
            zar_rate = None
        zar_source = nav_freshness.get("zar_source") or "fresh"
        zar_age_candidate = nav_freshness.get("zar_age_seconds")
        if isinstance(zar_age_candidate, (int, float)):
            zar_age_seconds = float(zar_age_candidate)
    if zar_rate is None:
        cached_rate, cached_source, cached_age = cached_usd_to_zar()
        if isinstance(cached_rate, (int, float)):
            zar_rate = float(cached_rate)
            zar_source = cached_source or "cache"
            zar_age_seconds = float(cached_age) if isinstance(cached_age, (int, float)) else None
            if zar_status is None or str(zar_status).upper() == "MISSING":
                if zar_source == "cache" or (zar_age_seconds is not None and zar_age_seconds > 6 * 3600):
                    zar_status = "STALE"
                else:
                    zar_status = "FRESH"
    if zar_rate is not None and isinstance(nav_value, (int, float)):
        nav_zar_value = float(nav_value) * float(zar_rate)
        nav_zar_text = f"≈ {format_currency(nav_zar_value, 'R')}"
        if zar_status and str(zar_status).upper() == "STALE":
            nav_zar_text = f"{nav_zar_text} · STALE"

    if total_equity_usd is not None and zar_rate is not None:
        try:
            total_equity_zar = float(total_equity_usd) * float(zar_rate)
        except Exception:
            total_equity_zar = None

    prev_nav_zar = None
    if zar_rate is not None and isinstance(prev_nav_value, (int, float)):
        prev_nav_zar = float(prev_nav_value) * float(zar_rate)

    nav_delta_html = compare_nav_zar(prev_nav_zar, nav_zar_value)

    nav_display = f"{nav_value:,.2f} USD" if isinstance(nav_value, (int, float)) else "n/a"

    def _tooltip_span(label_html: str, tooltip_text: Optional[str], classes: str = "") -> str:
        tooltip_text = tooltip_text or ""
        safe_tooltip = html.escape(tooltip_text, quote=True)
        base_classes = (classes or "").strip()
        class_attr = f"{base_classes} dash-tooltip".strip() if base_classes else "dash-tooltip"
        return f"<span class='{class_attr}' data-tooltip=\"{safe_tooltip}\">{label_html}</span>"

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
    .dash-tooltip {
        position: relative;
        cursor: help;
    }
    .dash-tooltip::after {
        content: attr(data-tooltip);
        position: absolute;
        left: 50%;
        top: calc(100% + 8px);
        transform: translateX(-50%);
        background: rgba(15, 23, 42, 0.92);
        color: #f8fafc;
        padding: 0.45rem 0.6rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 500;
        line-height: 1.2;
        white-space: pre-wrap;
        max-width: 260px;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.15s ease-in-out;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.2);
        z-index: 1200;
    }
    .dash-tooltip::before {
        content: "";
        position: absolute;
        left: 50%;
        top: calc(100% + 2px);
        transform: translateX(-50%);
        border-width: 6px;
        border-style: solid;
        border-color: rgba(15, 23, 42, 0.92) transparent transparent transparent;
        opacity: 0;
        transition: opacity 0.15s ease-in-out;
    }
    .dash-tooltip:hover::after,
    .dash-tooltip:hover::before {
        opacity: 1;
    }
    .dash-tooltip-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.15rem;
        height: 1.15rem;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.2);
        color: rgba(30, 41, 59, 0.85);
        font-size: 0.75rem;
        margin-left: 0.35rem;
    }
    </style>
    """

    nav_zar_tooltip_parts: List[str] = []
    if zar_rate is not None:
        source_label = f" ({zar_source})" if zar_source else ""
        nav_zar_tooltip_parts.append(f"Rate: R{zar_rate:.2f}/USD{source_label}")
    else:
        nav_zar_tooltip_parts.append("Rate unavailable")
    if zar_age_seconds is not None:
        nav_zar_tooltip_parts.append(f"Age: {format_age_seconds(zar_age_seconds)}")
    if zar_status:
        nav_zar_tooltip_parts.append(f"Status: {zar_status}")
    if nav_zar_value is not None:
        nav_zar_tooltip_parts.append(f"NAV ≈ {format_currency(nav_zar_value, 'R')}")
    if total_equity_zar is not None:
        nav_zar_tooltip_parts.append(f"Total equity ≈ {format_currency(total_equity_zar, 'R')}")
    nav_zar_tooltip = "\n".join(nav_zar_tooltip_parts)
    nav_zar_html = _tooltip_span(nav_zar_text, nav_zar_tooltip, "dash-nav-zar")

    nav_main_label = f"⚡ NAV {nav_usd_text} | {nav_zar_html} | {env_label}"
    rate_display = f"R{zar_rate:.2f}/USD" if zar_rate is not None else "R—/USD"

    drawdown_snapshot = nav_freshness.get("drawdown", {})
    heartbeat_snapshot = doctor_snapshot.get("heartbeats", {})

    nav_age_display = format_age_seconds(nav_age_seconds)
    nav_is_fresh = bool(nav_freshness.get("fresh"))
    nav_chip_class = "fresh" if nav_is_fresh else "stale"
    nav_chip_label = "Fresh" if nav_is_fresh else "STALE"
    nav_chip_label = f"{nav_chip_label} · {nav_age_display}"
    nav_chip_tooltip_parts = [f"Status: {'Fresh' if nav_is_fresh else 'Stale'}", f"Age {nav_age_display}"]
    threshold_val = nav_freshness.get("threshold")
    if isinstance(threshold_val, (int, float)):
        nav_chip_tooltip_parts.append(f"Fresh ≤ {format_age_seconds(threshold_val)}")
    nav_chip_tooltip = "\n".join(nav_chip_tooltip_parts)
    nav_chip_html = _tooltip_span(nav_chip_label, nav_chip_tooltip, f"dash-nav-chip {nav_chip_class}")

    dd_pct = drawdown_snapshot.get("dd_pct") or 0.0
    dd_abs = drawdown_snapshot.get("dd_abs") or 0.0
    dd_chip_class = "dd-ok" if not dd_pct or dd_pct <= 0 else "dd-warn"
    dd_chip_label = f"Drawdown · {dd_pct:.2f}%"
    dd_chip_title = f"Daily drawdown {dd_pct:.2f}%"
    dd_chip_html = _tooltip_span(dd_chip_label, dd_chip_title, f"dash-nav-chip {dd_chip_class}")

    treasury_display = format_currency(treasury_total_usd, "$") if treasury_total_usd is not None else "—"
    reserves_display = format_currency(reserves_total_usd, "$") if reserves_total_usd is not None else "—"
    nav_tooltip_text = (
        "Trading NAV (USD)\n"
        f"Exchange: {exchange_nav_display} (source {nav_info.get('source', 'n/a')})\n"
        f"Treasury: {treasury_display}\n"
        f"Reserves: {reserves_display}"
        if nav_display != "n/a"
        else "Trading NAV (USD)"
    )
    nav_span = _tooltip_span(nav_main_label, nav_tooltip_text, "dash-nav-main")
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

    # ---- AUM breakdown ----------------------------------------------------------
    aum_rows = []

    def _add_segment(label: str, value: Optional[float]) -> None:
        if value is None:
            return
        try:
            numeric = float(value)
        except Exception:
            return
        if numeric <= 0:
            return
        aum_rows.append({"Segment": label, "USD": numeric})

    _add_segment("Trading NAV", nav_value if isinstance(nav_value, (int, float)) else nav_trading_usd)
    _add_segment("Treasury", treasury_total_usd)
    _add_segment("Reserves", reserves_total_usd)

    try:
        cached_sources, _ = compute_total_nav_cached()
    except Exception:
        cached_sources = {}
    for key, label in (("spot", "Spot"), ("poly", "Polymarket")):
        nav_entry = cached_sources.get(key) if isinstance(cached_sources, dict) else {}
        nav_val = nav_entry.get("nav") if isinstance(nav_entry, dict) else None
        _add_segment(label, nav_val)
    if aum_rows:
        with st.container():
            st.markdown("#### AUM Breakdown")
            if alt is not None:
                df_aum = pd.DataFrame(aum_rows)
                segment_order = df_aum["Segment"].tolist()
                palette = {
                    "Trading NAV": "#2563eb",
                    "Treasury": "#f97316",
                    "Reserves": "#14b8a6",
                    "Spot": "#06b6d4",
                    "Polymarket": "#a855f7",
                }
                color_scale = alt.Scale(
                    domain=segment_order,
                    range=[palette.get(seg, "#94a3b8") for seg in segment_order],
                )
                chart = (
                    alt.Chart(df_aum)
                    .mark_arc(innerRadius=60, stroke="white")
                    .encode(
                        theta=alt.Theta(field="USD", type="quantitative"),
                        color=alt.Color(
                            field="Segment",
                            type="nominal",
                            legend=alt.Legend(title=None, orient="right"),
                            scale=color_scale,
                        ),
                        tooltip=[
                            alt.Tooltip("Segment:N"),
                            alt.Tooltip("USD:Q", format=",.2f"),
                        ],
                    )
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(pd.DataFrame(aum_rows), hide_index=True, use_container_width=True)

    # ---- Read-only Reserve KPI ---------------------------------------------------
    RESERVE_BTC = get_env_float("DASHBOARD_RESERVE_BTC", 0.13)
    price_cache = cached_coingecko_prices()
    btc_price = float(price_cache.get("BTC") or 0.0)
    if not btc_price:
        btc_price = fetch_mark_price_usdt("BTCUSDT")
    reserve_usdt = RESERVE_BTC * btc_price if btc_price and btc_price > 0 else 0.0
    reserve_zar = reserve_usdt * float(zar_rate) if zar_rate is not None and reserve_usdt else None

    # ---- Tabs layout --------------------------------------------------------------
    (
        tab_overview,
        tab_execution,
        tab_router,
        tab_signals,
        tab_ml_insights,
        tab_ml_models,
        tab_ml_confidence,
        tab_rl_pilot,
        tab_doctor,
        tab_portfolio_corr,
        tab_factor_fusion,
    ) = st.tabs(
        [
            "Overview",
            "Execution",
            "Router Health",
            "Signals",
            "ML Insights",
            "ML Models",
            "ML Confidence",
            "RL Pilot",
            "Doctor",
            "Portfolio Correlation",
            "Factor Fusion",
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

        doc_row2 = st.columns(5)
        doc_row2[0].metric("Positions", positions_info.get("count", 0), None)
        doc_row2[2].metric(
            "ZAR Rate",
            f"{zar_rate:.2f}" if zar_rate is not None else "n/a",
            zar_source if zar_rate is not None else "none",
        )
        doc_row2[3].metric(
            "Reserves",
            treasury_display,
            None,
        )
        total_equity_display = format_currency(total_equity_usd, "$") if total_equity_usd is not None else "—"
        total_equity_delta = (
            f"≈ R{total_equity_zar:,.0f} @ {zar_rate:.2f}/USD"
            if total_equity_zar is not None and zar_rate is not None
            else None
        )
        doc_row2[4].metric(
            "Total Equity (USD)",
            total_equity_display,
            total_equity_delta,
        )

        st.markdown("#### Treasury (Firestore)")
        treasury_cols = st.columns([1, 2])
        treasury_delta = format_age_seconds(treasury_age_seconds) if treasury_age_seconds is not None else "n/a"
        treasury_cols[0].metric(
            "Treasury (USDT)",
            format_currency(treasury_total_usd, "$") if treasury_total_usd is not None else "—",
            f"Age {treasury_delta}" if treasury_delta not in {"n/a", "—"} else None,
        )
        if treasury_assets_rows:
            df_treasury = pd.DataFrame(treasury_assets_rows)
            # format values for readability
            display_df = df_treasury.copy()
            display_df["Balance"] = display_df["Balance"].map(lambda v: f"{v:.6f}".rstrip("0").rstrip(".") if isinstance(v, (int, float)) else "—")
            display_df["Price (USDT)"] = display_df["Price (USDT)"].map(
                lambda v: f"{v:,.2f}" if isinstance(v, (int, float)) else "—"
            )
            display_df["USD Value"] = display_df["USD Value"].map(
                lambda v: format_currency(v, "$") if isinstance(v, (int, float)) else "—"
            )
            treasury_cols[1].dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            treasury_cols[1].info("No treasury assets available.")
        treasury_caption_parts = []
        if treasury_updated_at:
            treasury_caption_parts.append(f"updated {treasury_updated_at}")
        if treasury_source:
            treasury_caption_parts.append(f"source {treasury_source}")
        if treasury_caption_parts:
            st.caption(" · ".join(treasury_caption_parts))

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

        if zar_rate is not None and nav_zar_value:
            st.markdown(
                f"<div class='zar-note'>{format_currency(nav_value, '$')} ≈ {format_currency(nav_zar_value, 'R')} @ R{zar_rate:.2f}/USD</div>",
                unsafe_allow_html=True,
            )
        if zar_rate is not None and isinstance(reserve_zar, (int, float)) and reserve_zar:
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

            exec_kpi_data = execution_kpis()
            maker_col, fallback_col, q25_col, q50_col, q75_col = st.columns(5)
            maker_col.metric("Maker Fill %", format_percent(exec_kpi_data.get("maker_fill_ratio")))
            fallback_col.metric("Fallback Rate %", format_percent(exec_kpi_data.get("fallback_ratio")))
            q25_col.metric("Slippage Q25", format_bps(exec_kpi_data.get("slip_q25")))
            q50_col.metric("Slippage Q50", format_bps(exec_kpi_data.get("slip_q50")))
            q75_col.metric("Slippage Q75", format_bps(exec_kpi_data.get("slip_q75")))

            st.markdown("#### Execution Health Overview")
            health_snapshot = execution_health(None)
            router = health_snapshot.get("router") or {}
            risk = health_snapshot.get("risk") or {}
            vol = health_snapshot.get("vol") or {}
            sizing = health_snapshot.get("sizing") or {}

            r1c1, r1c2, r1c3 = st.columns(3)
            r1c1.metric("Maker Fill % (7d)", format_percent(router.get("maker_fill_ratio")))
            r1c2.metric("Fallback Rate % (7d)", format_percent(router.get("fallback_ratio")))
            r1c3.metric("Median Slippage (bps)", format_bps(router.get("slip_q50")))

            r2c1, r2c2, r2c3 = st.columns(3)
            sharpe_val = sizing.get("sharpe_7d")
            r2c1.metric(
                "Sharpe (7d)",
                "—" if sharpe_val is None else f"{sharpe_val:.2f}",
                help=f"State: {risk.get('sharpe_state', 'unknown')}",
            )
            dd_today = risk.get("dd_today_pct")
            r2c2.metric("DD Today (%)", "—" if dd_today is None else f"{dd_today:.2f}")
            r2c3.metric("ATR Regime", (vol.get("atr_regime") or "unknown").capitalize())

            r3c1, r3c2, r3c3 = st.columns(3)
            r3c1.metric("Sharpe Size Mult", f"{(sizing.get('size_mult_sharpe') or 1.0):.2f}x")
            r3c2.metric("Regime Size Mult", f"{(sizing.get('size_mult_regime') or 1.0):.2f}x")
            r3c3.metric("Combined Size Mult", f"{(sizing.get('size_mult_combined') or 1.0):.2f}x")

            warnings_lines = []
            router_warnings = router.get("router_warnings") or []
            risk_flags = risk.get("risk_flags") or []
            if router_warnings:
                warnings_lines.append(f"Router: {', '.join(router_warnings)}")
            if risk_flags:
                warnings_lines.append(f"Risk: {', '.join(risk_flags)}")
            if warnings_lines:
                st.warning(" | ".join(warnings_lines))

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

        st.markdown("#### Open Positions")
        positions_payload = positions_fs or []
        positions_df = pd.DataFrame(positions_payload)
        if positions_df.empty:
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
                c
                for c in [
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

        if not latency_summary:
            st.caption(
                "Latency metrics unavailable. Generate a cache with "
                "`python scripts/replay_logs.py --since <iso> --json logs/execution/replay_cache.json`."
            )

        st.markdown("#### Router Snapshot (mirror)")
        router_snapshot_items = router_snapshot.get("items") if isinstance(router_snapshot, dict) else []
        router_items = [
            dict(item)
            for item in router_snapshot_items
            if isinstance(item, dict) and item.get("kind") != "summary"
        ]
        if router_items:
            st.dataframe(pd.DataFrame(router_items).head(50), use_container_width=True, height=220)
        else:
            st.info("No router executions recorded yet.")

        st.markdown("#### Recent Trades (mirror)")
        trade_snapshot_items = []
        if isinstance(trades_snapshot, dict):
            items = trades_snapshot.get("items")
            if isinstance(items, list):
                trade_snapshot_items = items
        trade_items = [dict(item) for item in trade_snapshot_items if isinstance(item, dict)]
        if trade_items:
            st.dataframe(pd.DataFrame(trade_items).head(50), use_container_width=True, height=220)
        else:
            st.info("No mirrored trades available.")

        st.markdown("#### Latest Signals (mirror)")
        signal_snapshot_items = signals_snapshot.get("items") if isinstance(signals_snapshot, dict) else []
        signal_items = [dict(item) for item in signal_snapshot_items if isinstance(item, dict)]
        if signal_items:
            st.dataframe(pd.DataFrame(signal_items).head(50), use_container_width=True, height=220)
        else:
            st.info("No mirrored signals available.")

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
        clean_detail: Optional[str] = None
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
        telemetry_status_lines: List[str] = []
        if age_seconds is not None:
            telemetry_status_lines.append(f"Age: {int(age_seconds)} s")
        if telemetry_health:
            firestore_ok = telemetry_health.get("firestore_ok")
            firestore_label = "yes" if firestore_ok else "no"
            telemetry_status_lines.append(f"Firestore ok: {firestore_label}")
        if clean_detail:
            telemetry_status_lines.append(f"Issue: {clean_detail}")
        elif telemetry_error:
            detail_line = str(telemetry_error).splitlines()[0]
            if len(detail_line) > 80:
                detail_line = detail_line[:77] + "..."
            telemetry_status_lines.append(f"Issue: {detail_line}")
        if not telemetry_status_lines:
            telemetry_status_lines.append("No additional details")
        status_label_html = f"<strong>Status:</strong> {status_icon} {status_text}"
        telemetry_status_html = _tooltip_span(status_label_html, "\n".join(telemetry_status_lines), "dash-telemetry-status")
        telemetry_section.markdown(telemetry_status_html, unsafe_allow_html=True)

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
            router_health = load_router_health(
                window=300,
                snapshot=router_snapshot,
                trades_snapshot=trades_snapshot,
            )
        except Exception as exc:
            LOG.warning("[dash] router health load failed: %s", exc)
            st.error(f"Router health unavailable: {exc}")
            router_health = RouterHealthData(
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(columns=["time", "cum_pnl", "hit_rate", "confidence_weighted_cum_pnl", "rolling_sharpe"]),
                {
                    "count": 0,
                    "win_rate": 0.0,
                    "avg_pnl": 0.0,
                    "cum_pnl": 0.0,
                    "fill_rate_pct": 0.0,
                    "fees_total": 0.0,
                    "realized_pnl": 0.0,
                    "avg_confidence": None,
                    "confidence_weighted_cum_pnl": 0.0,
                    "normalized_sharpe": 0.0,
                    "volatility_scale": 1.0,
                    "rolling_sharpe_last": 0.0,
                },
                {},
            )

        summary = router_health.summary
        metrics = st.columns(6)
        metrics[0].metric("Trades", summary.get("count", 0))
        metrics[1].metric("Win %", f"{summary.get('win_rate', 0.0):.1f}%")
        metrics[2].metric("Avg PnL", f"{summary.get('avg_pnl', 0.0):.2f} USDT")
        metrics[3].metric("Cum PnL", f"{summary.get('cum_pnl', 0.0):.2f} USDT")
        metrics[4].metric("Conf-Weighted PnL", f"{summary.get('confidence_weighted_cum_pnl', 0.0):.2f} USDT")
        metrics[5].metric("Rolling Sharpe", f"{summary.get('rolling_sharpe_last', 0.0):.2f}")

        detail_metrics = st.columns(3)
        avg_conf = summary.get("avg_confidence")
        avg_conf_display = f"{float(avg_conf) * 100.0:.1f}%" if isinstance(avg_conf, (int, float)) else "n/a"
        detail_metrics[0].metric("Avg Confidence", avg_conf_display)
        fill_rate = summary.get("fill_rate_pct")
        fill_display = f"{float(fill_rate):.1f}%" if isinstance(fill_rate, (int, float)) else "n/a"
        detail_metrics[1].metric("Fill Rate", fill_display)
        vol_scale = summary.get("volatility_scale")
        vol_display = f"{float(vol_scale):.2f}×" if isinstance(vol_scale, (int, float)) else "n/a"
        detail_metrics[2].metric("Volatility Scale", vol_display)

        st.markdown("### PnL & Confidence-Weighted Curve")
        if router_health.pnl_curve.empty:
            st.info("No router executions recorded yet.")
        else:
            curve_df = router_health.pnl_curve.copy()
            if "confidence_weighted_cum_pnl" not in curve_df:
                curve_df["confidence_weighted_cum_pnl"] = curve_df["cum_pnl"]
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
                        .mark_line(color="#2ca02c")
                        .encode(
                            x=alt.X("time:T", title="Time"),
                            y=alt.Y(
                                "confidence_weighted_cum_pnl:Q",
                                title="Conf-Weighted PnL (USDT)",
                                axis=alt.Axis(titleColor="#2ca02c"),
                            ),
                        ),
                    )
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                plot_df = curve_df.set_index("time")[["cum_pnl", "confidence_weighted_cum_pnl"]]
                st.line_chart(plot_df, use_container_width=True)

        st.markdown("### Rolling Sharpe & Confidence")
        if router_health.pnl_curve.empty:
            st.info("Rolling Sharpe unavailable; collect more executions.")
        else:
            rolling_df = router_health.pnl_curve[["time", "rolling_sharpe", "hit_rate"]].copy()
            rolling_df["hit_pct"] = rolling_df["hit_rate"] * 100.0
            confidence_overlay = (router_health.overlays or {}).get("confidence") if router_health.overlays else None
            if confidence_overlay is not None and not confidence_overlay.empty:
                conf_df = confidence_overlay.copy()
                conf_df["confidence_pct"] = conf_df["rolling_confidence"] * 100.0
                rolling_df = pd.merge(rolling_df, conf_df[["time", "confidence_pct"]], on="time", how="left")
            if "confidence_pct" not in rolling_df.columns:
                rolling_df["confidence_pct"] = float("nan")
            if alt is not None:
                chart = (
                    alt.layer(
                        alt.Chart(rolling_df)
                        .mark_line(color="#d62728")
                        .encode(
                            x=alt.X("time:T", title="Time"),
                            y=alt.Y("rolling_sharpe:Q", title="Rolling Sharpe", axis=alt.Axis(titleColor="#d62728")),
                        ),
                        alt.Chart(rolling_df)
                        .mark_line(color="#2ca02c")
                        .encode(
                            x=alt.X("time:T", title="Time"),
                            y=alt.Y(
                                "confidence_pct:Q",
                                title="Confidence (%)",
                                axis=alt.Axis(titleColor="#2ca02c", orient="right"),
                            ),
                        ),
                        alt.Chart(rolling_df)
                        .mark_line(color="#ff7f0e", strokeDash=[4, 4])
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
                plot_df = rolling_df.set_index("time")[["rolling_sharpe", "confidence_pct", "hit_pct"]]
                st.line_chart(plot_df, use_container_width=True)

        st.markdown("### Per-Symbol Performance")
        if router_health.per_symbol.empty:
            st.info("No per-symbol performance data yet.")
        else:
            display_df = router_health.per_symbol.copy()
            if "win_rate" in display_df.columns:
                display_df["win_rate"] = display_df["win_rate"].apply(lambda v: f"{v:.1f}%")
            for col in ("avg_pnl", "cum_pnl", "confidence_weighted_pnl", "fees_total", "realized_pnl"):
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda v: f"{float(v):.2f}")
            if "sharpe" in display_df.columns:
                display_df["sharpe"] = display_df["sharpe"].apply(lambda v: f"{float(v):.2f}")
            if "normalized_sharpe" in display_df.columns:
                display_df["normalized_sharpe"] = display_df["normalized_sharpe"].apply(lambda v: f"{float(v):.2f}")
            if "volatility_scale" in display_df.columns:
                display_df["volatility_scale"] = display_df["volatility_scale"].apply(lambda v: f"{float(v):.2f}×")
            if "avg_confidence" in display_df.columns:
                display_df["avg_confidence"] = display_df["avg_confidence"].apply(
                    lambda v: f"{float(v) * 100.0:.1f}%" if pd.notna(v) else "n/a"
                )
            if "fill_rate_pct" in display_df.columns:
                display_df["fill_rate_pct"] = display_df["fill_rate_pct"].apply(
                    lambda v: f"{float(v):.1f}%" if v is not None else "n/a"
                )
            st.dataframe(display_df, use_container_width=True, height=420)
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

    # --------------------------- ML Confidence Tab ----------------------------
    with tab_ml_confidence:
        st.subheader("Model Confidence Telemetry")
        if telemetry_load is None or telemetry_aggregate is None:
            st.warning("Telemetry module unavailable; ensure ml.telemetry is installed.")
        else:
            telemetry_history = telemetry_load(limit=200)
            if not telemetry_history:
                st.info("No telemetry points recorded yet. Call ml.telemetry.record_confidence from your models.")
            else:
                aggregate = telemetry_aggregate(telemetry_history)
                metrics_cols = st.columns(3)
                avg_conf = aggregate.get("avg_confidence")
                latest_conf = aggregate.get("latest_confidence")
                metrics_cols[0].metric(
                    "Average Confidence",
                    f"{avg_conf * 100.0:.1f}%" if isinstance(avg_conf, (int, float)) else "n/a",
                )
                metrics_cols[1].metric(
                    "Latest Confidence",
                    f"{latest_conf * 100.0:.1f}%" if isinstance(latest_conf, (int, float)) else "n/a",
                )
                metrics_cols[2].metric("Samples", aggregate.get("count", 0))

                telemetry_df = pd.DataFrame(
                    {
                        "time": [point.ts for point in telemetry_history],
                        "confidence": [point.confidence for point in telemetry_history],
                        "model": [point.model for point in telemetry_history],
                    }
                ).sort_values("time")
                telemetry_df["rolling_confidence"] = telemetry_df["confidence"].rolling(window=20, min_periods=1).mean()
                telemetry_df["time"] = pd.to_datetime(telemetry_df["time"])

                st.markdown("### Confidence Trend")
                if alt is not None:
                    chart = (
                        alt.layer(
                            alt.Chart(telemetry_df)
                            .mark_line(color="#1f77b4")
                            .encode(
                                x=alt.X("time:T", title="Time"),
                                y=alt.Y("confidence:Q", title="Confidence", axis=alt.Axis(format=".2%")),
                                color="model:N",
                            ),
                            alt.Chart(telemetry_df)
                            .mark_line(color="#ff7f0e", strokeDash=[4, 4])
                            .encode(
                                x=alt.X("time:T", title="Time"),
                                y=alt.Y(
                                    "rolling_confidence:Q",
                                    title="Rolling Confidence",
                                    axis=alt.Axis(format=".2%", titleColor="#ff7f0e"),
                                ),
                            ),
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    plot_df = telemetry_df.set_index("time")[["confidence", "rolling_confidence"]]
                    st.line_chart(plot_df, use_container_width=True)

                top_features = aggregate.get("top_features") or []
                if top_features:
                    st.markdown("### Top Feature Attributions (abs sum)")
                    feat_df = pd.DataFrame(top_features, columns=["feature", "importance"])
                    feat_df["importance"] = feat_df["importance"].apply(lambda v: f"{float(v):.3f}")
                    st.table(feat_df)

                latest_point = telemetry_history[-1]
                if latest_point.metadata:
                    st.caption("Latest metadata payload")
                    st.json(latest_point.metadata)

    # --------------------------- RL Pilot Tab ---------------------------------
    with tab_rl_pilot:
        st.subheader("RL Sizer Pilot")
        rl_log_path = RL_LOG_DIR / "episodes.jsonl"
        if not rl_log_path.exists():
            st.info("No RL pilot runs logged yet. Execute `python -m research.rl_sizer.runner` to generate episodes.")
        else:
            try:
                with rl_log_path.open("r", encoding="utf-8") as handle:
                    lines = [line.strip() for line in handle if line.strip()]
                episodes = [json.loads(line) for line in lines[-200:]]
            except Exception as exc:
                st.error(f"Failed to load RL pilot log: {exc}")
                episodes = []

            if not episodes:
                st.info("RL pilot log is empty. Run the dry-run trainer to populate metrics.")
            else:
                df_runs = pd.DataFrame(episodes)
                df_runs["episode"] = range(len(df_runs))
                df_runs = df_runs.sort_values("episode")
                metrics_cols = st.columns(4)
                metrics_cols[0].metric("Episodes", len(df_runs))
                metrics_cols[1].metric(
                    "Avg Normalized Sharpe",
                    f"{df_runs['normalized_sharpe'].mean():.2f}"
                    if "normalized_sharpe" in df_runs
                    else "n/a",
                )
                metrics_cols[2].metric(
                    "Best Reward",
                    f"{df_runs['total_reward'].max():.3f}"
                    if "total_reward" in df_runs
                    else "n/a",
                )
                metrics_cols[3].metric(
                    "Final Equity (median)",
                    f"{df_runs['equity_final'].median():.3f}"
                    if "equity_final" in df_runs
                    else "n/a",
                )

                st.markdown("### Episode Trajectories")
                df_runs["episode_index"] = df_runs.index
                if alt is not None and "normalized_sharpe" in df_runs:
                    chart = (
                        alt.Chart(df_runs)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("episode_index:Q", title="Episode"),
                            y=alt.Y("normalized_sharpe:Q", title="Normalized Sharpe"),
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)
                elif "normalized_sharpe" in df_runs:
                    st.line_chart(df_runs.set_index("episode_index")["normalized_sharpe"], use_container_width=True)

                st.markdown("### Recent Episodes")
                display_cols = [
                    col
                    for col in ["episode_index", "normalized_sharpe", "total_reward", "equity_final", "avg_position"]
                    if col in df_runs.columns
                ]
                if display_cols:
                    st.dataframe(df_runs[display_cols].tail(50), use_container_width=True, height=260)
                else:
                    st.json(episodes[-5:])

    # --------------------------- Doctor Tab --------------------------------------
    with tab_doctor:
        st.subheader("Doctor Snapshot")

        # Run scripts/doctor.py on demand (read-only, signed request inside the script)
        run = st.button("Run doctor.py", help="Gathers hedge/one-way, crosses, gates, and blocked_by reasons.")
        doctor_data = None
        if run:
            stream_placeholder = st.empty()
            try:
                output, error_msg = run_doctor_subprocess(timeout=30, placeholder=stream_placeholder)
                if error_msg:
                    st.warning(error_msg)
                try:
                    doctor_data = json.loads(output or "{}")
                except Exception:
                    st.code((output or "")[:4000], language="json")
            except TimeoutError as exc:
                st.error(f"doctor.py timed out: {exc}")
            except Exception as exc:
                st.error(f"doctor.py failed: {exc}")

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
            zpath = "logs/treasury.json"
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
                        if zar_rate is not None and isinstance(zval, (int, float)):
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
                                zar_val = usd_val * float(zar_rate) if zar_rate is not None and usd_val is not None else None
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


    # --------------------------- Portfolio Correlation Tab ----------------------
    with tab_portfolio_corr:
        st.subheader("Portfolio Correlation")
        corr_payload = correlation_async.get("payload") if isinstance(correlation_async, dict) else {}
        capital_payload = capital_async.get("payload") if isinstance(capital_async, dict) else {}
        if not corr_payload:
            st.info("No correlation snapshot available. Run `python -m research.correlation_matrix` to populate the cache.")
        else:
            asof = corr_payload.get("asof", "n/a")
            window = corr_payload.get("window")
            method = corr_payload.get("method")
            refreshed = _to_optional_float(correlation_async.get("refreshed_at")) if isinstance(correlation_async, dict) else None
            caption = f"Asof: {asof} · Window: {window} · Method: {method}"
            if refreshed is not None:
                caption += f" · Refreshed: {datetime.fromtimestamp(refreshed).isoformat()}"
            st.caption(caption)
            metrics_cols = st.columns(3)
            avg_corr = _to_optional_float(corr_payload.get("average_abs_correlation"))
            max_corr = _to_optional_float(corr_payload.get("max_correlation"))
            min_corr = _to_optional_float(corr_payload.get("min_correlation"))
            metrics_cols[0].metric("Avg |ρ|", f"{avg_corr:.3f}" if avg_corr is not None else "n/a")
            metrics_cols[1].metric("Max ρ", f"{max_corr:.3f}" if max_corr is not None else "n/a")
            metrics_cols[2].metric("Min ρ", f"{min_corr:.3f}" if min_corr is not None else "n/a")

            matrix_payload = corr_payload.get("matrix") or {}
            matrix_df = pd.DataFrame.from_dict(matrix_payload, orient="index")
            if not matrix_df.empty:
                matrix_df = matrix_df.reindex(sorted(matrix_df.index)).reindex(sorted(matrix_df.columns), axis=1)
                st.markdown("### Correlation Matrix")
                st.dataframe(matrix_df.round(2), use_container_width=True, height=360)
                if alt is not None:
                    melted = (
                        matrix_df.reset_index(names="base")
                        .melt(id_vars="base", var_name="paired", value_name="correlation")
                    )
                    heatmap = (
                        alt.Chart(melted)
                        .mark_rect()
                        .encode(
                            x=alt.X("paired:O", title="Strategy"),
                            y=alt.Y("base:O", title="Strategy"),
                            color=alt.Color("correlation:Q", scale=alt.Scale(scheme="redblue", domain=(-1, 1))),
                            tooltip=["base", "paired", alt.Tooltip("correlation:Q", format=".3f")],
                        )
                    )
                    st.altair_chart(heatmap, use_container_width=True)
            else:
                st.info("Correlation matrix empty — waiting for sufficient history.")

        if capital_payload:
            st.markdown("### Dynamic Capital Allocation")
            weights = capital_payload.get("weights") or {}
            if weights:
                weight_df = (
                    pd.DataFrame({"strategy": list(weights.keys()), "weight": [float(v) for v in weights.values()]})
                    .sort_values("weight", ascending=False, ignore_index=True)
                )
                st.bar_chart(weight_df.set_index("strategy"), use_container_width=True)
                st.dataframe(weight_df, use_container_width=True, height=220)
            scores = capital_payload.get("scores") or {}
            if scores:
                score_df = (
                    pd.DataFrame({"strategy": list(scores.keys()), "score": [float(v) for v in scores.values()]})
                    .sort_values("score", ascending=False, ignore_index=True)
                )
                st.markdown("#### Allocation Scores")
                st.dataframe(score_df, use_container_width=True, height=220)
            metadata = capital_payload.get("metadata") or {}
            if metadata:
                avg_meta = _to_optional_float(metadata.get("average_abs_correlation"))
                avg_meta_text = f"{avg_meta:.3f}" if avg_meta is not None else "n/a"
                st.caption(
                    f"Allocator metadata: avg |ρ|={avg_meta_text} · strategies={metadata.get('strategies', 'n/a')}"
                )

    # --------------------------- Factor Fusion Tab ------------------------------
    with tab_factor_fusion:
        st.subheader("Factor Fusion Diagnostics")
        telemetry_payload = telemetry_async.get("payload") if isinstance(telemetry_async, dict) else {}
        history_records = telemetry_payload.get("history") or []
        if not history_records:
            st.info("No telemetry records available. Factor fusion requires ML telemetry history.")
        elif len(history_records) < 20:
            st.info("Need at least 20 telemetry observations to fit the fusion layer.")
        else:
            features_list: List[Dict[str, Any]] = []
            confidences: List[float] = []
            timestamps: List[Optional[pd.Timestamp]] = []
            targets: List[float] = []
            prices: List[Optional[float]] = []
            volumes: List[Optional[float]] = []
            for record in history_records:
                features = record.get("features") or {}
                features_list.append({k: _safe_float(v) for k, v in features.items() if isinstance(v, (int, float))})
                conf_val = _safe_float(record.get("confidence"))
                confidences.append(conf_val if conf_val is not None else 0.0)
                timestamps.append(
                    pd.to_datetime(record.get("ts"), errors="coerce") if record.get("ts") else pd.NaT
                )
                price_candidate = record.get("price") or record.get("mark_price") or record.get("mark")
                prices.append(_safe_float(price_candidate))
                volumes.append(_safe_float(record.get("volume")))
                target_val = (
                    _safe_float(record.get("realized_pnl"))
                    or _safe_float(record.get("future_return"))
                    or _safe_float(record.get("pnl"))
                )
                if target_val is None:
                    target_val = (conf_val if conf_val is not None else 0.0) - 0.5
                targets.append(float(target_val))

            feature_frame = pd.DataFrame(features_list).fillna(0.0)
            feature_frame["ml_confidence"] = confidences
            index = pd.Index(timestamps)
            if index.isna().all():
                index = pd.RangeIndex(len(feature_frame))
            feature_frame.index = index

            price_series = pd.Series(prices, index=index, dtype="float64")
            volume_series = pd.Series(volumes, index=index, dtype="float64")
            if price_series.notna().sum() >= 10:
                ta_features = prepare_factor_frame(
                    price_series.fillna(method="ffill").fillna(method="bfill"),
                    ml_signal=feature_frame.get("ml_confidence"),
                    volume=volume_series,
                )
                feature_frame = feature_frame.join(ta_features, how="left").fillna(0.0)

            numeric_frame = feature_frame.select_dtypes(include=["number"]).fillna(0.0)
            target_series = pd.Series(targets, index=index, dtype="float64").fillna(0.0)
            if numeric_frame.empty:
                st.warning("Telemetry features did not contain numeric values to blend.")
            else:
                fusion = FactorFusion(FactorFusionConfig(regularization=1e-2, positive_weights=False, clip_output=4.0))
                try:
                    fusion_result = fusion.fit(numeric_frame, target_series)
                except ValueError as exc:
                    st.warning(f"Unable to fit fusion layer: {exc}")
                else:
                    metrics_cols = st.columns(3)
                    metrics_cols[0].metric("Information Coefficient", f"{fusion_result.ic:.3f}")
                    metrics_cols[1].metric("Signal Sharpe", f"{fusion_result.signal_sharpe:.2f}")
                    metrics_cols[2].metric("R²", f"{fusion_result.r_squared:.2f}")

                    fusion_df = pd.DataFrame(
                        {
                            "Fused Alpha": fusion_result.fused_signal,
                            "Target": target_series.reindex(fusion_result.fused_signal.index),
                        }
                    ).dropna().tail(200)
                    if not fusion_df.empty:
                        if alt is not None:
                            plot_df = fusion_df.reset_index(names="time").melt(
                                id_vars="time", var_name="series", value_name="value"
                            )
                            chart = (
                                alt.Chart(plot_df)
                                .mark_line()
                                .encode(
                                    x=alt.X("time:T", title="Time"),
                                    y=alt.Y("value:Q", title="Value"),
                                    color=alt.Color("series:N", title="Series"),
                                )
                            )
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.line_chart(fusion_df, use_container_width=True)

                    st.markdown("### Factor Weights")
                    weight_df = fusion_result.weights.reset_index()
                    weight_df.columns = ["factor", "weight"]
                    weight_df.sort_values("weight", ascending=False, inplace=True, ignore_index=True)
                    st.dataframe(weight_df, use_container_width=True, height=260)

                    aggregate = telemetry_payload.get("aggregate") or {}
                    if aggregate:
                        st.caption(
                            f"Telemetry points={aggregate.get('count', 'n/a')} · Avg confidence={aggregate.get('avg_confidence', 'n/a'):.3f}"
                            if isinstance(aggregate.get("avg_confidence"), (int, float))
                            else f"Telemetry points={aggregate.get('count', 'n/a')}"
                        )

    st.caption(
        "Flags: "
        f"{'FUTURES ' if flags.get('use_futures') else ''}"
        f"{'TESTNET ' if flags.get('testnet') else ''}"
        f"{'DRY_RUN ' if flags.get('dry_run') else ''}"
        f"{'HEDGE_MODE ' if flags.get('dualSide') else 'ONE-WAY'}"
    )

if __name__ == '__main__':
    main()
