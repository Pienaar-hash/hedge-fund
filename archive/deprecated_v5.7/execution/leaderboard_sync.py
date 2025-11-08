# execution/leaderboard_sync.py
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from firebase_admin import credentials, firestore, initialize_app

from execution.config_loader import get, load

STATE_FILE = "synced_state.json"
PEAK_FILE = "logs/cache/peak_state.json"
LOG_DIR = Path("logs")
EXEC_LOG_DIR = LOG_DIR / "execution"
LOCAL_METRICS_PATH = LOG_DIR / "leaderboard_metrics.json"
MAX_LOG_LINES = int(os.getenv("LEADERBOARD_LOG_LINES", "5000"))
WINDOW_SECONDS = int(os.getenv("LEADERBOARD_WINDOW_SECONDS", str(24 * 3600)))


def _load_json(path: str) -> Any:
    if not Path(path).exists():
        return {} if path.endswith(".json") else []
    with open(path, "r") as f:
        return json.load(f)


def _pct(n, d):
    try:
        return float(n) / float(d) if float(d) != 0 else 0.0
    except:  # noqa: E722
        return 0.0


def _coerce_ts(record: Dict[str, Any]) -> Optional[float]:
    for key in ("ts", "timestamp", "time", "t", "local_ts"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                continue
            try:
                return float(trimmed)
            except ValueError:
                try:
                    cleaned = trimmed.replace("Z", "+00:00") if trimmed.endswith("Z") else trimmed
                    dt = datetime.fromisoformat(cleaned)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.timestamp()
                except Exception:
                    continue
    return None


def _extract_strategy(record: Dict[str, Any]) -> str:
    for key in ("strategy", "strategy_name", "strategyId", "strategy_id", "source"):
        val = record.get(key)
        if val:
            return str(val)
    payload = record.get("intent") or record.get("nav_snapshot") or record.get("veto_detail") or {}
    if isinstance(payload, dict):
        for key in ("strategy", "strategy_name", "strategyId"):
            val = payload.get(key)
            if val:
                return str(val)
    return "unknown"


def _extract_symbol(record: Dict[str, Any]) -> str:
    val = record.get("symbol") or record.get("pair")
    if val:
        return str(val).upper()
    payload = record.get("intent") or record.get("veto_detail") or {}
    if isinstance(payload, dict):
        sym = payload.get("symbol")
        if sym:
            return str(sym).upper()
    return "UNKNOWN"


def _extract_request_id(record: Dict[str, Any]) -> Optional[str]:
    for key in ("request_id", "client_order_id", "clientOrderId"):
        value = record.get(key)
        if value:
            return str(value)
    payload = record.get("intent") or record.get("veto_detail") or record.get("normalized") or {}
    if isinstance(payload, dict):
        for key in ("request_id", "client_order_id", "clientOrderId"):
            value = payload.get(key)
            if value:
                return str(value)
    return None


def _read_recent_records(path: Path, max_lines: int) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return []
    except Exception:
        return []

    records: List[Dict[str, Any]] = []
    for line in reversed(lines[-max_lines:]):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return list(reversed(records))


def _find_strategy_for_execution(
    record: Dict[str, Any],
    attempts_by_request: Dict[str, str],
    attempts_by_symbol_bucket: Dict[Tuple[str, int], List[Tuple[float, str]]],
) -> Optional[str]:
    request_id = _extract_request_id(record)
    if request_id and request_id in attempts_by_request:
        return attempts_by_request[request_id]

    symbol = _extract_symbol(record)
    ts = _coerce_ts(record)
    if ts is None:
        return None
    bucket = int(ts // 60)
    for offset in (0, -1, 1):
        key = (symbol, bucket + offset)
        candidates = attempts_by_symbol_bucket.get(key)
        if not candidates:
            continue
        best = min(candidates, key=lambda item: abs(item[0] - ts))
        return best[1]
    return None


def _aggregate_exec_metrics() -> Dict[str, Any]:
    cutoff = time.time() - WINDOW_SECONDS
    attempts = _read_recent_records(EXEC_LOG_DIR / "orders_attempted.jsonl", MAX_LOG_LINES)
    executed = _read_recent_records(EXEC_LOG_DIR / "orders_executed.jsonl", MAX_LOG_LINES)

    attempts_counter: Dict[str, int] = defaultdict(int)
    executed_counter: Dict[str, int] = defaultdict(int)
    fills_counter: Dict[str, int] = defaultdict(int)
    attempts_by_request: Dict[str, str] = {}
    attempts_by_symbol_bucket: Dict[Tuple[str, int], List[Tuple[float, str]]] = defaultdict(list)

    for record in attempts:
        ts = _coerce_ts(record)
        if ts is None or ts < cutoff:
            continue
        strategy = _extract_strategy(record)
        symbol = _extract_symbol(record)
        bucket = int(ts // 60)
        attempts_counter[strategy] += 1
        req = _extract_request_id(record)
        if req:
            attempts_by_request[req] = strategy
        attempts_by_symbol_bucket[(symbol, bucket)].append((ts, strategy))

    for record in executed:
        ts = _coerce_ts(record)
        if ts is None or ts < cutoff:
            continue
        strategy = _find_strategy_for_execution(record, attempts_by_request, attempts_by_symbol_bucket)
        if strategy is None:
            continue
        executed_counter[strategy] += 1
        status = str(record.get("status") or record.get("order_status") or "").upper()
        if status == "FILLED":
            fills_counter[strategy] += 1

    rows: List[Dict[str, Any]] = []
    total_attempts = 0
    total_executed = 0
    total_fills = 0

    strategies = set(attempts_counter.keys()) | set(executed_counter.keys())
    for strategy in sorted(strategies, key=lambda s: -attempts_counter.get(s, 0)):
        att = attempts_counter.get(strategy, 0)
        exe = executed_counter.get(strategy, 0)
        fills = fills_counter.get(strategy, 0)
        fill_rate = fills / att if att else 0.0
        rows.append(
            {
                "strategy": strategy,
                "attempted_24h": att,
                "executed_24h": exe,
                "fills_24h": fills,
                "fill_rate": fill_rate,
            }
        )
        total_attempts += att
        total_executed += exe
        total_fills += fills

    overall_fill = total_fills / total_attempts if total_attempts else 0.0

    return {
        "rows": rows,
        "summary": {
            "attempted_24h": total_attempts,
            "executed_24h": total_executed,
            "fills_24h": total_fills,
            "fill_rate": overall_fill,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _publish_leaderboard_metrics(db, payload: Dict[str, Any]) -> None:
    payload_with_env = dict(payload)
    payload_with_env["env"] = get("env", "prod")
    if db is not None:
        try:
            db.collection("metrics").document("leaderboard").set(payload_with_env, merge=True)
        except Exception as exc:
            print(f"⚠️ Failed to publish leaderboard metrics to Firestore: {exc}")
    try:
        LOCAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_METRICS_PATH.write_text(json.dumps(payload_with_env, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"⚠️ Failed to write local leaderboard metrics: {exc}")


def compute_assets(state: dict) -> List[dict]:
    rows = []
    for sym, pos in state.items():
        qty = float(pos.get("qty") or 0.0)
        entry = float(pos.get("entry") or 0.0)
        last = float(pos.get("latest_price") or 0.0)
        pnl = (last - entry) * qty
        ret = _pct(last - entry, entry)
        rows.append(
            {
                "symbol": sym,
                "qty": qty,
                "entry": entry,
                "last": last,
                "unrealized": pnl,
                "return_pct": ret,
            }
        )
    rows.sort(key=lambda r: r["unrealized"], reverse=True)
    return rows


def compute_strategies(state: dict, peak: dict) -> List[dict]:
    # current value per strategy (sum MTM of owned symbols)
    values = {}
    for sym, pos in state.items():
        strat = pos.get("strategy") or "unknown"
        qty = float(pos.get("qty") or 0.0)
        last = float(pos.get("latest_price") or 0.0)
        values[strat] = values.get(strat, 0.0) + qty * last

    peaks = peak.get("strategies", {}) if isinstance(peak, dict) else {}
    rows = []
    for skey, val in values.items():
        pk = float(peaks.get(skey, 0.0))
        dd = 0.0 if pk == 0 else (val - pk) / pk
        rows.append(
            {
                "strategy": skey,
                "current_value": val,
                "peak_value": pk,
                "drawdown_pct": dd,
            }
        )
    rows.sort(key=lambda r: r["current_value"], reverse=True)
    return rows


def init_db():
    load()
    cred_path = get("runtime.FIREBASE_CREDS_PATH", "config/firebase_creds.json")
    if not Path(cred_path).exists():
        return None
    try:
        cred = credentials.Certificate(cred_path)
        initialize_app(cred)
        return firestore.client()
    except Exception:
        return None


def sync_once():
    if not get("execution.leaderboard_enabled", True):
        print("ℹ️ Leaderboard disabled.")
        return False

    state = _load_json(STATE_FILE)
    peak = _load_json(PEAK_FILE)
    assets = compute_assets(state)
    strats = compute_strategies(state, peak)
    updated = datetime.now(timezone.utc).isoformat()
    metrics_payload = _aggregate_exec_metrics()
    metrics_payload["updated_at"] = updated

    db = init_db()
    if db is None:
        print("ℹ️ No Firestore creds — skipping remote write (local only).")
        _publish_leaderboard_metrics(None, metrics_payload)
        return False

    try:
        db.collection("hedge_leaderboard").document("assets").set(
            {"rows": assets, "updated_at": updated}
        )
        db.collection("hedge_leaderboard").document("strategies").set(
            {"rows": strats, "updated_at": updated}
        )
        _publish_leaderboard_metrics(db, metrics_payload)
        print(f"✅ Leaderboard synced at {updated}")
        return True
    except Exception as e:
        print(f"❌ Leaderboard sync failed: {e}")
        _publish_leaderboard_metrics(None, metrics_payload)
        return False


if __name__ == "__main__":
    while True:
        sync_once()
        time.sleep(300)
