import json
import os
from datetime import datetime, timezone


def load_env_var(key, default=None):
    val = os.getenv(key)
    if val is None:
        print(f"⚠️ Environment variable {key} not set.")
    return val or default


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def log_trade(entry, path="logs/trade_log.json"):
    log = load_json(path)
    timestamp = datetime.now(timezone.utc).isoformat()
    log[timestamp] = entry
    save_json(path, log)


def load_local_state(path="synced_state.json"):
    """Loads local synced state from JSON."""
    return load_json(path)


def write_nav_snapshot(nav_usdt: float, breakdown: dict, path: str = "logs/nav_snapshot.json") -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "nav_usdt": float(nav_usdt),
            "breakdown": breakdown,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        save_json(path, payload)
    except Exception:
        pass


def write_nav_snapshots_pair(
    trading: tuple[float, dict], reporting: tuple[float, dict]
) -> None:
    try:
        nav_t, det_t = trading
        nav_r, det_r = reporting
        write_nav_snapshot(nav_t, det_t, "logs/nav_trading.json")
        write_nav_snapshot(nav_r, det_r, "logs/nav_reporting.json")
        # Legacy single snapshot (trading NAV)
        write_nav_snapshot(nav_t, det_t, "logs/nav_snapshot.json")
    except Exception:
        pass


def write_treasury_snapshot(
    val_usdt: float, breakdown: dict, path: str = "logs/nav_treasury.json"
) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "treasury_usdt": float(val_usdt),
            "breakdown": breakdown,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        save_json(path, payload)
    except Exception:
        pass
