import os
import json
from datetime import datetime

def load_env_var(key, default=None):
    val = os.getenv(key)
    if val is None:
        print(f"⚠️ Environment variable {key} not set.")
    return val or default

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def log_trade(entry, path="logs/trade_log.json"):
    log = load_json(path)
    timestamp = datetime.utcnow().isoformat()
    log[timestamp] = entry
    save_json(path, log)
