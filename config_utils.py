import json
from core.duckdb_query import get_top_momentum_symbols

CONFIG_PATH = "config/strategy_config.json"

def load_strategy_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_strategy_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

def update_strategy_config_with_top_momentum():
    config = load_strategy_config()
    top_symbols = get_top_momentum_symbols()

    if "momentum" in config:
        config["momentum"]["params"]["top_symbols"] = top_symbols
        print(f"✅ Injected top 3 momentum symbols based on sharpe: {top_symbols}")
    else:
        print("⚠️ 'momentum' strategy not found in config")

    save_strategy_config(config)

def normalize_strategy_config(config):
    for strat in config["strategies"]:
        if strat["name"] == "momentum":
            symbols = strat["params"].get("symbols", [])
            strat["params"]["symbols"] = [s.replace("_1h", "").upper() for s in symbols]
        elif strat["name"] == "volatility_target":
            for asset in strat["params"].get("assets", []):
                asset["symbol"] = asset["symbol"].upper()
                asset["tf"] = asset["tf"].upper()
        elif strat["name"] == "relative_value":
            strat["params"]["base"] = strat["params"]["base"].upper()
            strat["params"]["pairs"] = [p.upper() for p in strat["params"].get("pairs", [])]
