# === core/backtest_runner.py ===
import importlib
import json

def load_config(path="config/strategy_config.json"):
    with open(path, "r") as f:
        return json.load(f)

def run_strategy(strategy_name, params=None):
    try:
        print(f"\n🚀 Running strategy: {strategy_name}")
        print(f"🔧 Params:\n{json.dumps(params or {}, indent=2)}")

        mod = importlib.import_module(f"strategies.{strategy_name}")
        print("📂 Loaded module from:", mod.__file__)
        cls = getattr(mod, "StrategyImpl")
        strategy = cls()

        strategy.configure(params or {})
        strategy.run()
        strategy.log_results()
        print(f"✅ Completed strategy: {strategy_name}")

    except Exception as e:
        print(f"❌ Failed to run strategy '{strategy_name}': {e}")

if __name__ == "__main__":
    config = load_config()
    for strat in config["strategies"]:
        run_strategy(strat["name"], strat.get("params", {}))
