# === core/backtest_runner.py ===
import importlib
import traceback
from config_utils import load_strategy_config

def run_strategy(strategy_name, params):
    try:
        print(f"\n🚀 Running Strategy: {strategy_name}")
        module = importlib.import_module(f"strategies.{strategy_name}")
        importlib.reload(module)
        StrategyImpl = getattr(module, "StrategyImpl")
        strat = StrategyImpl()
        strat.configure(params)
        strat.run()
        strat.log_results()  # 🔧 PATCHED to log results after running
        print(f"✅ {strategy_name} completed.\n")
    except Exception as e:
        print(f"❌ Failed to run strategy '{strategy_name}': {e}")
        traceback.print_exc()

if __name__ == "__main__":
    config = load_strategy_config()
    for strat in config.get("strategies", []):
        name = strat["name"]
        params = strat.get("params", {})
        print(f"🧪 Passed params for {name}:", params, flush=True)
        run_strategy(name, params)
