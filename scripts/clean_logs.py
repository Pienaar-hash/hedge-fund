import os
import glob

def clear_backtest_logs(logs_dir="logs"):
    patterns = [
        "backtest_trades_*.csv",
        "equity_curve_*.csv",
        "momentum_grid_results.csv"
    ]
    deleted = []

    for pattern in patterns:
        for file in glob.glob(os.path.join(logs_dir, pattern)):
            try:
                os.remove(file)
                deleted.append(file)
            except Exception as e:
                print(f"❌ Failed to delete {file}: {e}")

    print("🧹 Deleted the following files:")
    for f in deleted:
        print(f"  - {f}")

if __name__ == "__main__":
    clear_backtest_logs()
