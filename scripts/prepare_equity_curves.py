# === scripts/prepare_equity_curves.py ===
import shutil
import os
import glob

log_dir = "logs"

# Attempt to find any existing equity curve for momentum strategy
momentum_candidates = glob.glob(os.path.join(log_dir, "equity_curve_momentum_*.csv"))
if momentum_candidates:
    latest_momentum = sorted(momentum_candidates, key=os.path.getmtime)[-1]
    dst_path = os.path.join(log_dir, "equity_curve_momentum.csv")
    shutil.copyfile(latest_momentum, dst_path)
    print(f"✅ Copied {os.path.basename(latest_momentum)} ➝ equity_curve_momentum.csv")
else:
    print("⚠️ No momentum equity curve found.")

# Ensure vol target file is aliased correctly
src_vol = os.path.join(log_dir, "equity_curve_vol_target_btcusdt.csv")
dst_vol = os.path.join(log_dir, "equity_curve_volatility_target.csv")
if os.path.exists(src_vol):
    shutil.copyfile(src_vol, dst_vol)
    print(f"✅ Copied {os.path.basename(src_vol)} ➝ equity_curve_volatility_target.csv")
else:
    print("⚠️ Source file missing: equity_curve_vol_target_btcusdt.csv")
