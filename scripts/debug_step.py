import os
import re

pattern = re.compile(r"backtest_ETH_BNB_1d_w\d+_ze[\d\.]+_zx[\d\.]+\.csv")
log_folder = "logs"

print("🔍 Matching files:")
for file in os.listdir(log_folder):
    if pattern.match(file):
        print(f"✅ {file}")

print("\n🚫 Non-matching files:")
for file in os.listdir(log_folder):
    if not pattern.match(file):
        print(f"❌ {file}")

