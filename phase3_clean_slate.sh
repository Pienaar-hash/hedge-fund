#!/bin/bash
set -e

echo "🚦 Phase 3 Clean Slate + Repo Refresh — Starting"

# 1️⃣ Stop all hedge services
echo "🛑 Stopping hedge services..."
sudo supervisorctl stop hedge-executor hedge-sync hedge-dashboard || true

# 2️⃣ Remove Python caches & redundant files from repo
echo "🧹 Removing __pycache__, .pyc, .pyo, .tmp..."
cd /root/hedge-fund
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.tmp" \) -delete

# 3️⃣ Clear old hedge logs
echo "🗑 Clearing /var/log/hedge logs..."
sudo rm -f /var/log/hedge/*.log

# 4️⃣ Pull latest from GitHub (without overwriting local config/state)
echo "📥 Pulling latest repo changes..."
git reset --hard
git pull origin sprint-phase2

# 5️⃣ Restart services clean
echo "🚀 Restarting hedge services..."
sudo supervisorctl start hedge-executor hedge-sync hedge-dashboard

# 6️⃣ Show status
echo "📋 Supervisor status:"
sudo supervisorctl status

echo "✅ Phase 3 Clean Slate + Repo Refresh complete!"

