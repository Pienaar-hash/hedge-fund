# **Stage Deployment Guide — v5.10 Execution Intelligence (RC1)**

**Environment:** STAGE (Hetzner VPS)
**Components:** executor, dashboard, sync_state
**Version:** v5.10.0 → v5.10.4 (Execution Intelligence Layer)

---

# 1. **Pre-Deployment Checks**

### 1.1 Ensure the repo is clean

```bash
cd ~/hedge-fund
git status
```

**Expected:** no uncommitted changes.

### 1.2 Fetch latest main + tags

```bash
git checkout main
git pull origin main
```

### 1.3 Confirm Python environment

```bash
source venv/bin/activate
python -V
pip freeze | grep numpy
```

If numpy missing (due to execution.utils.metrics dependency):

```bash
pip install numpy
```

---

# 2. **Update Runtime Configuration**

(v5.10 introduces no new required env vars; everything remains optional and backward-compatible.)

### 2.1 Validate `.env` contains:

```
HEDGE_ENV=stage
GOOGLE_APPLICATION_CREDENTIALS=/root/hedge-fund/config/firestore_creds.json
```

### 2.2 Validate runtime.yaml (optional tuning)

```bash
cat config/runtime.yaml
```

Look for:

* maker offset bounds
* ATR lookback bars
* routing thresholds (existing)

Ensure no missing indentation or stray tabs.

---

# 3. **Install Updated Dependencies**

From the repo root:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Optional but recommended:

```bash
pip install --upgrade google-cloud-firestore
```

---

# 4. **Restart Supervisor Services (Rolling Restart)**

We restart **sync_state**, then **executor**, then **dashboard** to minimize transient inconsistencies.

### 4.1 Restart sync_state

```bash
sudo supervisorctl restart hedge:sync_state
sudo supervisorctl status hedge:sync_state
```

### 4.2 Restart executor

(*This activates router policy + adaptive offset + intel sizing immediately.*)

```bash
sudo supervisorctl restart hedge:executor
sudo supervisorctl status hedge:executor
```

### 4.3 Restart dashboard

```bash
sudo supervisorctl restart hedge:dashboard
sudo supervisorctl status hedge:dashboard
```

---

# 5. **Live Logs & Telemetry Validation**

Run these immediately after the restart.

---

## 5.1 Executor Live Log Tail

Verify that executor starts cleanly:

```bash
sudo tail -n 150 /var/log/supervisor/executor.err
```

**Expected:**

* No ImportErrors
* No missing numpy
* Router policy printed into router_ctx
* Maker offset bps logged
* Execution intel publishing (every ~5 minutes)

Look for lines like:

```
[router_policy] BTCUSDC quality=good maker_first=True bias=prefer_maker
[intel] BTCUSDC score=0.54 size_factor=1.14
[offset] BTCUSDC maker_offset_bps=1.7
```

---

## 5.2 Dashboard Log Tail

```bash
sudo tail -n 150 /var/log/supervisor/dashboard.err
```

**Expected:**

* No attribute errors
* Streamlit launched cleanly
* Execution tab loads without import errors

---

## 5.3 Check Firestore Mirrors

### Intel mirror

```bash
firestore:
  hedge/stage/execution_intel/<SYMBOL>
```

### Router metrics mirror

```bash
hedge/stage/router_metrics/<SYMBOL_TIMESTAMP>
```

### Symbol toggles

```bash
hedge/stage/symbol_toggles/<SYMBOL>
```

**Expected:** Docs update within minutes.

---

# 6. **Functional Validation (Dashboard)**

On the dashboard’s **Execution Tab**, validate:

### 6.1 Execution Intelligence

* Symbol score (float)
* Component scores (Sharpe, ATR, router, DD)
* Hourly expectancy table (non-empty)
* Maker offset bps (visible)
* Router policy (good/degraded/broken)
* Maker-first vs taker bias

### 6.2 Execution Health

* `size_multiplier` (Sharpe-based)
* `intel_size_factor` (symbol score)
* `final_size_factor` (product)
* `policy_quality`
* `policy_taker_bias`

### 6.3 Routing Behavior

Trigger 1–2 paper trades (stage/test environment) and confirm:

* Maker-first applied when router quality=good
* Router falls back to taker-only when quality=broken
* Maker offsets adjust with volatility regime

---

# 7. **Operational Safety Validations**

### 7.1 Risk Veto Smoke Tests

Run the doctor:

```bash
python3 -m scripts.doctor -v | tail -n 50
```

Check:

* NAV freshness OK
* DD guard OK
* No missing telemetry keys
* No router policy exceptions

### 7.2 Latency / Slippage Check

After a few trades, inspect:

```
logs/execution/order_metrics.jsonl
```

Expect adaptive offsets reflected in the `maker_offset_bps` field.

---

# 8. **CI Test Sweep (Optional in STAGE)**

Run selective suite:

```bash
pytest -k "intel or router_policy or maker_offset or execution_health" -q
```

Should be green.
Ignore sandbox-related Firestore/no-network warnings.

---

# 9. **Rollback Procedure**

v5.10.x is fully backward compatible with 5.9.x.

### To rollback:

1. Checkout previous tag:

```bash
git checkout v5.9.4
```

2. Reinstall deps:

```bash
pip install -r requirements.txt
```

3. Restart supervisor:

```bash
sudo supervisorctl restart hedge:executor hedge:dashboard hedge:sync_state
```

No data migrations are required.

---

# 10. **Known Issues & Notes**

* Tests require numpy; production runtime already includes it.
* Firestore writes are best-effort and may log warnings if credentials invalid.
* Router policy and maker offsets run independently of each other; disabling one doesn’t affect the other.
* Symbol intelligence requires at least a few days of router metrics to stabilize.

---

# 11. **Deployment Verdict**

**v5.10 RC1 is production-ready for STAGE**
and introduces **zero breaking changes** while delivering:

* Adaptive sizing
* Adaptive maker quoting
* Router quality classification
* Execution intelligence telemetry & dashboard visibility

If stage behaves as expected, RC2 or final will follow after monitoring.
