# **prod_deployment_v5.10.md**

### **Production Deployment Guide — v5.10 Execution Intelligence (Final Release)**

**Environment:** PROD (live trading)
**Components:** executor, dashboard, sync_state
**Version:** v5.10.0 → v5.10.4

---

# 1. **Pre-Deployment Requirements**

### 1.1 Confirm Stage Stability

Before PROD deployment:

* ≥72 hours of Stage runtime
* No maker/taker routing anomalies
* Reasonable maker offsets (0.5–8.0 bps)
* Router policy oscillation minimal
* Health + Intel + Router tabs load without errors
* Firestore mirrors stable (execution_intel, router_metrics, symbol_toggles)

**Gate:** No critical errors in:

```
/var/log/supervisor/executor.err
/var/log/supervisor/dashboard.err
```

---

# 2. **Prepare Production Machine**

### 2.1 Connect to PROD

```bash
ssh root@<prod_ip>
cd ~/hedge-fund
```

### 2.2 Ensure repo clean

```bash
git status
```

**Must be clean** before pulling.

### 2.3 Checkout main and update

```bash
git checkout main
git pull origin main
```

### 2.4 Activate virtual environment

```bash
source venv/bin/activate
```

### 2.5 Install required dependencies

v5.10 requires `numpy` for metrics intelligence.

```bash
pip install -r requirements.txt
pip install numpy
```

### 2.6 Validate Python

```bash
python3 -V
```

3.10+ is fine.

---

# 3. **Update .env for PROD**

Ensure `.env` has:

```
HEDGE_ENV=prod
GOOGLE_APPLICATION_CREDENTIALS=/root/hedge-fund/config/firestore_creds.json
```

Optional tuning:

```
EXEC_INTEL_PUBLISH_INTERVAL_S=300
EXEC_HEALTH_PUBLISH_INTERVAL_S=120
```

Reload:

```bash
source .env
```

---

# 4. **Runtime Config Validation**

### 4.1 Inspect runtime.yaml

```bash
cat config/runtime.yaml
```

Confirm:

* maker offset bounds present
* ATR lookbacks correct
* No stray YAML indentation

### 4.2 Validate Firestore creds exist

```bash
ls -l config/firestore_creds.json
```

---

# 5. **Safety Snapshot Before Deployment**

### 5.1 Run doctor with verbose mode

```bash
python3 -m scripts.doctor -v | tail -n 50
```

Confirm:

* NAV freshness OK
* Drawdown < thresholds
* Exchange connectivity OK
* Firestore connectivity OK

### 5.2 Snapshot open positions (for rollback validation)

```bash
python3 scripts/doctor.py | grep Position
```

Record these before restart.

### 5.3 Snapshot live bot state

```bash
python3 scripts/doctor.py | tail -n 20
```

---

# 6. **Supervisor Rolling Deploy**

We restart in safe order: **sync_state → executor → dashboard**.

### 6.1 Restart sync_state

```bash
sudo supervisorctl restart hedge:sync_state
sudo supervisorctl status hedge:sync_state
```

### 6.2 Restart executor

```bash
sudo supervisorctl restart hedge:executor
sudo supervisorctl status hedge:executor
```

**This activates:**

* adaptive sizing
* adaptive maker offsets
* router policy engine
* execution intel publishers

### 6.3 Restart dashboard

```bash
sudo supervisorctl restart hedge:dashboard
sudo supervisorctl status hedge:dashboard
```

---

# 7. **Post-Deployment Validation (Live)**

## 7.1 Executor Log Tail

```bash
sudo tail -n 200 /var/log/supervisor/executor.err
```

**Expect:**

* `[router_policy]` lines
* `[maker_offset]` lines
* `[intel] score=… size_factor=…`
* No exceptions or import errors

If missing logs appear:

* ensure DEBUG level enabled
* check `.env` for `LOG_LEVEL`

---

## 7.2 Dashboard Validation (Critical)

Open PROD dashboard → Execution Tab:

### Validate:

#### **Execution Intelligence**

* Symbol score present
* ATR regime
* Router quality + maker-first status
* Maker offset bps
* Hourly expectancy table

#### **Execution Health**

* `size_multiplier` (old)
* `intel_size_factor`
* `final_size_factor`
* Router policy (quality, taker bias)

#### **Routing Path Behavior**

* Good router = maker-first enabled
* Broken router = taker-only
* Hot ATR regime = offsets widened

---

## 7.3 Firestore Integrity

Check Firestore collections:

```
hedge/prod/execution_health/<symbol_timestamp>
hedge/prod/execution_intel/<symbol>
hedge/prod/router_metrics/<symbol_timestamp>
hedge/prod/symbol_toggles/<symbol>
```

**Validation checklist:**

* New docs written within 2–5 minutes
* Correct `env=prod`
* Offset + router policy fields present
* Intel payload (score, components, expectancy) present

---

# 8. **Live Routing Validation**

Trigger a controlled micro-trade (0.001–0.01 size):

Use CLI or controlled signal:

### 8.1 Confirm maker-first behavior

If router quality = “good”:

* First attempt: maker
* Second attempt: tighter or fallback depending on offset
* If fallback: router logs must show fallback ratio increment

### 8.2 Confirm taker-only

If router quality = “broken”:

* Should skip maker-first entirely
* Direct taker submission
* No maker quote attempts

### 8.3 Confirm offset behavior

* Quiet regime → <2 bps
* Hot regime → >2 bps
* Panic regime → ≥3 bps

---

# 9. **Operational Risk Checks**

### 9.1 Re-run doctor after 10 minutes

```bash
python3 -m scripts.doctor -v | tail -n 40
```

Check:

* DD stable
* NAV delta reasonable
* Router effectiveness reflecting new trades

### 9.2 Tail router metrics log

```bash
tail -n 50 logs/execution/order_metrics.jsonl
```

Look for:

```
"maker_offset_bps": x.x  
"policy_quality": …
"slippage_bps": …
```

### 9.3 Ensure no symbol toggles triggered unexpectedly

```bash
firestore: hedge/prod/symbol_toggles
```

---

# 10. **Rollback Plan (Fast)**

v5.10.x is fully backward-compatible; rollback is straightforward.

### 10.1 Git rollback

```bash
git fetch
git checkout v5.9.4
pip install -r requirements.txt
```

### 10.2 Supervisor rollback

```bash
sudo supervisorctl restart hedge:executor hedge:dashboard hedge:sync_state
```

### 10.3 Confirm positions safe

Run:

```bash
python3 scripts/doctor.py | tail -n 20
```

---

# 11. **Expected Live Effects After Deployment**

* Maker offsets tighten or widen with volatility + slippage
* Fallback ratios should **decline** for good symbols
* Size adjusts slightly up/down by intel factor
* Router bias auto-switches (prefer_maker ↔ prefer_taker)
* Better consistency in realized slippage
* More stable fill behavior in volatile regimes
* Execution health + intel snapshots available in Firestore

---

# 12. **Deployment Success Criteria**

Deployment is considered **successful** if, within 2–6 hours:

### ✔ Executor runs without exceptions

### ✔ Dashboard renders Execution Intelligence fully

### ✔ Firestore receives new intel & health snapshots

### ✔ Maker offsets align with ATR regime

### ✔ Router policy toggles correctly by symbol

### ✔ Slippage and fallback ratios remain stable or improved

If any anomaly appears, fallback is immediate (Section 10).

---

# **Final Note**

**v5.10 brings the first true “intelligence layer” to routing and sizing.**
Safe, bounded, transparent — and fully observable.
This guide prepares PROD to run it with minimal disruption and maximum control.