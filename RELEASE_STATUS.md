# RELEASE STATUS — GPT-Hedge v7.9

**Generated:** 2026-02-26  
**Version:** v7.9  
**Branch:** v7.6-dev / main (synced at `5aefee47`)  
**HEAD:** `5aefee47` — fix: fund-ops structural reconciliation

---

## System State

| Component | Status |
|---|---|
| Executor | RUNNING (pid 42129) |
| Dashboard | RUNNING (pid 41908) |
| Sync State | RUNNING (pid 41909) |
| CLOB Client | RUNNING (pid 41910) |
| Round Observer | RUNNING (pid 41913) |

---

## Certification Window

| Field | Value |
|---|---|
| Status | **ACTIVE** |
| Day | 0/14 |
| Remaining | 13.87 days |
| Start | 2026-02-26T09:36:00Z |
| End | 2026-03-12T09:36:00Z |
| Manifest Integrity | INTACT |
| Config Integrity | INTACT |
| Drawdown Kill | 5.0% |
| DD Breached | No |
| Halted | No |

---

## Binary Lab

| Field | Value |
|---|---|
| Status | **ACTIVE (SHADOW)** |
| Day | 1/30 |
| Capital Allocated | $2,000.00 |
| PnL | $0.00 |
| Total Trades | 0 |
| Kill Line | OK (distance: $300) |
| Freeze Intact | Yes |
| Rule Violations | 0 |

---

## Portfolio

| Metric | Value |
|---|---|
| NAV | $9,786.00 |
| 30-day Return | -8.91% |
| 30-day Max Drawdown | 13.75% (peak-to-trough from nav_log) |
| Episodes | 692 total (32 scored, 660 pre-v6) |
| Win Rate | 10.5% (W:73 / L:619) |
| Realised PnL | -$759.23 |
| Regime | MEAN_REVERT (54%, 416 consecutive cycles) |

---

## Test Suite

```
3905 passed, 0 failed, 47 skipped
```

---

## Infrastructure

| Service | Schedule |
|---|---|
| Telegram Heartbeat | 06:00 UTC daily (`/etc/cron.d/hedge-telegram-heartbeat`) |
| Episode Rebuild | cron (`/etc/cron.d/hedge-episode-rebuild`) |
| Prediction Rotate | cron (`/etc/cron.d/hedge-prediction-rotate`) |
| Watchdog | cron (`/etc/cron.d/hedge_watchdog`) |

---

## Commits Since v7.6 Tag

129 commits. Key structural changes (last 30 days):

### Governance & Certification
- Activation Window v8.0 — 14-day full-stack certification protocol
- Phase C / DLE governance binding
- Dashboard certification panel

### Observability & Channel Discipline
- Telegram daily heartbeat (mechanical, 06:00 UTC, no commentary)
- Hard message limits: 24 lines / 1200 chars, truncation guard
- Silence-is-safe: send failures exit 0, no retries
- Fund-Ops monthly template generator (7-section pipeline)

### Binary Lab
- Binary Lab S1 shadow sleeve (30-day zero-capital forward test)
- Activation fix: event_type_override was masking ACTIVATE events

### Data Integrity Fixes
- NAV reads `total_equity` (not `nav_usd`)
- Daily summary field-name alignment: unrealized_pnl, portfolio_dd_pct, primary_regime, binary status
- Fund-ops drawdown: uses nav_log peak-to-trough (not daily snapshot)
- Conviction reporting: splits pre-scoring vs scored eras

### Execution & Risk
- DLE TTL fix (30→120s), stale permit filter
- Round observer memory leak fix (_tail_lines + RSS guardrail)
- True Edge v1: ATR×confidence mapping + fee gate v2
- Strategy hardening: sentinel, vol_target, screener, manifest

---

## Known Structural Issues

### 1. Win Rate: 10.5% Across 692 Episodes
89% of episodes (660) are from the pre-conviction era (Dec 2025) when the conviction engine did not exist. The 32 scored episodes (Feb 24–26) are too recent to evaluate separately. This is a legacy data issue, not current architecture failure.

### 2. Conviction Band Coverage
Only 32/692 episodes have conviction bands (low: 16, medium: 16). No `high` or `very_high` bands reached — hybrid scores range 0.34–0.52, below the 0.60 threshold for `medium`. Requires regime transition or stronger signal inputs to test upper bands.

### 3. Hybrid Score Dispersion: stdev=0.034
Expected for a 32-episode, 48-hour sample in a single regime (MEAN_REVERT). Not indicative of collapsed scoring. Needs more regime diversity to validate.

### 4. Drawdown: 13.75% Peak-to-Trough
NAV peaked at $10,805, troughed at $9,267 over the 30-day window. Episode ledger reports 9.55% max drawdown from episode PnL. The difference is unrealised mark-to-market fluctuation.

---

## Blockers

| Item | Status | Notes |
|---|---|---|
| Certification window | IN PROGRESS | Day 0/14, ends 2026-03-12 |
| Binary Lab observation | IN PROGRESS | Day 1/30, SHADOW mode |
| Fund-Ops monthly memo | **HOLD** | Insufficient scored episodes (32). Drawdown/conviction reconciled but sample too thin for institutional send. |
| Phase C (DLE enforcement) | **NOT STARTED** | Shadow-only in v7.9. Requires certification completion. |

---

## What Ships Next

1. **Certification window completes** (2026-03-12) — 14 days of integrity observation
2. **Binary Lab S1 completes** (~2026-03-27) — 30 days of shadow observation
3. **Conviction band dispersion** — requires regime transition data
4. **Fund-Ops monthly memo** — unblocked when scored episode count is statistically meaningful
5. **DLE Phase C enforcement** — gated on certification + postmortem protocol

---

## Verification Commands

```bash
# System health
make smoke

# Activation window status
PYTHONPATH=. python scripts/aw_status.py

# Telegram heartbeat (dry-run)
make heartbeat

# Fund-ops monthly (stdout)
make fund-ops

# Full test suite
PYTHONPATH=. pytest -q
```
