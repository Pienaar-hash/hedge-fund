# TESTNET_RESET_PROTOCOL.md

**System:** GPT-HEDGE v7.x
**Scope:** Binance Futures Testnet only
**Status:** Active
**Last Updated:** 2026-02-12

---

# I. PURPOSE

Binance Futures Testnet performs **periodic full account resets**.

A reset:

* Wipes all open positions
* Resets wallet balances
* Deletes order history
* Deletes trade fills

This protocol defines:

1. How resets are detected
2. How they are logged
3. How NAV continuity is preserved
4. How replay integrity is protected
5. What is strictly forbidden

This protocol does **not** apply to production.

---

# II. PRINCIPLE

> The exchange is not authoritative for history.
> Our logs are authoritative.

The system must never:

* Rewrite local history
* Merge pre-reset and post-reset equity curves
* Pretend the reset did not occur

A reset is an **environment discontinuity**, not a dataset failure.

---

# III. WHAT A RESET BREAKS

| Component             | Impact             |
| --------------------- | ------------------ |
| Exchange positions    | Wiped              |
| Exchange balance      | Reset to default   |
| Exchange fill history | Deleted            |
| Local logs            | Unaffected         |
| Regime detection      | Unaffected         |
| Episode ledger        | Unaffected (local) |

The only true risk is:

> NAV corruption through artificial balance jump.

---

# IV. RESET DETECTION

A reset is declared when **all** of the following occur simultaneously:

1. Exchange balance equals known testnet default
2. No open positions returned
3. Local logs indicate prior non-zero balance
4. Sudden balance delta > 50% without deposit log

Detection must occur before execution cycle continues.

---

## A. Detection Pseudocode

```
if env == "testnet":
    if exchange_balance == DEFAULT_TESTNET_BALANCE
       and exchange_positions == []
       and last_logged_nav > 0
       and abs(exchange_balance - last_logged_nav) > threshold:
           trigger TESTNET_RESET_EVENT
```

---

# V. REQUIRED ACTIONS ON RESET

Once detected:

---

## 1️⃣ Log Explicit Reset Event

Append to:

```
logs/execution/environment_events.jsonl
```

Example:

```json
{
  "ts": "2026-02-04T00:00:01Z",
  "event": "TESTNET_RESET",
  "environment": "binance_futures_testnet",
  "pre_reset_nav": 10238.44,
  "post_reset_balance": 10000.00,
  "cycle_id": "CYCLE_TEST_004"
}
```

This event is immutable.

---

## 2️⃣ Freeze Pre-Reset Ledger

Immediately:

* Archive current episode ledger
* Archive NAV state
* Close current test cycle

Example:

```
archive/testnet_cycle_004/
    episode_ledger.json
    nav_state.json
    positions_state.json
```

---

## 3️⃣ Start New Test Cycle

Create:

```
CYCLE_TEST_005
```

Reset:

* NAV baseline
* Drawdown tracking
* Cycle statistics

Do **not** modify historical logs.

---

## 4️⃣ Disable Historical Backfill

After reset:

Backfill scripts must:

* Refuse to fetch fills older than reset timestamp
* Never reconcile wiped exchange history

Any backfill attempt beyond reset boundary must:

* Raise explicit warning
* Abort silently
* Log event

---

# VI. NAV HANDLING RULES

NAV continuity must be handled as:

```
NAV_pre_reset  →  TERMINATION
NAV_post_reset →  NEW BASELINE
```

Never:

* Combine them into single equity curve
* Treat reset as profit or loss
* Adjust historical PnL

Testnet resets are **not economic events**.

---

# VII. EXPECTANCY & PERFORMANCE

Expectancy v6 derives from local episode ledger.

Two acceptable policies:

### Option A (Preferred)

Carry expectancy forward across resets
(Local episodes remain valid)

### Option B

Cold-start expectancy at new cycle

Either is allowed, but must be logged.

---

# VIII. WHAT IS FORBIDDEN

During or after reset:

❌ Rebuilding history from exchange
❌ Rewriting episode IDs
❌ Editing local execution logs
❌ Retroactively adjusting PnL
❌ Manual NAV smoothing
❌ Silent reset handling

Any of the above constitutes audit violation.

---

# IX. RELATION TO DATASET_ADMISSION_GATE

Testnet reset does **not** constitute:

* Dataset admission failure
* Dataset rollback
* Doctrine falsification

It is an environmental reset, not data corruption.

Dataset integrity remains intact because:

* Regime authority uses klines
* Fills are locally logged
* Replay determinism is internal

---

# X. RELATION TO DATASET_ROLLBACK_CLAUSE

This event does **not** trigger rollback.

Rollback applies to:

* Data instability
* Non-determinism
* Regime corruption

A testnet reset is:

> Exchange sandbox reset
> Not a dataset instability

---

# XI. PRODUCTION CONSTRAINT

This protocol must never:

* Trigger on production endpoints
* Be allowed to mask real production anomalies

Production balance discontinuity must be treated as:

> CRITICAL INCIDENT
> Not reset

---

# XII. OPTIONAL AUTOMATION

If automated:

* Reset detection must halt executor for one cycle
* Logging must precede resumption
* Supervisor restart is permitted after archive

---

# XIII. AUDIT CHECKLIST

After each reset:

* [ ] Reset event logged
* [ ] Pre-reset ledger archived
* [ ] New cycle created
* [ ] NAV baseline reset
* [ ] Backfill disabled pre-boundary
* [ ] No historical files modified

---

# XIV. FINAL LINE

> A reset handled transparently preserves trust.
> A reset hidden corrupts audit integrity.

Testnet is for experimentation.
Auditability is not.
