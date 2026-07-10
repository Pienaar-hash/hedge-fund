# CARD-HEDGE-RED-AUDIT-ROOT-CAUSE-CLASSIFICATION-001

**Type:** Read-only root-cause classification (no runtime changes made)
**Scope:** Convert the four RED findings from `LIVE_STATE_VERIFICATION_AUDIT_2026-07-10.md` into code-grounded defect classifications
**Method:** Static trace through `execution/*.py` + live log correlation against `/var/log/hedge-executor.err.log` and current state files
**Hard boundary respected:** No leverage resets, position closures, state-file rewrites, log cleanup, or restarts were performed. All commands were read-only (`grep`, `Read`, `git`, `python3 -m json.tool`, log tailing).

---

## Summary table

| # | Original RED finding | Classification | One-line verdict |
|---|---|---|---|
| 1 | Leverage 20× vs 3–4× policy | `confirmed_runtime_defect` + `telemetry_semantics_defect` | Exchange-side leverage is never set by the bot (dead code) and the risk engine validates the wrong quantity — it cannot see this class of drift even in principle |
| 2 | Exit pipeline suppressed by HALTED | `audit_interpretation_corrected` | Reduce-only orders bypass the risk gate unconditionally, by design — confirmed both in code and by live fills executing while risk_mode=HALTED |
| 3 | "No exits in 4+ hours" | `telemetry_semantics_defect` → superseded by finding #6 | The metric measures doctrine-driven exits only; a separate flip/trim mechanism (`auto_reduce`) was active throughout, but its own telemetry has been frozen since 18:55:19 |
| 4 | Veto reason always "unknown" | `audit_interpretation_corrected` | Both audits queried a non-existent JSON key (`reason`); the real key (`veto_reason`) is populated correctly in 100% of sampled entries |
| 5 | NAV wallet sync "stalled" (`balances_ok=false`) | `audit_interpretation_corrected` + `latent_safety_defect` | The flag is a double-negative (`stale_flags.balances_ok = NOT healthy`); `false` means healthy. The *real* mechanism is a latent defect: NAV freshness is refreshed opportunistically only when an order is sent, not on an independent timer |
| 6 (new, discovered live) | — | `confirmed_runtime_defect`, **currently active** | `positions_state.json`, `nav.json`, `diagnostics.json` have been frozen since 18:55:19 UTC and are now factually wrong — they show 2 open positions when only 1 remains |

---

## 1. Leverage — `confirmed_runtime_defect` + `telemetry_semantics_defect`

**Claim under test:** is `leverage: 20.0` in `positions_state.json` real 20× economic exposure, or the exchange's configured margin-leverage setting?

**Trace:**

- `config/risk_limits.json` → `per_symbol.BTCUSDT.max_leverage = 4`, `per_symbol.ETHUSDT.max_leverage = 3`. Confirmed real policy values.
- `execution/exchange_utils.py:1544` — `get_positions()` reads `"leverage": float(p.get("leverage") or 0)` directly from Binance's `/fapi/v2/positionRisk` response. This is the **account's configured margin-leverage setting** for that symbol/side, not a computed "% of NAV" ratio. This is what lands in `positions_state.json`.
- `execution/exchange_utils.py:1477` — `set_symbol_leverage(symbol, leverage)` is the function that would push a leverage setting to Binance via `/fapi/v1/leverage`. **It is defined but never called anywhere in the codebase, including tests** (`grep -rn "set_symbol_leverage" --include=*.py .` returns only its own definition).
- `execution/risk_limits.py:947` (`check_order`) enforces `lev > cfg.max_leverage` → veto. The `lev` value it checks comes from `execution/executor_live.py` (`lev = float(intent.get("leverage", 1) or 1)`) — a value **declared by the sizing/signal layer**, not read from the exchange. Confirmed via `execution/signal_screener.py:348` where reduce/flip intents are built with `"leverage": 1.0` hardcoded.

**Conclusion:** The risk engine's leverage cap validates the bot's *declared intent* leverage (which defaults to conservative values like 1.0–4.0), completely independent of whatever leverage is actually configured on the Binance account for that symbol. Since the bot never calls `set_symbol_leverage`, the account-side leverage setting is whatever was last configured manually (or defaulted by the exchange) and is **structurally invisible to the risk engine.**

- The **defect** is real: `set_symbol_leverage` dead code = the account's exchange-side leverage was never brought under policy control.
- The **audit correction** is also real: the observed `20.0` almost certainly does *not* mean 20× of NAV was economically deployed — actual gross notional was $2,844 against $6,791 NAV (≈0.42× portfolio exposure), consistent with the user's hypothesis. But this does **not** make the underlying condition acceptable: it's still policy-config drift with a real enforcement gap that would matter under adverse price moves (margin/liquidation math depends on the *account's* leverage setting, not the bot's internal accounting).

**Affected modules:** `execution/exchange_utils.py` (dead `set_symbol_leverage`), `execution/risk_limits.py::check_order` (validates wrong data source), `execution/executor_live.py::_send_order` (never reconciles exchange leverage before placing orders).

**Smallest safe fix (not implemented, per hard boundary):** call `set_symbol_leverage(symbol, cfg.max_leverage)` once at position-open time (or startup reconciliation) and fail-closed if the exchange rejects/mismatches it. This closes the loop the risk engine currently cannot see.

---

## 2. HALTED vs. reduce-only exits — `audit_interpretation_corrected`

**Claim under test:** does risk_mode=HALTED block reduce-only (exit) orders, explaining the apparent "no exits in 4+ hours"?

**Trace:**

- `execution/executor_live.py:2501-2502` (`_evaluate_order_risk`):
  ```python
  if reduce_only:
      return False, {}
  ```
  Reduce-only orders **skip the entire risk-gate call** (both legacy `check_order` and `RiskEngineV6.check_order`) unconditionally — they are never vetoed, regardless of risk_mode, NAV freshness, or leverage.
- `execution/executor_live.py` contains **zero references to the string "HALTED"** — the HALTED classification is a downstream label computed in `risk_engine_v6.py` / `nav_health_snapshot()` for reporting/dashboard purposes, not a direct branch the main executor loop checks before sending orders.
- **Live proof:** while `risk_snapshot.json` showed `risk_mode: HALTED` (`reason: nav_stale_age=101s` … `162s`), `logs/execution/orders_executed.jsonl` showed `reduceOnly: true, source: "auto_reduce"` fills executing at 19:06:50 and 19:12:19, and the executor log confirms a further fill at 19:18:08 that fully closed the BTCUSDT position (`positions: 2` → `positions: 1`).

**Conclusion:** HALTED does **not** suppress reduce-only/exit orders — by design. This is the correct, safe behavior (you always want to be able to de-risk). The original audit's "no exit activity" reading was a **misinterpretation of stale telemetry** (see #6), not a real suppression bug.

---

## 3. "No exits in 4+ hours" — superseded by #6

**Trace:**

- The `last_exit_trigger_ts` / `last_exit_scan_ts` fields live in `logs/state/diagnostics.json`, written by `write_runtime_diagnostics_state()`, called from `_pub_tick()` in `execution/executor_live.py:6521`.
- `_pub_tick()` also writes `positions_state.json` and `nav.json` (see #6). All three froze at the same instant: **2026-07-10T18:55:19 UTC**.
- Separately, `execution/episode_ledger.py:711` distinguishes a genuinely different signal source: `source == "auto_reduce"` (opposing-signal flip/trim, built in `execution/signal_screener.py::_reduce_plan`) from proper doctrine-driven exits (`TIME_STOP`, `REGIME_FLIP`, `STOP_LOSS_SEATBELT`, etc., resolved in `execution/doctrine_kernel.py` and logged to `logs/doctrine_events.jsonl`).

**Conclusion:** the diagnostic fields the original audit cited to claim "exit pipeline stalled" have been **frozen, not accurately reporting zero**, since 18:55:19. We cannot currently distinguish "doctrine exit scanner genuinely idle" from "diagnostics writer broken" using that file. What we *can* confirm independently (via raw order/log evidence) is that a **different** mechanism (`auto_reduce`, a flip-driven trim — not a stop-loss or take-profit) has continued reducing the BTCUSDT position. This is not equivalent to functioning TP/SL coverage (still 0% per `tp_sl_registered_count: 0`), so the original audit's underlying *concern* (no proper exit protection) is still valid — but the specific evidence ("4+ hour stall") is unreliable, being sourced from a frozen file.

---

## 4. Veto reason "unknown" — `audit_interpretation_corrected`

**Trace:**

- `execution/risk_limits.py:393` (`_emit_veto`): writes `"veto_reason": REASONS.get(reason, reason or "unknown")` and `"original_reason": reason`. The `REASONS` dict (lines 333–361) maps ~20 internal reason codes (`exceeds_leverage_cap`, `min_notional`, `symbol_cap`, `nav_stale`, etc.) to human-readable labels.
- **Direct inspection of raw `logs/execution/risk_vetoes.jsonl` entries** shows fully populated, correct fields: `"veto_reason": "symbol_cap"`, `"veto_reason": "min_notional"`, each with a full `veto_detail` block (thresholds, nav_health_diag, constraint_geometry, etc.).
- Sampled the last 500 veto entries programmatically: **0 entries have `veto_reason` missing or `"unknown"`** — distribution was `min_notional: 264`, `symbol_cap: 236`.

**Conclusion:** both this session's original audit and the prior `LIVE_STATE_VERIFICATION_AUDIT.md` (2026-06-18) queried a JSON key literally named `"reason"`, which **does not exist** in this schema — the real key is `veto_reason`. This was a parsing bug in the audit tooling, not a defect in the trading system. The veto pipeline is fully auditable and has been the whole time.

---

## 5. NAV "wallet sync stalled" — `audit_interpretation_corrected` + `latent_safety_defect`

**Trace (semantics correction):**

- `execution/nav.py:808` and `:855`: `record["stale_flags"] = {key: not val for key, val in health.items()}`. This is a **logical negation** — `stale_flags.balances_ok` is `NOT health.balances_ok`.
- Live cache file `logs/cache/nav_confirmed.json` (freshly written, mtime 19:17:33) shows `"source_health": {"balances_ok": true}` **and** `"stale_flags": {"balances_ok": false}` simultaneously — i.e., `false` under `stale_flags` means "not stale" = **healthy**. Both audits (this one included, in its first pass) misread `stale_flags.balances_ok: false` as "balances NOT ok," when it means the opposite.

**Trace (real mechanism — latent defect):**

- NAV cache refresh (`_persist_confirmed_nav`, called from `_futures_nav_usdt`) only happens when `compute_nav_pair`/`compute_trading_nav` is invoked.
- `_compute_nav()` is called from `execution/executor_live.py:4324`, inside `_send_order()` — i.e., **NAV is refreshed opportunistically whenever an order is about to be sent**, not on an independent fixed-interval timer.
- The HALTED threshold (`_NAV_FRESHNESS_SECONDS`, default 90s) is tighter than the natural gap between order-send events during quiet periods (churn guard: 120s min hold + 300s cooldown per symbol; sparse opposing signals for `auto_reduce`). Live log evidence: `[risk] nav_age=0.0s` right after an order send, followed by multi-minute gaps with no `[nav] snapshot` log lines until the next order attempt.

**Conclusion:** the system does not have a "wallet sync failure" — it has a **structural coupling defect**: NAV freshness is tied to order-send cadence rather than an independent poll loop. Given the 90s HALTED threshold is shorter than the natural inter-order gap during quiet trading, the system will predictably oscillate in and out of HALTED as a **designed side-effect**, not a fault. This is real and worth fixing (decouple NAV refresh from order-send timing), but it is a latent design defect, not a "stall" or "sync failure" as both prior audits characterized it.

---

## 6. NEW — Live state-publish freeze (discovered during this investigation) — `confirmed_runtime_defect`, **currently active**

This was not one of the five requested determinations; it surfaced while investigating #2/#3 and is more severe than any RED finding in the original audit.

**Evidence:**

```
File mtimes (checked 2026-07-10 19:20:53 UTC, executor confirmed alive):
  logs/state/risk_snapshot.json    19:20:32  (fresh, 21s old)
  logs/state/router_health.json    19:20:13  (fresh, 40s old)
  logs/state/diagnostics.json      18:55:19  (STALE, ~25 min old)
  logs/state/positions_state.json  18:55:19  (STALE, ~25 min old)
  logs/state/nav.json              18:55:19  (STALE, ~25 min old)
```

All three frozen files are written together by `_pub_tick()` (`execution/executor_live.py:5565` → `write_positions_ledger_state`, `write_runtime_diagnostics_state`, and the nav-state block ending at line 5742). `risk_snapshot.json`/`router_health.json` are written by a separate code path that continues to function.

**Timing correlation:** The last successful `_pub_tick`-family write logged was `[nav-state] AUM present: ...` at **18:55:19.419**. The very next log lines (18:55:21.220 onward, within ~2 seconds) are a burst of **35,050** `exit_reason_unmapped raw='unknown' source='episode_ledger' fallback=THESIS_INVALIDATED` warnings from an episode-ledger rebuild, all emitted within under a second. No traceback appears in the error log after this point (the only `Traceback` in the whole file predates the freeze by tens of thousands of lines), so this is **not** an unhandled exception — `_pub_tick()`'s write calls are individually try/excepted and would log at ERROR/DEBUG on failure, and none did.

**Proof the frozen files are now factually wrong:**

```
positions_state.json (frozen, updated_at=18:55:19):
  ETHUSDT SHORT -1.46
  BTCUSDT SHORT -0.0037     ← still shown open

Live executor log, 19:18:08: ORDER_FILL ... BTCUSDT BUY reduceOnly=True → position fully closed
Live executor log, 19:18:41: "positions: 1"   ← BTCUSDT no longer open
```

`positions_state.json` — the file the dashboard and any investor-facing position view reads — currently overstates open positions by one and is 25+ minutes stale during active trading, with no indication in its own `updated_at` field that anything is wrong (a naive freshness check would need to compare against wall-clock time, which nothing currently does for this specific file, given `diagnostics.json`'s own freshness fields are inside the same frozen file).

**Classification:** `confirmed_runtime_defect`. Root cause is correlated but not fully proven — the timing match with the episode-ledger rebuild's warning burst is exact to the second, but the causal mechanism inside `_pub_tick()` (why positions/nav/diagnostics writes specifically stopped while risk_snapshot/router_health continued) was not traced to a specific line before this report was due. This needs its own dedicated read-only trace as a follow-up.

**This is currently ongoing as of report time (19:22 UTC) and independent of anything discussed in the original RED audit.**

---

## Consolidated finding table (for the card)

| Finding | File(s) | Line(s) | Classification |
|---|---|---|---|
| `set_symbol_leverage` dead code | `execution/exchange_utils.py` | 1477 | `confirmed_runtime_defect` |
| Risk engine validates intent-leverage, not exchange-leverage | `execution/risk_limits.py`, `execution/executor_live.py` | 947, ~4160 | `confirmed_runtime_defect` |
| Reduce-only bypasses risk gate (by design, confirmed safe) | `execution/executor_live.py` | 2501-2502 | `audit_interpretation_corrected` |
| Exit-pipeline diagnostics frozen, cannot support "4h stall" claim | `logs/state/diagnostics.json` | — | `telemetry_semantics_defect` |
| Veto reason field misread by audit tooling | `execution/risk_limits.py` | 393 | `audit_interpretation_corrected` |
| `stale_flags.balances_ok` is a double negative, misread | `execution/nav.py` | 808, 855 | `audit_interpretation_corrected` |
| NAV refresh coupled to order-send cadence, not independent timer | `execution/nav.py`, `execution/executor_live.py` | 4324 | `latent_safety_defect` |
| **`positions_state.json`/`nav.json`/`diagnostics.json` frozen since 18:55:19, now factually wrong** | `execution/executor_live.py::_pub_tick` | 5565-5746 | **`confirmed_runtime_defect` (active)** |

---

## Recommended next cards (unchanged from the proposed remediation tranche, reprioritized)

1. **`_pub_tick` freeze investigation** (new — promote to P0): trace why positions/nav/diagnostics writes stopped at 18:55:19 while risk_snapshot/router_health continued. Currently actively misrepresenting investor-facing position state.
2. **Exchange leverage reconciliation**: wire `set_symbol_leverage` into the entry path; fail closed on mismatch against `config/risk_limits.json` per-symbol caps.
3. **NAV refresh decoupling**: run NAV wallet fetch on an independent timer (e.g., every `_NAV_FRESHNESS_SECONDS / 2`) rather than only on order-send, to stop the HALTED oscillation.
4. **Audit tooling fix**: any future audit script must read `veto_reason` (not `reason`) and must interpret `stale_flags.*` as "is-stale" booleans (true = bad), not health booleans.
5. **TP/SL registry** (unchanged from original audit): still 0% coverage; `auto_reduce` is not a substitute for stop-loss/take-profit protection.

No code was changed. No services were restarted. No leverage, positions, or config were modified as part of this investigation.
