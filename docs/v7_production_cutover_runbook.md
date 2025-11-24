## 📌 v7 Production Cutover Runbook

Goal:  
Promote the v7 sizing/risk contract from **testnet** to **production**, with safe rollback.

---

## 0. Assumptions

- v7 branch or tag exists and has passed:
  - v7 Testing Pack
  - v7 Runtime Verification on testnet
- You have SSH + sudo access to the prod box.
- Supervisor is managing executor/dashboard/sync/shadow processes under group `hedge`.

---

## 1. Pre-Cutover Checklist

1. **Freeze code / tag**

   - Ensure main contains v7 changes.
   - Create a tag (example):

     ```bash
     git tag v7.0-screener-sizing
     git push origin v7.0-screener-sizing
     ```

2. **Capture current prod state**

   - Snapshot:

     - `config/risk_limits.json`
     - `config/strategy_config.json`
     - `config/pairs_universe.json`
     - `.env`
   - Save `logs/state/*` for reference (NAV, positions, risk).

3. **Confirm there are no open critical positions** you aren’t comfortable carrying through a short restart.

---

## 2. Deploy v7 Code to Prod Box

On the prod box:

```bash
cd ~/hedge-fund
git fetch origin
git checkout main          # or the v7 tag/branch
git pull origin main
````

Optional sanity:

```bash
git log -1
```

Confirm the latest commit matches what you expect as v7.

---

## 3. Update Configuration

1. **risk_limits.json**

   * Ensure caps are in **fractions** (e.g., `0.02` for 2%).
   * Confirm per-symbol caps align with real portfolio size (e.g., 4k NAV vs 11k NAV).
   * Check:

     * `trade_equity_nav_pct`
     * `max_trade_nav_pct`
     * Per-symbol `max_nav_pct`
     * `min_notional_usdt` and per-symbol `min_notional`.

2. **strategy_config.json**

   * Confirm the v7 strategy config (e.g., 15m momentum for BTC/ETH/SOL).
   * Ensure no stale sizing knobs (size_model, RiskGate, etc) remain in active use.

3. **.env**

   For production:

   * `BINANCE_TESTNET=0`
   * `DRY_RUN=1` for soft launch / DRY phase.
   * Confirm API keys point to **production** futures account.

   After the DRY phase, you will switch `DRY_RUN=0` (Step 6).

---

## 4. Restart Services

```bash
sudo supervisorctl stop hedge:
sudo supervisorctl status   # everything should be STOPPED
```

Then:

```bash
sudo supervisorctl start hedge:
sudo supervisorctl status
```

Ensure:

* `hedge:hedge-executor` RUNNING
* `hedge:hedge-dashboard` RUNNING
* `hedge:hedge-sync_state` RUNNING
* `hedge:hedge-pipeline-shadow-heartbeat` RUNNING
* `hedge:hedge-pipeline-compare` RUNNING

---

## 5. DRY-RUN Smoke Test (Prod Data, No Live Orders)

With `DRY_RUN=1`:

1. Tail logs:

   ```bash
   sudo tail -f /var/log/hedge-executor.out.log
   ```

2. Check:

   * NAV lines show prod futures balance (4k-ish, or whatever is correct).
   * `[screener] attempted/emitted/submitted` look healthy.
   * `[screener->executor]` payloads show **non-zero** `gross_usd` and `qty`.
   * `[risk]` vetos are consistent and attributable solely to risk caps.
   * There are **no actual orders** going to the exchange (by contract `DRY_RUN=1` must block order sends).

3. Visit the dashboard:

   * NAV, positions, and risk stats are sane.
   * No unexpected spikes.

If this DRY-RUN phase looks wrong, **stop and roll back** (see Step 8).

---

## 6. Enable Live Trading (Prod)

Once DRY-RUN phase is clean and you’re satisfied:

1. Edit `.env`:

   * Set `DRY_RUN=0`.

2. Reload environment for supervisor (if necessary) and restart:

   ```bash
   sudo supervisorctl restart hedge:hedge-executor
   sudo supervisorctl status
   ```

3. Tail logs:

   ```bash
   sudo tail -f /var/log/hedge-executor.out.log
   ```

4. Confirm:

   * Same healthy screener → executor → risk flow as in DRY-RUN.
   * Orders that pass risk now show as **submitted** to exchange (check Binance futures prod).

---

## 7. Post-Cutover Monitoring

For at least the first **24–72 hours**:

1. Watch:

   * NAV trajectory and daily PnL.
   * Number of trades per day and average notional.
   * Max symbol exposure vs caps.
   * Drawdown and daily loss vs configured limits.

2. Logs to watch:

   * `/var/log/hedge-executor.out.log`
   * `/var/log/hedge-executor.err.log`
   * Dashboard app logs (for NAV/snapshot issues).

3. Shadow pipeline:

   * Ensure parity metrics do not show persistent drift between live and shadow.

---

## 8. Rollback Plan

If something is fundamentally wrong (sizing, caps, unexpected behavior):

1. **Stop trading immediately**:

   ```bash
   sudo supervisorctl stop hedge:hedge-executor
   ```

2. Optionally close open positions manually on Binance.

3. Roll back code:

   ```bash
   cd ~/hedge-fund
   git checkout <previous-stable-tag>   # e.g. v6.x
   git pull origin <branch-containing-that-tag>  # Optional
   ```

4. Restore configs from the snapshot taken in Step 1.

5. Set `.env` back to known-good values.

6. Restart:

   ```bash
   sudo supervisorctl start hedge:hedge-executor
   ```

7. Verify old behavior is restored (logs + dashboard + exchange).

---

## 9. Final Sign-off

Cutover is considered complete when:

* v7 sizing contract is live on production.
* Trades are flowing with correct notional and leverage.
* Risk vetos are understandable and aligned with `risk_limits.json`.
* No unexpected re-sizing, RiskGate artifacts, or size_model behavior appears in logs.
* NAV and risk metrics look coherent over multiple days of runtime.

Document the cutover date, tag, and any manual overrides in a short note (e.g.,
`docs/v7_cutover_log.md`) for future audits.