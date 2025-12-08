# 📌 **RISK TUNING PROMPT (CLEAN VERSION)**

## **Objective**

Tune the v7 risk caps so that valid trades (≈1–3% NAV) pass, while oversize trades block cleanly, using a **single**, **consistent**, **fraction-based** cap system.

## **Required Outcomes**

1. Trades in the **1%–3% NAV range** must pass.
2. Trades above the configured **per-trade NAV cap** must block.
3. Per-symbol caps must be respected consistently.
4. No double-normalization of percent values.
5. No cap enforcement outside the risk engine.
6. Screener sizing and executor passthrough must remain untouched.

---

## **Files Codex may modify**

* `config/risk_limits.json`
* `config/strategy_config.json` (sizing values only)
* `execution/risk_loader.py` (normalization only)
* `execution/risk_limits.py` (thresholds and cap math only)

## **Files Codex must not modify**

* `execution/signal_screener.py`
* `execution/executor_live.py`
* `execution/router/*.py`
* `execution/nav.py`
* Any sizing logic outside risk engine
* Any NAV sourcing logic
* Any runtime flow / pipeline logic

---

## **Cap Model (Target Contract)**

All caps use **fractions**:

```
per_trade_nav_pct:   ~0.01 – 0.03
max_trade_nav_pct:   ~0.03 – 0.05
trade_equity_nav_pct: same range
symbol max_nav_pct:  BTC 0.10–0.15, ETH 0.08–0.12, SOL 0.05–0.08
portfolio gross cap: 0.30–0.50
```

Codex must tune caps to these ranges so that:

* “BTC SELL 1342 USD” (≈12% margin, ≈3% NAV notional) → **should pass**
* ETH and SOL intents around 1.5–3% NAV → **pass**
* Anything above symbol max_nav_pct or max_trade_nav_pct → **clean block**

---

## **Normalization Rules**

Codex must enforce:

### 1. Values > 1 → treat as percent → divide by 100

Examples:
`15 → 0.15`
`5 → 0.05`

### 2. Values 0–1 → treat as fractions (leave untouched)

### 3. 0 → absolute cap (block)

### 4. Remove all other normalization paths

This produces a **single, correct, stable representation**.

---

## **Behaviour Requirements**

* Screener gross_usd and qty are authoritative (no executor resizes).
* Risk engine is the **sole enforcement layer** for caps.
* NAV for risk must be `nav_health_snapshot.nav_total`.
* Block reasons must be single and clean:

  * `"symbol_cap"`
  * `"trade_gt_equity_cap"`
  * `"max_trade_nav_pct"`
  * `"portfolio_cap"`

No stacked vetoes unless multiple caps are genuinely exceeded.

---

## **Deliverables Codex must output**

* A single diff patch updating:

  * config caps
  * normalization in risk_loader
  * cap enforcement thresholds in risk_limits
  * any tests that expect old percent math

Patch should contain **only the edits** and preserve existing structure.

---

## **Validation**

After tuning:

* BTC/ETH/SOL testnet intent streams must produce **submitted=1** when sizing is within the tuned caps.
* No symbol_cap blocks unless *actual* notional exceeds per-symbol limits.
* No trade_gt_equity_cap blocks for trades < max_trade_nav_pct.
* NAV aging and stale detection must remain correct.
