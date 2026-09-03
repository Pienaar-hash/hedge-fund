# Weekly Production Architecture

## Entrypoint

```text
scripts/run_weekly_cycle.py
```

This is the only production entrypoint for the weekly system.

`execution/executor_live.py` is legacy-present-but-unreachable for this path.

## Allowed Module Graph

```text
scripts/run_weekly_cycle.py
├── execution/weekly_market_data.py
├── execution/weekly_momentum_engine.py
├── execution/weekly_position_sizer.py
├── execution/weekly_risk_gate.py
├── execution/weekly_execution_planner.py
├── execution/weekly_audit_ledger.py
├── execution/exchange_utils.py
└── execution/exchange_precision.py
```

No other production imports are allowed.

## Allowed Configuration

```text
config/weekly_momentum.json
config/weekly_risk_limits.json
config/weekly_universe.json
```

The weekly runner must reject legacy strategy/config inputs that can reactivate:

- Hydra
- Sentinel-X
- intraday signals
- adaptive sizing
- dynamic leverage
- ML or sentiment paths
- shadow execution
- strategy registries

## Execution Sequence

```text
completed weekly data
-> weekly momentum features
-> cross-sectional ranking
-> conviction classification
-> candidate selection
-> fixed position sizing
-> deterministic risk gate
-> weekly execution plan
-> order intents
-> audit record
```

## Invariants

- Only completed UTC weeks are eligible. Partial current-week candles are rejected.
- Symbol ordering is deterministic.
- No fallback values, no imputation, no discretionary overrides.
- Identical inputs must produce byte-identical plans and audit records.
- Order intents must respect exchange precision and minimum-notional rules.
- Legacy environment variables and strategy flags cannot alter weekly behaviour.
- The default production command must not invoke the old executor.
- Focused weekly tests must run without importing legacy subsystems.

## Runtime Classification

Everything outside the allowed graph is:

```text
legacy_present_but_unreachable
```

This includes Hydra, Sentinel-X, intraday strategies, regime logic, adaptive budgets, sub-weekly paths, ML, sentiment, funding/basis/order-book features, and shadow strategy execution.

## Non-Goals

- Simplifying `execution/executor_live.py`
- Preserving backward compatibility with legacy strategy flags
- Supporting intraday entries, re-ranking, or resizing
- Dynamic sizing, dynamic leverage, or regime-dependent behaviour
- Reusing the legacy strategy registry as a control plane
