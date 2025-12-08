## Execution Audit — Pass A (feature/v6.1-audit)

- execution/risk_engine_v6.py  
  - Added fail-closed guard around `check_order`; logs and surfaces `risk_engine_error` diagnostics instead of propagating exceptions.  
  - Risk/edge: prior path could crash executor/shadow pipelines on malformed config; now blocked with explicit veto signal.  
  - Follow-up: capture nav/position snapshots into diagnostics to make operator triage faster.

- execution/pipeline_v6_shadow.py  
  - Shadow intents now derive `open_positions_count` and `symbol_open_qty` from provided position snapshots when signal/nav hints are absent.  
  - Risk/edge: shadow risk checks previously assumed zero exposure when upstream did not pass counts, understating vetoes.  
  - Follow-up: align positions schema ingestion with synced state payloads to cover alt keys (e.g., hedged legs).

- execution/order_router.py  
  - `monitor_and_refresh` now ignores terminal orders before cancelling/repricing to avoid acting on filled/canceled orders.  
  - Risk/edge: watcher could previously fire cancellations on stale terminal states from UMFutures pollers.  
  - Follow-up: consider logging low-fill cancellations for visibility.

- execution/executor_live.py  
  - Reviewed; no code change. Key risk: shadow compare heartbeat assumes `_LAST_POSITIONS_STATE` schema; add schema validation in later pass.

- execution/risk_limits.py  
  - Reviewed; no code change. Nav freshness gate relies solely on cached snapshots; consider threading live nav_state when available.

- execution/exchange_utils.py  
  - Reviewed; no code change. UM client stub lacks recent error cache telemetry; consider exporting for executor health logs.

- execution/pipeline_v6_compare.py  
  - Reviewed; no code change. Comparison uses last live order per symbol; may miss sequencing—consider intent-id alignment in pass B.
