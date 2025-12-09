# PATCHSET V7.6-P1 — Router Microstructure Intelligence

## Objective

Upgrade the router to an institutional microstructure engine with:
- Per-symbol slippage & latency stats
- Slippage drift and latency buckets using config thresholds
- TWAP usage statistics and child-order fill ratios
- A richer router_health.json surface (global + per_symbol quality scores)
- Correct last_router_event_ts wiring into runtime diagnostics

Do **not** change high-level trading/risk semantics.

---

## Files to Inspect First

- execution/order_router.py
- execution/state_publish.py
- execution/diagnostics_metrics.py
- dashboard/state_v7.py (router loaders/panels)
- config/strategy_config.json (router_quality block)
- docs/v7.6_State_Contract.md
- docs/v7.6_Architecture.md
- docs/v7.6_Runtime_Diagnostics.md
- v7_manifest.json
- tests/integration/test_state_files_schema.py
- tests/integration/test_manifest_state_contract.py

---

## Changes to Implement

### 1) RouterStats helper in order_router

- Introduce a small RouterStats accumulator (new class or helper) inside execution/order_router.py (or a tiny helper module).
- Responsibilities:
  - `update_on_fill(symbol, intended_price, fill_price, ts_sent, ts_fill, is_twap_child, notional)`
  - Maintain a bounded rolling window per symbol using config.router_quality.stats_window (seconds + min_events).
  - Compute for each symbol:
    - avg_slippage_bps
    - slippage_drift_bps (e.g. EMA or mean minus baseline)
    - avg_latency_ms
    - twap_usage_ratio (TWAP notional / total notional)
  - Provide `snapshot(now)` returning a dict of RouterSymbolStats ready for state_publish.

Do not perform any file I/O here.

### 2) Extend write_router_health_state in state_publish

- Modify `write_router_health_state(...)` to:
  - Accept a RouterStats snapshot.
  - Look up config.router_quality:
    - base_score, min_score, max_score
    - slippage_drift_bps_thresholds.green/yellow
    - latency_ms_thresholds.fast/normal (new)
    - bucket_penalties, twap_skip_penalty
  - For each symbol:
    - Compute slippage_drift_bucket: GREEN/YELLOW/RED from thresholds.
    - Compute latency_bucket: FAST/NORMAL/SLOW from thresholds.
    - Determine router_bucket (A_HIGH/B_MEDIUM/C_LOW) from symbol tier or existing router config.
    - Compute quality_score:
      - Start from base_score.
      - Apply router_bucket penalty.
      - Apply penalties for high slippage_drift_bps, slow latency, and low twap_usage_ratio (using twap_skip_penalty).
      - Clamp to [min_score, max_score].
    - Build a per_symbol entry including:
      - quality_score
      - avg_slippage_bps
      - slippage_drift_bps
      - slippage_drift_bucket
      - avg_latency_ms
      - latency_bucket
      - last_order_ts, last_fill_ts
      - twap_usage_ratio
      - child_orders.count, child_orders.fill_ratio
      - router_bucket
  - Compute global aggregates (notional-weighted averages) and a global quality_score and buckets.
  - Write router_health.json atomically with schema:

    ```jsonc
    {
      "updated_ts": "...",
      "router_health": {
        "global": { ... },
        "per_symbol": { ... }
      }
    }
    ```

### 3) Diagnostics wiring (diagnostics_metrics)

- Ensure `runtime_diagnostics.exit_pipeline.last_router_event_ts` is updated whenever the router processes an order event (send, fill, cancel).
- If not already, compute `router_idle_seconds` in liveness.details using config.diagnostics.liveness.max_idle_router_events_seconds.
- Do not add side effects; diagnostics must be observational only.

### 4) Dashboard loader and panel

- In dashboard/state_v7.py:
  - Update the router_health loader to parse the new schema.
  - Provide safe defaults if fields are missing.
  - Add helper functions:
    - get_router_global_quality()
    - get_router_symbol_quality(symbol)
- Update the router panel to show:
  - Global quality_score and buckets.
  - Table of per-symbol quality_score, avg_slippage_bps, avg_latency_ms, buckets, twap_usage_ratio.

No writes from dashboard.

### 5) Docs & Manifest

- docs/v7.6_State_Contract.md:
  - Update the router_health row with the new fields and invariants (updated_ts required, router_health.global required, per_symbol object allowed empty).
- docs/v7.6_Architecture.md:
  - Document RouterStats → state_publish → router_health.json path.
- docs/v7.6_Runtime_Diagnostics.md:
  - Clarify last_router_event_ts semantics and relationship to router_idle_seconds.
- v7_manifest.json:
  - Update router_health description to mention slippage, latency, TWAP, quality scores.

### 6) Tests

- Add unit tests:
  - tests/unit/test_router_stats.py:
    - Verify windowing, averages, twap_usage_ratio.
  - tests/unit/test_router_quality_scoring.py:
    - Given stats + config thresholds, assert expected buckets and quality_score.
- Add integration tests:
  - tests/integration/test_state_router_health_schema.py:
    - Create a fake router_health payload via state_publish and assert it matches the new schema.
  - tests/integration/test_diagnostics_router_liveness.py:
    - Simulate router events + idle periods and validate last_router_event_ts and liveness.details.router_idle_seconds.

---

## Invariants & Safety

- Single-writer rule: router_health.json remains written only via state_publish from executor.
- Atomic writes only; use existing state_publish helpers.
- No changes to trading, risk, or sizing semantics.
- Diagnostics remain read-only/observational.
- Loaders must tolerate missing or partial router_health structure.

---

## Commands to Run

- make test-fast
- make test-runtime  # if you added runtime/stateful tests
