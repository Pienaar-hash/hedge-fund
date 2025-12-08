## v7 Router Model (current repo)

### Sources
- `execution/order_router.py`
- Intel/policy: `execution/intel/maker_offset.py`, `execution/intel/router_policy.py`, `execution/intel/router_autotune_v6.py`, `execution/intel/router_autotune_apply_v6.py`
- Runtime config: `execution/runtime_config.py` (loads `config/runtime.yaml`)

### Flow
1. `route_intent` builds router settings from runtime.yaml/env (post-only default, offsets, min child notional, fill windows).
2. Maker-first submit via `submit_limit` with postOnly=GTX; offset from `suggest_maker_offset_bps` + router_policy/autotune suggestions; spread clamps from runtime flags.
3. On post-only rejection: retries up to `router_rejects_max`, then fallback to taker if slippage/spread thresholds exceeded.
4. Chunking: `chunk_qty` splits qty based on min child notional (runtime), ensures no child below threshold.
5. Monitoring: `monitor_and_refresh` polls fills, cancels/refreshes if fill ratio below `min_fill_ratio` within `low_fill_window_s`; can pivot to taker if needed.
6. Effective price: `effective_px` adjusts for maker/taker fees (bps from runtime or env).
7. Health/metrics: errors logged via `execution.utils.execution_health.record_execution_error`; events logged via `log_event` to router logs.

### Constraints/tunables (runtime.yaml/env)
- `post_only_default`, `router_slip_max_bps`, `router_rejects_max`, `router_max_spread_bps`, `router_offset_spread_clamp_bps`, `min_child_notional` (offpeak/priority sections), `low_fill_window_s`, `min_fill_ratio`, fees (`TAKER_FEE_BPS`, `MAKER_FEE_BPS`).

### Telemetry
- Order events logged via `log_event` with safe_dump payloads.
- `router_metrics.py` captures success/failure counts, slippage, rejects, and can publish to state metrics.

### Future tuning hooks
- Autotune suggestions (`router_autotune_v6.py`) + apply path (`router_autotune_apply_v6.py`) already wired; runtime_config exposes window/priority knobs for v7 tuning without code changes.
