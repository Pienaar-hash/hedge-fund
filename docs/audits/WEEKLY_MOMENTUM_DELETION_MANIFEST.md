# Weekly Momentum Deletion Manifest

This branch pivots the production path from the legacy multi-strategy executor to a weekly-only stack.

## Removed From Active Production Path

| Path | Current Purpose | Replacement | Retained Dependency Check | Test References Removed | Configuration References Removed | Runtime References Removed |
| --- | --- | --- | --- | --- | --- | --- |
| `execution/executor_live.py` | Legacy live multi-strategy executor | `scripts/run_weekly_cycle.py` | Weekly runner does not import it | `Makefile` no longer runs legacy executor tests | Weekly runner ignores `strategy_config.json` and `runtime.yaml` | Weekly runner bypasses executor loop |
| `execution/hydra_engine.py` | Hydra strategy orchestration | None | Weekly import graph validator blocks Hydra imports | Legacy Hydra tests removed from active targets | No weekly config references | No weekly runtime references |
| `execution/hydra_integration.py` | Hydra execution bridge | None | Unused by weekly modules | Removed from active targets | None | None |
| `execution/hydra_monotonicity.py` | Hydra telemetry | None | Unused by weekly modules | Removed from active targets | None | None |
| `execution/hydra_pnl.py` | Hydra budget throttling | None | Unused by weekly modules | Removed from active targets | None | None |
| `execution/signal_generator.py` | Intraday signal generation | `execution/weekly_momentum_engine.py` | Weekly modules do not import it | Removed from active targets | No weekly config references | No weekly runtime references |
| `execution/signal_screener.py` | Intraday intent screening | `execution/weekly_momentum_engine.py` | Weekly modules do not import it | Removed from active targets | None | None |
| `execution/sentinel_x.py` | Regime classification | None | Weekly validator blocks sentinel imports | Removed from active targets | No weekly config references | No weekly runtime references |
| `execution/strategy_adaptation.py` | Adaptive strategy allocation | None | Weekly sizing is fixed | Removed from active targets | Dynamic sizing flags removed from weekly config | No weekly runtime references |
| `execution/alpha_miner.py` | Experimental alpha mining and ML feature path | None | Weekly validator blocks alpha miner imports | Removed from active targets | No weekly config references | No weekly runtime references |
| `execution/cross_pair_engine.py` | Relative-value and crossfire logic | None | Weekly selector uses one symbol per correlation group only | Removed from active targets | No weekly config references | No weekly runtime references |
| `execution/strategies/vol_target.py` | Volatility target strategy | None | Weekly risk gate has fixed stop/risk rules | Removed from active targets | No weekly config references | No weekly runtime references |
| `config/strategy_config.json` | Legacy strategy and intraday settings | `config/weekly_momentum.json` | Weekly runner rejects explicit legacy config inputs | Legacy config parsing tests removed from active targets | Replaced by weekly config set | Weekly runner never reads it |
| `config/runtime.yaml` | Legacy runtime toggles | None | Weekly runner does not read it | Runtime-target suite replaced | Replaced by fixed weekly schedule in JSON | Weekly runner never reads it |
| `config/risk_limits.json` | Legacy broad risk config | `config/weekly_risk_limits.json` | Weekly modules only load weekly risk JSON | Legacy risk suite removed from active targets | Weekly risk JSON replaces it | Weekly runner never reads it |
| `scripts/run_executor_once.sh` | One-shot legacy executor wrapper | `scripts/run_weekly_cycle.py` | Weekly runner bypasses it | Removed from active targets | None | None |
| `scripts/strategy_probe.py` | Legacy strategy probe | `scripts/validate_weekly_engine.py` | Weekly validator replaces production check | Removed from active targets | None | None |
| `tests/unit/*legacy weekly-unrelated*` | Legacy unit coverage | Weekly unit suite | New targets only include weekly tests | Active targets replaced | None | None |
| `tests/integration/*legacy weekly-unrelated*` | Legacy integration coverage | Weekly integration suite | New targets only include weekly tests | Active targets replaced | None | None |

## Verification Notes

- Active validation now runs only the weekly modules and weekly tests.
- `scripts/validate_weekly_engine.py` proves the weekly import graph does not pull banned legacy modules.
- Legacy files may remain in the repository for audit history, but they are outside the active production and validation path for this branch.
