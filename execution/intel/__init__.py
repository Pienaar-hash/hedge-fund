"""
Execution Intelligence (v5.10+).

This package holds higher-level analytics and policies that sit on top of the
core execution and telemetry stack. All logic here should be:

- Read-only or pure functions where possible.
- Driven by existing logs and metrics (router_metrics, fills, PnL, etc.).
- Covered by focused pytest modules in tests/test_*intel*.py.
"""

__all__ = ["expectancy_map", "symbol_score", "maker_offset", "router_policy"]
