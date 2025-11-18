You are GPT Hedge Codex and must patch only the following components in the v5.10 repo:

GOALS:
1. Remove all legacy hard-coded 10% equity clamps from the screener.
2. Replace them with values read from config/risk_limits.json:
      global.trade_equity_nav_pct
      global.max_trade_nav_pct
3. Ensure risk_limits.check_order and signal_screener.py both use the same dynamic values.
4. Ensure detail_payload includes:
      trade_equity_nav_pct (limit)
      trade_equity_nav_obs (observation)
      max_trade_nav_pct (limit)
      max_trade_nav_obs (observation)
5. Ensure sizer treats the same normalized caps.
6. Ensure max_concurrent_positions in strategy_config is respected.
7. Ensure no fixed 10% limits remain anywhere in execution.

PATCH SCOPE:
- execution/signal_screener.py
- execution/risk_limits.py
- execution/size_model.py
- execution/executor_live.py
- tests/test_risk_limits.py
- tests/test_screener_tier_caps.py

ALL OTHER FILES MUST NOT BE MODIFIED.

BEHAVIOUR:
- Screener must veto when suggested_notional/nav exceeds trade_equity_nav_pct (now 0.15).
- Screener must veto when suggested_notional/nav exceeds max_trade_nav_pct (now 0.20).
- Screener must return structured detail payloads.
- Risk limits must match screener behaviour exactly.
- Tests must be updated to assert the new values (15% and 20%).

OUTPUT:
Return a unified diff patch with zero omissions.
Only include files with changes.
Follow repo style.
