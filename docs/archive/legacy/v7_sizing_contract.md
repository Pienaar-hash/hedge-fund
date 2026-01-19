## New Sizing Contract (v6.2)

- Screener is the single source of sizing. Each intent carries `gross_usd`, `qty`, `per_trade_nav_pct`, `min_notional`, `leverage`, and sizing context (floors, nav_used, price_used).
- Executor is a pass-through: it validates required fields, snaps price/qty to exchange filters, and forwards to the risk engine. It must not re-size or re-cap.
- RiskEngineV6 / risk_limits are the only cap enforcers (min_notional_usdt, max_order_notional, per-symbol max_nav_pct, trade_equity_nav_pct, max_trade_nav_pct, portfolio gross, tier caps, leverage, cooldown, nav freshness).
- Shadow pipeline consumes the screener-sized intent and runs risk + router for telemetry; it does not re-size.
- NAV source is unified via `nav_health_snapshot.nav_total`; stale NAV is vetoed in the risk layer.
