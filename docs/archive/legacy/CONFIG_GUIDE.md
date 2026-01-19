# Config Guide (Live Trading)

## Key blocks
### `sizing`
- `min_gross_usd_per_order` — global per-order **gross** floor (USDT).
- `per_symbol_min_gross_usd` — map to override the floor for specific symbols (e.g., `"BTCUSDT": 150`).
- `max_gross_exposure_pct`, `max_symbol_exposure_pct`, `max_open_positions` — caps enforced by **RiskGate**.
- `default_leverage`, `per_symbol_leverage` — leverage map; risk checks use **gross** (notional × leverage).

### `ml`
- `lookback_bars` — **≤ 1500** (Binance USD-M cap). For deeper history, paginate.
- `prob_threshold` — probability gate; screener emits only if `p ≥ threshold` *and* all non-ML gates pass.

### `nav`
- Trading NAV (risk) uses **futures wallet** only. Reserves/off-exchange treasury are **not** included in risk.

### Environment flags
- `EVENT_GUARD=1` — runtime scaler to tighten portfolio/symbol caps during events.
- `FIRESTORE_ENABLED=0` — disable Firestore until ADC is configured.

## Examples
```json
"sizing": {
  "min_gross_usd_per_order": 110,
  "per_symbol_min_gross_usd": { "BTCUSDT": 150 },
  "max_gross_exposure_pct": 40,
  "max_symbol_exposure_pct": 20,
  "max_open_positions": 1
},
"ml": { "lookback_bars": 1500, "prob_threshold": 0.68 }
```
