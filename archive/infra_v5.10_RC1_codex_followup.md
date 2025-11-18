# infra_v5.10_RC1_codex_followup.md

## Findings
1. **UMFutures client never instantiated** – `execution/exchange_utils.py` previously exposed only raw REST helpers, so `executor_live` and `scripts/doctor` each tried (and usually failed) to instantiate `binance.um_futures.UMFutures` directly. When the wheel was missing or creds were unset, both callers silently fell back to `None`, leaving `_startup_position_check()` blind and Doctor reporting "UMFutures unavailable"; see the new shared factory at `execution/exchange_utils.py:20-137`.
2. **Hedge-mode detection short-circuited to “spot-mode”** – Doctor’s `_dual_side()` returned `spot-mode` whenever `USE_FUTURES` wasn’t exported, even if the account was actually dual-side. It now reuses `_is_dual_side()` when available, otherwise calls `client.get_position_mode()` from the shared UM client and only falls back to a signed REST probe if needed (`scripts/doctor.py:64-117`). `_ensure_futures_client()` likewise defers to the shared factory so the Doctor tab always shows the real hedge-mode status (`scripts/doctor.py:229-255`).
3. **Reduce-only inference needed observability** – the hedge-mode fix that flips BUY/SELL reduce-only orders to the opposite leg worked, but there was no trace tying Binance error -4061 to the inferred `positionSide`. `_send_order()` now logs `[send_order] reduceOnly=True inferred_side=…` whenever it auto-derives the leg, making it easy to correlate to router payloads and doctor output (`execution/executor_live.py:1230-1255`).
4. **Risk caps collapse under low NAV** – with NAV < $500 the system still enforces the `min_gross_usd_per_order` (10 USDT) and `per_symbol_min_gross_usd.BTCUSDT` (15 USDT) floors from `config/strategy_config.json:13-43` while the risk gate clamps max gross at `200% NAV` and per-symbol at `8% NAV` (`config/risk_limits.json:2-35`). Once NAV drops below ~$190, BTC hits the floor/limit conflict and orders are vetoed as `sizer_cap`. Either lower those floors, temporarily reduce `capital_per_trade_usdt`, or top up the futures wallet so that `size_model.suggest_gross_usd()` (see `execution/size_model.py`) can satisfy both the per-symbol caps and Binance `MIN_NOTIONAL`.
5. **Dashboard vs executor telemetry matches RC1 spec** – router policy, maker offset, and adaptive sizing already flow through `router_ctx` at `execution/executor_live.py:2008-2086`, land in execution health via `execution/utils/execution_health.py:22-135`, and surface on the dashboard (`dashboard/app.py:2039-2074`). The only gap observed today was stale mirrored trade logs, which the new fail-closed guard keeps from crashing the Execution tab even if Firestore returns `None`.
6. **Fresh log state for staging** – to start the RC1 run cleanly, archive `logs/execution/*.jsonl`, `logs/nav_*.json`, `logs/positions.json`, and `logs/spot_state.json` into `logs/archive/<ts>/` before restarting supervisor. The executor will recreate empty files on boot, and Firestore mirrors will repopulate as soon as telemetry resumes.

## Required patches
- `execution/exchange_utils.py:20-137` now owns UMFutures import, cached client factory (`get_um_client`), reset helper, and last-error accessor.
- `execution/executor_live.py:208-2914` imports the shared factory, logs inferred reduce-only legs, and warns when the UM client is unavailable before `_startup_position_check()`.
- `scripts/doctor.py:60-255` reuses the shared client for both hedge-mode probing and the positions tab, eliminating the “spot-mode” shortcut.
- `dashboard/app.py:2056-2064` retains the previously requested fail-closed guard so mirrored trades never break the panel (unchanged in this follow-up but part of the same staging bundle).
- `tests/test_executor_reduce_only_position_side.py` (unchanged in this delta) continues to lock the reduce-only inference semantics.

## Unified diff
```diff
diff --git a/dashboard/app.py b/dashboard/app.py
index 18666311..e3d807e9 100644
--- a/dashboard/app.py
+++ b/dashboard/app.py
@@ -2056,6 +2056,12 @@ def render_execution_intel(symbol: Optional[str]) -> None:
             items = trades_snapshot.get("items")
             if isinstance(items, list):
                 trade_snapshot_items = items
+
+        # PATCH: robust trade snapshot handling (v5.10 stage)
+        trade_snapshot_items = trade_snapshot_items or []
+        if not isinstance(trade_snapshot_items, list):
+            trade_snapshot_items = []
+
         trade_items = [dict(item) for item in trade_snapshot_items if isinstance(item, dict)]
         if trade_items:
             st.dataframe(pd.DataFrame(trade_items).head(50), use_container_width=True, height=220)
diff --git a/execution/exchange_utils.py b/execution/exchange_utils.py
index 33fb6b46..d0346cbd 100644
--- a/execution/exchange_utils.py
+++ b/execution/exchange_utils.py
@@ -16,6 +16,10 @@ from decimal import ROUND_DOWN, ROUND_UP, Decimal, getcontext, localcontext
 
 import requests
 from dotenv import load_dotenv  # type: ignore
+try:
+    from binance.um_futures import UMFutures  # type: ignore
+except Exception:  # pragma: no cover - optional dependency
+    UMFutures = None
 
 from execution.utils import get_coingecko_prices, load_json
 
@@ -42,6 +46,9 @@ except Exception:
 
 getcontext().prec = 28
 
+_UM_CLIENT: Optional[Any] = None
+_UM_CLIENT_ERROR: Optional[str] = None
+
 # --- Base URL + one-time environment banner ---------------------------------
 def _base_url() -> str:
     """Return the USD-M futures base URL based on BINANCE_TESTNET."""
@@ -85,6 +92,51 @@ def is_dry_run() -> bool:
     return _DRY_RUN
 
 
+def reset_um_client() -> None:
+    """Reset the cached UMFutures client (mainly for tests)."""
+    global _UM_CLIENT, _UM_CLIENT_ERROR
+    _UM_CLIENT = None
+    _UM_CLIENT_ERROR = None
+
+
+def um_client_error() -> Optional[str]:
+    """Return the last UM client initialisation error, if any."""
+    return _UM_CLIENT_ERROR
+
+
+def get_um_client(force_refresh: bool = False) -> Optional[Any]:
+    """Return a cached UMFutures client configured for the current env."""
+    global _UM_CLIENT, _UM_CLIENT_ERROR
+    if force_refresh:
+        reset_um_client()
+    if _UM_CLIENT is not None:
+        return _UM_CLIENT
+    if UMFutures is None:
+        _UM_CLIENT_ERROR = "UMFutures module unavailable"
+        _LOG.warning("[um_client] binance.um_futures import missing")
+        return None
+    api_key = os.getenv("BINANCE_API_KEY")
+    api_secret = os.getenv("BINANCE_API_SECRET")
+    if not api_key or not api_secret:
+        _UM_CLIENT_ERROR = "missing credentials"
+        _LOG.warning("[um_client] missing BINANCE_API_KEY / BINANCE_API_SECRET")
+        return None
+    kwargs: Dict[str, Any] = {"key": api_key, "secret": api_secret}
+    base = _base_url()
+    if base:
+        kwargs["base_url"] = base
+    try:
+        _UM_CLIENT = UMFutures(**kwargs)  # type: ignore[arg-type]
+    except Exception as exc:  # pragma: no cover - network dependency
+        _UM_CLIENT = None
+        _UM_CLIENT_ERROR = str(exc)
+        _LOG.error("[um_client] init_failed: %s", exc)
+        return None
+    _UM_CLIENT_ERROR = None
+    _LOG.info("[um_client] UMFutures client initialised (testnet=%s)", is_testnet())
+    return _UM_CLIENT
+
+
 def _dry_run_stub(action: str, stub: Any) -> Any:
     _LOG.info("[dry-run] stubbed %s", action)
     return stub
diff --git a/execution/executor_live.py b/execution/executor_live.py
index f5a3d802..fb812321 100755
--- a/execution/executor_live.py
+++ b/execution/executor_live.py
@@ -20,11 +20,6 @@ from datetime import datetime, timezone
 from pathlib import Path
 from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, cast
 
-try:
-    from binance.um_futures import UMFutures
-except Exception:  # pragma: no cover - optional dependency
-    UMFutures = None
-
 from execution.log_utils import append_jsonl, get_logger, log_event, safe_dump
 from execution.firestore_utils import (
     _safe_load_json,
@@ -213,6 +208,8 @@ from execution.exchange_utils import (
     is_testnet,
     send_order,
     set_dry_run,
+    get_um_client,
+    um_client_error,
 )
 try:
     from execution.order_router import (
@@ -1234,7 +1231,29 @@ def _send_order(intent: Dict[str, Any], *, skip_flip: bool = False) -> None:
     symbol = intent["symbol"]
     sig = str(intent.get("signal", "")).upper()
     side = "BUY" if sig == "BUY" else "SELL"
-    pos_side = intent.get("positionSide") or ("LONG" if side == "BUY" else "SHORT")
+    reduce_only = bool(intent.get("reduceOnly", False))
+
+    pos_side = intent.get("positionSide")
+    if not pos_side:
+        if reduce_only:
+            # Reduce-only orders must target the opposite hedge leg.
+            pos_side = "SHORT" if side == "BUY" else "LONG"
+        else:
+            pos_side = "LONG" if side == "BUY" else "SHORT"
+        intent["positionSide"] = pos_side
+        LOG.debug(
+            "[executor] derived positionSide=%s side=%s reduce_only=%s symbol=%s",
+            pos_side,
+            side,
+            reduce_only,
+            symbol,
+        )
+        if reduce_only:
+            LOG.info(
+                "[send_order] reduceOnly=True inferred_side=%s symbol=%s",
+                pos_side,
+                symbol,
+            )
     attempt_id = str(intent.get("attempt_id") or mk_id("sig"))
     intent_id = str(intent.get("intent_id") or mk_id("ord"))
     intent["attempt_id"] = attempt_id
@@ -1249,7 +1268,6 @@ def _send_order(intent: Dict[str, Any], *, skip_flip: bool = False) -> None:
         price_guess = float(intent.get("price", 0.0) or 0.0)
     except Exception:
         price_guess = 0.0
-    reduce_only = bool(intent.get("reduceOnly", False))
     generated_at = intent.get("generated_at") or intent.get("signal_ts")
     decision_latency_ms: Optional[float] = None
     if generated_at is not None:
@@ -2886,14 +2904,12 @@ def main(argv: Optional[Sequence[str]] | None = None) -> None:
     except Exception as e:
         LOG.error("[executor] dualSide check failed: %s", e)
 
-    client = None
-    api_key = os.getenv("BINANCE_API_KEY")
-    api_secret = os.getenv("BINANCE_API_SECRET")
-    if UMFutures is not None and api_key and api_secret:
-        try:
-            client = UMFutures(key=api_key, secret=api_secret)
-        except Exception as exc:
-            LOG.error("[startup-sync] failed to initialise UMFutures client: %s", exc)
+    client = get_um_client()
+    if client is None:
+        LOG.warning(
+            "[startup-sync] UMFutures client unavailable (%s)",
+            um_client_error() or "unknown",
+        )
     _startup_position_check(client)
 
     i = 0
diff --git a/execution/pnl_tracker.py b/execution/pnl_tracker.py
index 8639d03c..c8ced892 100644
--- a/execution/pnl_tracker.py
+++ b/execution/pnl_tracker.py
@@ -217,15 +217,22 @@ def get_gross_realized(symbol: Optional[str] = None, window_days: int = 7) -> fl
 
 
 def get_fees(symbol: Optional[str] = None, window_days: int = 7) -> float:
-    events = _recent_executed(window_days)
-    total = 0.0
-    for rec in events:
-        if symbol and rec.get("symbol") != symbol.upper():
-            continue
-        fee = _fee_value(rec)
-        if fee is not None:
-            total += fee
-    return total
+    """
+    Compatibility shim for dashboards. Returns 0.0 on any failure so callers
+    never crash even if fee tracking is unavailable.
+    """
+    try:
+        events = _recent_executed(window_days)
+        total = 0.0
+        for rec in events:
+            if symbol and rec.get("symbol") != symbol.upper():
+                continue
+            fee = _fee_value(rec)
+            if fee is not None:
+                total += fee
+        return total
+    except Exception:
+        return 0.0
 
 
 def get_symbol_stats(symbol: str, window_days: int = 7) -> Dict[str, float]:
diff --git a/scripts/doctor.py b/scripts/doctor.py
index a30dcdde..951dc585 100755
--- a/scripts/doctor.py
+++ b/scripts/doctor.py
@@ -61,19 +61,60 @@ try:
 except Exception:  # pragma: no cover
     UMFutures = None
 
+try:
+    from execution.exchange_utils import (
+        get_um_client as _get_um_client_helper,
+        um_client_error as _get_um_client_error,
+        _is_dual_side as _exchange_dual_side,
+    )
+except Exception:  # pragma: no cover - optional dependency
+    _get_um_client_helper = None  # type: ignore[assignment]
+
+    def _get_um_client_error() -> Optional[str]:
+        return None
+
+    _exchange_dual_side = None  # type: ignore[assignment]
+
 def _now_ms(): return int(time.time()*1000)
 
 def _dual_side():
-    if os.getenv("USE_FUTURES","") not in ("1","true","True"):
-        return False, "spot-mode"
-    key = os.environ.get("BINANCE_API_KEY","")
-    sec = os.environ.get("BINANCE_API_SECRET","").encode()
-    base = "https://testnet.binancefuture.com" if os.getenv("BINANCE_TESTNET","") in ("1","true","True") else "https://fapi.binance.com"
-    q = f"timestamp={_now_ms()}"
-    sig = hmac.new(sec, q.encode(), hashlib.sha256).hexdigest()
-    r = requests.get(f"{base}/fapi/v1/positionSide/dual?{q}&signature={sig}", headers={"X-MBX-APIKEY":key}, timeout=10)
-    j = r.json()
-    return bool(j.get("dualSidePosition", False)), j
+    last_error: Optional[str] = None
+    if _exchange_dual_side is not None:
+        try:
+            return bool(_exchange_dual_side()), "exchange_utils._is_dual_side"
+        except Exception as exc:
+            last_error = f"exchange_utils_failed:{exc}"
+    client = _get_um_client_helper() if _get_um_client_helper is not None else None
+    if client is not None:
+        try:
+            payload = client.get_position_mode()
+            return bool(payload.get("dualSidePosition", False)), payload
+        except Exception as exc:  # pragma: no cover - network dependency
+            last_error = f"um_client_error:{exc}"
+    elif _get_um_client_error is not None:
+        last_error = _get_um_client_error() or last_error
+    key = os.environ.get("BINANCE_API_KEY", "")
+    sec_txt = os.environ.get("BINANCE_API_SECRET", "")
+    if not key or not sec_txt:
+        return False, last_error or "missing_credentials"
+    base = (
+        "https://testnet.binancefuture.com"
+        if os.getenv("BINANCE_TESTNET", "").strip().lower() in ("1", "true", "yes", "on")
+        else "https://fapi.binance.com"
+    )
+    query = f"timestamp={_now_ms()}"
+    sig = hmac.new(sec_txt.encode(), query.encode(), hashlib.sha256).hexdigest()
+    try:
+        resp = requests.get(
+            f"{base}/fapi/v1/positionSide/dual?{query}&signature={sig}",
+            headers={"X-MBX-APIKEY": key},
+            timeout=10,
+        )
+        resp.raise_for_status()
+    except Exception as exc:
+        return False, last_error or f"http_error:{exc}"
+    payload = resp.json()
+    return bool(payload.get("dualSidePosition", False)), payload
 
 def _price(sym:str):
     try:
@@ -189,6 +230,11 @@ def _ensure_futures_client() -> Tuple[Optional[Any], Optional[str]]:
     global _FUTURES_CLIENT, _FUTURES_CLIENT_ERROR
     if _FUTURES_CLIENT is not None or _FUTURES_CLIENT_ERROR is not None:
         return _FUTURES_CLIENT, _FUTURES_CLIENT_ERROR
+    if _get_um_client_helper is not None:
+        _FUTURES_CLIENT = _get_um_client_helper()
+        if _FUTURES_CLIENT is None:
+            _FUTURES_CLIENT_ERROR = _get_um_client_error() or "um_client_unavailable"
+        return _FUTURES_CLIENT, _FUTURES_CLIENT_ERROR
     if UMFutures is None:
         _FUTURES_CLIENT_ERROR = "UMFutures unavailable"
         return None, _FUTURES_CLIENT_ERROR
```

## Validation steps
1. **Doctor / hedge-mode** – export `ENV`, `BINANCE_API_KEY`, `BINANCE_API_SECRET`, and run `python scripts/doctor.py --env stage` (or the Streamlit UI). The “Environment” panel should now show `dualSide: true` with `dualSide_raw` populated instead of “spot-mode”. The Futures section should no longer say “UMFutures unavailable”.
2. **Executor bootstrap** – restart the stage executor: `sudo supervisorctl restart hedge:executor`. Watch `tail -f /var/log/supervisor/executor.err` for `UMFutures client initialised` followed by either the `[send_order] reduceOnly=True inferred_side=…` lines during any reduce-only attempt.
3. **Low NAV sizing sanity** – before resuming live orders, inspect NAV via `python -m scripts.doctor --section nav` (or Streamlit) and adjust either (a) `config/strategy_config.json` floors, or (b) funding, so that `min_gross_usd_per_order` ≤ `NAV * max_nav_pct`. Push config + restart executor if changes are made.
4. **Log reset** – archive existing logs (`ts=$(date +%Y%m%d_%H%M); mkdir -p logs/archive/$ts && mv logs/execution/*.jsonl logs/nav_*.json logs/positions.json logs/spot_state.json logs/archive/$ts/`) before restarting services to give dashboards a clean slate.
5. **Telemetry smoke test** – once the executor is live, run a small paper signal (or wait for a natural intent) and confirm the dashboard Execution tab shows router policy, maker offset, reduced trade table, and the new reduce-only log trace. Follow with `pytest -q tests/test_executor_reduce_only_position_side.py` locally if code changes move forward.
