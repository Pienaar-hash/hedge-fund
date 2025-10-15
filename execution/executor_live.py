#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import requests
# Optional .env so Supervisor doesn't need to export everything
try:
    from dotenv import load_dotenv

    load_dotenv() or load_dotenv("/root/hedge-fund/.env")
except Exception:
    pass

LOG = logging.getLogger("exutil")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [exutil] %(message)s"
)

# ---- Exchange utils (binance) ----
from execution.exchange_utils import (
    _is_dual_side,
    build_order_payload,
    get_balances,
    get_positions,
    get_price,
    is_testnet,
    send_order,
    set_dry_run,
)
try:
    from execution.order_router import route_order as _route_order
except Exception:
    _route_order = None  # type: ignore[assignment]
from execution.risk_limits import RiskState, check_order, RiskGate
from execution.nav import compute_nav_pair, compute_treasury_only, PortfolioSnapshot
from execution.utils import (
    load_json,
    write_nav_snapshots_pair,
    write_treasury_snapshot,
    save_json,
)

# ---- Screener (best effort) ----
def _default_signal_source() -> Iterable[Dict[str, Any]]:
    return []


generate_signals_from_config: Callable[[], Iterable[Dict[str, Any]]] = _default_signal_source
try:
    from execution.signal_screener import generate_signals_from_config as _gen

    generate_signals_from_config = _gen
except Exception:
    pass

_generate_intents: Optional[
    Callable[[float, Sequence[str] | None, Mapping[str, Any] | None], List[Mapping[str, Any]]]
] = None
try:
    from execution.signal_generator import generate_intents as _imported_generate_intents
except Exception:
    _generate_intents = None
else:
    _generate_intents = _imported_generate_intents

# ---- Firestore publisher handle (revisions differ) ----
_PUB: Any
try:
    from execution.state_publish import (
        StatePublisher,
        publish_close_audit,
        publish_intent_audit,
        publish_order_audit,
    )

    _PUB = StatePublisher(interval_s=int(os.getenv("FS_PUB_INTERVAL", "60")))
except Exception:

    class _Publisher:
        def __init__(self, interval_s: int = 60):
            self.interval_s = interval_s

    def publish_intent_audit(intent: Dict[Any, Any]) -> None:
        pass

    def publish_order_audit(symbol: str, event: Dict[Any, Any]) -> None:
        pass

    def publish_close_audit(
        symbol: str, position_side: str, event: Dict[Any, Any]
    ) -> None:
        pass

    _PUB = _Publisher(interval_s=int(os.getenv("FS_PUB_INTERVAL", "60")))

# ---- risk limits config ----
_RISK_CFG_PATH = os.getenv("RISK_LIMITS_CONFIG", "config/risk_limits.json")
_RISK_CFG: Dict[str, Any] = {}
try:
    with open(_RISK_CFG_PATH, "r") as fh:
        _RISK_CFG = json.load(fh) or {}
except Exception as e:
    logging.getLogger("exutil").warning(
        "[risk] config load failed (%s): %s", _RISK_CFG_PATH, e
    )

_RISK_STATE = RiskState()

# Build a minimal RiskGate config adapter from legacy risk_limits.json
def _mk_gate_cfg(raw: Dict[str, Any]) -> Dict[str, Any]:
    g = (raw.get("global") or {}) if isinstance(raw, dict) else {}
    # Legacy configs kept caps at top-level; prefer `global` section when present
    if not g and isinstance(raw, dict):
        for key in ("daily_loss_limit_pct", "max_gross_exposure_pct"):
            if key in raw:
                g[key] = raw[key]
    sizing = {
        "min_notional_usdt": g.get("min_notional_usdt", 5.0),
        # Accept either key for portfolio gross NAV cap
        "max_gross_exposure_pct": (
            g.get("max_gross_exposure_pct")
            or g.get("max_portfolio_gross_nav_pct")
            or g.get("max_gross_nav_pct")
            or 0.0
        ),
        # Fallback sensible default if not provided
        "max_symbol_exposure_pct": g.get("max_symbol_exposure_pct", 50.0),
        "max_trade_nav_pct": g.get("max_trade_nav_pct", 0.0),
    }
    risk = {
        "daily_loss_limit_pct": g.get("daily_loss_limit_pct", 5.0),
        # Derive cooldown (minutes) from error circuit cooldown (seconds) if present
        "cooldown_minutes_after_stop": (
            g.get("cooldown_minutes_after_stop")
            or max(
                0.0,
                float(((g.get("error_circuit") or {}).get("cooldown_sec") or 0))
                / 60.0,
            )
        ),
        "max_trades_per_symbol_per_hour": g.get("max_trades_per_symbol_per_hour", 6),
    }
    return {"sizing": sizing, "risk": risk}

_RISK_GATE = RiskGate(_mk_gate_cfg(_RISK_CFG))
_PORTFOLIO_SNAPSHOT = PortfolioSnapshot(load_json("config/strategy_config.json"))
_RISK_GATE.nav_provider = _PORTFOLIO_SNAPSHOT

# ---- knobs ----
SLEEP = int(os.getenv("LOOP_SLEEP", "60"))
MAX_LOOPS = int(os.getenv("MAX_LOOPS", "0") or 0)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "describe", "--tags", "--always"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _read_dry_run_flag() -> bool:
    return os.getenv("DRY_RUN", "1").lower() in ("1", "true", "yes")


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


DRY_RUN = _read_dry_run_flag()
set_dry_run(DRY_RUN)
INTENT_TEST = _truthy_env("INTENT_TEST", "0")

LOG.info(
    "[executor] starting loop ENV=%s DRY_RUN=%s commit=%s",
    os.getenv("ENV", "dev"),
    DRY_RUN,
    _git_commit(),
)


def _sync_dry_run() -> None:
    global DRY_RUN
    current = _read_dry_run_flag()
    if current != DRY_RUN:
        LOG.info("[executor] DRY_RUN flag changed -> %s", current)
        DRY_RUN = current
    set_dry_run(DRY_RUN)

# ------------- helpers -------------


def _account_snapshot() -> None:
    try:
        bals = get_balances() or []
        assets = sorted({b.get("asset", "?") for b in bals})
        pos = [
            p
            for p in get_positions()
            if float(p.get("qty", p.get("positionAmt", 0)) or 0) != 0
        ]
        LOG.info(
            "[executor] account OK — futures=%s testnet=%s dry_run=%s balances: %s positions: %d",
            True,
            is_testnet(),
            DRY_RUN,
            assets,
            len(pos),
        )
    except Exception as e:
        LOG.exception("[executor] preflight_error: %s", e)


def _compute_nav() -> float:
    cfg = load_json("config/strategy_config.json") or {}

    try:
        trading, reporting = compute_nav_pair(cfg)
        write_nav_snapshots_pair(trading, reporting)
        tre_val, tre_detail = compute_treasury_only()
        write_treasury_snapshot(tre_val, tre_detail)
        return float(trading[0])
    except Exception as exc:
        LOG.error("[executor] compute_nav not available: %s", exc)

    nav_val = 0.0
    try:
        from execution.exchange_utils import get_account

        acc = get_account()
        nav_val = float(
            acc.get("totalMarginBalance")
            or (
                float(acc.get("totalWalletBalance", 0) or 0)
                + float(acc.get("totalUnrealizedProfit", 0) or 0)
            )
        )
    except Exception as e:
        LOG.error("[executor] account NAV error: %s", e)
    # As a last resort try capital_base
    if not nav_val and cfg:
        return float(cfg.get("capital_base_usdt", 0.0) or 0.0)
    return float(nav_val or 0.0)


def _gross_and_open_qty(
    symbol: str, pos_side: str, positions: Iterable[Dict[str, Any]]
) -> tuple[float, float]:
    gross = 0.0
    open_qty = 0.0
    for p in positions or []:
        try:
            qty = float(p.get("qty", p.get("positionAmt", 0)) or 0.0)
            entry = float(p.get("entryPrice") or 0.0)
            gross += abs(qty) * abs(entry)
            if str(p.get("symbol")) == symbol:
                ps = p.get("positionSide", "BOTH")
                if ps == pos_side or (ps == "BOTH" and pos_side in ("LONG", "SHORT")):
                    open_qty = max(open_qty, abs(qty))
        except Exception:
            continue
    return gross, open_qty


def _opposite_position(
    symbol: str, desired_side: str, positions: Iterable[Dict[str, Any]]
) -> tuple[str | None, float, float]:
    sym = str(symbol).upper()
    desired = str(desired_side).upper()
    opposite = "SHORT" if desired == "LONG" else "LONG"
    for p in positions or []:
        try:
            if str(p.get("symbol", "")).upper() != sym:
                continue
            ps = str(p.get("positionSide", "BOTH")).upper()
            if ps != opposite:
                continue
            qty = float(p.get("qty", p.get("positionAmt", 0.0)) or 0.0)
            if qty == 0.0:
                continue
            mark = float(p.get("markPrice") or p.get("entryPrice") or 0.0)
            return opposite, abs(qty), abs(mark)
        except Exception:
            continue
    return None, 0.0, 0.0


def _update_risk_state_counters(
    positions: Iterable[Dict[str, Any]],
    portfolio_gross: float,
) -> None:
    open_positions = 0
    for p in positions or []:
        try:
            qty = float(p.get("qty", p.get("positionAmt", 0.0)) or 0.0)
        except Exception:
            qty = 0.0
        if qty != 0.0:
            open_positions += 1

    _RISK_STATE.open_notional = float(portfolio_gross)
    _RISK_STATE.open_positions = int(open_positions)
    try:
        loss_pct = float(_RISK_GATE._daily_loss_pct())
    except Exception:
        loss_pct = 0.0
    _RISK_STATE.daily_pnl_pct = -loss_pct


def _send_order(intent: Dict[str, Any], *, skip_flip: bool = False) -> None:
    symbol = intent["symbol"]
    sig = str(intent.get("signal", "")).upper()
    side = "BUY" if sig == "BUY" else "SELL"
    pos_side = intent.get("positionSide") or ("LONG" if side == "BUY" else "SHORT")
    cap = float(intent.get("capital_per_trade", 0) or 0)
    lev = float(intent.get("leverage", 1) or 1)
    gross_target = float(intent.get("gross_usd") or (cap * lev))
    if lev <= 0:
        lev = 1.0
    try:
        cfg = load_json("config/strategy_config.json")
        sizing_cfg = (cfg.get("sizing") or {})
        floor_gross = float((sizing_cfg.get("min_gross_usd_per_order", 0.0)) or 0.0)
        per_symbol_cfg = sizing_cfg.get("per_symbol_min_gross_usd") or {}
        sym_floor = per_symbol_cfg.get(symbol.upper())
        if sym_floor is not None:
            floor_gross = max(floor_gross, float(sym_floor or 0.0))
        gross_target = max(gross_target, floor_gross)
    except Exception:
        pass
    margin_target = gross_target / max(lev, 1.0)
    reduce_only = bool(intent.get("reduceOnly", False))

    def _persist_veto(reason: str, price_hint: float, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "positionSide": pos_side,
            "reason": reason,
            "gross_usd": gross_target,
            "price": price_hint,
            "ts": time.time(),
        }
        if extra:
            payload.update(extra)
        try:
            save_json(f"logs/veto_exec_{symbol}.json", payload)
        except Exception:
            try:
                import pathlib as _pl

                _pl.Path("logs").mkdir(exist_ok=True)
                _pl.Path(f"logs/veto_exec_{symbol}.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return payload

    try:
        _PORTFOLIO_SNAPSHOT.refresh()
    except Exception:
        pass

    if os.environ.get("KILL_SWITCH", "0").lower() in ("1", "true", "yes", "on"):
        price_hint = float(intent.get("price", 0.0) or 0.0)
        LOG.warning("[risk] kill switch active; veto %s %s", symbol, side)
        _persist_veto("kill_switch_on", price_hint, {"intent": intent})
        try:
            publish_order_audit(
                symbol,
                {
                    "phase": "blocked",
                    "side": side,
                    "positionSide": pos_side,
                    "reason": "kill_switch_on",
                    "notional": gross_target,
                },
            )
        except Exception:
            pass
        return

    LOG.info(
        "[executor] INTENT symbol=%s side=%s ps=%s margin=%.4f gross=%.4f lev=%.2f reduceOnly=%s",
        symbol,
        side,
        pos_side,
        margin_target,
        gross_target,
        lev,
        reduce_only,
    )

    if not reduce_only:
        # Shared gross-notional gate (canonical taxonomy)
        try:
            allowed, veto = _RISK_GATE.allowed_gross_notional(symbol, gross_target)
        except Exception:
            allowed, veto = True, ""
        if not allowed:
            price_hint = float(intent.get("price", 0.0) or 0.0)
            _persist_veto(veto or "blocked", price_hint, {"intent": intent})
            return

    try:
        positions = list(get_positions() or [])
    except Exception:
        positions = []

    force_direct_send = False

    # Flip handling: flatten any opposing hedge-mode position via a dedicated
    # reduce-only order, then fall through to emit the new opening leg.
    if not reduce_only and not skip_flip:
        opp_side, opp_qty, opp_mark = _opposite_position(symbol, pos_side, positions)
        if opp_side and opp_qty > 0:
            reduce_price = opp_mark if opp_mark > 0 else float(intent.get("price", 0.0) or 0.0)
            if reduce_price <= 0:
                try:
                    reduce_price = float(get_price(symbol))
                except Exception:
                    reduce_price = 0.0
            if reduce_price <= 0:
                LOG.warning("Invalid reduce_price, skipping reduce-only intent")
                return

            reduce_notional = abs(opp_qty) * reduce_price
            reduce_signal = "BUY" if opp_side == "SHORT" else "SELL"
            try:
                reduce_payload, reduce_meta = build_order_payload(
                    symbol=symbol,
                    side=reduce_signal,
                    price=reduce_price,
                    desired_gross_usd=reduce_notional,
                    reduce_only=True,
                    position_side=opp_side,
                )
            except Exception as exc:
                LOG.error("[executor] reduce_only_build_failed %s %s", symbol, exc)
                return

            LOG.info(
                "[executor] flip flatten symbol=%s side=%s notional=%.4f",
                symbol,
                opp_side,
                reduce_notional,
            )
            try:
                send_order(**reduce_payload)
            except Exception as exc:
                LOG.error("[executor] reduce_only_send_failed %s %s", symbol, exc)
                return
            try:
                publish_order_audit(
                    symbol,
                    {
                        "phase": "flip_reduce",
                        "side": reduce_signal,
                        "positionSide": opp_side,
                        "payload": {
                            k: reduce_payload[k]
                            for k in ("type", "quantity", "reduceOnly", "positionSide")
                            if k in reduce_payload
                        },
                        "normalized": {
                            "price": reduce_meta.get("normalized_price"),
                            "qty": reduce_meta.get("normalized_qty"),
                        },
                    },
                )
            except Exception:
                pass

            try:
                positions = list(get_positions() or [])
            except Exception:
                positions = []
            opp_after_side, opp_after_qty, _ = _opposite_position(symbol, pos_side, positions)
            if opp_after_qty > 0:
                LOG.warning(
                    "[executor] flip flatten incomplete symbol=%s side=%s qty=%.6f",
                    symbol,
                    opp_after_side,
                    opp_after_qty,
                )
                return
            force_direct_send = True
    nav = _compute_nav()
    current_gross, sym_open_qty = _gross_and_open_qty(symbol, pos_side, positions)
    _update_risk_state_counters(positions, current_gross)

    try:
        _RISK_STATE.note_attempt(time.time())
    except Exception:
        pass

    if reduce_only:
        ok = True
        details: Dict[str, Any] = {"reasons": []}
    else:
        ok, details = check_order(
            symbol=symbol,
            side=side,
            requested_notional=gross_target,
            price=0.0,
            nav=nav,
            open_qty=sym_open_qty,
            now=time.time(),
            cfg=_RISK_CFG,
            state=_RISK_STATE,
            current_gross_notional=current_gross,
            lev=lev,
        )
    reasons = details.get("reasons", []) if isinstance(details, dict) else []
    if not ok:
        reason = reasons[0] if reasons else "blocked"
        price_hint = float(intent.get("price", 0.0) or 0.0)
        block_info = {
            "symbol": symbol,
            "side": side,
            "reason": reason,
            "notional": gross_target,
            "price": price_hint,
        }
        LOG.warning("[risk] block %s", block_info)
        try:
            from execution.telegram_utils import send_telegram

            send_telegram(
                f"Risk‑block {symbol} {side}: {reason}\nnotional={gross_target:.2f} price={price_hint:.2f}",
                silent=True,
            )
        except Exception:
            pass
        try:
            audit = {
                "phase": "blocked",
                "side": side,
                "positionSide": pos_side,
                "reason": reason,
                "reasons": reasons,
                "notional": gross_target,
                "nav": nav,
                "open_qty": sym_open_qty,
                "gross": current_gross,
            }
            if isinstance(details, dict) and "cooldown_until" in details:
                audit["cooldown_until"] = details.get("cooldown_until")
            publish_order_audit(symbol, audit)
        except Exception:
            pass
        _persist_veto(reason, price_hint, {"reasons": reasons, "intent": intent})
        return

    price_hint = float(intent.get("price", 0.0) or 0.0)
    if price_hint <= 0:
        try:
            price_hint = float(get_price(symbol))
        except Exception as exc:
            LOG.error("[executor] price_fetch_err %s %s", symbol, exc)
            return
    try:
        payload, meta = build_order_payload(
            symbol=symbol,
            side=side,
            price=price_hint,
            desired_gross_usd=gross_target,
            reduce_only=reduce_only,
            position_side=pos_side,
        )
    except Exception as exc:
        LOG.error(
            "[executor] SIZE_ERR %s side=%s gross=%.4f err=%s",
            symbol,
            side,
            gross_target,
            exc,
        )
        try:
            publish_order_audit(
                symbol,
                {
                    "phase": "size_error",
                    "side": side,
                    "positionSide": pos_side,
                    "error": str(exc),
                    "requested_gross": gross_target,
                },
            )
        except Exception:
            pass
        return

    def _meta_float(val: Any, fallback: float) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return fallback

    if not reduce_only:
        # Ensure the opening order never carries the reduceOnly flag
        payload.pop("reduceOnly", None)

    norm_price = _meta_float(meta.get("normalized_price"), price_hint)
    norm_qty = _meta_float(meta.get("normalized_qty"), _meta_float(payload.get("quantity"), 0.0))
    payload_view = {
        k: payload[k]
        for k in ("type", "quantity", "reduceOnly", "positionSide")
        if k in payload
    }

    if not reduce_only:
        try:
            allowed_now, veto_now = _RISK_GATE.allowed_gross_notional(symbol, gross_target)
        except Exception:
            allowed_now, veto_now = True, ""
        if not allowed_now:
            LOG.warning(
                "[risk] final gate veto %s %s reason=%s", symbol, side, veto_now or "blocked"
            )
            _persist_veto(veto_now or "blocked", norm_price, {
                "intent": intent,
                "normalized_qty": norm_qty,
                "payload": payload_view,
                "meta": meta,
            })
            try:
                publish_order_audit(
                    symbol,
                    {
                        "phase": "blocked",
                        "side": side,
                        "positionSide": pos_side,
                        "reason": veto_now or "blocked",
                        "notional": gross_target,
                    },
                )
            except Exception:
                pass
            return

    normalized_ctx = {"price": norm_price, "qty": norm_qty, **meta}

    try:
        publish_intent_audit(
            {
                **intent,
                "t": time.time(),
                "side": side,
                "positionSide": pos_side,
                "notional": gross_target,
                "reduceOnly": reduce_only,
                "normalized": normalized_ctx,
            }
        )
    except Exception:
        pass

    if DRY_RUN:
        LOG.info("[executor] DRY_RUN — skipping SEND_ORDER")
        try:
            publish_order_audit(
                symbol,
                {
                    "phase": "dry_run",
                    "side": side,
                    "positionSide": pos_side,
                    "notional": gross_target,
                    "normalized": normalized_ctx,
                    "payload": payload_view,
                },
            )
        except Exception:
            pass
        return

    LOG.info("[executor] SEND_ORDER %s %s payload=%s meta=%s", symbol, side, payload_view, meta)
    try:
        publish_order_audit(
            symbol,
            {
                "phase": "request",
                "side": side,
                "positionSide": pos_side,
                "notional": gross_target,
                "reduceOnly": reduce_only,
                "normalized": normalized_ctx,
                "payload": payload_view,
            },
        )
    except Exception:
        pass

    resp: Dict[str, Any] = {}
    router_error: Optional[str] = None

    if force_direct_send:
        try:
            resp = send_order(**payload)
        except Exception as exc:
            router_error = str(exc)
            LOG.error("[executor] flip open send_failed %s %s", symbol, exc)
            return
        try:
            publish_order_audit(
                symbol,
                {
                    "phase": "flip_open",
                    "side": side,
                    "positionSide": pos_side,
                    "payload": {
                        k: payload[k]
                        for k in ("type", "quantity", "positionSide")
                        if k in payload
                    },
                    "normalized": {"price": norm_price, "qty": norm_qty},
                },
            )
        except Exception:
            pass

    if _route_order is not None and not force_direct_send:
        router_intent = {
            **intent,
            "symbol": symbol,
            "side": side,
            "positionSide": pos_side,
            "reduceOnly": reduce_only,
            "quantity": payload.get("quantity"),
            "type": payload.get("type"),
            "price": payload.get("price", price_hint),
        }
        router_ctx = {
            "payload": payload,
            "price": price_hint,
            "positionSide": pos_side,
            "reduceOnly": reduce_only,
        }
        try:
            result = _route_order(router_intent, router_ctx, DRY_RUN)
        except Exception as exc:
            router_error = str(exc)
        else:
            if result.get("accepted"):
                resp = result.get("raw") or {}
                if result.get("price") is not None:
                    resp.setdefault("avgPrice", result.get("price"))
                if result.get("qty") is not None:
                    resp.setdefault("executedQty", str(result.get("qty")))
            else:
                router_error = str(result.get("reason") or "router_reject")

    if not resp:
        if router_error and not force_direct_send:
            LOG.error(
                "[executor] router_reject %s %s reason=%s", symbol, side, router_error
            )
        try:
            resp = send_order(**payload)
        except requests.HTTPError as exc:
            try:
                _RISK_STATE.note_error(time.time())
            except Exception:
                pass
            err_code = None
            try:
                if exc.response is not None:
                    err_code = exc.response.json().get("code")
            except Exception:
                err_code = None
            LOG.error(
                "[executor] ORDER_ERR code=%s symbol=%s side=%s meta=%s payload=%s err=%s",
                err_code,
                symbol,
                side,
                meta,
                payload_view,
                exc,
            )
            try:
                publish_order_audit(
                    symbol,
                    {
                        "phase": "error",
                        "side": side,
                        "positionSide": pos_side,
                        "error": str(exc),
                        "code": err_code,
                        "normalized": normalized_ctx,
                        "payload": payload_view,
                    },
                )
            except Exception:
                pass
            if err_code == -1111:
                LOG.error(
                    "[executor] ORDER_PRECISION ctx=%s payload=%s",
                    normalized_ctx,
                    payload_view,
                )
                return
            raise
        except Exception as exc:
            try:
                _RISK_STATE.note_error(time.time())
            except Exception:
                pass
            try:
                publish_order_audit(
                    symbol,
                    {
                        "phase": "error",
                        "side": side,
                        "positionSide": pos_side,
                        "error": str(exc),
                    },
                )
            except Exception:
                pass
            raise

    oid = resp.get("orderId")
    avg = resp.get("avgPrice", "0.00")
    qty = resp.get("executedQty", resp.get("origQty", "0"))
    st = resp.get("status")
    LOG.info("[executor] ORDER_REQ 200 id=%s avgPrice=%s qty=%s", oid, avg, qty)
    try:
        publish_order_audit(
            symbol,
            {
                "phase": "response",
                "side": side,
                "positionSide": pos_side,
                "status": st,
                "orderId": oid,
                "avgPrice": avg,
                "qty": qty,
                "normalized": normalized_ctx,
                "payload": payload_view,
            },
        )
    except Exception:
        pass

    if st == "FILLED":
        LOG.info(
            "[executor] ORDER_FILL id=%s status=%s avgPrice=%s qty=%s",
            oid,
            st,
            avg,
            qty,
        )
        try:
            _RISK_STATE.note_fill(symbol, time.time())
        except Exception:
            pass
        if (
            reduce_only
            or (side == "SELL" and pos_side == "LONG")
            or (side == "BUY" and pos_side == "SHORT")
        ):
            try:
                publish_close_audit(
                    symbol,
                    pos_side,
                    {"orderId": oid, "avgPrice": avg, "qty": qty, "status": st},
                )
            except Exception:
                pass



def _pub_tick() -> None:
    """
    Try StatePublisher.{tick,run_once,step,publish,update}; if no such method,
    publish a minimal NAV+positions snapshot inline so the dashboard stays live.
    """
    # Try common method names on the object first
    for meth in ("tick", "run_once", "step", "publish", "update"):
        m = getattr(_PUB, meth, None)
        if callable(m):
            try:
                m()
                return
            except Exception as e:
                LOG.error("[executor] publisher.%s error: %s", meth, e)

    # Inline minimal publish (NAV + positions) — Firestore-first dashboard compatible
    try:
        nav_val = None
        try:
            from execution.state_publish import compute_nav

            nav_val = float(compute_nav())
        except Exception as e:
            LOG.error("[executor] compute_nav not available: %s", e)
            try:
                from execution.exchange_utils import get_account

                acc = get_account()
                nav_val = float(
                    acc.get("totalMarginBalance")
                    or (
                        float(acc.get("totalWalletBalance", 0) or 0)
                        + float(acc.get("totalUnrealizedProfit", 0) or 0)
                    )
                )
            except Exception as ee:
                LOG.error("[executor] account NAV error: %s", ee)

        # Positions -> normalized rows
        try:
            raw: Iterable[Dict[str, Any]] = get_positions()
        except Exception as e:
            LOG.error("[executor] get_positions error: %s", e)
            raw = []
        rows: List[Dict[str, Any]] = []
        for p in raw:
            try:
                rows.append(
                    {
                        "symbol": p.get("symbol"),
                        "positionSide": p.get("positionSide", "BOTH"),
                        "qty": float(p.get("qty", p.get("positionAmt", 0)) or 0.0),
                        "entryPrice": float(p.get("entryPrice") or 0.0),
                        "unrealized": float(
                            p.get("unRealizedProfit", p.get("unrealized", 0)) or 0.0
                        ),
                        "leverage": float(p.get("leverage") or 0.0),
                    }
                )
            except Exception:
                continue

        # Firestore write
        try:
            if os.environ.get("FIRESTORE_ENABLED", "1") == "0":
                try:
                    from execution.state_publish import publish_nav_value, publish_positions

                    if rows:
                        publish_positions(rows)
                    if nav_val is not None:
                        publish_nav_value(float(nav_val))
                except Exception as inner:
                    LOG.error("[executor] local publish fallback error: %s", inner)
                return

            from google.cloud import firestore

            ENV = os.getenv("ENV", "prod")
            db = firestore.Client()

            if nav_val is not None:
                doc_nav = db.document(f"hedge/{ENV}/state/nav")
                snap = doc_nav.get()
                series: List[Dict[str, Any]] = []
                if snap.exists:
                    d = snap.to_dict() or {}
                    series = d.get("series") or d.get("rows") or []
                series.append({"t": time.time(), "nav": float(nav_val)})
                series = series[-20000:]
                doc_nav.set({"series": series}, merge=False)

            db.document(f"hedge/{ENV}/state/positions").set({"rows": rows}, merge=False)
        except Exception as e:
            LOG.error("[executor] Firestore publish error: %s", e)
    except Exception as e:
        LOG.error("[executor] publisher fallback error: %s", e)


def _loop_once(i: int) -> None:
    _sync_dry_run()
    _account_snapshot()

    try:
        baseline_positions = list(get_positions() or [])
    except Exception:
        baseline_positions = []
    gross_total, _ = _gross_and_open_qty("", "", baseline_positions)
    _update_risk_state_counters(baseline_positions, gross_total)

    if INTENT_TEST:
        intent = {
            "symbol": "BTCUSDT",
            "signal": "BUY",
            "capital_per_trade": 120.0,
            "leverage": 1,
            "positionSide": "LONG",
            "reduceOnly": False,
        }
        LOG.info("[screener->executor] %s", intent)
        _send_order(intent)
    else:
        if callable(generate_signals_from_config):
            try:
                for intent in generate_signals_from_config() or []:
                    LOG.info("[screener->executor] %s", intent)
                    _send_order(intent)
            except Exception as e:
                LOG.error("[screener] error: %s", e)
        else:
            LOG.error("[screener] missing signal generator")

    if callable(_generate_intents):
        try:
            cfg = load_json("config/strategy_config.json") or {}
        except Exception:
            cfg = {}
        universe = cfg.get("universe") or []
        try:
            auto_intents = list(_generate_intents(time.time(), universe, cfg) or [])
        except Exception as e:
            LOG.error("[signal-gen] error: %s", e)
            auto_intents = []
        for auto_intent in auto_intents:
            LOG.info("[signal-gen] %s", auto_intent)
            _send_order(dict(auto_intent))

    _pub_tick()


def main() -> None:
    _sync_dry_run()
    try:
        if not _is_dual_side():
            LOG.warning("[executor] WARNING — account not in hedge (dualSide) mode")
    except Exception as e:
        LOG.error("[executor] dualSide check failed: %s", e)

    i = 0
    while True:
        _loop_once(i)
        i += 1
        if MAX_LOOPS and i >= MAX_LOOPS:
            LOG.info("[executor] MAX_LOOPS reached — exiting.")
            break
        time.sleep(SLEEP)


if __name__ == "__main__":
    main()
