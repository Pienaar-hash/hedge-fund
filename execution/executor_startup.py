from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

_LOG = logging.getLogger("executor_startup")


def acquire_startup_lock(acquire_executor_lock: Callable[[], int | None]) -> int | None:
    return acquire_executor_lock()


def check_startup_twap_recovery(
    *,
    check_twap_recovery: Callable[[], dict[str, Any] | None],
    clear_twap_state: Callable[[], None],
    logger: Any,
) -> None:
    stale_twap = check_twap_recovery()
    if not stale_twap:
        return
    logger.warning(
        "[executor] TWAP_RECOVERY: incomplete TWAP found on startup — "
        "symbol=%s side=%s completed=%d/%d started_ts=%s. "
        "State file cleared; operator should reconcile.",
        stale_twap.get("symbol"),
        stale_twap.get("side"),
        len(stale_twap.get("completed_slices", [])),
        stale_twap.get("total_slices"),
        stale_twap.get("started_ts"),
    )
    clear_twap_state()


def sync_dry_run(
    *,
    current: bool,
    previous: bool,
    logger: Any,
    set_dry_run: Callable[[bool], None],
    check_calibration_window: Callable[[], dict[str, Any]] | None = None,
    check_activation_window: Callable[[], dict[str, Any]] | None = None,
) -> bool:
    if current != previous:
        logger.info("[executor] DRY_RUN flag changed -> %s", current)
    set_dry_run(current)
    if check_calibration_window is not None:
        try:
            cw_status = check_calibration_window()
            if cw_status.get("halted"):
                logger.info(
                    "[calibration_window] episode cap reached (%d/%d) — KILL_SWITCH active",
                    cw_status.get("episodes_completed", 0),
                    cw_status.get("episode_cap", 0),
                )
        except Exception:
            pass
    if check_activation_window is not None:
        try:
            aw_status = check_activation_window()
            if aw_status.get("halted"):
                logger.info(
                    "[activation_window] HALT — %s (day %.1f/%d)",
                    aw_status.get("halt_reason", "unknown"),
                    aw_status.get("elapsed_days", 0),
                    aw_status.get("duration_days", 14),
                )
        except Exception:
            pass
    return current


def clean_testnet_caches(*, repo_root: str, testnet_enabled: bool, logger: Any) -> None:
    if not testnet_enabled:
        return
    cache_paths = [
        Path(repo_root) / "logs" / "cache" / "risk_state.json",
        Path(repo_root) / "logs" / "cache" / "nav_confirmed.json",
    ]
    for path in cache_paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.debug("[executor][testnet] cache cleanup skipped for %s", path)
    logger.info("[executor][testnet] cleaned stale risk/nav cache for fresh start")


def run_order_reconciliation(
    *,
    cancel_all_open_orders: Callable[[], Any],
    log_path: str = "logs/execution/orders_executed.jsonl",
    lookback_seconds: float = 4 * 3600,
    logger: Any = None,
) -> None:
    """
    Startup reconciliation for crash-window fill gaps.

    Sequence:
        1. Scan recent unresolved order_ack events and backfill synthetic
           order_fill / order_close events for trades that happened during
           the crash window.
        2. Cancel all remaining open orders (those with no fills are simply
           dead; those with partial fills already have their fills recorded).
        3. Write a reconciliation_summary event to the execution log.

    This must run BEFORE the main executor loop begins placing new orders.
    """
    log = logger or _LOG
    try:
        from execution.startup_reconciliation import reconcile_unresolved_acks
        from execution.events import write_event
        summary = reconcile_unresolved_acks(
            log_path=log_path,
            lookback_seconds=lookback_seconds,
        )
        log.info(
            "[startup-reconcile] done: unresolved=%d cases=%s errors=%d",
            summary.get("unresolved_count", 0),
            summary.get("case_counts", {}),
            len(summary.get("errors", [])),
        )
    except Exception as exc:
        log.warning("[startup-reconcile] failed, continuing: %s", exc, exc_info=True)
        summary = {"event_type": "reconciliation_summary", "error": str(exc)}

    try:
        cancel_all_open_orders()
    except Exception as exc:
        log.warning("[startup-reconcile] cancel_all_open_orders failed: %s", exc)

    try:
        write_event("reconciliation_summary", summary)
    except Exception as exc:
        log.debug("[startup-reconcile] summary_write_failed: %s", exc)


def startup_position_check(
    client: Any,
    *,
    logger: Any,
    get_live_positions: Callable[[Any], list[dict[str, Any]]],
    sync_tp_sl_registry: Callable[[list[dict[str, Any]]], None],
    allow_open_positions: bool,
    sleep_fn: Callable[[int], None],
    retry_interval: int = 30,
    max_exchange_retries: int = 5,
) -> None:
    if client is None or getattr(client, "is_stub", False):
        logger.info("[startup-sync] unable to check positions (client unavailable)")
        return

    logger.info("[startup-sync] checking open positions …")
    first_warning = True
    exchange_retries = 0
    while True:
        try:
            client.get_position_risk()
            break
        except Exception as exc:
            exchange_retries += 1
            if exchange_retries > max_exchange_retries:
                raise RuntimeError(
                    f"[startup-sync] AUDIT-1.3d: exchange unreachable after "
                    f"{max_exchange_retries} retries — cannot reconcile positions"
                ) from exc
            logger.warning(
                "[startup-sync] exchange probe failed (attempt %d/%d): %s — retrying in %ds",
                exchange_retries, max_exchange_retries, exc, retry_interval,
            )
            sleep_fn(retry_interval)

    while True:
        live = get_live_positions(client)
        if not live:
            if not first_warning:
                logger.info("[startup-sync] all positions cleared -> resuming trading loop")
            else:
                logger.info("[startup-sync] no open positions detected")
            sync_tp_sl_registry([])
            return

        logger.warning(
            "[startup-sync] open positions detected (n=%d) -> trading init paused; will retry every %ss",
            len(live),
            retry_interval,
        )
        for pos in live:
            logger.warning(
                "[startup-sync] %s side=%s amt=%.6f entry=%.4f upnl=%.2f",
                pos.get("symbol"),
                pos.get("positionSide"),
                pos.get("positionAmt"),
                pos.get("entryPrice"),
                pos.get("unRealizedProfit"),
            )

        if allow_open_positions:
            logger.info(
                "[startup-sync] ALLOW_OPEN_POSITIONS=1 -> proceeding with %d open positions",
                len(live),
            )
            sync_tp_sl_registry(live)
            return

        first_warning = False
        sleep_fn(retry_interval)


__all__ = [
    "acquire_startup_lock",
    "check_startup_twap_recovery",
    "clean_testnet_caches",
    "run_order_reconciliation",
    "startup_position_check",
    "sync_dry_run",
]
