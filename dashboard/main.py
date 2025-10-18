import os
import sys
import json
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

# Ensure the project root is importable
PROJECT_ROOT = "/root/hedge-fund"
if PROJECT_ROOT not in sys.path and os.path.isdir(PROJECT_ROOT):
    sys.path.insert(0, PROJECT_ROOT)

# Local helpers
from dashboard.dashboard_utils import (
    get_firestore_connection,
    fetch_state_document,
    parse_nav_to_df_and_kpis,
    positions_sorted,
    fetch_mark_price_usdt,
    get_env_float,
)

# Read-only exchange helpers

# Doctor helper
# removed: doctor runs via subprocess now

# --------------------------- Page config -------------------------------------
st.set_page_config(page_title="Hedge — Portfolio Dashboard", layout="wide")
ENV = os.getenv("ENV", "prod")
REFRESH_SEC = int(os.getenv("DASHBOARD_REFRESH_SEC", "60"))
TRADE_LOG = os.getenv("TRADE_LOG", "trade_log.json")

# log-tail behavior
TAIL_BYTES = int(os.getenv("DASHBOARD_LOG_TAIL_BYTES", "200000"))
TAIL_LINES = int(os.getenv("DASHBOARD_SIGNAL_LINES", "80"))
WANT_TAGS = tuple((os.getenv("DASHBOARD_SIGNAL_TAGS") or "[screener],[screener->executor],[decision]").split(","))

# stable-coins to include in equity calc
STABLES = [s.strip() for s in (os.getenv("DASHBOARD_STABLES", "USDT,FDUSD,BFUSD,USDC").split(",")) if s.strip()]

LATENCY_CACHE_PATH = Path(os.getenv("EXEC_LATENCY_CACHE", "logs/execution/replay_cache.json"))

# Optional auto-refresh
try:
    from streamlit_extras.st_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SEC * 1000, key="auto_refresh")
except Exception:
    pass  # manual refresh only

st.title("📊 Hedge — Portfolio Dashboard")
st.caption(f"ENV = {ENV} · refresh ≈ {REFRESH_SEC}s")

# --------------------------- Small utilities ---------------------------------
def load_json(path: str, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default

def parse_iso_ts(ts: str):
    if not ts:
        return None
    try:
        cleaned = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None

def format_latency(value):
    try:
        if value is None or (isinstance(value, float) and value != value):
            return "—"
        return f"{float(value):.0f} ms"
    except Exception:
        return "—"

def human_age(ts) -> str:
    if not ts:
        return "–"
    try:
        now = int(time.time())
        d = now - int(ts)
        if d < 60:
            return f"{d}s"
        if d < 3600:
            return f"{d//60}m"
        return f"{d//3600}h"
    except Exception:
        return "–"

def tail_text(path: str, max_bytes: int = TAIL_BYTES) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes), os.SEEK_SET)
            return f.read().decode(errors="ignore")
    except Exception as e:
        return f"(cannot read {path}: {e})"

def rle_compact(lines: List[str], min_run: int = 3) -> List[str]:
    if not lines:
        return lines
    out = []
    prev = lines[0]
    count = 1
    for ln in lines[1:]:
        if ln == prev:
            count += 1
        else:
            out.append(prev if count < min_run else f"{prev}  × {count}")
            prev = ln
            count = 1
    out.append(prev if count < min_run else f"{prev}  × {count}")
    return out

# --------------------------- Load Firestore ----------------------------------
status = st.empty()
status.info("Loading data from Firestore…")

try:
    db = get_firestore_connection()
    nav_doc = fetch_state_document("nav", env=ENV)
    pos_doc = fetch_state_document("positions", env=ENV)
    lb_doc  = fetch_state_document("leaderboard", env=ENV)
except Exception as e:
    st.error(f"Firestore read failed: {e}")
    st.stop()

nav_df, kpis = parse_nav_to_df_and_kpis(nav_doc)
positions_fs = positions_sorted(pos_doc.get("items") or [])
leaderboard = lb_doc.get("items") or []

status.success(f"Loaded · updated_at={nav_doc.get('updated_at','—')}")

# ---- Read-only Reserve KPI ---------------------------------------------------
RESERVE_BTC = get_env_float("DASHBOARD_RESERVE_BTC", 0.13)
btc_mark = fetch_mark_price_usdt("BTCUSDT")
reserve_usdt = RESERVE_BTC * btc_mark if btc_mark > 0 else 0.0

# ---- Tabs layout --------------------------------------------------------------
tab_overview, tab_positions, tab_execution, tab_leader, tab_signals, tab_ml, tab_doctor = st.tabs(
    ["Overview", "Positions", "Execution", "Leaderboard", "Signals", "ML", "Doctor"]
)

# --------------------------- Overview Tab ------------------------------------
with tab_overview:
    st.subheader("Portfolio KPIs")
    equity = kpis.get("total_equity")
    peak_equity = kpis.get("peak_equity")
    drawdown_pct = kpis.get("drawdown")
    realized_pnl = kpis.get("realized_pnl")
    unrealized_pnl = kpis.get("unrealized_pnl")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Equity (USDT)", f"{equity:,.0f}" if equity is not None else "—")
    k2.metric("Peak (USDT)", f"{peak_equity:,.0f}" if peak_equity is not None else "—")
    k3.metric("Drawdown (%)", f"{drawdown_pct:.2f}%" if drawdown_pct is not None else "—")
    k4.metric("Realized PnL", f"{realized_pnl:,.0f}" if realized_pnl is not None else "—")
    k5.metric("Unrealized PnL", f"{unrealized_pnl:,.0f}" if unrealized_pnl is not None else "—")

    reserve_caption = f"~{reserve_usdt:,.0f} USDT" if reserve_usdt > 0 else "—"
    k6.metric("Reserve (BTC)", f"{RESERVE_BTC:.3f} BTC", reserve_caption)

    if nav_df.empty:
        st.info("No NAV points yet. Run executor + sync_state to populate.")
    else:
        st.line_chart(nav_df["equity"], use_container_width=True)

# --------------------------- Positions Tab -----------------------------------
with tab_positions:
    st.subheader("Open Positions")
    positions_df = pd.DataFrame(positions_fs)
    if positions_df is None or positions_df.empty:
        st.info("No open positions.")
    else:
        st.dataframe(positions_df, use_container_width=True, height=420)

# --------------------------- Execution Tab ----------------------------------
with tab_execution:
    st.subheader("Execution Health")

    exec_stats = (
        pos_doc.get("exec_stats")
        or nav_doc.get("exec_stats")
        or {}
    )
    latency_cache = load_json(str(LATENCY_CACHE_PATH), default={})
    latency_summary = (
        (latency_cache.get("summary") or {}).get("latency")
        or latency_cache.get("latency")
        or {}
    )

    heartbeats = exec_stats.get("last_heartbeats") or {}
    now_ts = time.time()
    hb_rows = []
    stale = False
    for svc in ("executor_live", "sync_daemon"):
        ts_iso = heartbeats.get(svc)
        ts_val = parse_iso_ts(ts_iso) if ts_iso else None
        if ts_val is not None:
            age = now_ts - ts_val
            hb_rows.append((svc, age))
            if age > 180:
                stale = True
        else:
            hb_rows.append((svc, None))
            stale = True

    hb_text = []
    for svc, age in hb_rows:
        if age is None:
            hb_text.append(f"{svc}: n/a")
        elif age < 60:
            hb_text.append(f"{svc}: {int(age)}s")
        elif age < 3600:
            hb_text.append(f"{svc}: {int(age // 60)}m")
        else:
            hb_text.append(f"{svc}: {int(age // 3600)}h")

    banner_message = " · ".join(hb_text) if hb_text else "No heartbeat data"
    if stale:
        st.error(f"Heartbeat stale · {banner_message}")
    else:
        st.success(f"Heartbeats healthy · {banner_message}")

    if not exec_stats:
        st.info("Execution stats unavailable. Ensure executor and sync daemon telemetry is publishing.")
    else:
        col_attempts, col_exec, col_veto, col_fill, col_p50, col_p90 = st.columns(6)
        attempted = exec_stats.get("attempted_24h")
        executed = exec_stats.get("executed_24h")
        vetoes = exec_stats.get("vetoes_24h")
        fill_rate = exec_stats.get("fill_rate")

        col_attempts.metric("Attempted (24h)", f"{attempted:,}" if attempted is not None else "—")
        col_exec.metric("Executed (24h)", f"{executed:,}" if executed is not None else "—")
        col_veto.metric("Vetoes (24h)", f"{vetoes:,}" if vetoes is not None else "—")

        if fill_rate is None:
            fill_display = "—"
        else:
            try:
                fr = float(fill_rate)
                fill_display = f"{fr*100:.1f}%" if fr <= 1 else f"{fr:.1f}%"
            except Exception:
                fill_display = "—"
        col_fill.metric("Fill Rate", fill_display)

        col_p50.metric("Latency p50", format_latency(latency_summary.get("p50_ms")))
        col_p90.metric("Latency p90", format_latency(latency_summary.get("p90_ms")))

        top_vetoes = exec_stats.get("top_vetoes") or []
        if top_vetoes:
            st.markdown("#### Top Veto Reasons (24h)")
            df_veto = pd.DataFrame(top_vetoes)
            if not df_veto.empty:
                df_veto = df_veto.rename(columns={"reason": "Reason", "count": "Count"})
                st.dataframe(df_veto, use_container_width=True, height=280)
            else:
                st.info("No veto data to display.")
        else:
            st.info("No veto data in the last 24 hours.")

        if not latency_summary:
            st.caption(
                "Latency metrics unavailable. Generate a cache with "
                "`python scripts/replay_logs.py --since <iso> --json logs/execution/replay_cache.json`."
            )
# --------------------------- Leaderboard Tab ---------------------------------
with tab_leader:
    st.subheader("Leaderboard")
    leader_df = pd.DataFrame(leaderboard)
    if leader_df is None or leader_df.empty:
        st.info("No leaderboard entries yet.")
    else:
        st.dataframe(leader_df, use_container_width=True, height=420)

# --------------------------- Signals Tab -------------------------------------
with tab_signals:
    st.subheader("Signals Tail (last N)")
    N = int(os.environ.get("DASHBOARD_SIGNAL_LINES", "150"))
    log_path = "/var/log/hedge/hedge-executor.out.log"
    patterns = ("[screener]", "[screener->executor]", "[decision]")
    lines = []

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f.readlines():
                if any(p in ln for p in patterns):
                    lines.append(ln.rstrip())
        tail = lines[-N:]
        if not tail:
            st.info("No screener/decision breadcrumbs yet.")
        else:
            compact = []
            count = 1
            for i in range(len(tail)):
                if i+1 < len(tail) and tail[i+1] == tail[i]:
                    count += 1
                else:
                    compact.append(tail[i] if count == 1 else f"{tail[i]}  ×{count}")
                    count = 1
            st.code("\n".join(compact), language="text")
    except Exception as e:
        st.warning(f"Could not read signals log: {e}")

# --------------------------- ML Tab -----------------------------------------
with tab_ml:
    st.header("ML — Models & Evaluation")
    try:
        reg_path = "models/registry.json"
        if os.path.exists(reg_path):
            registry = json.load(open(reg_path, "r"))
            if registry:
                st.subheader("Registry")
                st.dataframe(pd.DataFrame(registry).T)
            else:
                st.info("Registry is empty. Run ML retrain to populate.")
        else:
            st.info("No registry yet. Run ML retrain to populate.")

        rpt_path = "models/last_train_report.json"
        if os.path.exists(rpt_path):
            st.subheader("Last Retrain Report")
            st.json(json.load(open(rpt_path, "r")))

        eval_path = "models/signal_eval.json"
        if os.path.exists(eval_path):
            st.subheader("Signal Evaluation (ML vs RULE)")
            report = json.load(open(eval_path, "r"))
            st.json(report.get("aggregate", {}))

            symbols = report.get("symbols") or []
            rows = []
            for entry in symbols:
                if "error" in entry:
                    continue
                rows.append(
                    {
                        "symbol": entry.get("symbol"),
                        "ml_f1": entry.get("ml", {}).get("f1"),
                        "ml_cov": entry.get("ml", {}).get("coverage"),
                        "ml_hit": entry.get("ml", {}).get("hit_rate"),
                        "rule_f1": entry.get("rule", {}).get("f1"),
                        "rule_cov": entry.get("rule", {}).get("coverage"),
                        "rule_hit": entry.get("rule", {}).get("hit_rate"),
                        "n": entry.get("n"),
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows))

            errors = report.get("errors") or []
            if errors:
                st.warning(f"Evaluation skipped for {len(errors)} symbol(s):")
                for entry in errors[:20]:
                    st.write(f"- {entry.get('symbol')}: {entry.get('error')}")
    except Exception as exc:
        st.error(f"ML tab error: {exc}")

# --------------------------- Doctor Tab --------------------------------------
with tab_doctor:
    st.subheader("Doctor Snapshot")

    # Run scripts/doctor.py on demand (read-only, signed request inside the script)
    run = st.button("Run doctor.py", help="Gathers hedge/one-way, crosses, gates, and blocked_by reasons.")
    doctor_data = None
    if run:
        try:
            out = subprocess.check_output(
                ["python3", "scripts/doctor.py"], stderr=subprocess.STDOUT, timeout=20
            ).decode()
            try:
                doctor_data = json.loads(out)
            except Exception:
                st.code(out[:4000], language="json")
        except Exception as e:
            st.error(f"doctor.py failed: {e}")

    # Derive flags from env and doctor output (dualSide comes from doctor if available)
    def _b(name: str) -> bool:
        return str(os.getenv(name, "")).strip().lower() in ("1","true","yes","on")

    flags = {
        "use_futures": _b("USE_FUTURES"),
        "testnet": _b("BINANCE_TESTNET"),
        "dry_run": _b("DRY_RUN"),
        "dualSide": bool(((doctor_data or {}).get("env") or {}).get("dualSide", False)),
    }

    cols = st.columns(4)
    cols[0].metric("Futures", "Yes" if flags.get("use_futures") else "No")
    cols[1].metric("Testnet", "Yes" if flags.get("testnet") else "No")
    cols[2].metric("Dry-run", "Yes" if flags.get("dry_run") else "No")
    cols[3].metric("Hedge Mode", "Yes" if flags.get("dualSide") else "No")

    if doctor_data:
        with st.expander("Raw doctor output", expanded=False):
            st.json(doctor_data, expanded=False)

    # Tier diagnostics and veto stats
    with st.expander("Doctor — Universe & Risk", expanded=False):
        tpath = "logs/nav_trading.json"
        rpath = "logs/nav_reporting.json"
        zpath = "logs/nav_treasury.json"
        spath = "logs/nav_snapshot.json"
        if any(os.path.exists(p) for p in (tpath, rpath, zpath, spath)):
            try:
                if os.path.exists(tpath):
                    tnav = json.load(open(tpath, "r"))
                    tval = float(tnav.get("nav_usdt", 0.0) or 0.0)
                    st.markdown(f"**Trading NAV (USDT, used for risk):** {tval:.2f}")
                    tbr = tnav.get("breakdown", {})
                    tfw = tbr.get("futures_wallet_usdt")
                    if tfw is not None:
                        st.caption(f"Futures wallet: {float(tfw):.2f} USDT")
                    else:
                        st.caption("Futures wallet: n/a")
                if os.path.exists(rpath):
                    rnav = json.load(open(rpath, "r"))
                    rval = float(rnav.get("nav_usdt", 0.0) or 0.0)
                    st.markdown(f"**Reporting NAV (USDT):** {rval:.2f}")
                if os.path.exists(zpath):
                    znav = json.load(open(zpath, "r"))
                    zval = float(znav.get("treasury_usdt", 0.0) or 0.0)
                    st.markdown(
                        "**Treasury (off-exchange, excluded from NAV):** "
                        f"{zval:.2f} USDT"
                    )
                    zbr = znav.get("breakdown", {})
                    tre = zbr.get("treasury", {})
                    miss = zbr.get("missing_prices", {})
                    if tre:
                        st.write("Holdings (manual-seeded):")
                        for asset, data in tre.items():
                            qty = data.get("qty")
                            px = data.get("px")
                            val = data.get("val_usdt")
                            line = f"- {asset}: qty {qty}"
                            if px is not None:
                                line += f" × {px}"
                            if val is not None:
                                line += f" = {val:.2f} USDT"
                            st.write(line)
                    if miss:
                        st.warning(
                            "Missing prices for treasury symbols (skipped): "
                            + ", ".join(sorted(miss.keys()))
                        )
                if (not os.path.exists(tpath)) and (not os.path.exists(rpath)) and os.path.exists(spath):
                    nav = json.load(open(spath, "r"))
                    sval = float(nav.get("nav_usdt", 0.0) or 0.0)
                    st.markdown(f"**NAV (legacy single):** {sval:.2f}")
            except Exception as exc:
                st.caption(f"nav snapshot unavailable: {exc}")
        # Tier counts from config/symbol_tiers.json
        tier_counts = {}
        try:
            tiers_cfg = load_json(os.getenv("SYMBOL_TIERS_CONFIG", "config/symbol_tiers.json"), {})
            for t, arr in (tiers_cfg.items() if isinstance(tiers_cfg, dict) else []):
                if isinstance(arr, list):
                    tier_counts[str(t)] = len(arr)
        except Exception:
            tier_counts = {}
        if tier_counts:
            st.write({"tier_counts": tier_counts})

        # Open positions by tier
        by_tier = {}
        try:
            # Build tier map
            tmap = {s: t for t, arr in (tiers_cfg.items() if isinstance(tiers_cfg, dict) else []) for s in (arr or [])}
        except Exception:
            tmap = {}
        try:
            # positions_fs already parsed above
            for r in positions_fs:
                sym = str(r.get("symbol") or "").upper()
                t = tmap.get(sym, "?")
                by_tier[t] = by_tier.get(t, 0) + 1
        except Exception:
            by_tier = {}
        if by_tier:
            st.write({"open_positions_by_tier": by_tier})

        # Veto reasons dominated in last 24h (from risk collection)
        veto_counts = {}
        try:
            from dashboard.dashboard_utils import get_firestore_connection

            db = get_firestore_connection()
            docs = list(
                db.collection("hedge")
                .document(ENV)
                .collection("risk")
                .order_by("ts", direction="DESCENDING")
                .limit(1000)
                .stream()
            )
            import time as _t

            now = _t.time()
            for d in docs:
                x = d.to_dict() or {}
                if x.get("env") is not None and str(x.get("env")) != ENV:
                    continue
                ts = x.get("ts") or x.get("time")
                tnum = float(ts) if isinstance(ts, (int, float)) else 0.0
                if tnum > 1e12:
                    tnum /= 1000.0
                if (now - tnum) > 24 * 3600:
                    continue
                # reason (single) or reasons (list)
                if isinstance(x.get("reasons"), list):
                    for r in x.get("reasons"):
                        veto_counts[str(r)] = veto_counts.get(str(r), 0) + 1
                elif x.get("reason") is not None:
                    veto_counts[str(x.get("reason"))] = veto_counts.get(str(x.get("reason")), 0) + 1
        except Exception:
            veto_counts = {}
        if veto_counts:
            # Show top reasons
            top = sorted(veto_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
            st.write({"veto_top_24h": dict(top)})


st.caption(
    "Flags: "
    f"{'FUTURES ' if flags.get('use_futures') else ''}"
    f"{'TESTNET ' if flags.get('testnet') else ''}"
    f"{'DRY_RUN ' if flags.get('dry_run') else ''}"
    f"{'HEDGE_MODE ' if flags.get('dualSide') else 'ONE-WAY'}"
)
