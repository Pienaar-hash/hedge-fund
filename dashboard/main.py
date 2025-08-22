import os
import sys
import json
import time
import subprocess
from typing import Any, Dict, List, Tuple

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
    read_trade_log_tail,
    fmt_ccy,
    fmt_pct,
    fetch_mark_price_usdt,
    get_env_float,
)

# Read-only exchange helpers
from execution.exchange_utils import get_account_overview

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
tab_overview, tab_positions, tab_leader, tab_signals, tab_doctor = st.tabs(
    ["Overview", "Positions", "Leaderboard", "Signals", "Doctor"]
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
    log_path = "/var/log/hedge-executor.out.log"
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


st.caption(
    "Flags: "
    f"{'FUTURES ' if flags.get('use_futures') else ''}"
    f"{'TESTNET ' if flags.get('testnet') else ''}"
    f"{'DRY_RUN ' if flags.get('dry_run') else ''}"
    f"{'HEDGE_MODE ' if flags.get('dualSide') else 'ONE-WAY'}"
)
