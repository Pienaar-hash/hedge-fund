from dashboard.dashboard_utils import (
    get_firestore_connection,
    fetch_state_document,
    parse_nav_to_df_and_kpis,
    positions_sorted,
    fetch_mark_price_usdt,
    get_env_float,
)
from dashboard.data_sources import get_latest_nav, per_symbol_kpis

import math
import os
import sys
import json
import time
import subprocess
from typing import List

import pandas as pd
import streamlit as st

# Ensure the project root is importable
PROJECT_ROOT = "/root/hedge-fund"
if PROJECT_ROOT not in sys.path and os.path.isdir(PROJECT_ROOT):
    sys.path.insert(0, PROJECT_ROOT)

# Local helpers

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


def _format_age_label(age: float) -> str:
    try:
        age_float = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(age_float):
        return "unknown"
    if age_float < 1:
        return "<1s"
    if age_float < 60:
        return f"{int(age_float)}s"
    if age_float < 3600:
        return f"{int(age_float // 60)}m"
    if age_float < 86400:
        return f"{int(age_float // 3600)}h"
    return f"{int(age_float // 86400)}d"


latest_nav_snapshot = get_latest_nav()
nav_positions_snapshot = per_symbol_kpis(latest_nav_snapshot.get("positions") or [])
try:
    nav_age_seconds = float(latest_nav_snapshot.get("age_sec", float("inf")))
except (TypeError, ValueError):
    nav_age_seconds = float("inf")
nav_age_label = _format_age_label(nav_age_seconds)
nav_is_stale = bool(latest_nav_snapshot.get("is_stale"))
nav_badge_color = "#dc3545" if nav_is_stale else "#198754"
nav_badge_text = "STALE" if nav_is_stale else "FRESH"
nav_source = str(latest_nav_snapshot.get("source") or "unknown")

st.markdown(
    f"""
    <div style='display:flex;gap:0.5rem;align-items:center;margin:0.75rem 0;'>
        <span style='padding:0.15rem 0.7rem;border-radius:999px;font-size:0.75rem;font-weight:600;color:#ffffff;background:{nav_badge_color};'>
            {nav_badge_text}
        </span>
        <span style='padding:0.15rem 0.7rem;border-radius:999px;font-size:0.75rem;font-weight:500;color:#495057;background:#f1f3f5;'>
            source: {nav_source}
        </span>
        <span style='font-size:0.8rem;color:#6c757d;'>age ≈ {nav_age_label}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

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
tab_overview, tab_positions, tab_leader, tab_signals, tab_ml, tab_doctor = st.tabs(
    ["Overview", "Positions", "Leaderboard", "Signals", "ML", "Doctor"]
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
    if nav_positions_snapshot:
        st.markdown("**Latest Snapshot (cached feed)**")
        st.caption(f"source: {nav_source} · age ≈ {nav_age_label} · state: {nav_badge_text}")

        def _fmt_number(value, decimals: int = 2) -> str:
            try:
                num = float(value)
            except (TypeError, ValueError):
                return "—"
            if not math.isfinite(num):
                return "—"
            return f"{num:.{decimals}f}"

        table_rows = []
        for entry in nav_positions_snapshot:
            symbol = entry.get("symbol") or "—"
            veto_tail = entry.get("veto_tail") or []
            veto_html = ""
            if veto_tail:
                items = "".join(
                    f"<li>{line}</li>" for line in veto_tail if isinstance(line, str)
                )
                if items:
                    veto_html = (
                        "<ul style='margin:0.25rem 0 0 0.9rem;padding:0;"
                        "list-style:disc;color:#6c757d;font-size:0.72rem;'>"
                        f"{items}</ul>"
                    )
            symbol_cell = f"<div><strong>{symbol}</strong>{veto_html}</div>"
            rr = entry.get("rr")
            rr_display = _fmt_number(rr * 100.0 if isinstance(rr, (int, float)) else rr, 2)
            if rr_display != "—":
                rr_display = f"{rr_display}%"
            age_display = _format_age_label(entry.get("age_sec"))
            stale_display = "❌" if entry.get("stale") else "✅"
            row_html = (
                "<tr>"
                f"<td>{symbol_cell}</td>"
                f"<td>{entry.get('side') or '—'}</td>"
                f"<td style='text-align:right;'>{_fmt_number(entry.get('size'), 4)}</td>"
                f"<td style='text-align:right;'>{_fmt_number(entry.get('entry'), 4)}</td>"
                f"<td style='text-align:right;'>{_fmt_number(entry.get('mark'), 4)}</td>"
                f"<td style='text-align:right;'>{_fmt_number(entry.get('unrealized'), 2)}</td>"
                f"<td style='text-align:right;'>{_fmt_number(entry.get('realized'), 2)}</td>"
                f"<td style='text-align:right;'>{rr_display}</td>"
                f"<td style='text-align:center;'>{stale_display}</td>"
                f"<td style='text-align:right;'>{age_display}</td>"
                "</tr>"
            )
            table_rows.append(row_html)

        table_html = (
            "<div style='overflow-x:auto;'>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>"
            "<thead style='background:#f8f9fa;'>"
            "<tr>"
            "<th style='text-align:left;padding:0.4rem;'>Symbol</th>"
            "<th style='text-align:left;padding:0.4rem;'>Side</th>"
            "<th style='text-align:right;padding:0.4rem;'>Size</th>"
            "<th style='text-align:right;padding:0.4rem;'>Entry</th>"
            "<th style='text-align:right;padding:0.4rem;'>Mark</th>"
            "<th style='text-align:right;padding:0.4rem;'>Unreal.</th>"
            "<th style='text-align:right;padding:0.4rem;'>Realized</th>"
            "<th style='text-align:right;padding:0.4rem;'>R/R</th>"
            "<th style='text-align:center;padding:0.4rem;'>Stale</th>"
            "<th style='text-align:right;padding:0.4rem;'>Age</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
            + "".join(table_rows)
            + "</tbody></table></div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.caption("Latest snapshot has no open positions.")

    st.markdown("---")

    positions_df = pd.DataFrame(positions_fs)
    if positions_df is None or positions_df.empty:
        st.info("No open positions from Firestore.")
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
