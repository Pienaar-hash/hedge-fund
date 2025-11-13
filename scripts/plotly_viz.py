"""
plotly_viz.py — Fund Overview generator (v2.7.6)
------------------------------------------------
Inputs: config/assets.json,
        logs/nav_log.json (optional), logs/reserves.json (optional), logs/doctor_summary.json (optional),
        data/binance_week_daily_curve.csv, data/binance_week_by_symbol.csv,
        data/binance_week_overview.csv
Output: reports/fund_overview_v2_7.html
"""

import os, json, re, base64
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

DATA_PATH = "data"
LOG_PATH = "logs"
CONFIG_PATH = "config"
REPORT_PATH = "reports"
FX_FALLBACK = 17.36

ASSET_COLORS = {
    "BTC": "#f2a900",
    "XAUT": "#e5c07b",
    "USDT": "#26a17b",
    "USDC": "#2775ca",
}
FUTURES_COLOR = "#2ebd85"
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "XAUT": "tether-gold",
    "USDT": "tether",
    "USDC": "usd-coin",
}
SYSTEM_COMPONENTS = ["Executor", "Sync", "Dashboard", "Firestore"]
DEFAULT_STATUS = {
    "Executor": {"state": "OK", "emoji": "✅", "color": "#00cc96", "note": "Fresh heartbeat"},
    "Sync": {"state": "OK", "emoji": "✅", "color": "#00cc96", "note": "Fresh heartbeat"},
    "Dashboard": {"state": "WARN", "emoji": "⚠️", "color": "#f39c12", "note": "Restart required"},
    "Firestore": {"state": "OK", "emoji": "✅", "color": "#00cc96", "note": "Fresh heartbeat"},
}
BORDER_GREY = "#7f8c8d"
PRICE_CACHE = {}


def _decode_typed_array(obj):
    if not isinstance(obj, dict) or "bdata" not in obj or "dtype" not in obj:
        return None
    dtype_map = {
        "f8": "<f8",
        "f4": "<f4",
        "i1": "<i1",
        "i2": "<i2",
        "i4": "<i4",
        "u1": "<u1",
        "u2": "<u2",
        "u4": "<u4",
    }
    dtype = dtype_map.get(obj["dtype"])
    if dtype is None:
        return None
    try:
        raw = base64.b64decode(obj["bdata"])
        arr = np.frombuffer(raw, dtype=dtype)
        shape = obj.get("shape")
        if shape:
            arr = arr.reshape(shape)
        return arr.tolist()
    except Exception:
        return None


def _load_json_file(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def get_usd_to_zar():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "usd", "vs_currencies": "zar"}
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            rate = float(data.get("usd", {}).get("zar", FX_FALLBACK))
            if rate > 0:
                return rate
    except Exception:
        pass
    return FX_FALLBACK


def fetch_prices_for(symbols):
    symbols = [s.upper() for s in symbols if s.upper() in COINGECKO_IDS]
    to_fetch = [s for s in symbols if s not in PRICE_CACHE]
    if to_fetch:
        ids = ",".join({COINGECKO_IDS[s] for s in to_fetch})
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": ids, "vs_currencies": "usd"}
        try:
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                payload = resp.json()
                for sym in to_fetch:
                    price_val = payload.get(COINGECKO_IDS[sym], {}).get("usd")
                    if price_val:
                        PRICE_CACHE[sym] = float(price_val)
        except Exception:
            pass
    return {s: PRICE_CACHE.get(s) for s in symbols}


def load_assets_aum():
    path = os.path.join(CONFIG_PATH, "assets.json")
    payload = _load_json_file(path) or {}
    valuations = []
    missing = []
    for sym, meta in payload.items():
        sym_up = sym.upper()
        qty = float((meta or {}).get("qty") or 0.0)
        price = meta.get("price_usd") if isinstance(meta, dict) else None
        if not price:
            missing.append(sym_up)
        valuations.append({"symbol": sym_up, "qty": qty, "price": price})
    if missing:
        price_map = fetch_prices_for(missing)
        for asset in valuations:
            price_guess = price_map.get(asset["symbol"])
            if price_guess:
                asset["price"] = price_guess
    for asset in valuations:
        asset["price"] = float(asset.get("price") or 0.0)
        asset["value"] = asset["qty"] * asset["price"]
    total_aum = sum(a["value"] for a in valuations)
    return valuations, total_aum


def fetch_pct_from_coingecko(symbol):
    coin_id = COINGECKO_IDS.get(symbol.upper())
    if not coin_id:
        return pd.Series(dtype=float)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": 7}
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200:
            return pd.Series(dtype=float)
        prices = resp.json().get("prices", [])
        if not prices:
            return pd.Series(dtype=float)
        df = pd.DataFrame(prices, columns=["ts", "price"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.strftime("%Y-%m-%d")
        daily = df.groupby("date")["price"].last()
        base_candidates = daily[daily > 0]
        base = base_candidates.iloc[0] if not base_candidates.empty else (daily.iloc[0] if len(daily) else None)
        if not base or pd.isna(base):
            return pd.Series(dtype=float)
        pct = (daily / base - 1) * 100
        return pct
    except Exception:
        return pd.Series(dtype=float)


def get_crypto_pct_change(df, symbol):
    if df.empty:
        return None
    symbol = symbol.lower().strip()
    cols = {c.lower(): c for c in df.columns}
    date_col = next((cols[key] for key in cols if any(tag in key for tag in ("date", "__dt", "day", "ts"))), None)
    sym_col = cols.get(symbol)
    if not date_col or not sym_col:
        return None
    prices = pd.to_numeric(df[sym_col], errors="coerce").ffill().bfill()
    if prices.empty:
        return None
    base = prices.iloc[0]
    if pd.isna(base) or base == 0:
        non_zero = prices.replace(0, np.nan).dropna()
        if non_zero.empty:
            return None
        base = non_zero.iloc[0]
    if base == 0 or pd.isna(base):
        return None
    pct = (prices / base - 1) * 100
    dates = pd.to_datetime(df[date_col])
    return dates, pct


def load_hourly_notional_from_week_files():
    candidates = [
        os.path.join(DATA_PATH, "binance_week_by_symbol.csv"),
        os.path.join(DATA_PATH, "binance_week_overview.csv"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df.columns = [c.strip().lower() for c in df.columns]
        if {"hour", "notional"}.issubset(df.columns):
            df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
            df["notional"] = pd.to_numeric(df["notional"], errors="coerce").abs()
            hourly = df.dropna(subset=["hour"])
            hourly = hourly[hourly["hour"].between(0, 23)]
            hourly_sum = hourly.groupby("hour")["notional"].sum()
            if not hourly_sum.empty:
                return {int(hr): float(val) for hr, val in hourly_sum.items()}
        hour_cols = []
        for col in df.columns:
            match = re.search(r"hour[_\s]*(\d{1,2})", col)
            if match:
                hour_cols.append((int(match.group(1)) % 24, col))
        if hour_cols:
            cleaned = {}
            for hr, col in hour_cols:
                cleaned[hr] = cleaned.get(hr, 0.0) + float(pd.to_numeric(df[col], errors="coerce").abs().sum())
            if cleaned:
                return cleaned
    return None


def load_hourly_notional_fallback_trust_pack():
    html_path = os.path.join(REPORT_PATH, "trust_pack_v2.html")
    if not os.path.exists(html_path):
        return None
    try:
        text = open(html_path, encoding="utf-8").read()
    except Exception:
        return None
    traces_json = []
    idx = 0
    while True:
        start_call = text.find("Plotly.newPlot(", idx)
        if start_call == -1:
            break
        data_start = text.find("[", start_call)
        if data_start == -1:
            break
        depth = 0
        for pos in range(data_start, len(text)):
            char = text[pos]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    traces_json.append(text[data_start:pos + 1])
                    idx = pos
                    break
        else:
            break
        idx += 1
    for block in traces_json:
        try:
            traces = json.loads(block)
        except Exception:
            continue
        if not isinstance(traces, list):
            continue
        for trace in traces:
            if str(trace.get("name")) != "Trade Notional":
                continue
            x_obj = trace.get("x")
            y_obj = trace.get("y")
            x_vals = _decode_typed_array(x_obj) if isinstance(x_obj, dict) else x_obj
            y_vals = _decode_typed_array(y_obj) if isinstance(y_obj, dict) else y_obj
            if not (x_vals and y_vals):
                continue
            mapping = {}
            for hour, val in zip(x_vals, y_vals):
                try:
                    hour_int = int(round(float(hour))) % 24
                    raw_usdt = float(val) * 1000  # stored as k USDT
                    mapping[hour_int] = raw_usdt
                except Exception:
                    continue
            if mapping:
                return mapping
    return None


def load_system_status():
    path = os.path.join(LOG_PATH, "doctor_summary.json")
    payload = _load_json_file(path)
    if isinstance(payload, list) and payload:
        payload = payload[-1]

    status_map = {comp: DEFAULT_STATUS[comp].copy() for comp in SYSTEM_COMPONENTS}
    if not isinstance(payload, dict):
        return status_map

    for comp in SYSTEM_COMPONENTS:
        entry = payload.get(comp) or payload.get(comp.lower())
        if not entry:
            continue
        if isinstance(entry, dict):
            status = entry.get("status") or entry.get("state") or "Unknown"
            note = entry.get("note") or entry.get("message") or DEFAULT_STATUS[comp]["note"]
        else:
            status = entry or "Unknown"
            note = DEFAULT_STATUS[comp]["note"]
        status_lower = str(status).lower()
        if any(token in status_lower for token in ("ok", "up", "healthy")):
            state = "OK"
            emoji = "✅"
            value = 1.0
            color = "#00cc96"
        elif any(token in status_lower for token in ("warn", "stale", "delay", "sync")):
            state = "WARN"
            emoji = "⚠️"
            value = 0.6
            color = "#f39c12"
        elif any(token in status_lower for token in ("fail", "down", "error")):
            state = "ERROR"
            emoji = "❌"
            value = 0.2
            color = "#e74c3c"
        else:
            state = "UNKNOWN"
            emoji = "⬜"
            value = 0.4
            color = BORDER_GREY
        status_map[comp] = {"state": state, "emoji": emoji, "value": value, "color": color, "note": note}
    return status_map


def _format_amount(val):
    if val is None or pd.isna(val):
        return ""
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"{val / 1_000:.1f}K"
    return f"{val:,.0f}"


def format_currency(val):
    if val is None or pd.isna(val):
        return "$0"
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"${val / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"${val / 1_000:.1f}K"
    return f"${val:,.0f}"


def compute_fund_pct_series(week_daily, date_col, daily_col, cum_col, final_equity=None):
    log_message = None
    dates = week_daily[date_col] if (date_col and date_col in week_daily) else []

    def pct_from_equity(equity_series):
        eq = pd.to_numeric(equity_series, errors="coerce").ffill().bfill()
        if eq.isna().all():
            return None
        base_candidates = eq[eq > 0]
        base = base_candidates.iloc[0] if not base_candidates.empty else eq.iloc[0]
        if not base or pd.isna(base):
            return None
        return (eq / base - 1) * 100, eq

    equity_cols = ["equity_usd", "equity", "total_equity", "equity_value"]
    for col in equity_cols:
        if col in week_daily:
            result = pct_from_equity(week_daily[col])
            if result is not None:
                pct, eq = result
                return dates, pct, log_message, eq

    if cum_col and cum_col in week_daily:
        start_cols = ["equity_start_usd", "start_equity", "equity_start"]
        start_val = None
        for col in start_cols:
            if col in week_daily:
                series = pd.to_numeric(week_daily[col], errors="coerce").dropna()
                if not series.empty:
                    start_val = series.iloc[0]
                    break
        cum = pd.to_numeric(week_daily[cum_col], errors="coerce").ffill().fillna(0)
        if start_val is None and len(cum):
            if final_equity is not None:
                start_val = final_equity - cum.iloc[-1]
            else:
                start_val = max(abs(cum.iloc[0]), 1.0)
        start_val = start_val or 1.0
        pct = (cum / start_val) * 100
        equity_series = start_val + cum
        return dates, pct, log_message, equity_series

    if daily_col and daily_col in week_daily:
        daily = pd.to_numeric(week_daily[daily_col], errors="coerce").fillna(0)
        cumsum = daily.cumsum()
        if len(cumsum) and final_equity is not None:
            start_equity = final_equity - cumsum.iloc[-1]
        else:
            start_equity = abs(cumsum.iloc[0]) if len(cumsum) else 1.0
        start_equity = max(start_equity, 1.0)
        equity_series = start_equity + cumsum
        pct = (equity_series / start_equity - 1) * 100
        log_message = "[viz] fund % overlay: using inferred equity"
        return dates, pct, log_message, equity_series

    log_message = "[viz] fund % overlay: insufficient fields, drew flat baseline"
    zero_series = pd.Series([0.0] * len(week_daily)) if len(week_daily) else pd.Series(dtype=float)
    equity_series = pd.Series([1.0] * len(week_daily)) if len(week_daily) else pd.Series(dtype=float)
    return dates, zero_series, log_message, equity_series


def build_trust_pack():
    os.makedirs(REPORT_PATH, exist_ok=True)

    week_daily_path = os.path.join(DATA_PATH, "binance_week_daily_curve.csv")
    week_daily = pd.read_csv(week_daily_path) if os.path.exists(week_daily_path) else pd.DataFrame()
    week_daily.columns = [c.lower().strip() for c in week_daily.columns]

    overview_path = os.path.join(DATA_PATH, "binance_week_overview.csv")
    overview_df = pd.read_csv(overview_path) if os.path.exists(overview_path) else pd.DataFrame()
    overview_df.columns = [c.lower().strip() for c in overview_df.columns]

    weekly_changes = {"Fund": 0.0, "BTC": 0.0, "ETH": 0.0, "BNB": 0.0, "XAUT": 0.0}

    def _pick(col_opts):
        for name in col_opts:
            if name in week_daily.columns:
                return name
        return None

    date_col = _pick(["date", "__dt", "ts", "day"])
    daily_col = _pick(["daily_pnl", "daily_pnl_usdt", "daily"])
    cum_col = _pick(["cum_pnl", "cum_pnl_usdt", "cumulative"])

    if date_col and pd.api.types.is_string_dtype(week_daily[date_col]):
        week_daily[date_col] = pd.to_datetime(week_daily[date_col]).dt.strftime("%Y-%m-%d")

    if date_col and not week_daily.empty:
        week_daily = week_daily.sort_values(date_col).reset_index(drop=True)

    assets, total_aum_usd = load_assets_aum()
    usd_zar = get_usd_to_zar()
    total_aum_zar = total_aum_usd * usd_zar

    print(f"[viz] USD→ZAR rate {usd_zar:.2f}")
    print(f"[viz] Total AUM ${total_aum_usd:,.2f} → R{total_aum_zar:,.0f}")

    pie_labels = [asset["symbol"] for asset in assets] or ["No Data"]
    pie_values = [asset["value"] for asset in assets] or [1]
    pie_colors = [ASSET_COLORS.get(asset["symbol"], BORDER_GREY) for asset in assets] or [BORDER_GREY]
    pie_custom = [[value * usd_zar] for value in pie_values] or [[0.0]]

    fig_aum = go.Pie(
        labels=pie_labels,
        values=pie_values,
        hole=0.45,
        textinfo="label+percent",
        textfont=dict(color="white"),
        marker=dict(colors=pie_colors, line=dict(color="#0b0b0b", width=2)),
        customdata=pie_custom,
        hovertemplate="%{label}: %{percent:.1%} of AUM<br>$%{value:,.2f} USD (≈ R%{customdata[0]:,.0f})<extra></extra>",
        showlegend=False,
    )

    fund_dates, fund_pct_series, fund_log, equity_series = compute_fund_pct_series(
        week_daily, date_col, daily_col, cum_col, final_equity=total_aum_usd
    )
    if fund_log:
        print(fund_log)
    fund_y = fund_pct_series.tolist() if hasattr(fund_pct_series, "tolist") else list(fund_pct_series)
    if not fund_y:
        fund_y = [0.0] * len(week_daily)
    fund_x = list(fund_dates) if len(fund_dates) else (week_daily[date_col].tolist() if date_col in week_daily else list(range(len(fund_y))))
    weekly_changes["Fund"] = float(fund_y[-1]) if fund_y else 0.0

    equity_traces = [
        go.Scatter(
            x=fund_x,
            y=fund_y,
            mode="lines",
            line=dict(color="#00cc96", width=3),
            name="Fund",
            connectgaps=True,
            hovertemplate="%{x}<br>Fund: %{y:.1f}%<extra></extra>",
        )
    ]

    fallback_dates = fund_x
    bench_meta = {
        "BTC": {"color": "#f2a900"},
        "ETH": {"color": "#627eea"},
        "BNB": {"color": "#f3ba2f"},
        "XAUT": {"color": "#e5c07b"},
    }

    coingecko_used = False

    def align_series(x_vals, series_vals):
        if not fund_x:
            return list(series_vals)
        series = pd.Series(series_vals, index=x_vals)
        aligned = series.reindex(fund_x)
        if aligned.notna().any():
            aligned = aligned.ffill().bfill()
        else:
            aligned = pd.Series([0.0] * len(fund_x), index=fund_x)
        return aligned.fillna(0).tolist()

    for sym, meta in bench_meta.items():
        color = meta["color"]
        series = None
        crypto_data = get_crypto_pct_change(overview_df, sym)
        if crypto_data:
            dates, pct = crypto_data
            x_vals = dates.dt.strftime("%Y-%m-%d").tolist()
            series = align_series(x_vals, pct.tolist()) if not pct.empty else None
            if not pct.empty:
                weekly_changes[sym] = float(pct.iloc[-1])
        if series is None or not fund_x:
            fetched = fetch_pct_from_coingecko(sym)
            if not fetched.empty:
                coingecko_used = True
                series = align_series(fetched.index.tolist(), fetched.tolist())
                weekly_changes[sym] = float(fetched.iloc[-1]) if len(fetched) else weekly_changes.get(sym, 0.0)
        if series is None:
            series = [0.0] * len(fund_x)
        equity_traces.append(
            go.Scatter(
                x=fund_x,
                y=series,
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                name=sym,
                connectgaps=True,
                hovertemplate="%{x}<br>%{y:.1f}%<extra>" + sym + "</extra>",
                showlegend=True,
            )
        )

    if coingecko_used:
        print("[viz] fetched BTC/ETH/BNB/XAUT 7-day data via CoinGecko")

    daily_series = pd.to_numeric(week_daily[daily_col], errors="coerce") if (daily_col and daily_col in week_daily) else pd.Series(dtype=float)
    daily_series = daily_series.fillna(0)
    daily_dates = week_daily[date_col] if (date_col and date_col in week_daily) else []
    equity_series = pd.Series(equity_series).reset_index(drop=True)
    if len(equity_series) < len(daily_series):
        equity_series = equity_series.reindex(range(len(daily_series)), method="ffill")
    equity_series = equity_series.ffill().bfill()
    prev_equity = equity_series.shift(1)
    if len(prev_equity):
        prev_equity.iloc[0] = equity_series.iloc[0] if len(equity_series) else 1.0
    prev_equity = prev_equity.replace(0, np.nan).ffill().bfill().replace(0, 1.0)
    daily_pct = (daily_series / prev_equity) * 100 if len(prev_equity) else pd.Series([0.0] * len(daily_series))
    daily_pct = daily_pct.replace([np.inf, -np.inf], 0).fillna(0).clip(-10, 10)
    daily_colors = ["#2ebd85" if val >= 0 else "#e74c3c" for val in daily_pct]

    fig_pnl = go.Bar(
        x=daily_dates,
        y=daily_pct,
        marker_color=daily_colors if len(daily_colors) else "#2ebd85",
        name="Daily PnL (%)",
        hovertemplate="Date: %{x}<br>%{y:+.2f}%<extra></extra>",
        showlegend=False,
    )

    status_map = load_system_status()
    cell_labels = []
    cell_colors = []
    hover_text = []
    console_status = []
    for comp in SYSTEM_COMPONENTS:
        info = status_map.get(comp, DEFAULT_STATUS[comp])
        state = info.get("state", "UNKNOWN")
        emoji = info.get("emoji", "⬜")
        color = info.get("color", BORDER_GREY)
        note = info.get("note", "")
        cell_labels.append(f"{comp} {emoji}")
        cell_colors.append(color)
        hover_text.append(f"{comp} — Status: {state}{' (' + note + ')' if note else ''}")
        console_status.append(f"{comp} {state}")

    state_to_value = {"OK": 1.0, "WARN": 0.6, "ERROR": 0.2, "UNKNOWN": 0.4}
    tile_values = [state_to_value.get(status_map.get(comp, {}).get("state", "UNKNOWN"), 0.4) for comp in SYSTEM_COMPONENTS]

    fig_health = go.Heatmap(
        z=[tile_values],
        x=SYSTEM_COMPONENTS,
        y=["System"],
        text=[cell_labels],
        texttemplate="%{text}",
        textfont={"color": "white", "size": 18},
        colorscale=[
            [0.0, "#e74c3c"],
            [0.2, "#e74c3c"],
            [0.4, BORDER_GREY],
            [0.6, "#f39c12"],
            [1.0, "#00cc96"],
        ],
        zmin=0,
        zmax=1,
        showscale=False,
        customdata=[hover_text],
        hovertemplate="%{customdata}<extra></extra>",
    )

    hourly_map = load_hourly_notional_from_week_files()
    if hourly_map is None:
        hourly_map = load_hourly_notional_fallback_trust_pack()
        if hourly_map:
            print("[viz] using trust_pack_v2 hourly notional fallback")
    if hourly_map is None:
        print("[viz] notional-by-hour unavailable from current binance_week_* CSVs")
        trade_hourly = pd.DataFrame({"trade_hour": list(range(24)), "notional": [0.0] * 24})
    else:
        trade_hourly = pd.DataFrame({"trade_hour": list(range(24))})
        trade_hourly["notional"] = trade_hourly["trade_hour"].map(lambda hr: hourly_map.get(hr, 0.0))

    fig_trade_notional = go.Bar(
        x=trade_hourly["trade_hour"],
        y=trade_hourly["notional"] / 1000.0,
        marker_color="#f39c12",
        name="Trade Notional",
        hovertemplate="Hour %{x:02d}:00 UTC<br>Notional: $%{y:,.1f} k USDT<extra></extra>",
        showlegend=False,
    )

    calendar_vals = daily_series
    calendar_dates = daily_dates
    if len(calendar_vals):
        max_abs = max(abs(float(calendar_vals.max() or 0)), abs(float(calendar_vals.min() or 0)))
        if max_abs == 0:
            max_abs = 1
    else:
        max_abs = 1
    zmin, zmax = -max_abs, max_abs
    print("[viz] daily calendar rendered as USDT values (2dp)")

    fig_pnl_calendar = go.Heatmap(
        z=[calendar_vals.tolist()],
        x=calendar_dates,
        y=["PnL"],
        colorscale=[[0.0, "#e74c3c"], [0.5, "#7f8c8d"], [1.0, "#00cc96"]],
        zmin=zmin,
        zmax=zmax,
        zmid=0,
        showscale=False,
        text=[[f"${val:,.2f}" for val in calendar_vals]],
        texttemplate="%{text}",
        textfont={"color": "white"},
        hovertemplate="Date %{x}<br>PnL $%{z:,.2f}<extra></extra>",
        showlegend=False,
    )

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "pie"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "heatmap"}],
            [{"type": "xy"}, {"type": "heatmap"}],
        ],
        subplot_titles=[
            "Assets Under Management (AUM)",
            "Cumulative Performance vs BTC / ETH / BNB / XAUT (%)",
            "Daily PnL (%)",
            "System Status",
            "Trade Notional by Hour (UTC) — Last 7 Days",
            "Daily PnL Calendar (USDT)",
        ],
        vertical_spacing=0.12,
    )

    fig.add_trace(fig_aum, row=1, col=1)
    for trace in equity_traces:
        fig.add_trace(trace, row=1, col=2)
    fig.add_trace(fig_pnl, row=2, col=1)
    fig.add_trace(fig_health, row=2, col=2)
    fig.add_trace(fig_trade_notional, row=3, col=1)
    fig.add_trace(fig_pnl_calendar, row=3, col=2)

    fig.update_xaxes(title_text="Hour (UTC)", row=3, col=1, tickmode="linear", dtick=1)
    fig.update_yaxes(title_text="Trade Notional (k USDT)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=2)
    fig.update_yaxes(showticklabels=False, row=3, col=2)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Daily PnL (%)", row=2, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    subtitle_text = f"Total AUM ${total_aum_usd:,.2f} (≈ R{total_aum_zar:,.0f})"
    legend_config = dict(
        orientation="h",
        yanchor="bottom",
        y=0.98,
        xanchor="center",
        x=0.75,
        font=dict(size=11, color="#bbbbbb"),
        bgcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Update — {datetime.now():%d %b %Y} | Fund Overview (v2.7.6)"
                f"<br><span style='font-size:14px;'>{subtitle_text}</span>"
            ),
            x=0.5,
            font=dict(size=20, color="white"),
        ),
        paper_bgcolor="#0b0b0b",
        plot_bgcolor="#0b0b0b",
        font=dict(color="white", family="Arial"),
        showlegend=True,
        legend=legend_config,
        hovermode="x unified",
        height=1450,
        margin=dict(t=90, b=40, l=60, r=40),
    )

    html_path = os.path.join(REPORT_PATH, "fund_overview_v2_7.html")

    def footer_span(label, value):
        color = "#00cc96" if value >= 0 else "#e74c3c"
        return f"{label} <span style=\"color:{color};\">{value:+.2f}%</span>"

    footer_line = " | ".join(
        [
            footer_span("Week", weekly_changes["Fund"]),
            footer_span("BTC", weekly_changes["BTC"]),
            footer_span("ETH", weekly_changes["ETH"]),
            footer_span("BNB", weekly_changes["BNB"]),
            footer_span("XAUT", weekly_changes["XAUT"]),
        ]
    )
    footer_console = (
        "Fund {fund:+.2f}% vs BTC {btc:+.2f}% | ETH {eth:+.2f}% | BNB {bnb:+.2f}% | XAUT {xaut:+.2f}%".format(
            fund=weekly_changes.get("Fund", 0.0),
            btc=weekly_changes.get("BTC", 0.0),
            eth=weekly_changes.get("ETH", 0.0),
            bnb=weekly_changes.get("BNB", 0.0),
            xaut=weekly_changes.get("XAUT", 0.0),
        )
    )
    footer_div = (
        "<div style=\"text-align:center;color:white;font-family:Arial,sans-serif;"
        "font-size:14px;margin-top:18px;\">"
        f"{footer_line}"
        "</div>"
    )

    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
    if "</body>" in html_str:
        html_str = html_str.replace("</body>", f"{footer_div}</body>", 1)
    else:
        html_str += footer_div

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"[viz] system status: {' | '.join(console_status)}")
    print(f"[viz] {footer_console}")
    print(f"→ {html_path}\n")
    return fig


if __name__ == "__main__":
    build_trust_pack()
