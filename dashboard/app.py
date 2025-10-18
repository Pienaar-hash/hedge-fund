#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations
"""Streamlit dashboard (single "Overview" tab), read-only.
Firestore-first; falls back to local files for NAV.
Shows: KPIs, NAV chart, Positions, Signals (5), Trade log (5), Screener tail (10), BTC reserve.
"""
# Streamlit dashboard (single "Overview" tab), read-only.
# Firestore-first; falls back to local files for NAV.

# ---- tolerant dotenv ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import os
import json
import time
import logging
from typing import Any, Dict, List, Mapping, Optional
import pandas as pd
import streamlit as st
from pathlib import Path

try:
    from flask import Flask, jsonify, render_template_string, request
except Exception:  # pragma: no cover - dashboard fallback
    Flask = None  # type: ignore
    jsonify = None  # type: ignore
    render_template_string = None  # type: ignore
    request = None  # type: ignore

from execution.signal_generator import generate_intents, normalize_intent
from dashboard.dashboard_utils import (
    get_firestore_connection,
    fetch_state_document,
    fetch_telemetry_health,
    compute_total_nav_cached,
    parse_nav_to_df_and_kpis,
    positions_sorted,
    fetch_mark_price_usdt,
    get_env_float,
)
from dashboard.nav_helpers import (
    format_treasury_table,
    signal_attempts_summary,
)
from dashboard.live_helpers import get_nav_snapshot, get_caps, get_veto_counts, get_treasury

try:
    from scripts.polymarket_insiders import get_polymarket_snapshot
except Exception:
    def get_polymarket_snapshot() -> List[Dict[str, Any]]:
        return []

# Diagnostics container populated during data loads
_DIAG: Dict[str, Any] = {
    "source": None,
    "fs_project": None,
    "col_nav": None,
    "col_positions": None,
    "col_trades": None,
    "col_risk": None,
}

# ---- single env context ----
ENV = os.getenv("ENV", "prod")
TESTNET = os.getenv("BINANCE_TESTNET", "0") == "1"
ENV_KEY = f"{ENV}{'-testnet' if TESTNET else ''}"
DRY_RUN = os.getenv("DRY_RUN", "0")

RESERVE_BTC = float(os.getenv("RESERVE_BTC", "0.025"))
# Local-only log files when Firestore is not selected
LOG_PATH = os.getenv("EXECUTOR_LOG", f"logs/{ENV_KEY}/screener_tail.log")
TAIL_LINES = 10
RISK_CFG_PATH = os.getenv("RISK_LIMITS_CONFIG", "config/risk_limits.json")

LOGGER = logging.getLogger("dashboard.live")
DEFAULT_POLL_MS = 15000
DEFAULT_POLY_POLL_MS = 30000
STREAMLIT_REFRESH_MS = 15000
FOOTER_VERSION = "v4.2-stability-sync"

_INTENT_STATUS_STYLE = {
    "live": "background-color: #0f5132; color: #e6ffed;",
    "pending": "background-color: #8a6d1d; color: #fff4c2;",
    "vetoed": "background-color: #3c3c3c; color: #f2f2f2;",
}

def _status_cell_style(val: Any) -> str:
    if isinstance(val, str):
        return _INTENT_STATUS_STYLE.get(val.lower(), "")
    return ""

_LIVE_PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Live Signals</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 16px; background: #111; color: #f5f5f5; }
    header { margin-bottom: 20px; }
    h1 { margin: 0 0 8px 0; font-size: 1.6rem; }
    h2 { margin: 0 0 12px 0; font-size: 1.2rem; }
    section { margin-bottom: 24px; padding: 12px 16px; background: #1c1c1c; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    th, td { border-bottom: 1px solid #333; padding: 6px 8px; text-align: left; font-size: 0.95rem; }
    th { color: #bbb; }
    .badge { display: inline-block; background: #2c2c2c; padding: 4px 8px; margin-right: 8px; border-radius: 4px; font-size: 0.85rem; }
    #fetch-status { margin-top: 6px; font-size: 0.85rem; }
    .status-ok { color: #7ed957; }
    .status-warn { color: #ff9800; }
    ul { margin: 8px 0 0 16px; padding: 0; }
    li { margin-bottom: 4px; }
    tfoot td { border-top: 1px solid #333; font-weight: 600; }
    tfoot td:last-child { text-align: right; }
  </style>
</head>
<body>
  <header>
    <h1>Unified Live Signals</h1>
    <div>
      <span class="badge">ENV: {{ env_label }}</span>
      <span class="badge">DRY_RUN: {{ dry_run_label }}</span>
      <span class="badge">Poll: <span id="poll-interval">{{ default_poll }}</span> ms</span>
    </div>
    <div id="fetch-status" class="status-warn">Waiting for first update…</div>
  </header>
  <main>
    <section>
      <h2>Signals (polled)</h2>
      <div>Last poll: <span id="last-poll">–</span></div>
      <div>Attempted: <span id="attempted">0</span> | Emitted: <span id="emitted">0</span></div>
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Timeframe</th>
            <th>Signal</th>
            <th>Gross USD</th>
            <th>Reduce Only</th>
            <th>Position Side</th>
            <th>Veto</th>
          </tr>
        </thead>
        <tbody id="signals-body">
          <tr><td colspan="7">Waiting for data…</td></tr>
        </tbody>
      </table>
    </section>
    <section>
      <h2>Risk &amp; NAV</h2>
      <div>NAV: <span id="nav-nav">0</span> | Equity: <span id="nav-equity">0</span> | Snapshot: <span id="nav-ts">–</span></div>
      <div>max_trade_nav_pct: <span id="caps-trade">0</span></div>
      <div>max_gross_exposure_pct: <span id="caps-gross">0</span></div>
      <div>max_symbol_exposure_pct: <span id="caps-symbol">0</span></div>
      <div>min_notional: <span id="caps-min-notional">0</span></div>
    </section>
    <section>
      <h2>Treasury</h2>
      <table>
        <thead>
          <tr>
            <th>Asset</th>
            <th>Balance</th>
            <th>Price (USD)</th>
            <th>USD Value</th>
          </tr>
        </thead>
        <tbody id="treasury-body">
          <tr><td colspan="4">Waiting for data…</td></tr>
        </tbody>
        <tfoot>
          <tr>
            <td colspan="3">Total USD</td>
            <td id="treasury-total">$0.00</td>
          </tr>
        </tfoot>
      </table>
    </section>
    <section>
      <h2>Veto summary</h2>
      <ul id="veto-summary">
        <li>Waiting for data…</li>
      </ul>
    </section>
  </main>
  <script>
    (function() {
      const params = new URLSearchParams(window.location.search);
      const defaultPoll = parseInt('{{ default_poll }}', 10) || 10000;
      const pollMs = parseInt(params.get('poll') || defaultPoll, 10) || defaultPoll;
      document.getElementById('poll-interval').textContent = pollMs;
      const statusEl = document.getElementById('fetch-status');
      const formatTs = (value) => {
        if (!value) { return '–'; }
        const num = Number(value);
        if (!Number.isFinite(num)) { return String(value); }
        return new Date(num * 1000).toLocaleString();
      };
      const updateSignals = (intents) => {
        const body = document.getElementById('signals-body');
        body.innerHTML = '';
        if (!intents || intents.length === 0) {
          body.innerHTML = '<tr><td colspan="7">No intents emitted.</td></tr>';
          return;
        }
        intents.slice(0, 25).forEach((intent) => {
          const row = document.createElement('tr');
          const cells = [
            intent.symbol || intent.pair || '',
            intent.timeframe || intent.tf || '',
            intent.signal || '',
            (Number(intent.gross_usd || 0)).toFixed(2),
            intent.reduceOnly ? 'yes' : 'no',
            intent.positionSide || intent.side || '',
            (intent.veto && intent.veto.length ? intent.veto.join(', ') : '')
          ];
          cells.forEach((text) => {
            const td = document.createElement('td');
            td.textContent = text;
            row.appendChild(td);
          });
          body.appendChild(row);
        });
      };
      const updateVeto = (counts) => {
        const list = document.getElementById('veto-summary');
        list.innerHTML = '';
        if (!counts || Object.keys(counts).length === 0) {
          list.innerHTML = '<li>No veto activity.</li>';
          return;
        }
        Object.entries(counts)
          .sort((a, b) => Number(b[1]) - Number(a[1]))
          .forEach(([reason, count]) => {
            const li = document.createElement('li');
            li.textContent = reason + ': ' + count;
            list.appendChild(li);
          });
      };
      const updateTreasury = (treasury) => {
        const body = document.getElementById('treasury-body');
        if (!body) { return; }
        body.innerHTML = '';
        const assets = (treasury && Array.isArray(treasury.assets)) ? treasury.assets : [];
        if (!assets.length) {
          body.innerHTML = '<tr><td colspan="4">No treasury data.</td></tr>';
        } else {
          assets.forEach((entry) => {
            const row = document.createElement('tr');
            const balance = Number(entry.balance ?? 0);
            const price = Number(entry.price ?? 0);
            const usd = Number(entry.usd ?? 0);
            const cells = [
              entry.asset || '',
              balance.toFixed(6),
              '$' + price.toFixed(2),
              '$' + usd.toFixed(2)
            ];
            cells.forEach((text, idx) => {
              const td = document.createElement('td');
              td.textContent = text;
              if (idx > 0) {
                td.style.textAlign = 'right';
              }
              row.appendChild(td);
            });
            body.appendChild(row);
          });
        }
        const totalEl = document.getElementById('treasury-total');
        if (totalEl) {
          const total = Number(treasury && treasury.total_usd ? treasury.total_usd : 0);
          totalEl.textContent = '$' + total.toFixed(2);
        }
      };
      const applyData = (data) => {
        const now = new Date();
        document.getElementById('last-poll').textContent = now.toLocaleTimeString();
        document.getElementById('attempted').textContent = data.attempted ?? 0;
        document.getElementById('emitted').textContent = data.emitted ?? 0;
        updateSignals(data.intents || []);
        const nav = data.nav || {};
        document.getElementById('nav-nav').textContent = (Number(nav.nav || 0)).toFixed(2);
        document.getElementById('nav-equity').textContent = (Number(nav.equity || nav.nav || 0)).toFixed(2);
        document.getElementById('nav-ts').textContent = formatTs(nav.ts || data.ts);
        const caps = data.caps || {};
        document.getElementById('caps-trade').textContent = caps.max_trade_nav_pct ?? 0;
        document.getElementById('caps-gross').textContent = caps.max_gross_exposure_pct ?? 0;
        document.getElementById('caps-symbol').textContent = caps.max_symbol_exposure_pct ?? 0;
        document.getElementById('caps-min-notional').textContent = caps.min_notional ?? 0;
        updateVeto(data.veto_counts || {});
        updateTreasury(data.treasury || {});
        statusEl.textContent = 'Last update ' + now.toLocaleTimeString();
        statusEl.className = 'status-ok';
      };
      const fetchData = async () => {
        try {
          const resp = await fetch('{{ api_path }}');
          if (!resp.ok) {
            throw new Error('HTTP ' + resp.status);
          }
          const data = await resp.json();
          applyData(data);
        } catch (err) {
          statusEl.textContent = 'Fetch failed: ' + (err && err.message ? err.message : err);
          statusEl.className = 'status-warn';
          console.error(err);
        }
      };
      fetchData();
      setInterval(fetchData, pollMs);
    })();
  </script>
</body>
</html>
"""

_POLY_PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Polymarket Insights</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 16px; background: #0f1115; color: #f4f4f4; }
    header { margin-bottom: 20px; }
    h1 { margin: 0 0 8px 0; font-size: 1.5rem; }
    section { margin-bottom: 24px; padding: 14px 18px; background: #1b1e26; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.35); }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    th, td { border-bottom: 1px solid #2a2e3b; padding: 8px 10px; text-align: left; font-size: 0.95rem; }
    th { color: #9fa6b6; text-transform: uppercase; letter-spacing: 0.04em; font-size: 0.85rem; }
    tr:last-child td { border-bottom: none; }
    .muted { color: #7a8194; font-size: 0.85rem; }
    .badge { display: inline-block; background: #272b38; padding: 4px 8px; border-radius: 4px; margin-right: 6px; font-size: 0.85rem; }
    .status { margin-top: 6px; font-size: 0.85rem; }
    .status-ok { color: #6dd07f; }
    .status-warn { color: #ffb55a; }
    td.numeric { text-align: right; }
  </style>
</head>
<body>
  <header>
    <h1>Polymarket Insights</h1>
    <div>
      <span class="badge">Poll: <span id="poly-poll-interval">{{ default_poll }}</span> ms</span>
    </div>
    <div id="poly-status" class="status status-warn">Waiting for first update…</div>
  </header>
  <main>
    <section>
      <div class="muted">Latest curated signals from scripts/polymarket_insiders.py</div>
      <table>
        <thead>
          <tr>
            <th>Market</th>
            <th>Side</th>
            <th>Confidence</th>
            <th>Updated</th>
          </tr>
        </thead>
        <tbody id="poly-body">
          <tr><td colspan="4">Loading…</td></tr>
        </tbody>
      </table>
    </section>
  </main>
  <script>
    (function() {
      const params = new URLSearchParams(window.location.search);
      const defaultPoll = parseInt('{{ default_poll }}', 10) || 30000;
      const pollMs = parseInt(params.get('poll') || defaultPoll, 10) || defaultPoll;
      document.getElementById('poly-poll-interval').textContent = pollMs;
      const statusEl = document.getElementById('poly-status');
      const formatScore = (value) => {
        if (value === null || value === undefined) { return '–'; }
        const num = Number(value);
        if (Number.isFinite(num)) {
          if (Math.abs(num) < 1) {
            return (num * 100).toFixed(1) + '%';
          }
          return num.toFixed(2);
        }
        return String(value);
      };
      const formatTime = (value) => {
        if (!value) { return '–'; }
        const num = Number(value);
        if (Number.isFinite(num)) {
          if (num > 1e12) {
            return new Date(num).toLocaleString();
          }
          if (num > 1e3) {
            return new Date(num * 1000).toLocaleString();
          }
        }
        const date = new Date(value);
        if (!Number.isNaN(date.getTime())) {
          return date.toLocaleString();
        }
        return String(value);
      };
      const updateTable = (rows) => {
        const body = document.getElementById('poly-body');
        body.innerHTML = '';
        if (!rows || !rows.length) {
          body.innerHTML = '<tr><td colspan="4">No signals available.</td></tr>';
          return;
        }
        rows.slice(0, 50).forEach((row) => {
          const tr = document.createElement('tr');
          const cells = [
            row.market || '',
            row.side || '',
            formatScore(row.score),
            formatTime(row.updated_at)
          ];
          cells.forEach((text, idx) => {
            const td = document.createElement('td');
            td.textContent = text;
            if (idx === 2) {
              td.classList.add('numeric');
            }
            tr.appendChild(td);
          });
          body.appendChild(tr);
        });
      };
      const applyData = (payload) => {
        updateTable(payload.markets || []);
        statusEl.textContent = 'Last refresh ' + new Date().toLocaleTimeString();
        statusEl.className = 'status status-ok';
      };
      const fetchData = async () => {
        try {
          const resp = await fetch('{{ api_path }}');
          if (!resp.ok) {
            throw new Error('HTTP ' + resp.status);
          }
          const data = await resp.json();
          applyData(data);
        } catch (err) {
          statusEl.textContent = 'Fetch failed: ' + (err && err.message ? err.message : err);
          statusEl.className = 'status status-warn';
          console.error(err);
        }
      };
      fetchData();
      setInterval(fetchData, pollMs);
    })();
  </script>
</body>
</html>
"""


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


if Flask is not None:
    flask_app = Flask(__name__)

    @flask_app.route("/live")
    def live_signals_page():
        env_label = ENV_KEY
        dry_label = str(DRY_RUN if DRY_RUN not in (None, "") else "0")
        return render_template_string(
            _LIVE_PAGE_TEMPLATE,
            env_label=env_label,
            dry_run_label=dry_label,
            default_poll=DEFAULT_POLL_MS,
            api_path="/api/live_signals",
        )

    @flask_app.route("/api/live_signals")
    def api_live_signals():
        now = time.time()
        try:
            raw_intents = generate_intents(now)
        except Exception as exc:
            LOGGER.error("generate_intents failed: %s", exc, exc_info=True)
            raw_intents = []

        intents_payload: List[Dict[str, Any]] = []
        emitted = 0
        for raw in raw_intents:
            try:
                normalized = normalize_intent(raw)
            except Exception as exc:
                LOGGER.debug("intent normalization failed: %s", exc)
                continue
            veto_raw = normalized.get("veto")
            if isinstance(veto_raw, str):
                veto_list = [veto_raw]
            elif isinstance(veto_raw, (list, tuple, set)):
                veto_list = [str(item) for item in veto_raw if item]
            else:
                veto_list = []
            sanitized = {
                "symbol": str(normalized.get("symbol") or "").upper(),
                "timeframe": str(normalized.get("timeframe") or normalized.get("tf") or ""),
                "tf": str(normalized.get("tf") or normalized.get("timeframe") or ""),
                "signal": str(normalized.get("signal") or ""),
                "gross_usd": _safe_float(normalized.get("gross_usd")),
                "reduceOnly": bool(normalized.get("reduceOnly")),
                "positionSide": str(normalized.get("positionSide") or normalized.get("side") or ""),
                "timestamp": str(normalized.get("timestamp") or ""),
                "price": _safe_float(normalized.get("price")),
                "capital_per_trade": _safe_float(normalized.get("capital_per_trade")),
                "leverage": _safe_float(normalized.get("leverage")),
                "veto": veto_list,
            }
            intents_payload.append(sanitized)
            if not veto_list:
                emitted += 1

        attempted = len(raw_intents)
        nav_snapshot = get_nav_snapshot()
        caps = get_caps()
        veto_counts = get_veto_counts()
        treasury = get_treasury()

        response = {
            "attempted": attempted,
            "emitted": emitted if attempted else len(intents_payload),
            "intents": intents_payload,
            "nav": nav_snapshot,
            "caps": caps,
            "veto_counts": veto_counts,
            "treasury": treasury,
            "ts": now,
        }
        return jsonify(response)

    @flask_app.route("/polymarket")
    def polymarket_page():
        return render_template_string(
            _POLY_PAGE_TEMPLATE,
            default_poll=DEFAULT_POLY_POLL_MS,
            api_path="/api/polymarket",
        )

    @flask_app.route("/api/polymarket")
    def api_polymarket():
        now = time.time()
        try:
            snapshot = get_polymarket_snapshot()
        except Exception as exc:
            LOGGER.error("polymarket snapshot failed: %s", exc, exc_info=True)
            snapshot = []

        entries: List[Dict[str, Any]] = []
        for item in snapshot:
            if not isinstance(item, Mapping):
                continue
            market = str(item.get("market") or item.get("name") or "")
            side = str(item.get("side") or item.get("position") or "")
            score_raw = item.get("score", item.get("edge", item.get("confidence")))
            if isinstance(score_raw, (int, float)):
                score: Any = float(score_raw)
            elif score_raw in (None, ""):
                score = None
            else:
                score = str(score_raw)
            updated_raw = item.get("updated_at") or item.get("ts") or item.get("timestamp")
            updated = str(updated_raw) if updated_raw not in (None, "") else ""
            entries.append(
                {
                    "market": market,
                    "side": side,
                    "score": score,
                    "updated_at": updated,
                }
            )

        return jsonify({"markets": entries, "ts": now})

    app = flask_app
else:
    flask_app = None
    app = None

# ---------- helpers ----------
def btc_24h_change() -> float | None:
    import requests
    # Use TESTNET flag from ENV wiring; default to mainnet
    base = "https://testnet.binancefuture.com" if TESTNET else "https://fapi.binance.com"
    try:
        r = requests.get(base + "/fapi/v1/ticker/24hr", params={"symbol":"BTCUSDT"}, timeout=6)
        r.raise_for_status()
        return float(r.json().get("priceChangePercent", 0.0))
    except Exception:
        try:
            r = requests.get("https://api.binance.com/api/v3/ticker/24hr", params={"symbol":"BTCUSDT"}, timeout=6)
            r.raise_for_status()
            return float(r.json().get("priceChangePercent", 0.0))
        except Exception:
            return None

def _load_json(path: str, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

# Firestore (best effort)
def _fs_client():
    """Return Firestore client if libs + credentials are available.
    Firestore is authoritative only if:
    - google.cloud.firestore imports OK
    - And credentials file path is present: FIREBASE_CREDS_PATH or GOOGLE_APPLICATION_CREDENTIALS
    """
    try:
        from google.cloud import firestore  # type: ignore
    except Exception:
        return None

    creds_path = os.getenv("FIREBASE_CREDS_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).exists():
        try:
            return firestore.Client()
        except Exception:
            return None
    return None

# --- Firestore paths helper (correct sibling collections) ---
def fs_paths(db):
    ROOT_DOC = db.collection("hedge").document(ENV)
    STATE_COL = ROOT_DOC.collection("state")
    return {
        "root_doc": ROOT_DOC,
        "state_col": STATE_COL,
        "nav_doc": STATE_COL.document("nav"),
        "positions_doc": STATE_COL.document("positions"),
        "trades_col": ROOT_DOC.collection("trades"),
        "risk_col": ROOT_DOC.collection("risk"),
    }

def _fs_get_state(doc: str):
    """Back-compat read for single-doc state if present under old path.
    This is best-effort and namespacing by ENV only. Prefer collection helpers below.
    """
    cli = _fs_client()
    if not cli:
        return None
    try:
        snap = cli.document(f"hedge/{ENV}/state/{doc}").get()
        return snap.to_dict() if getattr(snap, "exists", False) else None
    except Exception:
        return None

# ---- Firestore helpers (authoritative when selected) ----
def _fs_pick_collection(cli, candidates: List[str]) -> Optional[str]:
    """Pick the first collection that appears to have docs. Returns name or None.
    Document choice in Doctor panel later.
    """
    for name in candidates:
        try:
            it = cli.collection(name).limit(1).stream()
            for _ in it:
                return name
        except Exception:
            continue
    return None

def _filter_env_fields(doc: Dict[str, Any]) -> bool:
    """Client-side filter: if env/testnet fields exist, they must match.
    Always applied for generic collections.
    """
    d_env = doc.get("env")
    d_tn = doc.get("testnet")
    if d_env is not None and str(d_env) != ENV:
        return False
    if d_tn is not None and bool(d_tn) != TESTNET:
        return False
    return True

def _is_mixed_namespaces(docs: List[Dict[str, Any]]) -> bool:
    env_vals = set()
    tn_vals = set()
    for d in docs:
        if "env" in d:
            env_vals.add(str(d.get("env")))
        if "testnet" in d:
            tn_vals.add(bool(d.get("testnet")))
    return (len(env_vals) > 1) or (len(tn_vals) > 1)

def _get_firestore_project_id() -> Optional[str]:
    try:
        from google.auth import default  # type: ignore
        creds, proj = default()
        return proj
    except Exception:
        return None

def _load_risk_cfg():
    try:
        return json.loads(Path(RISK_CFG_PATH).read_text())
    except Exception:
        return {}

# ---------- sources ----------
def _select_data_source() -> Dict[str, Any]:
    """Pick single data source for this render and cache useful paths.
    - If Firestore client available AND creds path present -> use Firestore exclusively.
    - Else use local namespaced files under state/{ENV_KEY} and logs/{ENV_KEY}.
    """
    cli = _fs_client()
    source = "firestore" if cli else "local"
    local_paths = {
        "nav": f"state/{ENV_KEY}/nav_log.json",
        "risk": f"logs/{ENV_KEY}/risk.log",
        "screener": f"logs/{ENV_KEY}/screener_tail.log",
    }
    _DIAG["source"] = source
    if source == "firestore":
        _DIAG["fs_project"] = _get_firestore_project_id()
    return {"source": source, "cli": cli, "local": local_paths}

_DS = _select_data_source()

def load_nav_series() -> List[Dict[str,Any]]:
    # Authoritative selection: Firestore (nested doc) OR Local, not both
    if _DS["source"] == "firestore" and _DS.get("cli"):
        cli = _DS["cli"]
        try:
            p = fs_paths(cli)
            doc = p["nav_doc"].get()
            if not getattr(doc, "exists", False):
                series: List[Dict[str, Any]] = []
            else:
                data = doc.to_dict() or {}
                series = []
                for _, v in (data.items() if isinstance(data, dict) else []):
                    if not isinstance(v, dict):
                        continue
                    try:
                        nav = float(v.get("nav", 0.0))
                    except Exception:
                        nav = 0.0
                    t = v.get("t") or v.get("ts") or v.get("time")
                    if hasattr(t, "timestamp"):
                        try:
                            ts = float(t.timestamp())
                        except Exception:
                            ts = None
                    elif isinstance(t, (int, float)):
                        ts = float(t) / 1000.0 if float(t) > 1e12 else float(t)
                    else:
                        ts = None
                    if ts is None:
                        continue
                    series.append({"ts": ts, "nav": nav})
                series.sort(key=lambda r: r["ts"])  # oldest->newest
            _DIAG["col_nav"] = f"hedge/{ENV}/state/nav"
            return series
        except Exception:
            return []
    # Local fallback (single file)
    p = _DS["local"]["nav"]
    js = _load_json(p, [])
    if isinstance(js, dict):
        return js.get("rows") or js.get("series") or js.get("nav") or []
    return js if isinstance(js, list) else []

def load_positions() -> List[Dict[str,Any]]:
    """Return ONLY open futures positions (non-zero qty). Columns:
    symbol, side, qty, entryPrice, markPrice, unrealizedPnl, leverage, updatedAt
    Read from single document hedge/{ENV}/state/positions. Accept {rows:[..]} or mapping.
    """
    rows: List[Dict[str, Any]] = []
    if _DS["source"] == "firestore" and _DS.get("cli"):
        cli = _DS["cli"]
        try:
            p = fs_paths(cli)
            snap = p["positions_doc"].get()
            data = snap.to_dict() if getattr(snap, "exists", False) else None
            if isinstance(data, dict):
                if isinstance(data.get("rows"), list):
                    rows = list(data.get("rows") or [])
                else:
                    rows = []
                    for sym, rec in data.items():
                        if isinstance(rec, dict):
                            rec = {**rec}
                            rec.setdefault("symbol", sym)
                            rows.append(rec)
            _DIAG["col_positions"] = f"hedge/{ENV}/state/positions"
        except Exception:
            rows = []
    else:
        rows = []

    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            qty_val = r.get("positionAmt") if r.get("positionAmt") is not None else r.get("qty")
            qty = float(qty_val or 0)
            if abs(qty) <= 0:
                continue
            entry = float(r.get("entryPrice") or r.get("avgEntryPrice") or r.get("entry_price") or 0)
            mark = float(r.get("markPrice") or r.get("mark") or r.get("lastPrice") or r.get("mark_price") or 0)
            upnl = float(r.get("unrealizedPnl") or r.get("uPnl") or r.get("unrealized") or r.get("u_pnl") or 0)
            lev = float(r.get("leverage") or r.get("lev") or 0)
            ts = r.get("updatedAt") or r.get("ts") or r.get("time")
            side = "LONG" if qty > 0 else "SHORT"
            out.append({
                "symbol": r.get("symbol"),
                "side": side,
                "qty": abs(qty),
                "entryPrice": entry,
                "markPrice": mark,
                "unrealizedPnl": upnl,
                "leverage": lev,
                "updatedAt": ts,
            })
        except Exception:
            continue
    # sort by updatedAt/ts desc
    def _sort_key(x: Dict[str, Any]):
        ts = x.get("updatedAt") or x.get("ts") or x.get("time")
        return _to_epoch_seconds(ts)
    out.sort(key=_sort_key, reverse=True)
    return out

def _literal_tail(tag: str, n: int) -> List[str]:
    """Return last n lines from local screener log containing the literal tag.
    Only used when local files are the selected source.
    """
    if _DS["source"] != "local":
        return []
    try:
        with open(_DS["local"]["screener"], "r", errors="ignore") as f:
            lines = f.readlines()[-10000:]
    except Exception:
        return []
    hits = [ln.rstrip("\n") for ln in lines if tag in ln]
    return hits[-n:]

def load_signals_table(limit: int = 100) -> pd.DataFrame:
    """Return recent signals using generate_intents feed; fallback to screener tail when empty."""
    rows: List[Dict[str, Any]] = []
    now = time.time()
    def _status_from_intent(gross: float, vetoed: bool) -> str:
        if vetoed:
            return "vetoed"
        if gross > 0:
            return "live"
        return "pending"
    try:
        intents = generate_intents(now)
    except Exception as exc:
        LOGGER.debug("generate_intents unavailable for dashboard: %s", exc, exc_info=True)
        intents = []

    if intents:
        for intent in intents:
            if not isinstance(intent, Mapping):
                continue
            ts_raw = intent.get("timestamp") or intent.get("t") or intent.get("ts") or intent.get("time")
            t_epoch = _to_epoch_seconds(ts_raw) or now
            if not is_recent(t_epoch, 24 * 3600):
                continue
            veto_field = intent.get("veto") or []
            if isinstance(veto_field, (list, tuple, set)):
                veto_display = ", ".join(str(item) for item in veto_field if item)
            elif veto_field in (None, ""):
                veto_display = ""
            else:
                veto_display = str(veto_field)
            gross_val = _safe_float(intent.get("gross_usd"))
            status = _status_from_intent(gross_val, bool(veto_display))
            rows.append(
                {
                    "ts": t_epoch,
                    "symbol": intent.get("symbol"),
                    "tf": intent.get("timeframe") or intent.get("tf"),
                    "signal": intent.get("signal"),
                    "price": _safe_float(intent.get("price")),
                    "cap": _safe_float(intent.get("capital_per_trade")),
                    "gross_usd": _safe_float(intent.get("gross_usd")),
                    "lev": _safe_float(intent.get("leverage")),
                    "reduceOnly": bool(intent.get("reduceOnly")),
                    "veto": veto_display,
                    "status": status,
                }
            )
        rows.sort(key=lambda x: x.get("ts") or 0.0, reverse=True)
        return pd.DataFrame(rows[:limit])

    # Fallback: local screener tail parser (legacy flow)
    if _DS["source"] != "local":
        return pd.DataFrame(rows)

    import ast

    tag = "[screener->executor]"
    raw = _literal_tail(tag, 200)
    for ln in raw:
        try:
            payload = ln.split(tag, 1)[1].strip() if tag in ln else None
            if not payload:
                continue
            obj: Optional[Dict[str, Any]] = None
            if payload.startswith("{"):
                try:
                    obj = json.loads(payload)
                except Exception:
                    try:
                        obj = ast.literal_eval(payload)
                    except Exception:
                        obj = None
            if not isinstance(obj, dict):
                continue
            ts = obj.get("timestamp") or obj.get("t") or obj.get("time")
            t_epoch = _to_epoch_seconds(ts)
            if not is_recent(t_epoch, 24 * 3600):
                continue
            rows.append(
                {
                    "ts": t_epoch,
                    "symbol": obj.get("symbol"),
                    "tf": obj.get("timeframe"),
                    "signal": obj.get("signal"),
                    "price": _safe_float(obj.get("price")),
                    "cap": _safe_float(obj.get("capital_per_trade")),
                    "gross_usd": _safe_float(obj.get("gross_usd")),
                    "lev": _safe_float(obj.get("leverage")),
                    "reduceOnly": bool(obj.get("reduceOnly")),
                    "veto": "",
                    "status": "pending",
                }
            )
        except Exception:
            continue
    rows.sort(key=lambda x: x.get("ts") or 0.0, reverse=True)
    return pd.DataFrame(rows[:limit])

def load_trade_log(limit: int = 100) -> pd.DataFrame:
    """Trades (last 24h) — Use hedge/{ENV}/trades only. Local returns empty.
    Apply env/testnet filter if fields exist. Order desc.
    """
    rows: List[Dict[str, Any]] = []
    if _DS["source"] == "firestore" and _DS.get("cli"):
        cli = _DS["cli"]
        try:
            p = fs_paths(cli)
            col = p["trades_col"]
            docs = list(col.order_by("ts", direction="DESCENDING").limit(1000).stream())
            for d in docs:
                x = d.to_dict() or {}
                # env/testnet filter if present
                if (x.get("env") is not None and str(x.get("env")) != ENV) or (
                    x.get("testnet") is not None and bool(x.get("testnet")) != TESTNET
                ):
                    continue
                ts = _to_epoch_seconds(x.get("ts") or x.get("time"))
                if not is_recent(ts, 24*3600):
                    continue
                rows.append({
                    "ts": ts,
                    "symbol": x.get("symbol"),
                    "side": x.get("side"),
                    "qty": x.get("qty"),
                    "price": x.get("price"),
                    "pnl": x.get("pnl"),
                })
            _DIAG["col_trades"] = f"hedge/{ENV}/trades"
        except Exception:
            rows = []
        rows.sort(key=lambda x: x.get("ts") or 0.0, reverse=True)
        return pd.DataFrame(rows[:limit])
    # Local (no trades file in this simplified app)
    return pd.DataFrame(rows[:limit])

def load_blocked_orders(limit: int = 200) -> pd.DataFrame:
    """Risk blocks (last 24h) — Use hedge/{ENV}/risk only for Firestore.
    Local: parse risk log if any; else empty.
    """
    rows: List[Dict[str, Any]] = []
    if _DS["source"] == "firestore" and _DS.get("cli"):
        cli = _DS["cli"]
        try:
            p = fs_paths(cli)
            col = p["risk_col"]
            docs = list(col.order_by("ts", direction="DESCENDING").limit(2000).stream())
            for d in docs:
                x = d.to_dict() or {}
                # env/testnet filter if present
                if (x.get("env") is not None and str(x.get("env")) != ENV) or (
                    x.get("testnet") is not None and bool(x.get("testnet")) != TESTNET
                ):
                    continue
                ts = _to_epoch_seconds(x.get("ts") or x.get("t") or x.get("time"))
                if not is_recent(ts, 24*3600):
                    continue
                # keep only blocked if phase exists; otherwise keep all
                if "phase" in x and str(x.get("phase")) != "blocked":
                    continue
                rows.append({
                    "ts": ts,
                    "symbol": x.get("symbol"),
                    "side": x.get("side"),
                    "reason": x.get("reason"),
                    "notional": x.get("notional"),
                    "open_qty": x.get("open_qty"),
                    "gross": x.get("gross"),
                    "nav": x.get("nav"),
                })
            _DIAG["col_risk"] = f"hedge/{ENV}/risk"
        except Exception:
            rows = []
        rows.sort(key=lambda x: x.get("ts") or 0.0, reverse=True)
        return pd.DataFrame(rows[:limit])
    # Local fallback: parse risk log if configured
    if _DS["source"] == "local":
        try:
            with open(_DS["local"]["risk"], "r", errors="ignore") as f:
                lines = f.readlines()[-5000:]
            for ln in lines:
                try:
                    x = json.loads(ln.strip())
                except Exception:
                    continue
                if (x.get("phase") or "") != "blocked":
                    continue
                t_epoch = _to_epoch_seconds(x.get("t") or x.get("ts") or x.get("time"))
                if not is_recent(t_epoch, 24*3600):
                    continue
                rows.append({
                    "ts": t_epoch,
                    "symbol": x.get("symbol"),
                    "side": x.get("side"),
                    "reason": x.get("reason"),
                    "notional": x.get("notional"),
                    "open_qty": x.get("open_qty"),
                    "gross": x.get("gross"),
                    "nav": x.get("nav"),
                })
        except Exception:
            rows = []
    rows.sort(key=lambda x: x.get("ts") or 0.0, reverse=True)
    return pd.DataFrame(rows[:limit])

# ---- time helpers / recency ----
NOW = time.time()

def _to_epoch_seconds(ts: Any) -> float:
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        x = float(ts)
        return x / 1000.0 if x > 1e12 else x
    try:
        return pd.to_datetime(str(ts), utc=True).timestamp()
    except Exception:
        return 0.0

def is_recent(ts: float, window_sec: int) -> bool:
    if not ts:
        return False
    return (NOW - float(ts)) <= float(window_sec)

def humanize_ago(ts: Optional[float]) -> str:
    if not ts:
        return "(no recent data)"
    delta = max(0.0, NOW - ts)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta//60)}m ago"
    if delta < 86400:
        return f"{int(delta//3600)}h ago"
    return f"{int(delta//86400)}d ago"

# ---------- KPIs ----------
def compute_nav_kpis(series: List[Dict[str,Any]]) -> Dict[str,Any]:
    if not series:
        return {"nav": None, "nav_24h": None, "delta": 0.0, "delta_pct": 0.0, "dd": None}
    def _ts(v):
        t = v.get("t") or v.get("ts") or v.get("time")
        # Allow epoch seconds/ms or ISO strings
        if isinstance(t, (int,float)):
            x = float(t)
            if x > 1e12:   # ms
                return x/1000.0
            return x       # seconds
        if isinstance(t, str):
            try:
                return pd.to_datetime(t, utc=True).timestamp()
            except Exception:
                return 0.0
        return 0.0
    def _val(v):
        for k in ("nav","value","equity","v"):
            if k in v:
                try:
                    return float(v[k])
                except Exception:
                    pass
        return None

    rows = [( _ts(x), _val(x)) for x in series if isinstance(x, dict)]
    rows = [(t,v) for (t,v) in rows if t and v is not None]
    if not rows:
        return {"nav": None, "nav_24h": None, "delta": 0.0, "delta_pct": 0.0, "dd": None}
    rows.sort(key=lambda x: x[0])
    nav_now = rows[-1][1]
    cutoff = time.time() - 24*3600
    past_vals = [v for (t,v) in rows if t <= cutoff]
    nav_24h = past_vals[-1] if past_vals else nav_now
    peak = max(v for _,v in rows)
    dd = 0.0 if not peak else (peak - nav_now) / peak * 100.0
    delta = nav_now - nav_24h
    delta_pct = (delta / nav_24h * 100.0) if nav_24h else 0.0
    return {"nav": nav_now, "nav_24h": nav_24h, "delta": delta, "delta_pct": delta_pct, "dd": dd}

# ---------- UI ----------
st.set_page_config(page_title="Hedge — Overview", layout="wide")
refresh_counter = 0
try:
    refresh_counter = st_autorefresh(interval=STREAMLIT_REFRESH_MS, key="live-refresh")
except Exception:
    refresh_counter = 0
st.title("Hedge — Overview")

# Source banner with namespace
source_label = "Firestore" if _DS["source"] == "firestore" else "Local"
_top_banner = st.empty()
fs_marker = "N/A"
try:
    if _DS["source"] == "firestore":
        from scripts.fs_doctor import run as fs_check  # type: ignore

        ok, _info = fs_check(ENV)
        fs_marker = "OK" if ok else "Mixed"
except Exception:
    fs_marker = "N/A"
refresh_seconds = STREAMLIT_REFRESH_MS // 1000
_top_banner.caption(
    f"Source: {source_label} • ENV_KEY: {ENV_KEY} • FS: {fs_marker} • Refresh: {refresh_seconds}s"
)

# Live data pulls
nav_snapshot = get_nav_snapshot()
polymarket_entries = get_polymarket_snapshot()
try:
    nav_snapshot_nav = float(nav_snapshot.get("nav")) if nav_snapshot.get("nav") not in (None, "") else None
except Exception:
    nav_snapshot_nav = None
try:
    nav_snapshot_equity = float(nav_snapshot.get("equity")) if nav_snapshot.get("equity") not in (None, "") else None
except Exception:
    nav_snapshot_equity = None
try:
    nav_snapshot_ts = float(nav_snapshot.get("ts")) if nav_snapshot.get("ts") not in (None, "") else None
except Exception:
    nav_snapshot_ts = None

veto_counts = get_veto_counts()
series = load_nav_series()
k = compute_nav_kpis(series)

nav_sources, total_nav = compute_total_nav_cached()
futures_source = nav_sources.get("futures") or {}
spot_source = nav_sources.get("spot") or {}
treasury_source = nav_sources.get("treasury") or {}
poly_source = nav_sources.get("poly") or {}

futures_nav = float(futures_source.get("nav") or 0.0)
spot_nav = float(spot_source.get("nav") or 0.0)
treasury_nav = float(treasury_source.get("nav") or 0.0)
poly_nav = float(poly_source.get("nav") or 0.0)
non_futures_nav = float(max(total_nav - futures_nav, 0.0))
treasury_assets = treasury_source.get("assets") or []

if _DS["source"] == "firestore":
    try:
        telemetry_health = fetch_telemetry_health(ENV)
    except Exception as exc:  # pragma: no cover - health is best-effort
        LOGGER.debug("telemetry health fetch failed: %s", exc, exc_info=True)
        telemetry_health = {"error": str(exc)}
else:
    telemetry_health = {}

# KPI helpers derived from positions (defined after functions above)
pos_now = load_positions()
open_cnt = sum(1 for p in pos_now if abs(p.get('qty') or 0) > 0)
df_sig = load_signals_table(100)
df_tr = load_trade_log(100)
df_rb = load_blocked_orders(200)

nav_series_latest_ts = None
if series:
    try:
        nav_series_latest_ts = max(
            _to_epoch_seconds(x.get("ts") or x.get("t") or x.get("time"))
            for x in series
            if isinstance(x, dict)
        ) or None
    except Exception:
        nav_series_latest_ts = None

nav_latest_ts = nav_series_latest_ts or (nav_snapshot_ts if (nav_snapshot_nav or nav_snapshot_equity) else None)
tr_latest_ts = float(df_tr["ts"].max()) if isinstance(df_tr, pd.DataFrame) and not df_tr.empty and "ts" in df_tr else None
rb_latest_ts = float(df_rb["ts"].max()) if isinstance(df_rb, pd.DataFrame) and not df_rb.empty and "ts" in df_rb else None
last_sync_candidates = [ts for ts in (nav_latest_ts, tr_latest_ts, rb_latest_ts) if ts]
last_sync_ts = max(last_sync_candidates) if last_sync_candidates else None

sidebar_status = st.sidebar.container()
with sidebar_status:
    dry_label = DRY_RUN if DRY_RUN not in (None, "", "None") else "0"
    st.subheader("Session")
    st.markdown(f"**ENV**: `{ENV_KEY}`")
    st.markdown(f"**DRY_RUN**: `{dry_label}`")
    st.markdown(
        f"**Last sync**: {humanize_ago(last_sync_ts) if last_sync_ts else 'waiting…'}"
    )
    st.caption(f"Auto-refresh every {refresh_seconds}s • refresh #{refresh_counter}")

data_ready = any(
    [
        nav_latest_ts,
        nav_snapshot_nav,
        nav_snapshot_equity,
        sum(veto_counts.values()) > 0 if veto_counts else False,
        bool(series),
        bool(pos_now),
        not df_tr.empty,
        not df_rb.empty,
    ]
)
if not data_ready:
    st.info("Waiting for Firestore sync… data will populate automatically once available.")

primary_nav_value = k.get("nav")
primary_drawdown = k.get("dd")

latest_nav_display = (
    f"{primary_nav_value:,.2f}" if isinstance(primary_nav_value, (int, float)) else "—"
)
if latest_nav_display == "—" and isinstance(total_nav, (int, float)):
    latest_nav_display = f"{total_nav:,.2f}"

if isinstance(nav_snapshot_equity, (int, float)):
    latest_equity_display = f"{nav_snapshot_equity:,.2f}"
elif isinstance(total_nav, (int, float)):
    latest_equity_display = f"{total_nav:,.2f}"
else:
    latest_equity_display = "—"

drawdown_display = (
    f"{primary_drawdown:.2f}%" if isinstance(primary_drawdown, (int, float)) else "—"
)
veto_total = int(sum(veto_counts.values())) if veto_counts else 0
if veto_counts:
    top_reason, top_count = max(veto_counts.items(), key=lambda item: item[1])
    veto_delta = f"{top_reason}: {top_count}"
else:
    veto_delta = "waiting…"
refresh_note = humanize_ago(nav_latest_ts) if nav_latest_ts else "waiting…"

c_nav, c_eq, c_dd, c_veto = st.columns(4)
c_nav.metric("NAV (latest)", latest_nav_display, refresh_note)
c_eq.metric("Equity", latest_equity_display, refresh_note)
c_dd.metric("Drawdown", drawdown_display)
c_veto.metric("Veto events", str(veto_total), veto_delta)

LOGGER.info(
    "[dashboard] polling ok | source=%s nav_rows=%s trades=%s risk=%s",
    _DS["source"],
    len(series),
    len(df_tr) if isinstance(df_tr, pd.DataFrame) else 0,
    len(df_rb) if isinstance(df_rb, pd.DataFrame) else 0,
)

delta_text = (
    f"{k['delta']:,.2f} / {k['delta_pct']:+.2f}%"
    if k.get('nav') is not None
    else ""
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Futures NAV", f"{futures_nav:,.2f}")
c2.metric("Non-Futures NAV", f"{non_futures_nav:,.2f}")
c3.metric("Total NAV", f"{total_nav:,.2f}", delta_text)
c4.metric("Poly NAV", f"{poly_nav:,.2f}")

btc_delta = btc_24h_change()
c5, c6 = st.columns(2)
c5.metric(
    "Reserve (BTC)",
    f"{RESERVE_BTC}",
    f"{btc_delta:+.2f}% (24h)" if btc_delta is not None else "",
)
c6.metric("Open positions", str(open_cnt))

treasury_df = pd.DataFrame(treasury_assets)

poly_rows: List[Dict[str, Any]] = []
poly_latest_ts: Optional[float] = None
if isinstance(polymarket_entries, list):
    for item in polymarket_entries:
        if not isinstance(item, Mapping):
            continue
        market = str(item.get("market") or item.get("name") or "").strip()
        side = str(item.get("side") or item.get("position") or "").upper()
        score_raw = item.get("score", item.get("edge", item.get("confidence")))
        if isinstance(score_raw, (int, float)):
            score_val: Any = float(score_raw)
        elif score_raw in (None, ""):
            score_val = None
        else:
            score_val = str(score_raw)
        ts_raw = item.get("updated_at") or item.get("timestamp") or item.get("ts")
        t_epoch = _to_epoch_seconds(ts_raw)
        if t_epoch:
            poly_latest_ts = max(poly_latest_ts or t_epoch, t_epoch)
        poly_rows.append(
            {
                "Market": market,
                "Side": side,
                "Score": score_val,
                "UpdatedEpoch": t_epoch,
            }
        )

if poly_rows:
    poly_rows.sort(
        key=lambda row: row.get("Score") if isinstance(row.get("Score"), (int, float)) else 0.0,
        reverse=True,
    )

cards_treasury, cards_poly = st.columns(2)
with cards_treasury:
    st.markdown("#### Treasury (live balances)")
    st.metric(
        "Total USD",
        f"${float(treasury_nav or 0.0):,.2f}",
    )
    st.caption(
        f"Spot ${spot_nav:,.2f} · Treasury ${treasury_nav:,.2f} · Poly ${poly_nav:,.2f}"
    )
    nav_mix_total = float(total_nav or 0.0)
    if nav_mix_total > 0:
        futures_pct = max(0.0, min(1.0, float(futures_nav or 0.0) / nav_mix_total))
        non_futures_pct = max(
            0.0,
            min(1.0, float(non_futures_nav) / nav_mix_total),
        )
        mix_bar = f"""
        <div style="margin-top:6px;">
          <div style="height:10px; background:#2d2d2d; border-radius:6px; overflow:hidden;">
            <div style="height:100%; width:{futures_pct*100:.2f}%; background:#147ad6; float:left;"></div>
            <div style="height:100%; width:{non_futures_pct*100:.2f}%; background:#13b26b; float:left;"></div>
          </div>
          <div style="font-size:0.75rem; margin-top:4px;">
            Futures {futures_pct*100:.1f}% · Non-Futures {non_futures_pct*100:.1f}%
          </div>
        </div>
        """
        st.markdown(mix_bar, unsafe_allow_html=True)
    if not treasury_df.empty:
        treasury_display = treasury_df.copy()
        rename_cols = {
            "Asset": "Symbol",
            "asset": "Symbol",
            "Units": "Qty",
            "units": "Qty",
            "usd_value": "USD Value",
            "value_usd": "USD Value",
        }
        treasury_display = treasury_display.rename(columns=rename_cols)
        for col in ("Qty", "USD Value"):
            if col in treasury_display.columns:
                try:
                    treasury_display[col] = treasury_display[col].astype(float)
                except Exception:
                    pass
        drop_cols = [col for col in ("Source", "price") if col in treasury_display.columns]
        if drop_cols:
            treasury_display = treasury_display.drop(columns=drop_cols)
        ordered_cols = [col for col in ["Symbol", "Qty", "USD Value"] if col in treasury_display.columns]
        remainder = [c for c in treasury_display.columns if c not in ordered_cols]
        treasury_display = treasury_display[ordered_cols + remainder]
        st.table(format_treasury_table(treasury_display))
    else:
        st.caption("Treasury balances unavailable.")

with cards_poly:
    st.markdown("#### Polymarket insights")
    poly_count = len(poly_rows)
    poly_delta = humanize_ago(poly_latest_ts) if poly_latest_ts else "waiting…"
    st.metric("Active signals", str(poly_count), poly_delta)
    if poly_rows:
        display_rows: List[Dict[str, Any]] = []
        for entry in poly_rows[:5]:
            score_val = entry.get("Score")
            if isinstance(score_val, (int, float)):
                score_display = f"{score_val:.2f}"
            elif score_val is None:
                score_display = "—"
            else:
                score_display = str(score_val)
            updated_display = humanize_ago(entry.get("UpdatedEpoch")) if entry.get("UpdatedEpoch") else "—"
            display_rows.append(
                {
                    "Market": entry.get("Market"),
                    "Side": entry.get("Side"),
                    "Score": score_display,
                    "Updated": updated_display,
                }
            )
        st.table(display_rows)
    else:
        st.caption("No Polymarket alerts detected.")

# Risk Status KPI (open gross vs cap)
risk_cfg = _load_risk_cfg()
max_gross_pct = float(((risk_cfg.get("global") or {}).get("max_gross_nav_pct") or 0.0))
open_gross = 0.0
for p in pos_now:
    try:
        open_gross += abs(float(p.get("qty") or 0.0)) * abs(float(p.get("entryPrice") or 0.0))
    except Exception:
        pass
nav_now = k.get('nav') or 0.0
cap = (float(nav_now) * (max_gross_pct/100.0)) if (nav_now and max_gross_pct>0) else 0.0
used_pct = (open_gross / cap * 100.0) if cap else 0.0
c5, = st.columns(1)
c5.metric("Risk Status", f"{open_gross:,.0f} / {cap:,.0f}", f"{used_pct:.1f}% used")

# NAV chart
if series:
    def _parse_ts_val(x):
        t = x.get("t") or x.get("ts") or x.get("time")
        v = x.get("nav") or x.get("value") or x.get("equity") or x.get("v")
        # pandas parse
        if isinstance(t, (int,float)):
            tnum=float(t)
            if tnum>1e12:  # ms
                ts=pd.to_datetime(tnum, unit="ms", utc=True)
            else:
                ts=pd.to_datetime(tnum, unit="s", utc=True)
        else:
            ts=pd.to_datetime(str(t), utc=True, errors="coerce")
        try:
            val=float(v)
        except Exception:
            val=None
        return ts, val

    df = pd.DataFrame([_parse_ts_val(x) for x in series], columns=["ts","nav"]).dropna()
    if not df.empty:
        st.line_chart(df.set_index("ts")["nav"])
    else:
        st.info("NAV series present but could not parse timestamps/values.")
else:
    st.info("No NAV series in Firestore; showing KPIs only (fallback mode).")

st.markdown("---")

# Positions
st.subheader("Positions")
pos = pos_now
if pos:
    dfp = pd.DataFrame(pos).sort_values(by=["symbol","side"])
    st.dataframe(dfp, use_container_width=True, height=260)
else:
    st.write("No open positions")

# Signals
st.subheader("Signals (last 24h)")
if not df_sig.empty:
    newest_sig = float(df_sig["ts"].max()) if "ts" in df_sig else None
    st.caption(f"Last updated: {humanize_ago(newest_sig)}")
    if newest_sig and not is_recent(newest_sig, 1800):
        st.warning("Signals data may be stale (>30m)")
    df_sig_display = df_sig.copy()
    if "ts" in df_sig_display.columns:
        df_sig_display["Age"] = df_sig_display["ts"].apply(
            lambda ts: humanize_ago(_to_epoch_seconds(ts)) if ts else "—"
        )
    if "reduceOnly" in df_sig_display.columns:
        df_sig_display["reduceOnly"] = df_sig_display["reduceOnly"].apply(lambda v: "yes" if v else "no")
    status_series = (
        df_sig_display["status"]
        if "status" in df_sig_display.columns
        else pd.Series(["pending"] * len(df_sig_display))
    )
    df_sig_display["Status"] = status_series.apply(
        lambda s: str(s).title() if s not in (None, "") else "Pending"
    )
    rename_map = {
        "symbol": "Symbol",
        "tf": "TF",
        "signal": "Signal",
        "gross_usd": "Gross USD",
        "cap": "Capital",
        "price": "Price",
        "lev": "Lev",
        "reduceOnly": "Reduce-Only",
        "veto": "Veto",
    }
    df_sig_display = df_sig_display.rename(columns=rename_map)
    for col in ("ts", "status"):
        if col in df_sig_display.columns:
            df_sig_display = df_sig_display.drop(columns=[col])
    display_order = [
        "Status",
        "Symbol",
        "TF",
        "Signal",
        "Gross USD",
        "Capital",
        "Price",
        "Lev",
        "Reduce-Only",
        "Veto",
        "Age",
    ]
    df_sig_display = df_sig_display[
        [col for col in display_order if col in df_sig_display.columns]
    ]
    style = (
        df_sig_display.style.applymap(_status_cell_style, subset=["Status"])
        .format(
            {
                "Gross USD": "{:,.0f}",
                "Capital": "{:,.0f}",
                "Price": "{:,.4f}",
                "Lev": "{:.2f}",
            }
        )
    )
    st.dataframe(style, use_container_width=True, height=210)
else:
    st.write("No recent signals")

# Trade log
st.subheader("Trade Log (last 24h)")
if not df_tr.empty:
    newest_tr = float(df_tr["ts"].max()) if "ts" in df_tr else None
    st.caption(f"Last updated: {humanize_ago(newest_tr)}")
    if newest_tr and not is_recent(newest_tr, 1800):
        st.warning("Trades data may be stale (>30m)")
    st.dataframe(df_tr, use_container_width=True, height=210)
else:
    st.write("No recent trades")

# Risk blocks
st.subheader("Risk Blocks (last 24h)")
if not df_rb.empty:
    newest_rb = float(df_rb["ts"].max()) if "ts" in df_rb else None
    st.caption(f"Last updated: {humanize_ago(newest_rb)}")
    if newest_rb and not is_recent(newest_rb, 1800):
        st.warning("Risk blocks may be stale (>30m)")
    st.dataframe(df_rb, use_container_width=True, height=210)
else:
    st.write("No recent risk blocks")

# Screener tail (last 10 lines) — literal tags, no regex
st.subheader("Screener Tail (recent)")
tail = []
if _DS["source"] == "local":
    # cap at 200 lines
    for tag in ("[screener]", "[decision]", "[screener->executor]"):
        tail.extend(_literal_tail(tag, 200))
    tail = tail[-200:]
    # Best-effort freshness from any timestamp-like token
    newest_ts: Optional[float] = None
    for ln in reversed(tail):
        # try to find a timestamp in json-ish payload
        try:
            if "{" in ln and "}" in ln:
                payload = ln.split("{",1)[1]
                payload = "{" + payload.split("}",1)[0] + "}"
                obj = json.loads(payload)
                cand = _to_epoch_seconds(obj.get("timestamp") or obj.get("t") or obj.get("time"))
                if cand:
                    newest_ts = cand
                    break
        except Exception:
            continue
    st.caption(f"Last updated: {humanize_ago(newest_ts)}")
    if newest_ts and not is_recent(newest_ts, 1800):
        st.warning("Screener tail may be stale (>30m)")
st.caption(signal_attempts_summary(tail))
st.code("\n".join(tail) if tail else "(empty)")

# Compact recency banner (NAV/Trades/Risk)
try:
    _top_banner.caption(
        f"Source: {'Firestore' if _DS['source']=='firestore' else 'Local'} • ENV_KEY: {ENV_KEY} "
        f"• FS: {fs_marker} • Refresh: {refresh_seconds}s • NAV: {humanize_ago(nav_latest_ts)} "
        f"• Trades: {humanize_ago(tr_latest_ts)} • Risk: {humanize_ago(rb_latest_ts)} "
        f"• Last sync: {humanize_ago(last_sync_ts) if last_sync_ts else 'waiting…'}"
    )
except Exception:
    # Keep original simple banner if anything goes wrong
    _top_banner.caption(
        f"Source: {'Firestore' if _DS['source']=='firestore' else 'Local'} • ENV_KEY: {ENV_KEY} • Refresh: {refresh_seconds}s"
    )

# Keep banner simple per acceptance

# ---- Doctor panel ----
with st.expander("Doctor", expanded=False):
    try:
        st.write(f"Source: {'Firestore' if _DS['source']=='firestore' else 'Local'} • ENV_KEY: {ENV_KEY}")
        # Paths
        st.write(f"Paths: hedge/{ENV}/state/nav, hedge/{ENV}/state/positions, hedge/{ENV}/trades, hedge/{ENV}/risk")
        pos_cnt = len(pos)
        tr_cnt = int(len(df_tr)) if isinstance(df_tr, pd.DataFrame) else 0
        rb_cnt = int(len(df_rb)) if isinstance(df_rb, pd.DataFrame) else 0
        nav_latest_ts = None
        if series:
            try:
                nav_latest_ts = max(_to_epoch_seconds(x.get('t') or x.get('ts') or x.get('time')) for x in series if isinstance(x, dict))
            except Exception:
                nav_latest_ts = None
        st.write(f"Counts: positions={pos_cnt}, nav={len(series)}, trades(24h)={tr_cnt}, risk(24h)={rb_cnt}")
        st.write(f"Newest NAV ts: {humanize_ago(nav_latest_ts)}")
        # Which collections used (if any)
        if _DS["source"] == "firestore":
            if _DIAG.get("col_trades"):
                st.write(f"Trades collection: {_DIAG.get('col_trades')}")
            if _DIAG.get("col_risk"):
                st.write(f"Risk collection: {_DIAG.get('col_risk')}")
        # Tier badges/counts
        try:
            tiers = _load_json("config/symbol_tiers.json", {}) or {}
            tier_counts = {k: (len(v) if isinstance(v, list) else 0) for k, v in tiers.items()}
            # Open positions by tier: map symbol->tier
            sym2tier = {}
            for t, arr in tiers.items():
                if isinstance(arr, list):
                    for s in arr:
                        sym2tier[str(s).upper()] = t
            pos_by_tier = {}
            for r in pos:
                try:
                    t = sym2tier.get(str(r.get("symbol")).upper(), "OTHER")
                    pos_by_tier[t] = pos_by_tier.get(t, 0) + 1
                except Exception:
                    continue
            st.write(
                "Tiers: "
                + ", ".join(f"{k}:{tier_counts.get(k,0)}" for k in ("CORE","SATELLITE","TACTICAL","ALT-EXT") if k in tier_counts)
            )
            if pos_by_tier:
                st.write(
                    "Open positions by tier: "
                    + ", ".join(f"{k}:{v}" for k, v in pos_by_tier.items())
                )
        except Exception:
            pass
    except Exception:
        st.write("(doctor diagnostics unavailable)")

# --- Exit plans loader (Firestore-first; local fallback) ---
def load_exit_plans() -> pd.DataFrame:
    plans = _fs_get_state("exit_plans")
    rows = []
    if isinstance(plans, dict) and "rows" in plans:
        rows = plans["rows"]
    elif isinstance(plans, list):
        rows = plans
    if not rows:
        local = _load_json("exit_plans.json", [])
        rows = (local.get("rows") if isinstance(local, dict) else local) or []

    out = []
    now = time.time()
    for r in rows:
        try:
            ts = r.get("created_ts") or r.get("ts") or r.get("time")
            if isinstance(ts,(int,float)):
                t_epoch = float(ts)
            else:
                try:
                    t_epoch = pd.to_datetime(ts, utc=True).timestamp()
                except Exception:
                    t_epoch = 0.0
            out.append({
                "symbol": r.get("symbol"),
                "side": r.get("side") or r.get("positionSide"),
                "entry_px": float(r.get("entry_px") or r.get("entryPrice") or 0),
                "sl_px": float(r.get("sl_px") or 0),
                "tp_px": float(r.get("tp_px") or 0),
                "age_min": round((now - t_epoch)/60.0, 1) if t_epoch else None
            })
        except Exception:
            continue
    return pd.DataFrame(out)

# --- UI block (fail-soft) ---
st.subheader("Exit plans (open)")
try:
    dfep = load_exit_plans()
    if not dfep.empty:
        st.dataframe(dfep, use_container_width=True, height=210)
    else:
        st.write("No exit plans recorded yet.")
except Exception as e:
    st.write(f"Exit plans unavailable: {e}")

footer = st.container()
with footer:
    if telemetry_health and isinstance(telemetry_health, Mapping):
        if "error" in telemetry_health and not telemetry_health.get("firestore_ok"):
            msg = str(telemetry_health.get("error"))
            st.caption(f"Backend health: telemetry unavailable ({msg}) • version {FOOTER_VERSION}")
        else:
            health_ok = bool(telemetry_health.get("firestore_ok"))
            health_ts_val = telemetry_health.get("ts") or telemetry_health.get("updated_at")
            ts_epoch = _to_epoch_seconds(health_ts_val)
            updated_str = humanize_ago(ts_epoch) if ts_epoch else "unknown"
            last_error = str(telemetry_health.get("last_error") or "none")
            if len(last_error) > 120:
                last_error = last_error[:117] + "…"
            uptime_keys = (
                "rolling_uptime_pct",
                "uptime_pct",
                "uptime_percent",
                "uptime24h",
                "uptime",
            )
            uptime_value: Optional[float] = None
            uptime_raw_str: Optional[str] = None
            for key in uptime_keys:
                raw = telemetry_health.get(key)
                if raw in (None, ""):
                    continue
                try:
                    uptime_value = float(raw)
                    break
                except Exception:
                    uptime_raw_str = str(raw)
                    break
            if uptime_value is not None:
                uptime_display = f"{uptime_value:.2f}%"
            elif uptime_raw_str:
                uptime_display = uptime_raw_str
            else:
                uptime_display = telemetry_health.get("uptime_display")
                if uptime_display in (None, ""):
                    uptime_display = "N/A"
                else:
                    uptime_display = str(uptime_display)
            status_word = "HEALTHY" if health_ok else "ATTENTION"
            color = "#16c784" if health_ok else "#f5a623"
            footer_html = (
                f"<div style='margin-top:12px; font-size:0.8rem;'>"
                f"<span style='color:{color}; font-weight:600;'>Backend health: {status_word}</span>"
                f" · firestore_ok={health_ok} · uptime {uptime_display}"
                f" · last error: {last_error}"
                f" · updated {updated_str}"
                f" · version {FOOTER_VERSION}"
                f"</div>"
            )
            st.markdown(footer_html, unsafe_allow_html=True)
    else:
        st.caption(f"Backend health: telemetry unavailable • version {FOOTER_VERSION}.")
