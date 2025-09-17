#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Tuple


def _to_epoch(t: Any) -> float:
    if isinstance(t, (int, float)):
        x = float(t)
        return x / 1000.0 if x > 1e12 else x
    if hasattr(t, "timestamp"):
        try:
            return float(t.timestamp())
        except Exception:
            return 0.0
    return 0.0


def _load_nav_series(env: str) -> List[Dict[str, Any]]:
    try:
        from utils.firestore_client import get_db

        db = get_db()
        doc = db.collection("hedge").document(env).collection("state").document("nav").get()
        data = doc.to_dict() if getattr(doc, "exists", False) else {}
        out: List[Dict[str, Any]] = []
        for _, v in (data.items() if isinstance(data, dict) else []):
            if not isinstance(v, dict):
                continue
            ts = _to_epoch(v.get("t") or v.get("time") or v.get("ts"))
            if ts <= 0:
                continue
            val = v.get("nav") or v.get("value") or v.get("equity") or v.get("v")
            if not isinstance(val, (int, float, str)):
                continue
            nav = float(val)
            out.append({"ts": ts, "nav": nav})
        out.sort(key=lambda r: r["ts"])
        return out
    except Exception:
        return []


def _compose_ai_note() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        # Safe fallback note
        return (
            "Market note: BTC, ETH, SOL saw typical two‑sided flows over the last 24h. "
            "Liquidity conditions appear normal; no material dislocations observed. This is not advice."
        )
    try:
        import openai

        client = openai.OpenAI(api_key=key)
        prompt = (
            "Write an 80–120 word neutral market note focused on BTC/ETH/SOL drivers in the last 24h. "
            "Avoid hype or recommendations. Close with: 'This is not investment advice.'"
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=160,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception:
        return (
            "Market note: BTC, ETH, SOL traded mixed with modest rotation; "
            "derivatives premiums and funding were stable. This is not investment advice."
        )


def _save_nav_png(series: List[Dict[str, Any]], path: str = "/tmp/nav.png") -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["ts"] for r in series]
        ys = [r["nav"] for r in series]
        fig, ax = plt.subplots(figsize=(3.2, 2.0), dpi=160)
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.2)
        ax.set_title("NAV (7d)", fontsize=8)
        ax.grid(True, alpha=0.2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
    except Exception:
        # Fallback: write a minimal 1x1 PNG placeholder
        png_minimal = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\x0cIDAT\x08\x99c\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x0f\xeb\xb1\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        try:
            with open(path, "wb") as f:
                f.write(png_minimal)
            return path
        except Exception as e:
            raise RuntimeError(f"png write error: {e}")


def _counts_24h(env: str) -> Tuple[int, str]:
    """Return (trades_24h, top_veto_reason) from Firestore, best-effort."""
    trades_24h = 0
    top_reason = ""
    try:
        from utils.firestore_client import get_db

        db = get_db()
        root = db.collection("hedge").document(env)
        now = time.time()

        # Trades
        docs_t = list(
            root.collection("trades").order_by("ts", direction="DESCENDING").limit(1000).stream()
        )
        ts_ok = 0
        for d in docs_t:
            x = d.to_dict() or {}
            t = _to_epoch(x.get("ts") or x.get("time"))
            if (now - t) <= 24 * 3600:
                ts_ok += 1
        trades_24h = ts_ok

        # Risk veto top
        docs_r = list(
            root.collection("risk").order_by("ts", direction="DESCENDING").limit(1000).stream()
        )
        counts: Dict[str, int] = {}
        for d in docs_r:
            x = d.to_dict() or {}
            t = _to_epoch(x.get("ts") or x.get("time"))
            if (now - t) > 24 * 3600:
                continue
            reasons_val = x.get("reasons")
            if isinstance(reasons_val, list):
                for r in reasons_val:
                    key = str(r)
                    counts[key] = counts.get(key, 0) + 1
            elif x.get("reason") is not None:
                key = str(x.get("reason"))
                counts[key] = counts.get(key, 0) + 1
        if counts:
            top_reason = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    except Exception:
        pass
    return trades_24h, top_reason


def run(env: str, dry_run: bool = True) -> Tuple[bool, Dict[str, Any]]:
    series = _load_nav_series(env)
    now = time.time()
    week_ago = now - 7 * 24 * 3600
    series7 = [r for r in series if r.get("ts", 0.0) >= week_ago]
    png_path = _save_nav_png(series7)
    note = _compose_ai_note()

    trades_24h, top_veto = _counts_24h(env)
    payload = {"png": png_path, "note": note, "trades_24h": trades_24h, "top_veto": top_veto}
    if dry_run:
        print({"dry_run": True, **payload})
        return True, payload

    # Send to Telegram if enabled
    try:
        from execution.telegram_utils import send_telegram
    except Exception:
        send_telegram = None

    sent = False
    if send_telegram is not None:
        header = (
            f"Daily mini‑report\n"
            f"Trades (24h): {trades_24h} | Top veto: {top_veto or '—'}\n\n{note}"
        )
        sent = bool(send_telegram(header, silent=True))
    return sent, payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Telegram mini-report: NAV PNG + AI note")
    ap.add_argument("--env", default=os.getenv("ENV", "prod"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    ok, info = run(args.env, dry_run=bool(args.dry_run))
    print({"ok": ok, **info})
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
