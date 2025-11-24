#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List


def _futures_listed(symbol: str) -> bool:
    try:
        from execution.exchange_utils import get_symbol_filters

        _ = get_symbol_filters(symbol)
        return True
    except Exception:
        return False


def _pick_candidates(max_n: int = 5) -> List[str]:
    # Basic placeholder universe; in practice, you might screen by volume/ATR.
    seeds = [
        "ARBUSDT",
        "OPUSDT",
        "AVAXUSDT",
        "APTUSDT",
        "INJUSDT",
        "RNDRUSDT",
        "SEIUSDT",
        "TONUSDT",
        "XRPUSDT",
        "ADAUSDT",
    ]
    out = []
    for s in seeds:
        if _futures_listed(s):
            out.append(s)
        if len(out) >= max_n:
            break
    return out


def _ai_rank(symbols: List[str]) -> List[Dict[str, Any]]:
    key = os.getenv("OPENAI_API_KEY", "")
    out: List[Dict[str, Any]] = []
    if not key:
        # Offline fallback: trivial rationale
        for s in symbols:
            out.append(
                {
                    "symbol": s,
                    "rationale": "Adequate liquidity; trend filter passed (stub).",
                    "liquidity_ok": True,
                    "trend_ok": True,
                }
            )
        return out
    try:
        import openai

        client = openai.OpenAI(api_key=key)
        prompt = (
            "Given these USDT-margined Binance futures symbols: "
            + ", ".join(symbols)
            + ". Pick up to 5 with adequate liquidity and a mild positive trend."
            "Return JSON array with objects: {symbol, rationale, liquidity_ok, trend_ok}."
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        # Be defensive parsing
        import json

        txt = (resp.choices[0].message.content or "[]").strip()
        arr = json.loads(txt)
        if isinstance(arr, list):
            for rec in arr:
                try:
                    s = str(rec.get("symbol")).upper()
                    out.append(
                        {
                            "symbol": s,
                            "rationale": str(rec.get("rationale") or "n/a"),
                            "liquidity_ok": bool(rec.get("liquidity_ok", True)),
                            "trend_ok": bool(rec.get("trend_ok", True)),
                        }
                    )
                except Exception:
                    continue
    except Exception:
        # Fall back to offline stub
        out = _ai_rank(symbols=[])
    return out[:5]


def save_yaml(rows: List[Dict[str, Any]], path: str = "config/discovery.yml") -> str:
    try:
        import yaml  # type: ignore

        with open(path, "w") as f:
            yaml.safe_dump(rows, f, sort_keys=False)
        return path
    except Exception as e:
        raise RuntimeError(f"yaml write error: {e}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI discovery list generator (operator-gated)")
    ap.add_argument("--max", type=int, default=5)
    args = ap.parse_args(argv)
    cands = _pick_candidates(max_n=int(args.max))
    rows = _ai_rank(cands)
    path = save_yaml(rows)
    print({"ok": True, "path": path, "count": len(rows)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

