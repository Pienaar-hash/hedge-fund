#!/usr/bin/env python3
"""Precompute doctor snapshot for dashboard caching."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from execution.log_utils import safe_dump
from scripts.doctor import collect_doctor_snapshot


def _resolve_env(default: str = "dev") -> str:
    value = (os.getenv("ENV") or os.getenv("ENVIRONMENT") or "").strip()
    return value or default


ENV = _resolve_env()
if ENV.lower() == "prod":
    allow_prod = os.getenv("ALLOW_PROD_WRITE", "0").strip().lower()
    if allow_prod not in {"1", "true", "yes"}:
        raise RuntimeError(
            "cron_doctor_cache.py refuses to write with ENV=prod. "
            "Set ALLOW_PROD_WRITE=1 to override explicitly."
        )

CACHE_PATH = Path("logs/cache/doctor.json")


def write_snapshot(snapshot: dict) -> None:
    payload = safe_dump(snapshot)
    payload["generated_at"] = time.time()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> None:
    snapshot = collect_doctor_snapshot()
    write_snapshot(snapshot)


if __name__ == "__main__":
    main()
