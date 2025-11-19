#!/usr/bin/env python3
"""Background service that periodically runs pipeline_v6_compare."""

from __future__ import annotations

import logging
import os
import time

from execution.intel import pipeline_v6_compare

LOG = logging.getLogger("pipeline_compare_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    interval = float(os.getenv("PIPELINE_COMPARE_INTERVAL", "900") or 900)
    shadow_limit = int(os.getenv("PIPELINE_COMPARE_SHADOW_LIMIT", "500") or 500)
    while True:
        try:
            summary = pipeline_v6_compare.compare_pipeline_v6(shadow_limit=shadow_limit)
            LOG.info("[pipeline_compare] summary=%s", summary)
        except Exception as exc:
            LOG.warning("[pipeline_compare] compare failed: %s", exc)
        time.sleep(max(interval, 5.0))


if __name__ == "__main__":
    main()
