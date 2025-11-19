#!/usr/bin/env python3
"""Background heartbeat writer for pipeline v6 shadow head."""

from __future__ import annotations

import os
import time
import logging

from execution import pipeline_v6_shadow
from execution.state_publish import write_pipeline_v6_shadow_state
from execution.v6_flags import log_v6_flag_snapshot

LOG = logging.getLogger("pipeline_shadow_heartbeat")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _write_summary() -> None:
    summary = pipeline_v6_shadow.build_shadow_summary(
        pipeline_v6_shadow.load_shadow_decisions(limit=100)
    )
    summary.setdefault("last_decision", None)
    write_pipeline_v6_shadow_state(summary)
    LOG.info("[heartbeat] pipeline_v6_shadow summary updated total=%s", summary.get("total"))


def main() -> None:
    interval = float(os.getenv("PIPELINE_SHADOW_HEARTBEAT_INTERVAL", "600") or 600)
    try:
        log_v6_flag_snapshot(LOG)
    except Exception:
        LOG.debug("v6 flag snapshot logging failed", exc_info=True)
    while True:
        try:
            _write_summary()
        except Exception as exc:
            LOG.warning("[heartbeat] update failed: %s", exc)
        time.sleep(max(interval, 5.0))


if __name__ == "__main__":
    main()
