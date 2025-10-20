import json
import os
import time
from pathlib import Path

import execution.risk_limits as risk_limits


def test_nav_age_selects_newest(tmp_path, monkeypatch):
    older = tmp_path / "nav_log.json"
    newer = tmp_path / "nav_confirmed.json"

    older.write_text(json.dumps({"ts": time.time() - 120}))
    newer.write_text(json.dumps({"ts": time.time(), "sources_ok": True}))

    os.utime(older, (time.time() - 120, time.time() - 120))
    os.utime(newer, None)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        risk_limits,
        "_NAV_SNAPSHOT_PATHS",
        [Path(older), Path(newer)],
        raising=False,
    )

    age, sources_ok = risk_limits.get_nav_freshness_snapshot()
    assert age is not None
    assert age < 5, f"Expected fresh age, got {age}"
    assert sources_ok is True
