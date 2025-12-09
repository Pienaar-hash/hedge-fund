from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.state_publish import compute_and_write_factor_diagnostics_state

pytestmark = [pytest.mark.integration]


def test_factor_diagnostics_state_publish_writes_surface(tmp_path: Path) -> None:
    hybrid_results = [
        {
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "regime": "normal",
            "hybrid_score": 0.4,
            "factor_vector": {
                "trend": 0.8,
                "carry": 0.2,
                "expectancy": 0.1,
                "router_quality": 0.9,
                "rv_momentum": 0.05,
                "vol_regime": 0.0,
            },
        }
    ]

    state = compute_and_write_factor_diagnostics_state(hybrid_results=hybrid_results, state_dir=tmp_path)

    out_path = tmp_path / "factor_diagnostics.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())

    assert "updated_ts" in state
    assert payload.get("raw_factors")
    assert payload.get("weights") is not None
    assert payload.get("pnl_attribution") is not None
