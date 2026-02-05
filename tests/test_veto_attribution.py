import pathlib
import sys

import pytest


SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

import veto_attribution  # type: ignore


def test_extract_fields_tolerates_non_dict_sections():
    record = {
        "symbol": "BTCUSDT",
        "veto_reason": "symbol_cap",
        "ts": 1700000000,  # non-string timestamp
        "veto_detail": "not-a-dict",
        "context": "executor",
    }

    out = veto_attribution.extract_fields(record)

    assert out["strategy"] == "unknown"
    assert out["date"] is None  # non-string ts is ignored
    assert out["requested_notional"] is None  # constraint_geometry absent


def test_generate_report_empty_records():
    report = veto_attribution.generate_report([])
    assert "No veto records found" in report


def test_generate_report_handles_missing_dates():
    record = {
        "symbol": "ETHUSDT",
        "veto_reason": "nav_stale",
        "ts": None,
        "veto_detail": {},
    }

    # Should not raise when dates are missing / None
    report = veto_attribution.generate_report([record])
    assert "Veto Attribution Analysis" in report
