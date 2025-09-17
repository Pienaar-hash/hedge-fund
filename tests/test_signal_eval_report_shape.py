import json
import pathlib
import pytest

def test_signal_eval_report_fields_when_present():
    path = pathlib.Path("models/signal_eval.json")
    if not path.exists():
        pytest.skip("no signal_eval.json yet")
    data = json.load(open(path, "r"))
    assert "aggregate" in data and "symbols" in data
    agg = data["aggregate"]
    for key in ["ml_f1", "rule_f1", "n_symbols_ok", "n_symbols_err"]:
        assert key in agg
    for entry in data["symbols"]:
        assert "symbol" in entry
        assert ("error" in entry) or ("ml" in entry and "rule" in entry)
