import json
import pathlib
import pytest

def test_report_schema_if_present():
    path = pathlib.Path("models/last_train_report.json")
    if not path.exists():
        pytest.skip("last_train_report.json not present yet")
    data = json.load(path.open())
    for key in [
        "started_at_utc",
        "finished_at_utc",
        "fit_rc",
        "eval_rc",
        "fit_result",
        "eval_result",
    ]:
        assert key in data
