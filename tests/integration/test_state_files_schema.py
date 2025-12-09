from __future__ import annotations

import json
import os

import pytest

from dashboard.state_v7 import iter_state_file_specs

pytestmark = [pytest.mark.integration, pytest.mark.runtime]


@pytest.mark.runtime
def test_state_files_have_minimal_schema():
    for name, spec in iter_state_file_specs():
        path = spec.get("path")
        required_keys = spec.get("required_keys", [])
        any_of_groups = spec.get("any_of_keys", [])
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        for key in required_keys:
            assert key in data, f"{name} missing required key: {key}"
        for group in any_of_groups:
            assert any(k in data for k in group), f"{name} missing one of required keys: {group}"
