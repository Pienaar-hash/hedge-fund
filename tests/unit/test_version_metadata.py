from __future__ import annotations

import json
from pathlib import Path

from execution.state_publish import write_engine_metadata_state
from execution.versioning import read_docs_version, read_version, version_alignment


def test_version_matches_manifest() -> None:
    version = read_version()
    docs_version = read_docs_version()
    engine, docs, aligned = version_alignment("v7.6")
    assert version == "v7.6"
    assert docs_version == "v7.6"
    assert engine == version
    assert docs == version
    assert aligned


def test_engine_metadata_state_written(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = write_engine_metadata_state({"git_commit": "abc123"}, state_dir=state_dir)
    path = state_dir / "engine_metadata.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["engine_version"]
    assert payload["engine_version"] == data["engine_version"]
    assert "updated_ts" in data
    assert data["git_commit"] == "abc123"
