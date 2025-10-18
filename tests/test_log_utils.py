import gzip
import json
from pathlib import Path

import pytest

from execution.log_utils import JsonlLogger, get_logger, safe_dump


def _read_lines(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_jsonl_write(tmp_path: Path) -> None:
    log_path = tmp_path / "app.log"
    logger = JsonlLogger(log_path, max_bytes=10_000, backup_count=2)

    logger.write({"foo": "bar"})
    logger.write({"baz": 123})
    lines = _read_lines(log_path)

    assert len(lines) == 2
    assert lines[0]["foo"] == "bar"
    assert lines[1]["baz"] == 123

    # safe_dump should make nested objects serializable
    complex_payload = {"data": safe_dump({"key": set([1, 2, 3])})}
    logger.write(complex_payload)
    lines = _read_lines(log_path)
    assert "data" in lines[-1]


def test_rotation_and_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "logs"
    root.mkdir()
    archive_dir = root / "archive"
    archive_dir.mkdir()
    log_path = root / "rotate.log"

    # Small max bytes to force rotation quickly
    logger = JsonlLogger(log_path, max_bytes=100, backup_count=2)

    for idx in range(10):
        payload = {"idx": idx, "text": "x" * 20}
        logger.write(payload)

    assert log_path.exists()
    rotated = log_path.with_name("rotate.1.log")
    assert rotated.exists()

    # The oldest file should be archived as gzip
    archives = list(archive_dir.glob("rotate.2*.log.gz"))
    assert archives, "Expected archived gzip file"

    with gzip.open(archives[0], "rt", encoding="utf-8") as fh:
        archived_lines = [json.loads(line) for line in fh if line.strip()]
    assert archived_lines, "Archived file should contain data"


def test_get_logger_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    log_dir = repo_root / "logs"
    log_dir.mkdir()

    from execution import log_utils

    monkeypatch.setattr(log_utils, "REPO_ROOT", repo_root)

    logger = get_logger("logs/test_rel.log", max_bytes=1000, backup_count=1)
    logger.write({"hello": "world"})

    target = repo_root / "logs" / "test_rel.log"
    assert target.exists()
    with target.open("r", encoding="utf-8") as fh:
        data = [json.loads(line) for line in fh if line.strip()]
    assert data[0]["hello"] == "world"
