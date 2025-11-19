from __future__ import annotations

from execution.log_utils import JsonlLogger


def test_jsonl_logger_rotates(tmp_path):
    log_path = tmp_path / "test.jsonl"
    logger = JsonlLogger(log_path, max_bytes=100, backup_count=1)
    for idx in range(20):
        logger.write({"idx": idx})
    archive_file = log_path.parent / "archive" / "test.1.jsonl.gz"
    assert log_path.exists()
    assert archive_file.exists()
