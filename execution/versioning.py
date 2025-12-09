from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = REPO_ROOT / "VERSION"
MANIFEST_FILE = REPO_ROOT / "v7_manifest.json"


def read_version(default: str = "") -> str:
    """Read the engine VERSION string from repo root."""
    try:
        value = VERSION_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return default
    return value or default


def read_docs_version(default: str = "") -> str:
    """Read docs_version from v7 manifest."""
    try:
        data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return default
    if not isinstance(data, Mapping):
        return default
    raw = data.get("docs_version")
    return str(raw).strip() if raw else default


def version_alignment(expected: str | None = None) -> Tuple[str, str, bool]:
    """
    Return (engine_version, docs_version, aligned flag).

    If expected is provided, both versions must equal it to be considered aligned.
    """
    engine = read_version(default=expected or "")
    docs = read_docs_version(default="")
    if expected:
        aligned = bool(engine) and bool(docs) and engine == expected and docs == expected
    else:
        aligned = bool(engine) and bool(docs) and engine == docs
    return engine, docs, aligned


def build_engine_metadata(
    *,
    run_id: str | None = None,
    git_commit: str | None = None,
    hostname: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Construct a minimal engine metadata payload with a fresh timestamp."""
    payload: Dict[str, Any] = {
        "engine_version": read_version(default="v7.6"),
        "git_commit": git_commit,
        "run_id": run_id,
        "hostname": hostname,
        "updated_ts": time.time(),
    }
    if extra:
        for key, value in extra.items():
            payload[key] = value
    return payload
