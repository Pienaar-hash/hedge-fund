"""
Binary Lab S2 runtime writer (state surface emitter).

Thin wrapper over the S1 runtime infrastructure, parameterised for S2:
- Separate config file (``config/binary_lab_limits_s2.json``)
- Separate state file (``logs/state/binary_lab_s2_state.json``)
- Separate limits-hash env var (``BINARY_LAB_S2_LIMITS_HASH``)

The underlying reducer (``execution.binary_lab_executor``) is sleeve-agnostic
and reused without modification — the ``sleeve_id`` in limits ``_meta``
distinguishes S1 / S2 in the state payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from execution.binary_lab_runtime import (
    BinaryLabRuntimeWriter,
    RuntimeLoopContext,        # re-export for S2 consumers
)

__all__ = [
    "BinaryLabS2RuntimeWriter",
    "RuntimeLoopContext",
    "DEFAULT_S2_LIMITS_PATH",
    "DEFAULT_S2_STATE_PATH",
]

DEFAULT_S2_LIMITS_PATH = Path("config/binary_lab_limits_s2.json")
DEFAULT_S2_STATE_PATH = Path("logs/state/binary_lab_s2_state.json")


class BinaryLabS2RuntimeWriter(BinaryLabRuntimeWriter):
    """BinaryLabRuntimeWriter pre-parameterised for the S2 sleeve."""

    def __init__(
        self,
        *,
        limits_path: Path = DEFAULT_S2_LIMITS_PATH,
        state_path: Path = DEFAULT_S2_STATE_PATH,
        expected_limits_hash: Optional[str] = None,
    ) -> None:
        import os

        if expected_limits_hash is None:
            expected_limits_hash = os.getenv("BINARY_LAB_S2_LIMITS_HASH")

        super().__init__(
            limits_path=limits_path,
            state_path=state_path,
            expected_limits_hash=expected_limits_hash,
        )
