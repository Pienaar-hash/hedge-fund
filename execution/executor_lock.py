"""
AUDIT-1.3b — Single-executor PID lock.

Prevents two executor instances from running concurrently, which would
cause contradictory orders and position duplication.

Usage (at executor startup):
    from execution.executor_lock import acquire_executor_lock
    _lock_fd = acquire_executor_lock()   # raises RuntimeError if held

The lock is released automatically when the process exits.
"""
from __future__ import annotations

import fcntl
import logging
import os

LOGGER = logging.getLogger(__name__)

_LOCK_PATH = os.path.join("logs", "state", "executor.lock")
_lock_fd: int | None = None


def acquire_executor_lock(lock_path: str = _LOCK_PATH) -> int:
    """Acquire an exclusive lock on *lock_path*.

    Returns the file descriptor (kept open for lifetime of process).
    Raises ``RuntimeError`` if another executor already holds the lock.
    """
    global _lock_fd  # noqa: PLW0603 — intentional singleton

    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise RuntimeError(
            f"Another executor instance is already running (lock file: {lock_path}). "
            "Cannot start a second instance — dual-executor creates position corruption risk."
        )

    # Write PID for diagnostics
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, f"{os.getpid()}\n".encode())

    _lock_fd = fd
    LOGGER.info("[executor_lock] Acquired exclusive lock (pid=%d, path=%s)", os.getpid(), lock_path)
    return fd


def release_executor_lock() -> None:
    """Explicitly release the lock (normally happens at process exit)."""
    global _lock_fd  # noqa: PLW0603
    if _lock_fd is not None:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            os.close(_lock_fd)
        except OSError:
            pass
        _lock_fd = None
