#!/usr/bin/env python3
"""
Loop timing diagnostics for executor performance auditing.

This module provides lightweight timing instrumentation to measure
where time is spent in each executor loop iteration.

Usage:
    LOOP_TIMING_DEBUG=1 python -m execution.executor_live

Environment variables:
    LOOP_TIMING_DEBUG=1     - Enable timing output (default: disabled)
    LOOP_TIMING_THRESHOLD=5 - Log sections taking longer than N seconds (default: 5)

Output is logged at INFO level when enabled, with summaries at DEBUG.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

LOG = logging.getLogger("loop_timing")

# Configuration from environment
ENABLED = os.getenv("LOOP_TIMING_DEBUG", "0").lower() in ("1", "true", "yes")
THRESHOLD_S = float(os.getenv("LOOP_TIMING_THRESHOLD", "5.0") or 5.0)
JSONL_PATH = os.getenv("LOOP_TIMING_LOG", "logs/execution/loop_timing.jsonl")


@dataclass
class TimingSection:
    """A single timed section within a loop iteration."""
    name: str
    start_ts: float
    end_ts: float = 0.0
    api_calls: int = 0
    symbols_processed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)


@dataclass
class LoopTiming:
    """Timing data for a single loop iteration."""
    iteration: int
    start_ts: float
    sections: List[TimingSection] = field(default_factory=list)
    end_ts: float = 0.0

    @property
    def total_duration_s(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)

    @property
    def work_duration_s(self) -> float:
        """Time spent in measured work (excluding sleep)."""
        return sum(s.duration_s for s in self.sections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "ts": self.start_ts,
            "total_s": round(self.total_duration_s, 3),
            "work_s": round(self.work_duration_s, 3),
            "sections": {
                s.name: {
                    "duration_s": round(s.duration_s, 3),
                    "api_calls": s.api_calls,
                    "symbols": s.symbols_processed,
                    **s.extra,
                }
                for s in self.sections
            },
        }


# Global timing state (singleton per process)
_current_loop: Optional[LoopTiming] = None
_current_section: Optional[TimingSection] = None
_recent_timings: List[Dict[str, Any]] = []
_MAX_RECENT = 100


def is_enabled() -> bool:
    """Check if timing diagnostics are enabled."""
    return ENABLED


def start_loop(iteration: int) -> None:
    """Begin timing a new loop iteration."""
    global _current_loop
    if not ENABLED:
        return
    _current_loop = LoopTiming(iteration=iteration, start_ts=time.time())


def end_loop() -> Optional[Dict[str, Any]]:
    """End timing the current loop iteration and log results."""
    global _current_loop, _recent_timings
    if not ENABLED or _current_loop is None:
        return None
    
    _current_loop.end_ts = time.time()
    timing_dict = _current_loop.to_dict()
    
    # Log to file
    _write_timing_log(timing_dict)
    
    # Identify slow sections
    slow_sections = [
        (s.name, s.duration_s)
        for s in _current_loop.sections
        if s.duration_s >= THRESHOLD_S
    ]
    
    if slow_sections:
        LOG.warning(
            "[loop_timing] iteration=%d total=%.2fs SLOW SECTIONS: %s",
            _current_loop.iteration,
            _current_loop.total_duration_s,
            ", ".join(f"{n}={d:.1f}s" for n, d in slow_sections),
        )
    elif _current_loop.total_duration_s >= THRESHOLD_S * 2:
        LOG.warning(
            "[loop_timing] iteration=%d total=%.2fs (exceeds 2x threshold)",
            _current_loop.iteration,
            _current_loop.total_duration_s,
        )
    else:
        LOG.debug(
            "[loop_timing] iteration=%d total=%.2fs work=%.2fs",
            _current_loop.iteration,
            _current_loop.total_duration_s,
            _current_loop.work_duration_s,
        )
    
    # Keep recent history
    _recent_timings.append(timing_dict)
    while len(_recent_timings) > _MAX_RECENT:
        _recent_timings.pop(0)
    
    result = timing_dict
    _current_loop = None
    return result


@contextmanager
def timed_section(
    name: str,
    api_calls: int = 0,
    symbols: int = 0,
    **extra: Any,
) -> Generator[TimingSection, None, None]:
    """
    Context manager to time a section of the loop.
    
    Usage:
        with timed_section("sentinel_compute", api_calls=1) as section:
            compute_sentinel_x()
            section.api_calls = 1  # can update during execution
    """
    global _current_section
    
    if not ENABLED or _current_loop is None:
        # Yield a dummy section when disabled
        yield TimingSection(name="noop", start_ts=0.0)
        return
    
    section = TimingSection(
        name=name,
        start_ts=time.time(),
        api_calls=api_calls,
        symbols_processed=symbols,
        extra=extra,
    )
    _current_section = section
    
    try:
        yield section
    finally:
        section.end_ts = time.time()
        _current_loop.sections.append(section)
        _current_section = None


def record_api_call(count: int = 1) -> None:
    """Record API call(s) in the current section."""
    if _current_section is not None:
        _current_section.api_calls += count


def record_symbols(count: int) -> None:
    """Record number of symbols processed in current section."""
    if _current_section is not None:
        _current_section.symbols_processed = count


def get_recent_timings() -> List[Dict[str, Any]]:
    """Get recent timing data for diagnostics."""
    return list(_recent_timings)


def get_timing_summary() -> Dict[str, Any]:
    """
    Get aggregate timing statistics from recent iterations.
    
    Returns:
        {
            "count": int,
            "avg_total_s": float,
            "avg_work_s": float,
            "max_total_s": float,
            "section_avgs": {"section_name": float, ...},
            "slowest_section": str,
        }
    """
    if not _recent_timings:
        return {"count": 0}
    
    count = len(_recent_timings)
    totals = [t.get("total_s", 0) for t in _recent_timings]
    works = [t.get("work_s", 0) for t in _recent_timings]
    
    # Aggregate section timings
    section_totals: Dict[str, List[float]] = {}
    for t in _recent_timings:
        for name, data in t.get("sections", {}).items():
            section_totals.setdefault(name, []).append(data.get("duration_s", 0))
    
    section_avgs = {
        name: sum(times) / len(times)
        for name, times in section_totals.items()
        if times
    }
    
    slowest = max(section_avgs.items(), key=lambda x: x[1])[0] if section_avgs else ""
    
    return {
        "count": count,
        "avg_total_s": round(sum(totals) / count, 3),
        "avg_work_s": round(sum(works) / count, 3),
        "max_total_s": round(max(totals), 3),
        "section_avgs": {k: round(v, 3) for k, v in section_avgs.items()},
        "slowest_section": slowest,
    }


def _write_timing_log(timing: Dict[str, Any]) -> None:
    """Append timing data to JSONL log."""
    try:
        os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)
        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps(timing) + "\n")
    except Exception as exc:
        LOG.debug("[loop_timing] failed to write log: %s", exc)


# Convenience function for quick one-liner timing
def time_block(name: str) -> "contextmanager[TimingSection]":
    """Alias for timed_section for cleaner imports."""
    return timed_section(name)
