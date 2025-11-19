"""Centralized parsing + logging for v6 runtime flags."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, Mapping, MutableMapping

_TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: str = "0") -> bool:
    raw = (os.getenv(name, default) or "").strip().lower()
    return raw in _TRUTHY


@dataclass(frozen=True)
class V6Flags:
    intel_v6_enabled: bool
    risk_engine_v6_enabled: bool
    pipeline_v6_shadow_enabled: bool
    router_autotune_v6_enabled: bool
    feedback_allocator_v6_enabled: bool
    router_autotune_v6_apply_enabled: bool


def _load_flags() -> V6Flags:
    return V6Flags(
        intel_v6_enabled=_env_bool("INTEL_V6_ENABLED"),
        risk_engine_v6_enabled=_env_bool("RISK_ENGINE_V6_ENABLED"),
        pipeline_v6_shadow_enabled=_env_bool("PIPELINE_V6_SHADOW_ENABLED"),
        router_autotune_v6_enabled=_env_bool("ROUTER_AUTOTUNE_V6_ENABLED"),
        feedback_allocator_v6_enabled=_env_bool("FEEDBACK_ALLOCATOR_V6_ENABLED"),
        router_autotune_v6_apply_enabled=_env_bool("ROUTER_AUTOTUNE_V6_APPLY_ENABLED"),
    )


_FLAGS = _load_flags()


def get_flags(*, refresh: bool = False) -> V6Flags:
    """Return the cached flag snapshot (refreshing from env when requested)."""
    global _FLAGS
    if refresh:
        _FLAGS = _load_flags()
    return _FLAGS


def flags_to_dict(flags: V6Flags | None = None) -> Dict[str, bool]:
    snapshot = asdict(flags or get_flags())
    return {
        "INTEL_V6_ENABLED": snapshot["intel_v6_enabled"],
        "RISK_ENGINE_V6_ENABLED": snapshot["risk_engine_v6_enabled"],
        "PIPELINE_V6_SHADOW_ENABLED": snapshot["pipeline_v6_shadow_enabled"],
        "ROUTER_AUTOTUNE_V6_ENABLED": snapshot["router_autotune_v6_enabled"],
        "FEEDBACK_ALLOCATOR_V6_ENABLED": snapshot["feedback_allocator_v6_enabled"],
        "ROUTER_AUTOTUNE_V6_APPLY_ENABLED": snapshot["router_autotune_v6_apply_enabled"],
    }


def log_v6_flag_snapshot(logger: logging.Logger, *, level: int = logging.INFO, flags: V6Flags | None = None) -> None:
    """Emit a single log line summarizing boolean flag values."""
    if logger is None:
        return
    snapshot = flags_to_dict(flags)
    payload = " ".join(f"{key}={int(value)}" for key, value in snapshot.items())
    logger.log(level, "[v6] flags %s", payload)


def enrich_payload(base: Mapping[str, bool] | None = None, *, flags: V6Flags | None = None) -> Dict[str, bool]:
    """Return payload merged with the flag snapshot (handy for runtime probes)."""
    payload: MutableMapping[str, bool] = dict(base or {})
    payload.update(flags_to_dict(flags))
    return dict(payload)


__all__ = ["V6Flags", "get_flags", "flags_to_dict", "log_v6_flag_snapshot", "enrich_payload"]
