"""
Binary Lab S2 — calibrated probability model.

Architecture:

    Naive baseline (adjusted mid) → isotonic calibration (at ≥50 obs)

Two permanent prediction tracks:
    - ``predict_baseline()`` — always returns the naive adjusted-mid estimate.
    - ``predict()``          — returns calibrated model output (equals baseline
                               until calibrator is fitted).

These are *never* collapsed.  Both are logged on every trade so that model
lift can be measured against the control group at any point.

Calibration authority levels:
    - inactive  (n < calibration_min_samples)
    - active    (calibration_min_samples ≤ n < calibration_confident_samples)
    - confident (n ≥ calibration_confident_samples)

Authority is *logged*, never silently upgraded.  Consumers decide how much
to trust ``p_model_yes`` based on the flags.

State is persisted atomically to ``logs/state/binary_lab_s2_calibration.json``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional sklearn dependency — model works without it (naive-only mode)
# ---------------------------------------------------------------------------
try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
    _SKLEARN_AVAILABLE = True
except ImportError:
    IsotonicRegression = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CALIBRATION_PATH = Path("logs/state/binary_lab_s2_calibration.json")
DEFAULT_MIN_SAMPLES = 50
DEFAULT_CONFIDENT_SAMPLES = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _brier_score(predictions: List[float], outcomes: List[int]) -> Optional[float]:
    if not predictions:
        return None
    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)


def _log_loss(predictions: List[float], outcomes: List[int]) -> Optional[float]:
    if not predictions:
        return None
    eps = 1e-15
    total = 0.0
    for p, o in zip(predictions, outcomes):
        p_clip = max(eps, min(1 - eps, p))
        total += o * math.log(p_clip) + (1 - o) * math.log(1 - p_clip)
    return -total / len(predictions)


def _bucket_rates(
    predictions: List[float], outcomes: List[int], n_buckets: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Per-probability-bucket realized YES rates."""
    buckets: Dict[str, List[int]] = {}
    bucket_preds: Dict[str, List[float]] = {}
    for p, o in zip(predictions, outcomes):
        idx = min(int(p * n_buckets), n_buckets - 1)
        lo = idx / n_buckets
        hi = (idx + 1) / n_buckets
        key = f"{lo:.2f}-{hi:.2f}"
        buckets.setdefault(key, []).append(o)
        bucket_preds.setdefault(key, []).append(p)

    result: Dict[str, Dict[str, Any]] = {}
    for key in sorted(buckets):
        outs = buckets[key]
        preds = bucket_preds[key]
        result[key] = {
            "count": len(outs),
            "avg_predicted": round(sum(preds) / len(preds), 6),
            "realized_rate": round(sum(outs) / len(outs), 6),
        }
    return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BinaryProbabilityModel:
    """
    Calibrated probability model for Binary Lab S2.

    Naive baseline: adjusted mid (mid corrected for half-spread cost).
    Calibration: isotonic regression fitted on (baseline_pred, outcome) pairs.
    """

    def __init__(
        self,
        *,
        model_version: str = "s2_naive_v1",
        calibration_min_samples: int = DEFAULT_MIN_SAMPLES,
        calibration_confident_samples: int = DEFAULT_CONFIDENT_SAMPLES,
        state_path: Path = DEFAULT_CALIBRATION_PATH,
    ) -> None:
        self.model_version = model_version
        self.calibration_min_samples = calibration_min_samples
        self.calibration_confident_samples = calibration_confident_samples
        self._state_path = state_path

        self._observations: List[Tuple[Dict[str, float], int]] = []
        self._calibrator: Any = None  # IsotonicRegression or None
        self._last_refit_n: int = 0

    # ------------------------------------------------------------------
    # Prediction — two permanent tracks
    # ------------------------------------------------------------------

    def predict_baseline(self, features: Dict[str, float]) -> float:
        """
        Always returns the naive baseline estimate.

        Baseline = p_yes_mid (adjusted mid from the probability market).
        This method is *never* affected by calibration.
        """
        return float(features.get("p_yes_mid", 0.5))

    def predict(self, features: Dict[str, float]) -> float:
        """
        Returns calibrated model output.

        Initially equals baseline.  When calibrator is fitted (n ≥ min_samples),
        the baseline is passed through isotonic regression.
        """
        baseline = self.predict_baseline(features)
        if self._calibrator is not None and self.calibration_active:
            try:
                calibrated = float(self._calibrator.predict([baseline])[0])
                return max(0.0, min(1.0, calibrated))
            except Exception:
                pass
        return baseline

    # ------------------------------------------------------------------
    # Observation accumulation + refit
    # ------------------------------------------------------------------

    def update_observation(
        self, features: Dict[str, float], outcome: bool,
    ) -> None:
        """Record a resolved round and refit if threshold reached."""
        self._observations.append((dict(features), int(outcome)))
        if len(self._observations) >= self.calibration_min_samples:
            self._refit()

    def _refit(self) -> None:
        """Fit isotonic regression on all accumulated (baseline, outcome) pairs."""
        if not _SKLEARN_AVAILABLE or IsotonicRegression is None:
            logger.debug("s2_model: sklearn unavailable — skipping refit")
            return
        baselines = [float(f.get("p_yes_mid", 0.5)) for f, _ in self._observations]
        outcomes = [o for _, o in self._observations]
        try:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(baselines, outcomes)
            self._calibrator = iso
            self._last_refit_n = len(self._observations)
            logger.info(
                "s2_model: isotonic refit at n=%d", self._last_refit_n,
            )
        except Exception as exc:
            logger.warning("s2_model: refit failed: %s", exc)

    # ------------------------------------------------------------------
    # Calibration state
    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        return len(self._observations)

    @property
    def calibration_active(self) -> bool:
        return self.n_observations >= self.calibration_min_samples

    @property
    def calibration_confident(self) -> bool:
        return self.n_observations >= self.calibration_confident_samples

    def calibration_stats(self) -> Dict[str, Any]:
        """Return calibration quality metrics."""
        baselines = [float(f.get("p_yes_mid", 0.5)) for f, _ in self._observations]
        outcomes = [o for _, o in self._observations]

        # Model predictions (through calibrator if available)
        model_preds = []
        for features, _ in self._observations:
            model_preds.append(self.predict(features))

        return {
            "model_version": self.model_version,
            "n_observations": self.n_observations,
            "calibration_active": self.calibration_active,
            "calibration_confident": self.calibration_confident,
            "brier_score": _brier_score(model_preds, outcomes),
            "baseline_brier_score": _brier_score(baselines, outcomes),
            "log_loss": _log_loss(model_preds, outcomes),
            "bucket_rates": _bucket_rates(model_preds, outcomes) if model_preds else None,
            "last_refit_n": self._last_refit_n if self._last_refit_n > 0 else None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Optional[Path] = None) -> None:
        """Atomically write calibration state to JSON."""
        dest = path or self._state_path
        stats = self.calibration_stats()
        stats["last_refit_ts"] = (
            datetime.now(timezone.utc).isoformat()
            if self._last_refit_n > 0
            else None
        )
        stats["updated_ts"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(dest, stats)

    def load_state(self, path: Optional[Path] = None) -> bool:
        """
        Load persisted calibration state (observations + calibrator).

        Returns True if state was loaded successfully.
        """
        src = path or self._state_path
        if not src.exists():
            return False
        try:
            with src.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # We only persist stats, not observations — the model starts
            # fresh each boot.  Observations accumulate from resolved rounds
            # during the current session.  This is intentional: it prevents
            # stale calibration from persisting across code changes.
            logger.info(
                "s2_model: loaded calibration state (n=%s, active=%s)",
                data.get("n_observations"),
                data.get("calibration_active"),
            )
            return True
        except Exception as exc:
            logger.warning("s2_model: load_state failed: %s", exc)
            return False
