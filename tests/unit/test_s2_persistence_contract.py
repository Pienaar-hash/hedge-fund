"""
S2 Persistence Contract — Experiment Continuity Guarantees.

Guards against Bug #2 (observations lost on restart).

Contract invariants:
    1. save → load → observations are byte-identical
    2. No truncation (count preserved)
    3. No duplication (no double-append on load)
    4. Order preserved (observation sequence is sacred)
    5. Calibrator state is restored (refit fires on load if n >= threshold)
    6. Predictions match after save → load cycle
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


from execution.binary_lab_s2_model import BinaryProbabilityModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(p_mid: float, spread: float = 0.01) -> Dict[str, float]:
    return {
        "p_yes_mid": p_mid,
        "depth_score": 1.0,
        "spread": spread,
        "trend_slope": -0.0005,
        "vol_regime_z": 0.2,
        "volume_z": -0.5,
    }


def _build_model_with_observations(
    n: int, *, min_samples: int = 10, state_path: Path,
) -> BinaryProbabilityModel:
    """Build a model with n observations and save state."""
    model = BinaryProbabilityModel(
        calibration_min_samples=min_samples, state_path=state_path,
    )
    for i in range(n):
        p_mid = 0.3 + (i / max(n - 1, 1)) * 0.5
        outcome = int(p_mid > 0.5) if i % 3 != 0 else int(p_mid > 0.4)
        model.update_observation(_make_features(p_mid, spread=0.008 + i * 0.0001), outcome)
    model.save_state()
    return model


# ===========================================================================
# 1. save → load → identical observations
# ===========================================================================

class TestSaveLoadIdentity:
    """Observations must survive a save → load cycle unchanged."""

    def test_observation_count_preserved(self, tmp_path: Path) -> None:
        state_path = tmp_path / "cal.json"
        _build_model_with_observations(30, state_path=state_path)

        model2 = BinaryProbabilityModel(
            calibration_min_samples=10, state_path=state_path,
        )
        model2.load_state()
        assert model2.n_observations == 30

    def test_observation_values_identical(self, tmp_path: Path) -> None:
        state_path = tmp_path / "cal.json"
        model = _build_model_with_observations(20, state_path=state_path)
        original_obs = list(model._observations)

        model2 = BinaryProbabilityModel(
            calibration_min_samples=10, state_path=state_path,
        )
        model2.load_state()

        assert len(model2._observations) == len(original_obs)
        for i, ((f1, o1), (f2, o2)) in enumerate(zip(original_obs, model2._observations)):
            assert o1 == o2, f"Outcome mismatch at index {i}: {o1} != {o2}"
            assert set(f1.keys()) == set(f2.keys()), f"Feature keys differ at index {i}"
            for k in f1:
                assert abs(f1[k] - f2[k]) < 1e-12, (
                    f"Feature '{k}' differs at index {i}: {f1[k]} != {f2[k]}"
                )

    def test_full_feature_dict_preserved(self, tmp_path: Path) -> None:
        """All feature fields survive, not just p_yes_mid."""
        state_path = tmp_path / "cal.json"
        model = BinaryProbabilityModel(
            calibration_min_samples=5, state_path=state_path,
        )
        rich_features = {
            "p_yes_mid": 0.55,
            "depth_score": 0.9,
            "spread": 0.012,
            "trend_slope": -0.00042,
            "vol_regime_z": 0.31,
            "volume_z": -1.7,
            "quote_age_s": 14.5,
            "p_yes_ask": 0.56,
            "p_yes_bid": 0.54,
        }
        model.update_observation(rich_features, 1)
        model.save_state()

        model2 = BinaryProbabilityModel(
            calibration_min_samples=5, state_path=state_path,
        )
        model2.load_state()

        loaded_features, loaded_outcome = model2._observations[0]
        assert loaded_outcome == 1
        for k, v in rich_features.items():
            assert k in loaded_features, f"Feature '{k}' missing after load"
            assert abs(loaded_features[k] - v) < 1e-12


# ===========================================================================
# 2. No truncation
# ===========================================================================

class TestNoTruncation:

    def test_large_observation_set_preserved(self, tmp_path: Path) -> None:
        state_path = tmp_path / "cal.json"
        _build_model_with_observations(200, state_path=state_path)

        model2 = BinaryProbabilityModel(
            calibration_min_samples=10, state_path=state_path,
        )
        model2.load_state()
        assert model2.n_observations == 200

    def test_state_file_contains_all_observations(self, tmp_path: Path) -> None:
        state_path = tmp_path / "cal.json"
        _build_model_with_observations(75, state_path=state_path)

        with state_path.open() as f:
            data = json.load(f)
        assert len(data["_observations"]) == 75


# ===========================================================================
# 3. No duplication
# ===========================================================================

class TestNoDuplication:

    def test_load_does_not_double_append(self, tmp_path: Path) -> None:
        """Loading state into a fresh model must not duplicate observations."""
        state_path = tmp_path / "cal.json"
        _build_model_with_observations(20, state_path=state_path)

        model2 = BinaryProbabilityModel(
            calibration_min_samples=10, state_path=state_path,
        )
        model2.load_state()
        assert model2.n_observations == 20

        # Loading again into the same model WOULD double — but that's expected.
        # The contract says: fresh model + one load = exact count.

    def test_double_save_no_duplication(self, tmp_path: Path) -> None:
        """Saving twice must not duplicate observations in the file."""
        state_path = tmp_path / "cal.json"
        model = _build_model_with_observations(15, state_path=state_path)
        model.save_state()  # second save

        with state_path.open() as f:
            data = json.load(f)
        assert len(data["_observations"]) == 15


# ===========================================================================
# 4. Order preserved
# ===========================================================================

class TestOrderPreserved:

    def test_observation_sequence_is_sacred(self, tmp_path: Path) -> None:
        state_path = tmp_path / "cal.json"
        model = BinaryProbabilityModel(
            calibration_min_samples=5, state_path=state_path,
        )
        # Use distinctive p_yes_mid values as sequence markers
        sequence = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88]
        for p in sequence:
            model.update_observation({"p_yes_mid": p}, int(p > 0.5))
        model.save_state()

        model2 = BinaryProbabilityModel(
            calibration_min_samples=5, state_path=state_path,
        )
        model2.load_state()

        loaded_mids = [f["p_yes_mid"] for f, _ in model2._observations]
        assert loaded_mids == sequence


# ===========================================================================
# 5. Calibrator restoration
# ===========================================================================

class TestCalibratorRestoration:

    def test_calibrator_restored_on_load(self, tmp_path: Path) -> None:
        state_path = tmp_path / "cal.json"
        model = _build_model_with_observations(50, min_samples=10, state_path=state_path)
        assert model._calibrator is not None

        model2 = BinaryProbabilityModel(
            calibration_min_samples=10, state_path=state_path,
        )
        model2.load_state()
        assert model2._calibrator is not None
        assert model2.calibration_active

    def test_calibrator_not_restored_below_threshold(self, tmp_path: Path) -> None:
        state_path = tmp_path / "cal.json"
        model = BinaryProbabilityModel(
            calibration_min_samples=50, state_path=state_path,
        )
        for i in range(20):
            model.update_observation({"p_yes_mid": 0.5}, i % 2)
        model.save_state()

        model2 = BinaryProbabilityModel(
            calibration_min_samples=50, state_path=state_path,
        )
        model2.load_state()
        assert model2._calibrator is None
        assert not model2.calibration_active


# ===========================================================================
# 6. Prediction identity after save → load
# ===========================================================================

class TestPredictionIdentity:

    def test_predictions_match_after_restore(self, tmp_path: Path) -> None:
        """The restored model must produce identical predictions."""
        state_path = tmp_path / "cal.json"
        model = _build_model_with_observations(50, min_samples=10, state_path=state_path)

        test_mids = [0.2, 0.35, 0.5, 0.65, 0.8]
        original_preds = [model.predict({"p_yes_mid": m}) for m in test_mids]

        model2 = BinaryProbabilityModel(
            calibration_min_samples=10, state_path=state_path,
        )
        model2.load_state()
        restored_preds = [model2.predict({"p_yes_mid": m}) for m in test_mids]

        for i, (o, r) in enumerate(zip(original_preds, restored_preds)):
            assert abs(o - r) < 1e-9, (
                f"Prediction mismatch at mid={test_mids[i]}: "
                f"original={o}, restored={r}"
            )

    def test_baseline_predictions_unaffected_by_restore(self, tmp_path: Path) -> None:
        state_path = tmp_path / "cal.json"
        _build_model_with_observations(30, state_path=state_path)

        model2 = BinaryProbabilityModel(
            calibration_min_samples=10, state_path=state_path,
        )
        model2.load_state()

        for mid in [0.3, 0.5, 0.7]:
            assert model2.predict_baseline({"p_yes_mid": mid}) == mid


# ===========================================================================
# 7. File format integrity
# ===========================================================================

class TestFileFormat:

    def test_state_file_is_valid_json(self, tmp_path: Path) -> None:
        state_path = tmp_path / "cal.json"
        _build_model_with_observations(10, state_path=state_path)

        with state_path.open() as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "_observations" in data
        assert "n_observations" in data
        assert "updated_ts" in data

    def test_observation_schema(self, tmp_path: Path) -> None:
        """Each persisted observation must have 'features' dict and 'outcome' int."""
        state_path = tmp_path / "cal.json"
        _build_model_with_observations(5, min_samples=10, state_path=state_path)

        with state_path.open() as f:
            data = json.load(f)

        for i, entry in enumerate(data["_observations"]):
            assert "features" in entry, f"Observation {i} missing 'features'"
            assert "outcome" in entry, f"Observation {i} missing 'outcome'"
            assert isinstance(entry["features"], dict)
            assert entry["outcome"] in (0, 1)
            assert "p_yes_mid" in entry["features"]

    def test_missing_state_file_returns_false(self, tmp_path: Path) -> None:
        model = BinaryProbabilityModel(
            state_path=tmp_path / "nonexistent.json",
        )
        assert model.load_state() is False
        assert model.n_observations == 0
