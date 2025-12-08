"""
Tests for regime heatmap and matrix (v7).

Test cases:
- correct 2D matrix structure
- correct regime assignment in matrix
- matrix position for all regime combinations
"""

import pytest
from execution.utils.vol import (
    ATR_REGIME_LOW,
    ATR_REGIME_NORMAL,
    ATR_REGIME_ELEVATED,
    ATR_REGIME_EXTREME,
    DD_REGIME_LOW,
    DD_REGIME_MODERATE,
    DD_REGIME_HIGH,
    DD_REGIME_CRITICAL,
    compute_regime_matrix,
    build_regime_snapshot,
)


class TestComputeRegimeMatrix:
    """Unit tests for compute_regime_matrix function."""

    def test_matrix_is_4x4(self):
        """Matrix should be 4x4"""
        matrix = compute_regime_matrix(0, 0)
        assert len(matrix) == 4
        for row in matrix:
            assert len(row) == 4

    def test_single_active_cell(self):
        """Matrix should have exactly one 1, rest 0s"""
        matrix = compute_regime_matrix(1, 2)
        flat = [cell for row in matrix for cell in row]
        assert flat.count(1) == 1
        assert flat.count(0) == 15

    def test_active_cell_position_00(self):
        """Active cell at [0][0] for Low DD, Low ATR"""
        matrix = compute_regime_matrix(
            atr_regime=ATR_REGIME_LOW, 
            dd_regime=DD_REGIME_LOW
        )
        assert matrix[0][0] == 1
        assert matrix[0][1] == 0
        assert matrix[1][0] == 0

    def test_active_cell_position_33(self):
        """Active cell at [3][3] for Critical DD, Extreme ATR"""
        matrix = compute_regime_matrix(
            atr_regime=ATR_REGIME_EXTREME,
            dd_regime=DD_REGIME_CRITICAL
        )
        assert matrix[3][3] == 1
        assert matrix[3][2] == 0
        assert matrix[2][3] == 0

    def test_active_cell_position_12(self):
        """Active cell at [1][2] for Moderate DD, Elevated ATR"""
        matrix = compute_regime_matrix(
            atr_regime=ATR_REGIME_ELEVATED,
            dd_regime=DD_REGIME_MODERATE
        )
        assert matrix[1][2] == 1

    def test_active_cell_position_21(self):
        """Active cell at [2][1] for High DD, Normal ATR"""
        matrix = compute_regime_matrix(
            atr_regime=ATR_REGIME_NORMAL,
            dd_regime=DD_REGIME_HIGH
        )
        assert matrix[2][1] == 1


class TestMatrixBoundaryHandling:
    """Test matrix boundary and edge cases."""

    def test_clamps_negative_dd(self):
        """Negative DD regime should clamp to 0"""
        matrix = compute_regime_matrix(atr_regime=1, dd_regime=-1)
        # Should clamp to row 0
        assert matrix[0][1] == 1

    def test_clamps_high_dd(self):
        """DD regime > 3 should clamp to 3"""
        matrix = compute_regime_matrix(atr_regime=1, dd_regime=10)
        # Should clamp to row 3
        assert matrix[3][1] == 1

    def test_clamps_negative_atr(self):
        """Negative ATR regime should clamp to 0"""
        matrix = compute_regime_matrix(atr_regime=-5, dd_regime=2)
        # Should clamp to column 0
        assert matrix[2][0] == 1

    def test_clamps_high_atr(self):
        """ATR regime > 3 should clamp to 3"""
        matrix = compute_regime_matrix(atr_regime=99, dd_regime=2)
        # Should clamp to column 3
        assert matrix[2][3] == 1


class TestAllRegimeCombinations:
    """Test all 16 regime combinations."""

    @pytest.mark.parametrize("dd_regime,atr_regime", [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ])
    def test_all_combinations(self, dd_regime, atr_regime):
        """Each combination should place 1 at correct position"""
        matrix = compute_regime_matrix(atr_regime=atr_regime, dd_regime=dd_regime)
        
        # Check the active cell is at correct position
        assert matrix[dd_regime][atr_regime] == 1
        
        # Check all other cells are 0
        for i in range(4):
            for j in range(4):
                if i == dd_regime and j == atr_regime:
                    continue
                assert matrix[i][j] == 0, f"Expected 0 at [{i}][{j}]"


class TestBuildRegimeSnapshotMatrix:
    """Test matrix in build_regime_snapshot."""

    def test_snapshot_contains_correct_matrix(self):
        """Snapshot matrix should match computed regimes"""
        # ATR 0.20 -> Normal (1)
        # DD 0.08 -> Moderate (1)
        snapshot = build_regime_snapshot(atr_value=0.20, dd_frac=0.08)
        
        matrix = snapshot["regime_matrix"]
        atr_regime = snapshot["atr_regime"]
        dd_regime = snapshot["dd_regime"]
        
        assert matrix[dd_regime][atr_regime] == 1

    def test_snapshot_matrix_low_low(self):
        """Low ATR + Low DD should be [0][0]"""
        snapshot = build_regime_snapshot(atr_value=0.10, dd_frac=0.02)
        matrix = snapshot["regime_matrix"]
        assert matrix[0][0] == 1

    def test_snapshot_matrix_extreme_critical(self):
        """Extreme ATR + Critical DD should be [3][3]"""
        snapshot = build_regime_snapshot(atr_value=0.50, dd_frac=0.35)
        matrix = snapshot["regime_matrix"]
        assert matrix[3][3] == 1

    def test_snapshot_matrix_elevated_high(self):
        """Elevated ATR + High DD should be [2][2]"""
        snapshot = build_regime_snapshot(atr_value=0.30, dd_frac=0.20)
        matrix = snapshot["regime_matrix"]
        # ATR 0.30 -> Elevated (2): p50=0.25 < 0.30 <= p80=0.40
        # DD 0.20 -> High (2): 0.15 <= 0.20 < 0.30
        assert matrix[2][2] == 1


class TestMatrixConsistency:
    """Test consistency between matrix and regime values."""

    def test_matrix_matches_regimes(self):
        """Matrix active cell should match atr_regime and dd_regime"""
        test_cases = [
            (0.10, 0.02),  # Low, Low
            (0.20, 0.08),  # Normal, Moderate
            (0.30, 0.20),  # Elevated, High
            (0.50, 0.35),  # Extreme, Critical
        ]
        
        for atr_val, dd_val in test_cases:
            snapshot = build_regime_snapshot(atr_value=atr_val, dd_frac=dd_val)
            matrix = snapshot["regime_matrix"]
            atr_regime = snapshot["atr_regime"]
            dd_regime = snapshot["dd_regime"]
            
            # Find the active cell
            active_pos = None
            for i in range(4):
                for j in range(4):
                    if matrix[i][j] == 1:
                        active_pos = (i, j)
                        break
            
            assert active_pos is not None, "No active cell found"
            assert active_pos == (dd_regime, atr_regime), (
                f"Matrix active at {active_pos}, expected ({dd_regime}, {atr_regime})"
            )


class TestMatrixImmutability:
    """Test that matrix operations don't have side effects."""

    def test_multiple_calls_independent(self):
        """Multiple calls should produce independent matrices"""
        matrix1 = compute_regime_matrix(0, 0)
        matrix2 = compute_regime_matrix(3, 3)
        
        # Modify matrix1
        matrix1[0][0] = 99
        
        # matrix2 should be unaffected
        assert matrix2[0][0] == 0
        assert matrix2[3][3] == 1

    def test_snapshot_matrices_independent(self):
        """Snapshot matrices should be independent"""
        snap1 = build_regime_snapshot(atr_value=0.10, dd_frac=0.02)
        snap2 = build_regime_snapshot(atr_value=0.50, dd_frac=0.35)
        
        # Modify snap1's matrix
        snap1["regime_matrix"][0][0] = 99
        
        # snap2's matrix should be unaffected
        assert snap2["regime_matrix"][0][0] == 0
