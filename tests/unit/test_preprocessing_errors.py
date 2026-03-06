"""
Unit tests for preprocessing error handling
"""

import dimod
import pytest

from dynex.preprocessing import scale_bqm_to_range


def test_scale_bqm_negative_max_coeff():
    """Test scaling with negative max_abs_coeff"""
    bqm = dimod.BinaryQuadraticModel({0: 1, 1: 2}, {(0, 1): -3}, 0.0, "BINARY")

    with pytest.raises(ValueError, match="max_abs_coeff must be positive, got -1.0"):
        scale_bqm_to_range(bqm, max_abs_coeff=-1.0)


def test_scale_bqm_zero_max_coeff():
    """Test scaling with zero max_abs_coeff"""
    bqm = dimod.BinaryQuadraticModel({0: 1, 1: 2}, {(0, 1): -3}, 0.0, "BINARY")

    with pytest.raises(ValueError, match="max_abs_coeff must be positive, got 0"):
        scale_bqm_to_range(bqm, max_abs_coeff=0)


def test_scale_bqm_tiny_max_coeff():
    """Test scaling with very small max_abs_coeff (0.0001 is valid, just very small)"""
    bqm = dimod.BinaryQuadraticModel({0: 1, 1: 2}, {(0, 1): -3}, 0.0, "BINARY")

    # 0.0001 is positive, so it should work (just scale a lot)
    scaled_bqm, factor = scale_bqm_to_range(bqm, max_abs_coeff=0.0001)
    assert factor > 0
    assert scaled_bqm is not None
