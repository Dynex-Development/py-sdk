import dimod
import pytest

from dynex.preprocessing import scale_bqm_to_range


def test_scale_bqm_basic():
    bqm = dimod.BinaryQuadraticModel.from_qubo({(0, 0): 100.0, (1, 1): -100.0, (0, 1): 50.0})

    scaled_bqm, factor = scale_bqm_to_range(bqm, max_abs_coeff=9.0)

    assert factor > 0
    assert all(abs(c) <= 9.0 for c in scaled_bqm.linear.values())
    assert all(abs(c) <= 9.0 for c in scaled_bqm.quadratic.values())


def test_scale_bqm_large_coefficients():
    bqm = dimod.BinaryQuadraticModel.from_qubo({(0, 0): 8500.0, (1, 1): -8500.0, (0, 1): 17000.0})

    scaled_bqm, factor = scale_bqm_to_range(bqm, max_abs_coeff=9.0)

    max_linear = max(abs(c) for c in scaled_bqm.linear.values())
    max_quad = max(abs(c) for c in scaled_bqm.quadratic.values())

    assert max_linear <= 9.01
    assert max_quad <= 9.01


def test_scale_bqm_zero_coefficients():
    bqm = dimod.BinaryQuadraticModel.from_qubo({(0, 0): 0.0, (1, 1): 0.0, (0, 1): 0.0})

    scaled_bqm, factor = scale_bqm_to_range(bqm, max_abs_coeff=9.0)

    assert factor == 1.0


def test_scale_bqm_invalid_max_coeff():
    bqm = dimod.BinaryQuadraticModel.from_qubo({(0, 0): 1.0})

    with pytest.raises(ValueError):
        scale_bqm_to_range(bqm, max_abs_coeff=0.0)

    with pytest.raises(ValueError):
        scale_bqm_to_range(bqm, max_abs_coeff=-1.0)


def test_scale_bqm_preserves_variables():
    bqm = dimod.BinaryQuadraticModel.from_qubo(
        {(0, 0): 100.0, (1, 1): -50.0, (2, 2): 75.0, (0, 1): 25.0, (1, 2): -30.0}
    )

    scaled_bqm, factor = scale_bqm_to_range(bqm, max_abs_coeff=9.0)

    assert len(scaled_bqm.variables) == 3
