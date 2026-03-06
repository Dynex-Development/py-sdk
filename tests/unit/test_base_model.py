"""Unit tests for dynex.models.base module."""

import dimod
import numpy as np
import pytest

from dynex import BQM, DynexConfig
from dynex.models.base import DynexModel


def test_base_class_str():
    """Test __str__ method of base class."""
    config = DynexConfig(compute_backend="cpu")
    bqm = dimod.BinaryQuadraticModel({0: 1}, {(0, 1): -1}, 0.0, "BINARY")
    model = BQM(bqm, config=config)

    # __str__ should return type_str
    result = str(model)
    assert isinstance(result, str)
    assert result == "BQM"


def test_max_precision():
    """Test _max_precision static method."""
    # Create BQM with known coefficients
    bqm = dimod.BinaryQuadraticModel({0: 1.5, 1: 2.3}, {(0, 1): -3.7}, 0.0, "BINARY")

    precision = DynexModel._max_precision(bqm)

    # Max abs coeff is 3.7, so precision should be 10^(floor(log10(3.7)) - 4)
    # log10(3.7) ≈ 0.568, floor(0.568) = 0, so 10^(0-4) = 10^-4 = 0.0001
    expected = 10 ** (np.floor(np.log10(3.7)) - 4)
    assert precision == expected


def test_max_precision_large_coeffs():
    """Test _max_precision with large coefficients."""
    bqm = dimod.BinaryQuadraticModel({0: 100.0, 1: 200.0}, {(0, 1): -500.0}, 0.0, "BINARY")

    precision = DynexModel._max_precision(bqm)

    # Max abs coeff is 500, log10(500) ≈ 2.699, floor = 2, so 10^(2-4) = 0.01
    expected = 10 ** (np.floor(np.log10(500.0)) - 4)
    assert precision == expected


def test_max_precision_zero_coeffs_raises():
    """Test _max_precision with all zero coefficients raises error."""
    bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 0.0}, {(0, 1): 0.0}, 0.0, "BINARY")

    with pytest.raises(Exception, match="At least one weight must be > 0.0"):
        DynexModel._max_precision(bqm)


def test_convert_bqm_to_qubo_direct_returns_correct_tuple():
    config = DynexConfig(compute_backend="cpu")
    bqm = dimod.BinaryQuadraticModel({0: 1.0, 1: -1.0}, {(0, 1): 2.0}, 0.0, "BINARY")
    model = BQM(bqm, config=config)

    result = model._convert_bqm_to_qubo_direct(bqm, logging=False)

    assert isinstance(result.clauses, list)
    assert isinstance(result.num_variables, (int, np.integer))
    assert isinstance(result.num_clauses, int)
    assert isinstance(result.var_mappings, dict)
    assert result.precision > 0
    assert result.precision <= 1.0
    assert result.bqm is not None


def test_convert_bqm_to_qubo_direct_with_logging():
    config = DynexConfig(compute_backend="cpu")
    bqm = dimod.BinaryQuadraticModel({0: 0.5}, {}, 0.0, "BINARY")
    model = BQM(bqm, config=config)

    result = model._convert_bqm_to_qubo_direct(bqm, logging=True)

    assert result.num_variables > 1
    assert result.precision > 0


def test_convert_bqm_to_qubo_direct_precision_calculation():
    config = DynexConfig(compute_backend="cpu")
    bqm = dimod.BinaryQuadraticModel({0: 0.00001}, {}, 0.0, "BINARY")
    model = BQM(bqm, config=config)

    result = model._convert_bqm_to_qubo_direct(bqm, logging=False)

    assert result.precision > 0
    assert result.precision <= 1.0


def test_convert_bqm_to_qubo_direct_with_quadratic():
    config = DynexConfig(compute_backend="cpu")
    bqm = dimod.BinaryQuadraticModel({0: 1.0, 1: 2.0, 2: 3.0}, {(0, 1): -1.0, (1, 2): -2.0}, 0.0, "BINARY")
    model = BQM(bqm, config=config)

    result = model._convert_bqm_to_qubo_direct(bqm, logging=False)

    assert len(result.clauses) > 0
    assert result.num_variables >= 3
    assert result.num_clauses == len(result.clauses)


def test_convert_bqm_to_qubo_direct_precision_warning():
    config = DynexConfig(compute_backend="cpu")
    bqm = dimod.BinaryQuadraticModel({0: 0.0001, 1: 0.0002}, {}, 0.0, "BINARY")
    model = BQM(bqm, config=config)

    result = model._convert_bqm_to_qubo_direct(bqm, logging=True)

    assert result.precision > 0
