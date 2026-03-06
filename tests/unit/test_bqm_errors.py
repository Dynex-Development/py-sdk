"""
Unit tests for BQM error handling and validation
"""

import dimod
import pytest

import dynex


def test_bqm_invalid_formula():
    """Test BQM with invalid formula parameter"""
    bqm = dimod.BinaryQuadraticModel({0: 1, 1: 1}, {(0, 1): -2}, 0.0, "BINARY")

    with pytest.raises(Exception, match="Unknown value of formula.*must be in \\[1, 2\\]"):
        dynex.BQM(bqm, formula=3, logging=False)


def test_bqm_empty_model():
    """Test BQM with no variables and clauses (raises different error about weights)"""
    empty_bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, "BINARY")

    with pytest.raises(Exception, match="At least one weight must be > 0.0"):
        dynex.BQM(empty_bqm, logging=False)


def test_bqm_invalid_formula_zero():
    """Test BQM with formula=0"""
    bqm = dimod.BinaryQuadraticModel({0: 1, 1: 1}, {(0, 1): -2}, 0.0, "BINARY")

    with pytest.raises(Exception, match="Unknown value of formula.*must be in \\[1, 2\\]"):
        dynex.BQM(bqm, formula=0, logging=False)


def test_bqm_invalid_formula_negative():
    """Test BQM with negative formula"""
    bqm = dimod.BinaryQuadraticModel({0: 1, 1: 1}, {(0, 1): -2}, 0.0, "BINARY")

    with pytest.raises(Exception, match="Unknown value of formula.*must be in \\[1, 2\\]"):
        dynex.BQM(bqm, formula=-1, logging=False)
