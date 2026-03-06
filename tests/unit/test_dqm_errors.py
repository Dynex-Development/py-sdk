"""
Unit tests for dynex.DQM error handling and validation
"""

import dimod
import pytest

import dynex


def test_dqm_invalid_formula():
    """Test DQM with invalid formula value"""
    dqm = dimod.DiscreteQuadraticModel()
    dqm.add_variable(3, label="var1")
    dqm.add_variable(3, label="var2")
    dqm.set_quadratic("var1", "var2", {(0, 0): -1, (1, 1): -1})

    with pytest.raises(Exception, match="Unknown value of formula.*must be in \\[1, 2\\]"):
        dynex.DQM(dqm, formula=3, logging=False)


def test_dqm_invalid_formula_zero():
    """Test DQM with formula=0"""
    dqm = dimod.DiscreteQuadraticModel()
    dqm.add_variable(2, label="var1")

    with pytest.raises(Exception, match="Unknown value of formula.*must be in \\[1, 2\\]"):
        dynex.DQM(dqm, formula=0, logging=False)


def test_dqm_invalid_formula_negative():
    """Test DQM with negative formula value"""
    dqm = dimod.DiscreteQuadraticModel()
    dqm.add_variable(2, label="var1")

    with pytest.raises(Exception, match="Unknown value of formula.*must be in \\[1, 2\\]"):
        dynex.DQM(dqm, formula=-1, logging=False)


def test_dqm_empty_model():
    """Test DQM with empty model (no variables)"""
    # Create empty DQM
    dqm = dimod.DiscreteQuadraticModel()

    with pytest.raises(Exception, match="At least one weight must be > 0.0"):
        dynex.DQM(dqm, logging=False)
