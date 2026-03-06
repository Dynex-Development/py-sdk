"""
Unit tests for dynex.CQM error handling and validation
"""

import dimod
import pytest

import dynex


def test_cqm_invalid_formula():
    """Test CQM with invalid formula value"""
    num_widget_a = dimod.Integer("num_widget_a", upper_bound=7)
    num_widget_b = dimod.Integer("num_widget_b", upper_bound=3)
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(-3 * num_widget_a - 4 * num_widget_b)
    cqm.add_constraint(num_widget_a + num_widget_b <= 5, label="total widgets")

    with pytest.raises(Exception, match="Unknown value of formula.*must be in \\[1, 2\\]"):
        dynex.CQM(cqm, formula=3, logging=False)


def test_cqm_invalid_formula_zero():
    """Test CQM with formula=0"""
    num_widget_a = dimod.Integer("num_widget_a", upper_bound=7)
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(-3 * num_widget_a)

    with pytest.raises(Exception, match="Unknown value of formula.*must be in \\[1, 2\\]"):
        dynex.CQM(cqm, formula=0, logging=False)


def test_cqm_invalid_formula_negative():
    """Test CQM with negative formula value"""
    num_widget_a = dimod.Integer("num_widget_a", upper_bound=7)
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(-3 * num_widget_a)

    with pytest.raises(Exception, match="Unknown value of formula.*must be in \\[1, 2\\]"):
        dynex.CQM(cqm, formula=-1, logging=False)


def test_cqm_empty_model():
    """Test CQM with empty model (no variables/constraints)"""
    # Create empty CQM
    cqm = dimod.ConstrainedQuadraticModel()

    with pytest.raises(Exception, match="At least one weight must be > 0.0"):
        dynex.CQM(cqm, logging=False)
