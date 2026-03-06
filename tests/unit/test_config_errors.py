"""
Unit tests for DynexConfig error handling and validation
"""

import pytest

from dynex import ComputeBackend, DynexConfig, QPUModel  # noqa: F401


def test_config_invalid_compute_backend_type():
    """Test config with invalid compute_backend type"""
    with pytest.raises(ValueError, match="compute_backend must be ComputeBackend enum or string"):
        DynexConfig(compute_backend=123)


def test_config_invalid_compute_backend_string():
    """Test config with invalid compute_backend string"""
    with pytest.raises(ValueError, match="compute_backend must be one of"):
        DynexConfig(compute_backend="invalid_backend")


def test_config_invalid_qpu_model_type():
    """Test config with invalid qpu_model type"""
    with pytest.raises(ValueError, match="qpu_model must be QPUModel enum or string"):
        DynexConfig(compute_backend="qpu", qpu_model=123)


def test_config_invalid_qpu_model_string():
    """Test config with invalid qpu_model string"""
    with pytest.raises(ValueError, match="qpu_model must be one of"):
        DynexConfig(compute_backend="qpu", qpu_model="invalid_model")


def test_config_qpu_without_model():
    """Test QPU backend without qpu_model specified"""
    with pytest.raises(ValueError, match="qpu_model is required when compute_backend='qpu'"):
        DynexConfig(compute_backend="qpu", qpu_model=None)


def test_config_qpu_enum_without_model():
    """Test QPU backend (enum) without qpu_model"""
    with pytest.raises(ValueError, match="qpu_model is required when compute_backend='qpu'"):
        DynexConfig(compute_backend=ComputeBackend.QPU, qpu_model=None)


def test_config_qpu_uppercase_without_model():
    """Test QPU backend (uppercase string) without qpu_model"""
    with pytest.raises(ValueError, match="qpu_model is required when compute_backend='qpu'"):
        DynexConfig(compute_backend="QPU", qpu_model=None)
