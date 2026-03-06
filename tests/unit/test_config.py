import pytest

from dynex import ComputeBackend, DynexConfig


def test_config_cpu_auto_mainnet():
    """Test that CPU backend automatically uses network mode"""
    config = DynexConfig(sdk_key="test_key", compute_backend=ComputeBackend.CPU)
    assert config.mainnet is True
    assert config.compute_backend == "cpu"


def test_config_gpu_auto_mainnet():
    """Test that GPU backend automatically uses network mode"""
    config = DynexConfig(sdk_key="test_key", compute_backend=ComputeBackend.GPU)
    assert config.mainnet is True
    assert config.compute_backend == "gpu"


def test_config_compute_backend_string():
    config = DynexConfig(sdk_key="test_key", compute_backend="qpu", qpu_model="apollo_rc1")
    assert config.compute_backend == "qpu"
    assert config.mainnet is True


def test_config_compute_backend_enum():
    config = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.QPU,
        qpu_model="apollo_rc1",
    )
    assert config.compute_backend == "qpu"
    assert config.mainnet is True


def test_config_compute_backend_enum_cpu():
    config = DynexConfig(sdk_key="test_key", compute_backend=ComputeBackend.CPU)
    assert config.compute_backend == "cpu"
    assert config.mainnet is True


def test_config_compute_backend_enum_gpu():
    config = DynexConfig(sdk_key="test_key", compute_backend=ComputeBackend.GPU)
    assert config.compute_backend == "gpu"
    assert config.mainnet is True


def test_config_defaults():
    config = DynexConfig(sdk_key="test_key")
    assert hasattr(config, "mainnet")
    assert hasattr(config, "logger")
    assert config.compute_backend == "unspecified"
    assert config.mainnet is True  # Default (UNSPECIFIED) uses network mode


def test_config_invalid_backend():
    with pytest.raises(ValueError):
        DynexConfig(sdk_key="test_key", compute_backend="invalid")


def test_qpu_requires_model():
    """Test that QPU backend requires qpu_model"""
    with pytest.raises(ValueError) as exc_info:
        DynexConfig(sdk_key="test_key", compute_backend=ComputeBackend.QPU)
    assert "qpu_model is required" in str(exc_info.value)


def test_qpu_always_mainnet():
    """Test that QPU backend automatically uses network mode"""
    config = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.QPU,
        qpu_model="apollo_rc1",
    )
    assert config.mainnet is True
    assert config.compute_backend == "qpu"


def test_local_always_offline(tmp_path):
    """Test that LOCAL backend uses offline mode (mainnet=False)."""
    solver_dir = tmp_path / "local"
    solver_dir.mkdir()
    (solver_dir / "dynexcore").touch()

    config = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.LOCAL,
        solver_path=str(solver_dir),
    )
    assert config.mainnet is False
    assert config.compute_backend == "local"
    assert config.solver_path is not None


def test_config_new_parameters_defaults():
    """Test new parameters have correct defaults"""
    config = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.CPU,
    )
    assert config.use_notebook_output is True
    assert config.default_timeout == 300.0
    assert config.default_description == "Dynex SDK Job"
    assert config.preserve_solutions is False


def test_config_new_parameters_custom():
    """Test new parameters accept custom values"""
    config = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.CPU,
        use_notebook_output=False,
        default_timeout=600.0,
        default_description="Custom Job",
        preserve_solutions=True,
    )
    assert config.use_notebook_output is False
    assert config.default_timeout == 600.0
    assert config.default_description == "Custom Job"
    assert config.preserve_solutions is True


def test_config_as_dict_includes_new_params():
    """Test as_dict includes new parameters"""
    config = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.CPU,
        use_notebook_output=False,
        default_timeout=600.0,
    )
    config_dict = config.as_dict()
    assert "use_notebook_output" in config_dict
    assert "default_timeout" in config_dict
    assert "default_description" in config_dict
    assert "preserve_solutions" in config_dict
    assert config_dict["use_notebook_output"] is False
    assert config_dict["default_timeout"] == 600.0
