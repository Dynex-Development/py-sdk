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


def test_sdk_key_masked_in_as_dict():
    """as_dict() must not expose the raw sdk_key."""
    key = "ABCDEF1234567890XYZ"
    config = DynexConfig(sdk_key=key, compute_backend="cpu")
    config_dict = config.as_dict()
    assert config_dict["sdk_key"] != key
    assert "..." in config_dict["sdk_key"]
    assert key[:6] in config_dict["sdk_key"]
    assert key[-4:] in config_dict["sdk_key"]


def test_sdk_key_short_masked():
    """Short sdk_key is fully masked."""
    config = DynexConfig(sdk_key="short", compute_backend="cpu")
    config_dict = config.as_dict()
    assert config_dict["sdk_key"] == "***"


def test_tmp_dir_default(tmp_path):
    """Without tmp_dir param, config.tmp_dir is None (./tmp used at runtime)."""
    config = DynexConfig(sdk_key="key", compute_backend="cpu")
    assert config.tmp_dir is None


def test_tmp_dir_custom(tmp_path):
    """Custom tmp_dir is stored and the directory is created."""
    custom = str(tmp_path / "custom_tmp")
    config = DynexConfig(sdk_key="key", compute_backend="cpu", tmp_dir=custom)
    assert config.tmp_dir == custom
    import os

    assert os.path.isdir(custom)


def test_grpc_use_tls_stored():
    """grpc_use_tls is stored on the config object."""
    config = DynexConfig(sdk_key="key", compute_backend="cpu", grpc_use_tls=False)
    assert config.grpc_use_tls is False

    config2 = DynexConfig(sdk_key="key", compute_backend="cpu", grpc_use_tls=True)
    assert config2.grpc_use_tls is True

    config3 = DynexConfig(sdk_key="key", compute_backend="cpu")
    assert config3.grpc_use_tls is None
