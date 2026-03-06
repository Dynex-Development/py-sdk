"""Unit tests for LOCAL compute backend and local solver integration."""

import os
import subprocess
import sys

import dimod
import pytest

from dynex import BQM, DynexConfig, DynexSampler
from dynex.compute_backend import ComputeBackend

LOCAL_BINARY = os.path.join(os.path.dirname(__file__), "..", "..", "testnet", "dynexcore")
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "testnet")
LOCAL_AVAILABLE = os.path.exists(LOCAL_BINARY) and sys.platform == "darwin"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_local_backend_enum_value():
    assert ComputeBackend.LOCAL.value == "local"


def test_local_config_mainnet_false(tmp_path):
    """LOCAL backend must set mainnet=False (offline mode)."""
    solver_dir = tmp_path / "local"
    solver_dir.mkdir()
    (solver_dir / "dynexcore").touch()

    config = DynexConfig(
        sdk_key="test",
        compute_backend=ComputeBackend.LOCAL,
        solver_path=str(solver_dir),
    )
    assert config.mainnet is False


def test_local_config_string_value(tmp_path):
    """String 'local' accepted as compute_backend."""
    solver_dir = tmp_path / "local"
    solver_dir.mkdir()
    (solver_dir / "dynexcore").touch()

    config = DynexConfig(
        sdk_key="test",
        compute_backend="local",
        solver_path=str(solver_dir),
    )
    assert config.compute_backend == "local"
    assert config.mainnet is False


def test_local_config_solver_path_resolved(tmp_path):
    """solver_path is resolved to absolute path."""
    solver_dir = tmp_path / "local"
    solver_dir.mkdir()
    (solver_dir / "dynexcore").touch()

    config = DynexConfig(
        sdk_key="test",
        compute_backend=ComputeBackend.LOCAL,
        solver_path=str(solver_dir),
    )
    assert os.path.isabs(config.solver_path)


def test_local_config_missing_solver_raises(tmp_path, monkeypatch):
    """Missing dynexcore binary raises FileNotFoundError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    monkeypatch.setattr("dynex.config.DynexConfig.DEFAULT_SOLVER_LOCATIONS", [])

    with pytest.raises(FileNotFoundError):
        DynexConfig(
            sdk_key="test",
            compute_backend=ComputeBackend.LOCAL,
            solver_path=str(empty_dir),
        )


def test_local_config_platform_prefix(tmp_path):
    """Platform prefix for LOCAL backend is 'LOCAL'."""
    solver_dir = tmp_path / "local"
    solver_dir.mkdir()
    (solver_dir / "dynexcore").touch()

    config = DynexConfig(
        sdk_key="test",
        compute_backend=ComputeBackend.LOCAL,
        solver_path=str(solver_dir),
    )
    assert config.get_platform_prefix() == "LOCAL"


# ---------------------------------------------------------------------------
# Binary smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LOCAL_AVAILABLE, reason="testnet/dynexcore not present")
def test_dynexcore_binary_runs():
    """dynexcore binary should run without crashing (no args → usage message)."""
    result = subprocess.run(
        [os.path.abspath(LOCAL_BINARY)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    output = result.stdout.decode("utf-8", errors="replace")
    assert "dynexcore" in output.lower() or "usage" in output.lower() or result.returncode in (0, 1)


# ---------------------------------------------------------------------------
# Sampling integration tests (require actual binary)
# ---------------------------------------------------------------------------


@pytest.fixture
def local_config():
    """DynexConfig for LOCAL backend using project testnet/ directory."""
    if not LOCAL_AVAILABLE:
        pytest.skip("testnet/dynexcore not present")
    return DynexConfig(
        sdk_key="test",
        compute_backend=ComputeBackend.LOCAL,
        solver_path=os.path.abspath(LOCAL_DIR),
        use_notebook_output=False,
    )


def test_local_simple_bqm(local_config):
    """Minimal 2-variable BQM: a=0,b=1 or a=1,b=0 minimises energy to -1."""
    bqm = dimod.BinaryQuadraticModel({"a": -1.0, "b": -1.0}, {("a", "b"): 2.0}, 0.0, "BINARY")
    model = BQM(bqm, config=local_config)
    sampler = DynexSampler(model, logging=False, config=local_config)
    ss = sampler.sample(num_reads=200, annealing_time=200)

    assert ss is not None
    assert ss.first.energy <= -0.9
    sample = ss.first.sample
    # exactly one of the two variables should be 1
    assert sample["a"] + sample["b"] == 1


def test_local_returns_sampleset(local_config):
    """Result is a dimod SampleSet with correct variable count."""
    bqm = dimod.BinaryQuadraticModel({0: -1.0, 1: -1.0}, {(0, 1): 2.0}, 0.0, "BINARY")
    model = BQM(bqm, config=local_config)
    sampler = DynexSampler(model, logging=False, config=local_config)
    ss = sampler.sample(num_reads=100, annealing_time=100)

    assert isinstance(ss, dimod.SampleSet)
    assert len(ss.variables) == 2
    assert len(ss) >= 1


def test_local_number_partitioning(local_config):
    """Number partitioning: [3,1,1,2,2,1] split into two equal-sum groups."""
    import pyqubo

    numbers = [3, 1, 1, 2, 2, 1]
    n = len(numbers)
    x = pyqubo.Array.create("x", shape=(n,), vartype="BINARY")
    cost = (sum(numbers[i] * (1 - 2 * x[i]) for i in range(n))) ** 2
    bqm_native = cost.compile().to_bqm()

    model = BQM(bqm_native, config=local_config)
    sampler = DynexSampler(model, logging=False, config=local_config)
    ss = sampler.sample(num_reads=1000, annealing_time=500)

    assert ss.first.energy <= 0.1  # optimal is 0.0


def test_local_qubo_dict(local_config):
    """BQM accepts plain QUBO dict and solves it locally."""
    Q = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0}
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    model = BQM(bqm, config=local_config)
    sampler = DynexSampler(model, logging=False, config=local_config)
    ss = sampler.sample(num_reads=200, annealing_time=200)

    assert ss.first.energy <= -0.9


def test_local_clones_not_supported(local_config):
    """clones > 1 raises an exception in LOCAL mode."""
    bqm = dimod.BinaryQuadraticModel({"a": -1.0}, {}, 0.0, "BINARY")
    model = BQM(bqm, config=local_config)
    sampler = DynexSampler(model, logging=False, config=local_config)

    with pytest.raises(Exception, match="network mode"):
        sampler.sample(num_reads=100, annealing_time=100, clones=2)


def test_local_logging_prefix(local_config, caplog):
    """Log messages use LOCAL prefix."""
    import logging

    bqm = dimod.BinaryQuadraticModel({"a": -1.0, "b": -1.0}, {("a", "b"): 2.0}, 0.0, "BINARY")
    model = BQM(bqm, config=local_config)
    sampler = DynexSampler(model, logging=True, config=local_config)
    with caplog.at_level(logging.INFO):
        sampler.sample(num_reads=100, annealing_time=100)

    assert any("LOCAL" in record.message for record in caplog.records)
