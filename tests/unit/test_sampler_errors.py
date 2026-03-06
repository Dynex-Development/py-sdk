"""
Unit tests for DynexSampler error handling and validation
"""

import dimod
import pytest

import dynex
from dynex import DynexConfig


def test_sampler_clones_too_small():
    """Test sampler with clones < 1"""
    # Use cpu backend (mainnet=True by default)
    config = DynexConfig(compute_backend="cpu")
    bqm = dimod.BinaryQuadraticModel({0: 1, 1: 1}, {(0, 1): -2}, 0.0, "BINARY")
    model = dynex.BQM(bqm, logging=False)
    sampler = dynex.DynexSampler(model, config=config, logging=False)

    with pytest.raises(Exception, match="Value of clones must be in range \\[1,128\\]"):
        sampler.sample(num_reads=10, annealing_time=10, clones=0)


def test_sampler_clones_too_large():
    """Test sampler with clones > 128"""
    config = DynexConfig(compute_backend="cpu")
    bqm = dimod.BinaryQuadraticModel({0: 1, 1: 1}, {(0, 1): -2}, 0.0, "BINARY")
    model = dynex.BQM(bqm, logging=False)
    sampler = dynex.DynexSampler(model, config=config, logging=False)

    with pytest.raises(Exception, match="Value of clones must be in range \\[1,128\\]"):
        sampler.sample(num_reads=10, annealing_time=10, clones=129)


def test_sampler_clones_local_not_supported(tmp_path, monkeypatch):
    """Test sampler with clones > 1 in LOCAL mode."""
    solver_dir = tmp_path / "local"
    solver_dir.mkdir(parents=True, exist_ok=True)
    solver_file = solver_dir / "dynexcore"
    solver_file.touch()
    solver_file.chmod(0o755)

    monkeypatch.setenv("DYNEX_SOLVER_PATH", str(solver_dir))

    config = DynexConfig(compute_backend="local", solver_path=str(solver_dir))
    bqm = dimod.BinaryQuadraticModel({0: 1, 1: 1}, {(0, 1): -2}, 0.0, "BINARY")
    model = dynex.BQM(bqm, logging=False)
    sampler = dynex.DynexSampler(model, config=config, logging=False)

    with pytest.raises(Exception, match="Clone sampling is only supported in network mode"):
        sampler.sample(num_reads=10, annealing_time=10, clones=2)
