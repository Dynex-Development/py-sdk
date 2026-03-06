"""Tests for DynexSampler configuration parameter integration."""

import dimod

from dynex import BQM, DynexConfig, DynexSampler
from dynex.compute_backend import ComputeBackend


def test_sampler_uses_config_defaults():
    """Test that sampler uses default values from config"""
    config = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.CPU,
        use_notebook_output=False,
        default_timeout=600.0,
        default_description="Custom Config Job",
        preserve_solutions=True,
    )

    bqm = dimod.BinaryQuadraticModel({0: 1.0}, {(0, 1): -1.0}, 0.0, "BINARY")
    model = BQM(bqm)
    sampler = DynexSampler(model, config=config)

    assert sampler.use_notebook_output is False
    assert sampler.timeout == 600.0
    assert sampler.description == "Custom Config Job"
    assert sampler.preserve_solutions is True


def test_sampler_overrides_config():
    """Test that explicit sampler parameters override config"""
    config = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.CPU,
        use_notebook_output=False,
        default_timeout=600.0,
        default_description="Config Job",
        preserve_solutions=True,
    )

    bqm = dimod.BinaryQuadraticModel({0: 1.0}, {(0, 1): -1.0}, 0.0, "BINARY")
    model = BQM(bqm)
    sampler = DynexSampler(
        model,
        config=config,
        description="Override Job",
        preserve_solutions=False,
    )

    assert sampler.use_notebook_output is False  # from config (cannot override)
    assert sampler.timeout == 600.0  # from config (cannot override)
    assert sampler.description == "Override Job"
    assert sampler.preserve_solutions is False


def test_sampler_backward_compatibility():
    """Test backward compatibility with default parameters"""
    bqm = dimod.BinaryQuadraticModel({0: 1.0}, {(0, 1): -1.0}, 0.0, "BINARY")
    model = BQM(bqm)

    # Create sampler without config (should use defaults)
    sampler = DynexSampler(model)

    assert sampler.use_notebook_output is True
    assert sampler.timeout == 300.0
    assert sampler.description == "Dynex SDK Job"
    assert sampler.preserve_solutions is False


def test_sampler_partial_override():
    """Test that partial parameter override works"""
    config = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.CPU,
        use_notebook_output=False,
        default_timeout=600.0,
        default_description="Config Job",
        preserve_solutions=True,
    )

    bqm = dimod.BinaryQuadraticModel({0: 1.0}, {(0, 1): -1.0}, 0.0, "BINARY")
    model = BQM(bqm)

    # Override only description, others should come from config
    sampler = DynexSampler(model, config=config, description="Override Job", preserve_solutions=False)

    assert sampler.use_notebook_output is False  # from config (cannot override)
    assert sampler.timeout == 600.0  # from config (cannot override)
    assert sampler.description == "Override Job"  # overridden
    assert sampler.preserve_solutions is False  # overridden


def test_timeout_from_config_only():
    """Test that timeout is only configurable via config, not sampler parameter"""
    config = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.CPU,
        default_timeout=1200.0,
    )

    bqm = dimod.BinaryQuadraticModel({0: 1.0}, {(0, 1): -1.0}, 0.0, "BINARY")
    model = BQM(bqm)
    sampler = DynexSampler(model, config=config)

    assert sampler.timeout == 1200.0  # from config

    # Timeout must be set via config, not as sampler parameter
    config2 = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.CPU,
        default_timeout=900.0,
    )
    sampler2 = DynexSampler(model, config=config2)
    assert sampler2.timeout == 900.0  # different timeout from different config


def test_use_notebook_output_from_config_only():
    """Test that use_notebook_output is only configurable via config, not sampler parameter"""
    config = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.CPU,
        use_notebook_output=False,
    )

    bqm = dimod.BinaryQuadraticModel({0: 1.0}, {(0, 1): -1.0}, 0.0, "BINARY")
    model = BQM(bqm)
    sampler = DynexSampler(model, config=config)

    assert sampler.use_notebook_output is False  # from config

    # use_notebook_output must be set via config, not as sampler parameter
    config2 = DynexConfig(
        sdk_key="test_key",
        compute_backend=ComputeBackend.CPU,
        use_notebook_output=True,
    )
    sampler2 = DynexSampler(model, config=config2)
    assert sampler2.use_notebook_output is True  # different value from different config
