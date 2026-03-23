import dimod
import pytest

import dynex


@pytest.mark.integration
@pytest.mark.slow
def test_simple_bqm_submission(sdk_credentials, run_integration_tests):
    """Test submitting a simple BQM problem through the full chain"""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    # Create simple BQM
    bqm = dimod.BinaryQuadraticModel.from_qubo({(0, 0): 1.0, (1, 1): -1.0, (0, 1): 0.5})

    # Initialize config and sampler
    config = dynex.DynexConfig(
        sdk_key=sdk_credentials["sdk_key"],
        grpc_endpoint=sdk_credentials["grpc_endpoint"],
        compute_backend="cpu",
        default_timeout=120.0,
    )

    model = dynex.BQM(bqm)
    sampler = dynex.DynexSampler(model, config=config)

    # Submit and get results
    sampleset = sampler.sample(num_reads=32, annealing_time=100)

    # Verify results
    assert sampleset is not None
    assert len(sampleset) > 0
    assert sampleset.first.energy < float("inf")
    assert len(sampleset.first.sample) == 2


@pytest.mark.integration
@pytest.mark.slow
def test_constrained_quadratic_model(sdk_credentials, run_integration_tests):
    """Test CQM problem submission"""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    # Create simple CQM (knapsack-like problem)
    x1 = dimod.Integer("x1", upper_bound=5)
    x2 = dimod.Integer("x2", upper_bound=5)

    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(-3 * x1 - 4 * x2)  # Maximize value
    cqm.add_constraint(x1 + x2 <= 7, label="capacity")

    # Initialize config and sampler
    config = dynex.DynexConfig(
        sdk_key=sdk_credentials["sdk_key"],
        grpc_endpoint=sdk_credentials["grpc_endpoint"],
        compute_backend="cpu",
        default_timeout=180.0,
    )

    model = dynex.CQM(cqm)
    sampler = dynex.DynexSampler(model, config=config)

    # Sample
    sampleset = sampler.sample(num_reads=32, annealing_time=100)

    # Verify
    assert sampleset is not None
    assert len(sampleset) > 0
    assert sampleset.first.energy < float("inf")
    assert "x1" in sampleset.first.sample
    assert "x2" in sampleset.first.sample
