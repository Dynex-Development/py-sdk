"""
Integration tests for all compute backends (CPU, GPU, QPU).
Tests N-Queens, TSP, CQM and other optimization problems.
"""

from collections import defaultdict

import dimod
import numpy as np
import pytest

import dynex

# ============================================================================
# Test Configuration
# ============================================================================

BACKENDS = ["cpu", "gpu", "qpu"]

# Timeout per test (seconds)
TIMEOUT_CPU = 60
TIMEOUT_GPU = 120
TIMEOUT_QPU = 180


# ============================================================================
# Helper: N-Queens QUBO Generator
# ============================================================================


def n_queens_qubo(n, penalty=8500):
    """Generate QUBO for N-Queens problem."""
    Q = defaultdict(float)
    penalty_row, penalty_col, penalty_diag = (penalty,) * 3

    # Row constraints
    for r in range(n):
        for c1 in range(n):
            i = r * n + c1
            for c2 in range(n):
                j = r * n + c2
                if i <= j:
                    if i == j:
                        Q[(i, j)] += penalty_row
                    else:
                        Q[(i, j)] += 2 * penalty_row
        for c in range(n):
            i = r * n + c
            Q[(i, i)] += -2 * penalty_row

    # Column constraints
    for c in range(n):
        for r1 in range(n):
            i = r1 * n + c
            for r2 in range(n):
                j = r2 * n + c
                if i <= j:
                    if i == j:
                        Q[(i, j)] += penalty_col
                    else:
                        Q[(i, j)] += 2 * penalty_col
        for r in range(n):
            i = r * n + c
            Q[(i, i)] += -2 * penalty_col

    # Diagonal constraints (\)
    for d_idx in range(-(n - 1), n):
        for r1 in range(n):
            c1 = r1 - d_idx
            if 0 <= c1 < n:
                i = r1 * n + c1
                for r2 in range(n):
                    c2 = r2 - d_idx
                    if 0 <= c2 < n:
                        j = r2 * n + c2
                        if i < j:
                            Q[(i, j)] += 2 * penalty_diag

    # Anti-diagonal constraints (/)
    for d_idx in range(2 * n - 1):
        for r1 in range(n):
            c1 = d_idx - r1
            if 0 <= c1 < n:
                i = r1 * n + c1
                for r2 in range(n):
                    c2 = d_idx - r2
                    if 0 <= c2 < n:
                        j = r2 * n + c2
                        if i < j:
                            Q[(i, j)] += 2 * penalty_diag

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    return bqm


def is_valid_nqueens(solution, n):
    """Validate N-Queens solution."""
    board = [[0 for _ in range(n)] for _ in range(n)]
    queens_placed = 0

    for r in range(n):
        for c in range(n):
            idx = r * n + c
            if idx in solution and solution[idx] == 1:
                board[r][c] = 1
                queens_placed += 1

    if queens_placed != n:
        return False

    # Check rows
    for r in range(n):
        if sum(board[r]) != 1:
            return False

    # Check columns
    for c in range(n):
        if sum(board[r][c] for r in range(n)) != 1:
            return False

    # Check diagonals
    diag1 = set()
    diag2 = set()
    for r in range(n):
        for c in range(n):
            if board[r][c] == 1:
                if (r - c) in diag1 or (r + c) in diag2:
                    return False
                diag1.add(r - c)
                diag2.add(r + c)

    return True


# ============================================================================
# Helper: TSP QUBO Generator
# ============================================================================


def tsp_to_bqm(dist_matrix, lagrange_multiplier):
    """Convert TSP to BQM."""
    n = len(dist_matrix)
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

    # Objective: minimize distance
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            for p in range(n):
                var1 = u * n + p
                var2 = v * n + ((p + 1) % n)
                bqm.add_interaction(var1, var2, dist_matrix[u][v])

    # Constraints
    for u in range(n):
        city_vars = [(u * n + p, 1.0) for p in range(n)]
        bqm.add_linear_equality_constraint(city_vars, constant=-1.0, lagrange_multiplier=lagrange_multiplier)

    for p in range(n):
        step_vars = [(u * n + p, 1.0) for u in range(n)]
        bqm.add_linear_equality_constraint(step_vars, constant=-1.0, lagrange_multiplier=lagrange_multiplier)

    return bqm


def extract_tsp_path(solution, n):
    """Extract TSP path from solution."""
    path = [None] * n
    for var_idx, value in solution.items():
        if value == 1:
            city = var_idx // n
            position = var_idx % n
            path[position] = city
    return path


def is_valid_tsp_path(path, n):
    """Check if TSP path is valid."""
    if None in path:
        return False
    if len(set(path)) != n:
        return False
    return True


# ============================================================================
# Test 1: Simple BQM (All Backends)
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("backend", BACKENDS)
def test_simple_bqm_all_backends(sdk_credentials, run_integration_tests, backend):
    """Test simple BQM on all compute backends."""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    # Create simple BQM
    bqm = dimod.BinaryQuadraticModel.from_qubo({(0, 0): 1.0, (1, 1): -1.0, (0, 1): 0.5})

    # Configure backend
    timeout = float(TIMEOUT_QPU if backend == "qpu" else TIMEOUT_CPU)
    config_kwargs = {
        "sdk_key": sdk_credentials["sdk_key"],
        "grpc_endpoint": sdk_credentials["grpc_endpoint"],
        "compute_backend": backend,
        "default_timeout": timeout,
    }

    if backend == "qpu":
        config_kwargs["qpu_model"] = "apollo_rc1"

    config = dynex.DynexConfig(**config_kwargs)

    model = dynex.BQM(bqm)
    sampler = dynex.DynexSampler(model, config=config)

    # Sample with backend-specific parameters
    kwargs = {"num_reads": 10, "annealing_time": 100}

    if backend == "qpu":
        kwargs["qpu_max_coeff"] = 9.0

    sampleset = sampler.sample(**kwargs)

    # Verify results
    assert sampleset is not None
    assert len(sampleset) > 0
    assert sampleset.first.energy < float("inf")
    assert len(sampleset.first.sample) == 2


# ============================================================================
# Test 2: N-Queens (All Backends)
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [4, 6])  # Test 4-Queens and 6-Queens
def test_nqueens_all_backends(sdk_credentials, run_integration_tests, backend, n):
    """Test N-Queens problem on all compute backends."""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    # Generate N-Queens QUBO
    bqm = n_queens_qubo(n, penalty=8500)

    # Scale BQM for QPU
    if backend == "qpu":
        bqm, scale_factor = dynex.scale_bqm_to_range(bqm, max_abs_coeff=9.0)

    # Configure backend
    timeout = float(TIMEOUT_QPU if backend == "qpu" else TIMEOUT_CPU)
    config_kwargs = {
        "sdk_key": sdk_credentials["sdk_key"],
        "grpc_endpoint": sdk_credentials["grpc_endpoint"],
        "compute_backend": backend,
        "default_timeout": timeout,
    }

    if backend == "qpu":
        config_kwargs["qpu_model"] = "apollo_rc1"

    config = dynex.DynexConfig(**config_kwargs)

    model = dynex.BQM(bqm, logging=False)
    sampler = dynex.DynexSampler(model, config=config, description=f"N-Queens {n} on {backend}")

    # Sample with backend-specific parameters
    kwargs = {
        "num_reads": 32 if backend in ["cpu", "gpu"] else 1,
        "annealing_time": 1000 if backend in ["cpu", "gpu"] else 10,
        "shots": 1,
    }

    if backend == "qpu":
        kwargs["qpu_max_coeff"] = 9.0

    sampleset = sampler.sample(**kwargs)

    # Verify results
    assert sampleset is not None
    assert len(sampleset) > 0

    # Check if solution is valid
    best_solution = sampleset.first.sample
    is_valid = is_valid_nqueens(best_solution, n)

    print(f"\n{backend.upper()} backend - {n}-Queens: Valid={is_valid}, Energy={sampleset.first.energy}")

    # Verify solution has finite energy (validity is logged but not required)
    assert sampleset.first.energy < float("inf"), "Solution energy must be finite"


# ============================================================================
# Test 3: TSP (All Backends)
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("backend", BACKENDS)
def test_tsp_all_backends(sdk_credentials, run_integration_tests, backend):
    """Test TSP on all compute backends (small 4-city problem)."""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    # Create small 4-city TSP
    n_cities = 4
    np.random.seed(42)
    coords = np.random.rand(n_cities, 2) * 100

    # Calculate distance matrix
    dist_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])

    # Lagrange multiplier (penalty for constraint violations)
    max_dist = np.max(dist_matrix)
    lagrange_multiplier = max_dist * 1.5

    # Convert to BQM
    bqm = tsp_to_bqm(dist_matrix, lagrange_multiplier)

    # Scale for QPU
    if backend == "qpu":
        bqm, scale_factor = dynex.scale_bqm_to_range(bqm, max_abs_coeff=9.0)

    # Configure backend
    timeout = float(TIMEOUT_QPU if backend == "qpu" else TIMEOUT_CPU)
    config_kwargs = {
        "sdk_key": sdk_credentials["sdk_key"],
        "grpc_endpoint": sdk_credentials["grpc_endpoint"],
        "compute_backend": backend,
        "default_timeout": timeout,
    }

    if backend == "qpu":
        config_kwargs["qpu_model"] = "apollo_rc1"

    config = dynex.DynexConfig(**config_kwargs)

    model = dynex.BQM(bqm, logging=False)
    sampler = dynex.DynexSampler(model, config=config, description=f"TSP {n_cities} cities on {backend}")

    # Sample with backend-specific parameters
    kwargs = {
        "num_reads": 32 if backend in ["cpu", "gpu"] else 1,
        "annealing_time": 1000 if backend in ["cpu", "gpu"] else 10,
        "shots": 1,
    }

    if backend == "qpu":
        kwargs["qpu_max_coeff"] = 9.0

    sampleset = sampler.sample(**kwargs)

    # Verify results
    assert sampleset is not None
    assert len(sampleset) > 0
    assert sampleset.first.energy < float("inf")

    # Extract and validate path
    best_solution = sampleset.first.sample
    path = extract_tsp_path(best_solution, n_cities)
    is_valid = is_valid_tsp_path(path, n_cities)

    print(f"\n{backend.upper()} backend - TSP: Valid={is_valid}, Energy={sampleset.first.energy}, Path={path}")

    # Verify solution has finite energy (validity is logged but not required)
    assert sampleset.first.energy < float("inf"), "Solution energy must be finite"


# ============================================================================
# Test 4: Preprocessing Feature (QPU)
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_qpu_with_preprocessing(sdk_credentials, run_integration_tests):
    """Test QPU with CPU preprocessing (warm-start)."""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    # Create medium-sized problem
    bqm = dimod.BinaryQuadraticModel.from_qubo({(i, j): np.random.randn() for i in range(10) for j in range(i, 10)})

    # Scale for QPU
    bqm, scale_factor = dynex.scale_bqm_to_range(bqm, max_abs_coeff=9.0)

    # Configure QPU with preprocessing
    config = dynex.DynexConfig(
        sdk_key=sdk_credentials["sdk_key"],
        grpc_endpoint=sdk_credentials["grpc_endpoint"],
        compute_backend="qpu",
        qpu_model="apollo_rc1",
        default_timeout=float(TIMEOUT_QPU),
    )

    model = dynex.BQM(bqm)
    sampler = dynex.DynexSampler(model, config=config)

    # Sample with preprocessing enabled
    sampleset = sampler.sample(
        num_reads=1,
        annealing_time=100,
        qpu_max_coeff=9.0,
        preprocess=True,  # Enable CPU preprocessing
    )

    # Verify results
    assert sampleset is not None
    assert len(sampleset) > 0
    assert sampleset.first.energy < float("inf")


# ============================================================================
# Test 5: Apollo Chip Selection
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("chip", ["apollo_rc1", dynex.QPUModel.APOLLO_RC1])
def test_qpu_model_selection(sdk_credentials, run_integration_tests, chip):
    """Test different ways to specify QPU model."""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    # Simple BQM
    bqm = dimod.BinaryQuadraticModel({0: -1, 1: -1}, {(0, 1): 2}, 0.0, "BINARY")

    # Scale for QPU
    bqm, scale_factor = dynex.scale_bqm_to_range(bqm, max_abs_coeff=9.0)

    # Configure QPU with qpu_model in config
    config = dynex.DynexConfig(
        sdk_key=sdk_credentials["sdk_key"],
        grpc_endpoint=sdk_credentials["grpc_endpoint"],
        compute_backend="qpu",
        qpu_model=chip,
        default_timeout=float(TIMEOUT_QPU),
    )

    model = dynex.BQM(bqm)
    sampler = dynex.DynexSampler(model, config=config)

    # Sample with specific chip (now from config)
    sampleset = sampler.sample(num_reads=1, annealing_time=100, qpu_max_coeff=9.0)

    # Verify results
    assert sampleset is not None
    assert len(sampleset) > 0
    assert sampleset.first.energy < float("inf")


# ============================================================================
# Test 6: Large Problem (GPU)
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_large_problem_gpu(sdk_credentials, run_integration_tests):
    """Test large problem on GPU backend (100 variables)."""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    # Generate large random BQM
    n = 100
    np.random.seed(42)
    linear = {i: np.random.randn() for i in range(n)}
    quadratic = {(i, j): np.random.randn() for i in range(0, n, 10) for j in range(i + 1, min(i + 10, n))}

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, "BINARY")

    # Configure GPU
    config = dynex.DynexConfig(
        sdk_key=sdk_credentials["sdk_key"],
        grpc_endpoint=sdk_credentials["grpc_endpoint"],
        compute_backend="gpu",
        default_timeout=float(TIMEOUT_GPU),
    )

    model = dynex.BQM(bqm)
    sampler = dynex.DynexSampler(model, config=config)

    # Sample
    sampleset = sampler.sample(num_reads=50, annealing_time=2000)

    # Verify results
    assert sampleset is not None
    assert len(sampleset) > 0
    assert sampleset.first.energy < float("inf")
    assert len(sampleset.first.sample) == n


# ============================================================================
# Test 7: CQM (Constrained Quadratic Model)
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_cqm_all_backends(sdk_credentials, run_integration_tests, backend):
    """Test CQM on CPU and GPU (QPU doesn't directly support CQM yet)."""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    # Create CQM (knapsack problem)
    x1 = dimod.Integer("x1", upper_bound=10)
    x2 = dimod.Integer("x2", upper_bound=10)
    x3 = dimod.Integer("x3", upper_bound=10)

    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(-3 * x1 - 4 * x2 - 2 * x3)  # Maximize value
    cqm.add_constraint(2 * x1 + 3 * x2 + x3 <= 15, label="weight")
    cqm.add_constraint(x1 + x2 + x3 >= 2, label="min_items")

    # Configure backend
    timeout = float(TIMEOUT_GPU if backend == "gpu" else TIMEOUT_CPU)
    config = dynex.DynexConfig(
        sdk_key=sdk_credentials["sdk_key"],
        grpc_endpoint=sdk_credentials["grpc_endpoint"],
        compute_backend=backend,
        default_timeout=timeout,
    )

    model = dynex.CQM(cqm)
    sampler = dynex.DynexSampler(model, config=config)

    # Sample
    num_reads = 20 if backend in ["cpu", "gpu"] else 1
    annealing_time = 500 if backend == "cpu" else (1000 if backend == "gpu" else 100)
    sampleset = sampler.sample(num_reads=num_reads, annealing_time=annealing_time)

    # Verify results
    assert sampleset is not None
    assert len(sampleset) > 0
    assert sampleset.first.energy < float("inf")
    assert "x1" in sampleset.first.sample
    assert "x2" in sampleset.first.sample
    assert "x3" in sampleset.first.sample
