"""
Integration tests for all examples from examples/ directory.
Tests run on CPU, GPU, and QPU backends.
"""

from collections import defaultdict

import dimod
import numpy as np
import pytest

import dynex

# Check if PennyLane is available for Circuit tests
try:
    import pennylane as qml
    from pennylane import numpy as pnp

    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


# ============================================================================
# Example 1: N-Queens (from nqueens_demo.py)
# ============================================================================


def n_queens_qubo(n, penalty=8500):
    """Generate QUBO for N-Queens problem (from nqueens_demo.py)."""
    Q = defaultdict(float)
    penalty_row, penalty_col, penalty_diag = (penalty,) * 3

    # Row constraints: exactly one queen per row
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

    # Column constraints: exactly one queen per column
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


@pytest.mark.integration
@pytest.mark.parametrize(
    "backend,n",
    [
        ("cpu", 4),
        ("gpu", 4),
        ("qpu", 4),
    ],
)
def test_nqueens_example(sdk_credentials, run_integration_tests, backend, n):
    """Test from nqueens_demo.py"""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    # Generate QUBO from nqueens_demo.py
    bqm = n_queens_qubo(n, penalty=8500)

    # Scale for QPU
    if backend == "qpu":
        bqm, _ = dynex.scale_bqm_to_range(bqm, max_abs_coeff=9.0)

    # Configure backend
    timeout = 180.0 if backend == "qpu" else 90.0
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
    sampler = dynex.DynexSampler(model, config=config, description=f"{n}-Queens on {backend}")

    # Sample from nqueens_demo.py
    kwargs = {
        "num_reads": 10 if backend in ["cpu", "gpu"] else 1,
        "annealing_time": 1000 if backend in ["cpu", "gpu"] else 100,
    }
    if backend == "qpu":
        kwargs["qpu_max_coeff"] = 9.0

    sampleset = sampler.sample(**kwargs)

    # Verify results
    assert sampleset is not None
    assert len(sampleset) > 0
    assert sampleset.first.energy < float("inf")

    print(f"\n{backend.upper()} backend - {n}-Queens: Energy={sampleset.first.energy}")


# ============================================================================
# Example 2: TSP (from tsp_demo.py)
# ============================================================================


def tsp_to_bqm(dist_matrix, lagrange_multiplier):
    """
    Converts TSP to BQM exactly as in tsp_demo.py.
    """
    n = len(dist_matrix)
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

    # Objective: Minimize tour distance
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            for p in range(n):
                var1 = u * n + p
                var2 = v * n + ((p + 1) % n)
                bqm.add_interaction(var1, var2, dist_matrix[u][v])

    # Constraints
    # 1. Each city must be visited exactly once
    for u in range(n):
        city_vars = [(u * n + p, 1.0) for p in range(n)]
        bqm.add_linear_equality_constraint(city_vars, constant=-1.0, lagrange_multiplier=lagrange_multiplier)

    # 2. Each step must be assigned to exactly one city
    for p in range(n):
        step_vars = [(u * n + p, 1.0) for u in range(n)]
        bqm.add_linear_equality_constraint(step_vars, constant=-1.0, lagrange_multiplier=lagrange_multiplier)

    return bqm


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["cpu", "gpu", "qpu"])
def test_tsp_example(sdk_credentials, run_integration_tests, backend):
    """Test from tsp_demo.py"""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    # 4-city problem from nqueens_demo.py
    num_cities = 4
    np.random.seed(42)
    coords = np.random.rand(num_cities, 2) * 100

    # Distance matrix
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])

    # Generate BQM from nqueens_demo.py
    lagrange = 1000.0
    bqm = tsp_to_bqm(dist_matrix, lagrange)

    # Scale for QPU
    if backend == "qpu":
        bqm, _ = dynex.scale_bqm_to_range(bqm, max_abs_coeff=9.0)

    # Configure backend
    timeout = 180.0 if backend == "qpu" else 90.0
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
    sampler = dynex.DynexSampler(model, config=config, description=f"TSP-{num_cities} on {backend}")

    # Sample
    kwargs = {
        "num_reads": 10 if backend in ["cpu", "gpu"] else 1,
        "annealing_time": 1000 if backend in ["cpu", "gpu"] else 100,
    }
    if backend == "qpu":
        kwargs["qpu_max_coeff"] = 9.0

    sampleset = sampler.sample(**kwargs)

    # Verify results
    assert sampleset is not None
    assert len(sampleset) > 0
    assert sampleset.first.energy < float("inf")

    print(f"\n{backend.upper()} backend - TSP: Energy={sampleset.first.energy}")


# ============================================================================
# Example 3: Quantum Adder (from adder_demo.py)
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not PENNYLANE_AVAILABLE, reason="PennyLane not installed")
def test_adder_example_qpu(sdk_credentials, run_integration_tests):
    """Test from adder_demo.py on QPU"""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    def Nqubits(a, b):
        return (a + b).bit_length()

    def Kfourier(k, wires):
        for j in range(len(wires)):
            qml.RZ(k * pnp.pi / (2**j), wires=wires[j])

    def FullAdder(params, state=True):
        a, b = params
        n_wires = Nqubits(a, b)

        # Initialize qubits with the first number
        qml.BasisEmbedding(a, wires=range(n_wires))

        # Apply QFT
        qml.QFT(wires=range(n_wires))

        # Apply phase rotations
        Kfourier(b, range(n_wires))

        # Apply inverse QFT
        qml.adjoint(qml.QFT)(wires=range(n_wires))

        # Return state
        if state:
            return qml.state()
        else:
            return qml.sample()

    # From examples/: (12, 3, 15, "6-bit")
    a, b, expected = 12, 3, 15
    params = [a, b]
    wires = Nqubits(a, b)

    # Configure QPU from nqueens_demo.py
    config = dynex.DynexConfig(
        sdk_key=sdk_credentials["sdk_key"],
        grpc_endpoint=sdk_credentials["grpc_endpoint"],
        compute_backend="qpu",
        qpu_model="apollo_rc1",
    )

    dnx_circuit = dynex.DynexCircuit(config=config)

    # Execute from nqueens_demo.py
    measure = dnx_circuit.execute(
        FullAdder,
        params,
        wires,
        method="measure",
        logging=False,
        num_reads=1,
        integration_steps=1000,
        shots=1,
        qpu_max_coeff=9.0,
    )

    # Verify result
    assert measure is not None
    bitStr = "".join(map(str, measure.astype(int)))
    result = int(bitStr, 2)

    print(f"\nQPU - Quantum Adder: {a} + {b} = {result} (expected {expected})")
    assert result == expected, f"Expected {expected}, got {result}"


# ============================================================================
# Example 4: Grover's Hash Reversal (from grover_demo.py)
# ============================================================================


def _1HotPenalty(y_vars, lam):
    """From grover_demo.py"""
    lin = {}
    quad = {}
    offset = lam * 1.0
    for v in y_vars:
        lin[v] = lin.get(v, 0.0) - lam
    for i in range(len(y_vars)):
        for j in range(i + 1, len(y_vars)):
            quad[(y_vars[i], y_vars[j])] = quad.get((y_vars[i], y_vars[j]), 0.0) + 2.0 * lam
    return lin, quad, offset


def _BuildRHOracle(weights, target_hash, modulus, alpha=1.0, gamma=2.0, prefix=""):
    """From grover_demo.py"""
    n = len(weights)
    wsum = sum(weights)
    mMax = wsum // modulus
    xV = [f"{prefix}x{i}" for i in range(n)]
    yV = [f"{prefix}m{m}" for m in range(mMax + 1)]
    coeffs = {}
    const = -float(target_hash)
    for v, w in zip(xV, weights):
        coeffs[v] = coeffs.get(v, 0.0) + float(w)
    for m, v in enumerate(yV):
        coeffs[v] = coeffs.get(v, 0.0) - float(modulus * m)
    lin = {}
    quad = {}
    offset = alpha * (const**2)
    for v, a in coeffs.items():
        lin[v] = lin.get(v, 0.0) + alpha * (a**2 + 2.0 * const * a)
    var_list = list(coeffs.items())
    for i in range(len(var_list)):
        v_i, a_i = var_list[i]
        for j in range(i + 1, len(var_list)):
            v_j, a_j = var_list[j]
            quad[(v_i, v_j)] = quad.get((v_i, v_j), 0.0) + alpha * 2.0 * a_i * a_j
    ohLIN, ohQUAD, ohOFF = _1HotPenalty(yV, gamma)
    for v, b in ohLIN.items():
        lin[v] = lin.get(v, 0.0) + b
    for (u, v), b in ohQUAD.items():
        if (u, v) in quad:
            quad[(u, v)] += b
        elif (v, u) in quad:
            quad[(v, u)] += b
        else:
            quad[(u, v)] = b
    offset += ohOFF
    bqm = dimod.BinaryQuadraticModel(lin, quad, offset, vartype=dimod.BINARY)
    return bqm


@pytest.mark.integration
@pytest.mark.slow
def test_grover_example_qpu(sdk_credentials, run_integration_tests):
    """Test from grover_demo.py on QPU"""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    # From examples/
    weights = [1, 2, 3, 4]
    target_hash = 7
    modulus = 10

    # Build oracle from nqueens_demo.py
    bqm = _BuildRHOracle(weights, target_hash, modulus, alpha=1.0, gamma=2.0)

    # Configure QPU from nqueens_demo.py
    config = dynex.DynexConfig(
        sdk_key=sdk_credentials["sdk_key"],
        grpc_endpoint=sdk_credentials["grpc_endpoint"],
        compute_backend="qpu",
        qpu_model="apollo_rc1",
        default_timeout=180.0,
    )

    model = dynex.BQM(bqm, logging=False)
    sampler = dynex.DynexSampler(model, config=config, description="Grover Hash Reversal")

    # Sample from nqueens_demo.py
    sampleset = sampler.sample(
        num_reads=1,
        annealing_time=100,
        debugging=False,
        shots=1,
        preprocess=True,
        rank=1,
        qpu_max_coeff=9.0,
    )

    # Verify results
    assert sampleset is not None
    assert len(sampleset) > 0
    assert sampleset.first.energy < float("inf")

    print(f"\nQPU - Grover: Energy={sampleset.first.energy}")
