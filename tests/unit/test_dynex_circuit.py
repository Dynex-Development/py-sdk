"""
Unit tests for dynex.dynex_circuit module
"""

from dynex import DynexConfig
from dynex.dynex_circuit import CircuitModel, DynexCircuit


def test_circuit_model_initialization():
    """Test CircuitModel initialization"""
    model = CircuitModel(circuit_str="test_circuit", wires=4, params={"a": 1})

    assert model.circuit_str == "test_circuit"
    assert model.wires == 4
    assert model.params == {"a": 1}
    assert model.type == "qasm"
    assert model.type_str == "QASM"
    assert model.qasm_circuit is None
    assert model.bqm is None
    assert model.clauses == []
    assert model.wcnf_offset == 0
    assert model.precision == 1.0


def test_circuit_model_default_initialization():
    """Test CircuitModel with default values"""
    model = CircuitModel()

    assert model.circuit_str is None
    assert model.wires is None
    assert model.params is None
    assert model.type == "qasm"


def test_dynex_circuit_initialization():
    """Test DynexCircuit initialization"""
    circuit = DynexCircuit()

    assert circuit.config is not None
    assert isinstance(circuit.config, DynexConfig)
    assert circuit.logger is not None


def test_dynex_circuit_with_config():
    """Test DynexCircuit with custom config"""
    config = DynexConfig(compute_backend="cpu")
    circuit = DynexCircuit(config=config)

    assert circuit.config is config
    # compute_backend is already a string after internal conversion
    assert circuit.config.compute_backend == "cpu"


def test_sol2state_dict_with_qubit_keys():
    """Test sol2state with dict containing qubit keys"""
    circuit = DynexCircuit()
    sample = {
        "q_0_real": 1,
        "q_0_imag": 0,
        "q_1_real": 0,
        "q_1_imag": 1,
    }

    state = circuit.sol2state(
        sample=sample,
        wires=2,
        is_qpe=False,
        is_grover=False,
        is_cqu=False,
        is_qu=False,
    )

    assert isinstance(state, list)
    assert len(state) == 2


def test_sol2state_dict_without_qubit_keys():
    """Test sol2state with dict without qubit keys (WCNF format)"""
    circuit = DynexCircuit()
    sample = {0: 1, 1: 0, 2: 1}

    state = circuit.sol2state(
        sample=sample,
        wires=3,
        is_qpe=False,
        is_grover=False,
        is_cqu=False,
        is_qu=False,
    )

    assert isinstance(state, list)
    assert len(state) == 3
    assert state[0] == 1
    assert state[1] == 0
    assert state[2] == 1


def test_sol2state_empty_sample():
    """Test sol2state with empty sample"""
    circuit = DynexCircuit()
    sample = {}

    state = circuit.sol2state(
        sample=sample,
        wires=2,
        is_qpe=False,
        is_grover=False,
        is_cqu=False,
        is_qu=False,
    )

    assert isinstance(state, list)
    assert len(state) == 2
    assert state[0] == 0
    assert state[1] == 0
