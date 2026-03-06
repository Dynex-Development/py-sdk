import dimod

import dynex
from dynex import DynexConfig


def test_bqm_creation():
    config = DynexConfig(sdk_key="test_key")
    bqm = dimod.BinaryQuadraticModel.from_qubo({(0, 0): 1.0, (1, 1): -1.0, (0, 1): 0.5})
    model = dynex.BQM(bqm, config=config, logging=False)

    assert model is not None
    assert model.num_variables > 0


def test_bqm_simple():
    config = DynexConfig(sdk_key="test_key")
    bqm = dimod.BinaryQuadraticModel.from_qubo({(0, 0): 2.0})
    model = dynex.BQM(bqm, config=config, logging=False)
    assert model is not None
