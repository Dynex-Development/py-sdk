from dynex.qpu_models import QPUModel


def test_qpu_model_enum():
    assert QPUModel.APOLLO_RC1 == "apollo_rc1"
    assert QPUModel.APOLLO_10000 == "apollo_10000"


def test_qpu_model_values():
    models = [model.value for model in QPUModel]
    assert "apollo_rc1" in models
    assert "apollo_10000" in models


def test_qpu_model_comparison():
    assert QPUModel.APOLLO_RC1 == "apollo_rc1"
    assert QPUModel.APOLLO_10000 != "apollo_rc1"
