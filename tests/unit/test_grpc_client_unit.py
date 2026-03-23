"""
Unit tests for dynex.grpc_client module
"""

import logging

import pytest

from dynex import DynexConfig
from dynex.grpc_client import DynexGrpcClient, _qubo_arrays_to_wcnf_bytes


def test_grpc_client_initialization():
    config = DynexConfig(compute_backend="cpu")
    client = DynexGrpcClient(config=config)
    assert client.config is config
    assert client.logger is None


def test_grpc_client_initialization_with_logger():
    config = DynexConfig(compute_backend="cpu")
    logger = logging.getLogger("test")
    client = DynexGrpcClient(config=config, logger=logger)
    assert client.logger is logger


def test_grpc_client_log_methods():
    config = DynexConfig(compute_backend="cpu")
    null_logger = logging.getLogger("test.grpc.null")
    null_logger.addHandler(logging.NullHandler())
    null_logger.propagate = False
    client = DynexGrpcClient(config=config, logger=null_logger)

    client._log_grpc_action("test_action", "test_details")
    client._log_success("test_success")
    client._log_error("test_error")
    client._log_warning("test_warning")
    client._log_debug("test_debug")


def test_build_job_options_uses_proto_fields():
    """_build_job_options must set job_metadata_json directly, not hack description."""
    config = DynexConfig(compute_backend="cpu")
    client = DynexGrpcClient(config=config)

    opts = {
        "annealing_time": 100,
        "num_reads": 32,
        "job_metadata": {"type": "qasm", "qpu_model": "apollo_rc1"},
    }
    result = client._build_job_options(opts)

    assert "qasm" in result.job_metadata_json
    # description must NOT contain hacked metadata
    assert "preprocess=True" not in result.description
    assert "job_metadata=" not in result.description


def test_build_job_options_no_metadata():
    config = DynexConfig(compute_backend="cpu")
    client = DynexGrpcClient(config=config)

    opts = {"annealing_time": 50, "num_reads": 64}
    result = client._build_job_options(opts)

    assert result.job_metadata_json == ""
    assert result.description == ""


def test_iter_create_job_from_data_requests_message_sequence():
    """Verify init message comes before job_data message."""
    config = DynexConfig(compute_backend="cpu", sdk_key="test-key")
    client = DynexGrpcClient(config=config)

    opts = {"annealing_time": 100, "num_reads": 32, "request_ip": "1.2.3.4"}
    rows = [0, 1]
    cols = [0, 1]
    vals = [1.5, -2.0]

    messages = list(
        client._iter_create_job_from_data_requests(opts, rows, cols, vals, offset=0.5, num_vars=2, filename="test.dnx")
    )

    assert len(messages) == 2
    assert messages[0].HasField("init")
    assert messages[1].HasField("job_data")

    jd = messages[1].job_data
    assert jd.num_vars == 2
    assert list(jd.row) == [0, 1]
    assert list(jd.col) == [0, 1]
    assert pytest.approx(list(jd.val), abs=1e-5) == [1.5, -2.0]
    assert pytest.approx(jd.offset, abs=1e-5) == 0.5
    assert jd.filename == "test.dnx"


def test_qubo_arrays_to_wcnf_bytes_header_and_lines():
    """Same wire format as QRE ReconstructWCNF (integration with legacy chunk path)."""
    b = _qubo_arrays_to_wcnf_bytes([0, 1, 2], [0, 1, 2], [1.5, -2.0, 0.5], 0.0, 3)
    lines = b.decode().strip().split("\n")
    assert lines[0] == "p qubo 3 3 0"
    assert lines[1] == "0 0 1.5"
    assert lines[2] == "1 1 -2"
    assert lines[3] == "2 2 0.5"
