"""
Unit tests for dynex.grpc_client module
"""

import logging

from dynex import DynexConfig
from dynex.grpc_client import DynexGrpcClient


def test_grpc_client_initialization():
    """Test DynexGrpcClient initialization"""
    config = DynexConfig(compute_backend="cpu")
    client = DynexGrpcClient(config=config)

    assert client.config is config
    assert client.logger is None


def test_grpc_client_initialization_with_logger():
    """Test DynexGrpcClient initialization with logger"""
    config = DynexConfig(compute_backend="cpu")
    logger = logging.getLogger("test")
    client = DynexGrpcClient(config=config, logger=logger)

    assert client.logger is logger


def test_grpc_client_log_methods():
    """Test logging methods don't raise exceptions"""
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
