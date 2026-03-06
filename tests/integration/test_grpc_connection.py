import pytest

import dynex


@pytest.mark.integration
def test_config_with_credentials(sdk_credentials, run_integration_tests):
    """Test that DynexConfig can be created with credentials"""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    config = dynex.DynexConfig(
        sdk_key=sdk_credentials["sdk_key"],
        grpc_endpoint=sdk_credentials["grpc_endpoint"],
    )

    assert config.sdk_key == sdk_credentials["sdk_key"]
    assert config.grpc_endpoint == sdk_credentials["grpc_endpoint"]
    assert config.mainnet is True  # Internal flag: network mode


@pytest.mark.integration
def test_sampler_initialization(sdk_credentials, run_integration_tests):
    """Test that DynexSampler can be initialized with valid credentials"""
    if not run_integration_tests:
        pytest.skip("Integration tests disabled or credentials not available")

    config = dynex.DynexConfig(
        sdk_key=sdk_credentials["sdk_key"],
        grpc_endpoint=sdk_credentials["grpc_endpoint"],
        default_timeout=60.0,
    )

    sampler = dynex.DynexSampler(config)

    assert sampler is not None
    assert sampler.config.mainnet is True  # Internal flag: network mode
