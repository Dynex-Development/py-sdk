import logging
import os

import pytest
from dotenv import load_dotenv

_mpl_cache = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "mpl")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load .env file if it exists (for local testing)"""
    load_dotenv()


@pytest.fixture(scope="session")
def sdk_credentials():
    """
    Get SDK credentials from environment variables.
    Returns None if not available (allows skipping integration tests).
    """
    sdk_key = os.getenv("DYNEX_SDK_KEY")
    grpc_endpoint = os.getenv("DYNEX_GRPC_ENDPOINT", "https://quantum-router.dynex.co:8091")

    if not sdk_key:
        return None

    return {"sdk_key": sdk_key, "grpc_endpoint": grpc_endpoint}


@pytest.fixture(scope="session")
def run_integration_tests():
    """
    Check if integration tests should run.
    Returns True if credentials are available, unless explicitly disabled.
    """
    has_credentials = bool(os.getenv("DYNEX_SDK_KEY"))

    # Allow disabling tests even with credentials (e.g., RUN_INTEGRATION_TESTS=false)
    run_flag = os.getenv("RUN_INTEGRATION_TESTS", "true" if has_credentials else "false").lower()

    return has_credentials and run_flag == "true"


def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "integration: Integration tests that require API access")
    config.addinivalue_line("markers", "slow: Tests that take more than 30 seconds")
    config.addinivalue_line("markers", "unit: Fast unit tests without external dependencies")
