"""
Dynex SDK (beta) Neuromorphic Computing Library
Copyright (c) 2021-2026, Dynex Developers

All rights reserved.

1. Redistributions of source code must retain the above copyright notice, this list of
    conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list
   of conditions and the following disclaimer in the documentation and/or other
   materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be
   used to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import logging
import os
from pathlib import Path
from typing import ClassVar, List, Optional, Union

from .compute_backend import ComputeBackend
from .exceptions import DynexValidationError
from .qpu_models import QPUModel

# Try to import python-dotenv, but make it optional
try:
    from dotenv import load_dotenv

    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False
    load_dotenv = None


class PlatformLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds platform prefix to all log messages."""

    def __init__(self, logger, config):
        super().__init__(logger, {})
        self.config = config

    def process(self, msg, kwargs):
        """Add platform prefix to message."""
        platform = self.config.get_platform_prefix()
        return f"[DYNEX-{platform}] {msg}", kwargs


class DynexConfig:
    """Configuration handler for Dynex SDK with ENV priority and validation.

    Supports loading configuration from:
    1. Function parameters (highest priority)
    2. Environment variables (DYNEX_*)
    3. .env file (if python-dotenv is installed)
    4. Default values (lowest priority)
    """

    # Default values
    DEFAULT_GRPC_ENDPOINT: ClassVar[str] = "127.0.0.1:9090"
    DEFAULT_SOLVER_LOCATIONS: ClassVar[List[str]] = ["testnet/"]
    ENV_PREFIX: ClassVar[str] = "DYNEX_"
    _dotenv_loaded: ClassVar[bool] = False

    def __init__(
        self,
        sdk_key: Optional[str] = None,
        grpc_endpoint: Optional[str] = None,
        solver_path: Optional[str] = None,
        retry_count: int = 5,
        remove_local_solutions: bool = False,
        dotenv_path: Optional[str] = None,
        compute_backend: Union[ComputeBackend, str] = ComputeBackend.UNSPECIFIED,
        qpu_model: Optional[Union[QPUModel, str]] = None,
        use_notebook_output: bool = True,
        default_timeout: float = 300.0,
        default_description: str = "Dynex SDK Job",
        preserve_solutions: bool = False,
        debug_save_solutions: bool = False,
    ) -> None:
        self.logger = self._init_logger()
        self.retry_count = retry_count
        self.remove_local_solutions = remove_local_solutions
        self.use_notebook_output = use_notebook_output
        self.default_timeout = default_timeout
        self.default_description = default_description
        self.preserve_solutions = preserve_solutions
        self.debug_save_solutions = debug_save_solutions

        # Validate and normalize compute_backend
        if isinstance(compute_backend, ComputeBackend):
            self.compute_backend = compute_backend.value
        elif isinstance(compute_backend, str):
            compute_backend_lower = compute_backend.lower()
            valid_backends = [b.value for b in ComputeBackend]
            if compute_backend_lower not in valid_backends:
                raise DynexValidationError(f"compute_backend must be one of {valid_backends}, got '{compute_backend}'")
            self.compute_backend = compute_backend_lower
        else:
            raise DynexValidationError(
                f"compute_backend must be ComputeBackend enum or string, got {type(compute_backend)}"
            )

        # Validate and normalize qpu_model
        if qpu_model is not None:
            if isinstance(qpu_model, QPUModel):
                self.qpu_model = qpu_model.value
            elif isinstance(qpu_model, str):
                qpu_model_lower = qpu_model.lower()
                valid_models = [m.value for m in QPUModel]
                if qpu_model_lower not in valid_models:
                    raise DynexValidationError(f"qpu_model must be one of {valid_models}, got '{qpu_model}'")
                self.qpu_model = qpu_model_lower
            else:
                raise DynexValidationError(f"qpu_model must be QPUModel enum or string, got {type(qpu_model)}")
        else:
            self.qpu_model = None

        is_qpu = self.compute_backend == "qpu"
        is_local = self.compute_backend == "local"

        if is_qpu and self.qpu_model is None:
            raise DynexValidationError(
                "qpu_model is required when compute_backend='qpu'. "
                "Available models: apollo_rc1, apollo_10000. "
                "Example: DynexConfig(compute_backend='qpu', qpu_model='apollo_rc1')"
            )

        # LOCAL backend uses local dynexcore binary (offline mode)
        # All other backends (CPU, GPU, QPU, UNSPECIFIED) use network mode

        self.mainnet = not is_local
        # Load .env file if available (only once per class)
        self._load_dotenv(dotenv_path)

        # Load configuration from parameters, ENV, .env, or defaults
        self.sdk_key = self._get_config_value(sdk_key, "SDK_KEY", required=self.mainnet)
        self.grpc_endpoint = self._get_config_value(grpc_endpoint, "GRPC_ENDPOINT", default=self.DEFAULT_GRPC_ENDPOINT)

        if self.mainnet:
            self._validate_grpc_endpoint(self.grpc_endpoint)

        if self.default_timeout <= 0:
            raise DynexValidationError(f"default_timeout must be positive, got {self.default_timeout}")
        if self.retry_count < 0:
            raise DynexValidationError(f"retry_count must be non-negative, got {self.retry_count}")

        # Resolve solver path for local mode (only for CPU/GPU with testnet)
        if not self.mainnet:
            self.solver_path = self._resolve_solver_path(solver_path)
            if not self.solver_path:
                raise FileNotFoundError("Solver file not found in testnet mode.")
        else:
            self.solver_path = None

        self._ensure_tmp_directory()

    def _init_logger(self):
        """Initialize logger with platform-aware adapter."""
        base_logger = logging.getLogger("dynex.config")
        if not base_logger.hasHandlers():
            handler = logging.StreamHandler()
            # Simple formatter - platform prefix is added by adapter
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        # Return a custom adapter that adds platform prefix to messages
        return PlatformLoggerAdapter(base_logger, self)

    def get_platform_prefix(self) -> str:
        """Get platform prefix for logging based on compute backend and QPU model."""
        backend = self.compute_backend.upper()

        if backend == "QPU" and self.qpu_model:
            qpu_name = self.qpu_model.upper().replace("_", "-")
            return qpu_name
        elif backend == "LOCAL":
            return "LOCAL"
        elif backend in ("CPU", "GPU"):
            return backend
        elif backend == "UNSPECIFIED":
            return "LOCAL" if not self.mainnet else "CPU"
        else:
            return "DYNEX"

    def update_logger_format(self):
        """Update logger formatter with platform prefix (for non-pytest usage)."""
        platform_prefix = f"DYNEX-{self.get_platform_prefix()}"
        for handler in self.logger.logger.handlers:
            formatter = logging.Formatter(f"[{platform_prefix}] %(message)s")
            handler.setFormatter(formatter)

    @classmethod
    def _load_dotenv(cls, dotenv_path: Optional[str] = None) -> None:
        """Load .env file if python-dotenv is available."""
        if cls._dotenv_loaded:
            return  # Already loaded

        if not _DOTENV_AVAILABLE:
            return  # python-dotenv not installed, skip silently

        # Determine .env file path
        if dotenv_path:
            env_path = Path(dotenv_path)
        else:
            # Look for .env in current directory and parent directories (up to 3 levels)
            current = Path.cwd()
            env_path = None
            for _ in range(4):  # Check current + 3 parent levels
                potential_path = current / ".env"
                if potential_path.exists():
                    env_path = potential_path
                    break
                current = current.parent
                if current == current.parent:  # Reached filesystem root
                    break

        if env_path and env_path.exists():
            try:
                load_dotenv(env_path, override=False)  # Don't override existing ENV vars
                cls._dotenv_loaded = True
            except Exception as e:
                # Silently fail if .env loading fails
                logger = logging.getLogger("dynex.config")
                logger.debug(f"Failed to load .env file: {e}")

    def _get_config_value(
        self, param_value: Optional[str], env_key: str, required: bool = False, default: Optional[str] = None
    ) -> str:
        """Get configuration value from parameter, ENV, .env file, or default.

        Priority order:
        1. Function parameter (highest)
        2. Environment variable (DYNEX_*)
        3. .env file (if loaded)
        4. Default value (lowest)
        """
        # Priority 1: Function parameter
        if param_value is not None:
            return param_value

        # Priority 2: Environment variable (includes .env if loaded)
        env_value = os.getenv(f"{self.ENV_PREFIX}{env_key}")
        if env_value is not None:
            return env_value

        # Priority 3: Default value
        if default is not None:
            return default

        if required:
            raise DynexValidationError(
                f"Required configuration '{env_key}' not provided. "
                f"Set {self.ENV_PREFIX}{env_key} environment variable, "
                f"add it to .env file, or pass as parameter."
            )

        return ""

    @staticmethod
    def _validate_grpc_endpoint(endpoint: str) -> None:
        """Validate gRPC endpoint format: host:port, with optional scheme."""
        raw = endpoint
        for prefix in ("https://", "http://", "grpc://"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break

        if ":" not in raw:
            raise DynexValidationError(
                f"grpc_endpoint must include a port (host:port), got '{endpoint}'"
            )
        host, _, port = raw.rpartition(":")
        if not host:
            raise DynexValidationError(f"grpc_endpoint has empty host, got '{endpoint}'")
        try:
            port_num = int(port)
        except ValueError:
            raise DynexValidationError(f"grpc_endpoint port must be numeric, got '{port}' in '{endpoint}'")
        if not (1 <= port_num <= 65535):
            raise DynexValidationError(f"grpc_endpoint port out of range (1-65535), got {port_num}")

    def _resolve_solver_path(self, solver_path: Optional[str]) -> Optional[str]:
        """Resolve solver path for LOCAL mode. Returns absolute path or None."""
        if solver_path:
            abs_path = os.path.abspath(solver_path)
            dynexcore_path = os.path.join(abs_path, "dynexcore")
            if os.path.exists(dynexcore_path):
                return abs_path + os.sep

        # Try default locations
        for location in self.DEFAULT_SOLVER_LOCATIONS:
            abs_location = os.path.abspath(location)
            dynexcore_path = os.path.join(abs_location, "dynexcore")
            if os.path.exists(dynexcore_path):
                return abs_location + os.sep

        return None

    def _ensure_tmp_directory(self) -> None:
        """Create tmp/ directory with write permissions."""
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        try:
            os.makedirs(tmp_dir, exist_ok=True)
            if not os.access(tmp_dir, os.W_OK):
                raise PermissionError(f"Cannot write to tmp/: {tmp_dir}")
        except OSError as e:
            self.logger.error(f"Failed to create tmp/: {e}")
            raise

    def as_dict(self) -> dict:
        """Return all config parameters as a dict."""
        return {
            "sdk_key": self.sdk_key,
            "grpc_endpoint": self.grpc_endpoint,
            "mainnet": self.mainnet,
            "solver_path": self.solver_path,
            "retry_count": self.retry_count,
            "remove_local_solutions": self.remove_local_solutions,
            "compute_backend": self.compute_backend,
            "qpu_model": self.qpu_model,
            "use_notebook_output": self.use_notebook_output,
            "default_timeout": self.default_timeout,
            "default_description": self.default_description,
            "preserve_solutions": self.preserve_solutions,
            "debug_save_solutions": self.debug_save_solutions,
        }
