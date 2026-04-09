"""Dynex SDK exception hierarchy."""


class DynexError(Exception):
    """Base exception for all Dynex SDK errors."""


class DynexValidationError(DynexError, ValueError):
    """Invalid configuration, parameters, or model input.

    Inherits from ValueError for backward compatibility with code
    that already catches ValueError from config/model validation.
    """


class DynexConnectionError(DynexError, ConnectionError):
    """Network or gRPC connectivity failure.

    Raised when the SDK cannot reach the backend, the gRPC channel
    drops, or a timeout occurs during communication.
    """


class DynexJobError(DynexError, RuntimeError):
    """Job-level failure on the backend side.

    Covers creation failures, solver errors, and unexpected
    responses from the compute infrastructure.
    """


class DynexSolverError(DynexError, RuntimeError):
    """Local solver execution failure (LOCAL backend only)."""


class DynexModelError(DynexError, ValueError):
    """Problem with the submitted model (empty, invalid formula, etc.)."""


class DynexAuthenticationError(DynexError, PermissionError):
    """Authentication failure (invalid API key, expired token, etc.)."""
