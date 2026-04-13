"""
Custom exception hierarchy for apps-benchmark.

All exceptions inherit from AppsBenchmarkError base class.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""


class AppsBenchmarkError(Exception):
    """Base exception for all apps-benchmark errors."""

    pass


# Backend errors
class BackendError(AppsBenchmarkError):
    """Base class for backend-related errors."""

    pass


class BackendNotFoundError(BackendError):
    """Backend not found in registry."""

    pass


class BackendValidationError(BackendError):
    """Backend failed interface validation."""

    pass


class BackendConnectionError(BackendError):
    """Backend is unreachable or connection failed."""

    pass


class BackendCredentialError(BackendError):
    """Backend credentials missing or invalid."""

    pass


# Benchmark errors
class BenchmarkError(AppsBenchmarkError):
    """Base class for benchmark-related errors."""

    pass


class BenchmarkNotFoundError(BenchmarkError):
    """Benchmark not found in registry."""

    pass


class BenchmarkValidationError(BenchmarkError):
    """Benchmark failed interface validation."""

    pass


class BenchmarkCaseError(BenchmarkError):
    """Problem instance JSON invalid or not found."""

    pass


# Config errors
class ConfigError(AppsBenchmarkError):
    """Base class for configuration errors."""

    pass


class ConfigNotFoundError(ConfigError):
    """Config file not found."""

    pass


class ConfigValidationError(ConfigError):
    """Config file invalid."""

    pass


# Registry errors
class RegistryError(AppsBenchmarkError):
    """Base class for registry errors."""

    pass
