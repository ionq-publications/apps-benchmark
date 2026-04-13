"""
Tests for custom exception hierarchy.

This module tests all custom exceptions defined in apps_benchmark.errors.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import pytest
from apps_benchmark.errors import (
    AppsBenchmarkError,
    BackendConnectionError,
    BackendCredentialError,
    BackendError,
    BackendNotFoundError,
    BackendValidationError,
    BenchmarkCaseError,
    BenchmarkError,
    BenchmarkNotFoundError,
    BenchmarkValidationError,
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
    RegistryError,
)


class TestAppsBenchmarkError:
    """Tests for base AppsBenchmarkError."""

    def test_apps_benchmark_error_is_exception(self):
        """Test that AppsBenchmarkError inherits from Exception."""
        assert issubclass(AppsBenchmarkError, Exception)

    def test_apps_benchmark_error_can_be_raised(self):
        """Test that AppsBenchmarkError can be raised and caught."""
        with pytest.raises(AppsBenchmarkError):
            raise AppsBenchmarkError("Test error")

    def test_apps_benchmark_error_message(self):
        """Test that AppsBenchmarkError preserves error message."""
        error_msg = "This is a test error"
        with pytest.raises(AppsBenchmarkError, match=error_msg):
            raise AppsBenchmarkError(error_msg)


class TestBackendErrors:
    """Tests for backend-related exceptions."""

    def test_backend_error_inheritance(self):
        """Test BackendError inherits from AppsBenchmarkError."""
        assert issubclass(BackendError, AppsBenchmarkError)
        assert issubclass(BackendError, Exception)

    def test_backend_not_found_error_inheritance(self):
        """Test BackendNotFoundError inherits correctly."""
        assert issubclass(BackendNotFoundError, BackendError)
        assert issubclass(BackendNotFoundError, AppsBenchmarkError)

    def test_backend_validation_error_inheritance(self):
        """Test BackendValidationError inherits correctly."""
        assert issubclass(BackendValidationError, BackendError)
        assert issubclass(BackendValidationError, AppsBenchmarkError)

    def test_backend_connection_error_inheritance(self):
        """Test BackendConnectionError inherits correctly."""
        assert issubclass(BackendConnectionError, BackendError)
        assert issubclass(BackendConnectionError, AppsBenchmarkError)

    def test_backend_credential_error_inheritance(self):
        """Test BackendCredentialError inherits correctly."""
        assert issubclass(BackendCredentialError, BackendError)
        assert issubclass(BackendCredentialError, AppsBenchmarkError)

    def test_backend_errors_can_be_caught_by_base(self):
        """Test that specific backend errors can be caught by BackendError."""
        with pytest.raises(BackendError):
            raise BackendNotFoundError("Backend not found")

        with pytest.raises(BackendError):
            raise BackendValidationError("Validation failed")

        with pytest.raises(BackendError):
            raise BackendConnectionError("Connection failed")

        with pytest.raises(BackendError):
            raise BackendCredentialError("Bad credentials")

    def test_backend_errors_can_be_caught_by_apps_benchmark_error(self):
        """Test that backend errors can be caught by AppsBenchmarkError."""
        with pytest.raises(AppsBenchmarkError):
            raise BackendNotFoundError("Test")

    def test_backend_error_messages(self):
        """Test that backend errors preserve messages."""
        msg = "Custom backend error message"
        with pytest.raises(BackendNotFoundError, match=msg):
            raise BackendNotFoundError(msg)


class TestBenchmarkErrors:
    """Tests for benchmark-related exceptions."""

    def test_benchmark_error_inheritance(self):
        """Test BenchmarkError inherits from AppsBenchmarkError."""
        assert issubclass(BenchmarkError, AppsBenchmarkError)
        assert issubclass(BenchmarkError, Exception)

    def test_benchmark_not_found_error_inheritance(self):
        """Test BenchmarkNotFoundError inherits correctly."""
        assert issubclass(BenchmarkNotFoundError, BenchmarkError)
        assert issubclass(BenchmarkNotFoundError, AppsBenchmarkError)

    def test_benchmark_validation_error_inheritance(self):
        """Test BenchmarkValidationError inherits correctly."""
        assert issubclass(BenchmarkValidationError, BenchmarkError)
        assert issubclass(BenchmarkValidationError, AppsBenchmarkError)

    def test_benchmark_case_error_inheritance(self):
        """Test BenchmarkCaseError inherits correctly."""
        assert issubclass(BenchmarkCaseError, BenchmarkError)
        assert issubclass(BenchmarkCaseError, AppsBenchmarkError)

    def test_benchmark_errors_can_be_caught_by_base(self):
        """Test that specific benchmark errors can be caught by BenchmarkError."""
        with pytest.raises(BenchmarkError):
            raise BenchmarkNotFoundError("Benchmark not found")

        with pytest.raises(BenchmarkError):
            raise BenchmarkValidationError("Validation failed")

        with pytest.raises(BenchmarkError):
            raise BenchmarkCaseError("Problem instance error")

    def test_benchmark_errors_can_be_caught_by_apps_benchmark_error(self):
        """Test that benchmark errors can be caught by AppsBenchmarkError."""
        with pytest.raises(AppsBenchmarkError):
            raise BenchmarkNotFoundError("Test")

    def test_benchmark_error_messages(self):
        """Test that benchmark errors preserve messages."""
        msg = "Custom benchmark error message"
        with pytest.raises(BenchmarkValidationError, match=msg):
            raise BenchmarkValidationError(msg)


class TestConfigErrors:
    """Tests for config-related exceptions."""

    def test_config_error_inheritance(self):
        """Test ConfigError inherits from AppsBenchmarkError."""
        assert issubclass(ConfigError, AppsBenchmarkError)
        assert issubclass(ConfigError, Exception)

    def test_config_not_found_error_inheritance(self):
        """Test ConfigNotFoundError inherits correctly."""
        assert issubclass(ConfigNotFoundError, ConfigError)
        assert issubclass(ConfigNotFoundError, AppsBenchmarkError)

    def test_config_validation_error_inheritance(self):
        """Test ConfigValidationError inherits correctly."""
        assert issubclass(ConfigValidationError, ConfigError)
        assert issubclass(ConfigValidationError, AppsBenchmarkError)

    def test_config_errors_can_be_caught_by_base(self):
        """Test that specific config errors can be caught by ConfigError."""
        with pytest.raises(ConfigError):
            raise ConfigNotFoundError("Config not found")

        with pytest.raises(ConfigError):
            raise ConfigValidationError("Validation failed")

    def test_config_errors_can_be_caught_by_apps_benchmark_error(self):
        """Test that config errors can be caught by AppsBenchmarkError."""
        with pytest.raises(AppsBenchmarkError):
            raise ConfigNotFoundError("Test")

    def test_config_error_messages(self):
        """Test that config errors preserve messages."""
        msg = "Custom config error message"
        with pytest.raises(ConfigNotFoundError, match=msg):
            raise ConfigNotFoundError(msg)


class TestRegistryErrors:
    """Tests for registry-related exceptions."""

    def test_registry_error_inheritance(self):
        """Test RegistryError inherits from AppsBenchmarkError."""
        assert issubclass(RegistryError, AppsBenchmarkError)
        assert issubclass(RegistryError, Exception)

    def test_registry_error_can_be_raised(self):
        """Test that RegistryError can be raised and caught."""
        with pytest.raises(RegistryError):
            raise RegistryError("Registry error")

    def test_registry_error_can_be_caught_by_apps_benchmark_error(self):
        """Test that registry error can be caught by AppsBenchmarkError."""
        with pytest.raises(AppsBenchmarkError):
            raise RegistryError("Test")

    def test_registry_error_message(self):
        """Test that registry error preserves message."""
        msg = "Custom registry error message"
        with pytest.raises(RegistryError, match=msg):
            raise RegistryError(msg)


class TestErrorHierarchy:
    """Tests for overall error hierarchy."""

    def test_all_errors_inherit_from_apps_benchmark_error(self):
        """Test that all custom errors inherit from AppsBenchmarkError."""
        error_classes = [
            BackendError,
            BackendNotFoundError,
            BackendValidationError,
            BackendConnectionError,
            BackendCredentialError,
            BenchmarkError,
            BenchmarkNotFoundError,
            BenchmarkValidationError,
            BenchmarkCaseError,
            ConfigError,
            ConfigNotFoundError,
            ConfigValidationError,
            RegistryError,
        ]

        for error_class in error_classes:
            assert issubclass(error_class, AppsBenchmarkError)
            assert issubclass(error_class, Exception)

    def test_error_hierarchy_organization(self):
        """Test that errors are properly organized by category."""
        # Backend errors
        backend_errors = [
            BackendNotFoundError,
            BackendValidationError,
            BackendConnectionError,
            BackendCredentialError,
        ]
        for error in backend_errors:
            assert issubclass(error, BackendError)

        # Benchmark errors
        benchmark_errors = [
            BenchmarkNotFoundError,
            BenchmarkValidationError,
            BenchmarkCaseError,
        ]
        for error in benchmark_errors:
            assert issubclass(error, BenchmarkError)

        # Config errors
        config_errors = [ConfigNotFoundError, ConfigValidationError]
        for error in config_errors:
            assert issubclass(error, ConfigError)

    def test_can_catch_all_with_apps_benchmark_error(self):
        """Test that AppsBenchmarkError can catch all custom errors."""
        errors_to_test = [
            BackendError("test"),
            BackendNotFoundError("test"),
            BackendValidationError("test"),
            BackendConnectionError("test"),
            BackendCredentialError("test"),
            BenchmarkError("test"),
            BenchmarkNotFoundError("test"),
            BenchmarkValidationError("test"),
            BenchmarkCaseError("test"),
            ConfigError("test"),
            ConfigNotFoundError("test"),
            ConfigValidationError("test"),
            RegistryError("test"),
        ]

        for error in errors_to_test:
            with pytest.raises(AppsBenchmarkError):
                raise error

    def test_can_distinguish_error_categories(self):
        """Test that we can distinguish between error categories."""
        # Backend errors should not be caught by BenchmarkError
        with pytest.raises(BackendError):
            try:
                raise BackendNotFoundError("test")
            except BenchmarkError:
                pytest.fail("BackendError caught by BenchmarkError")

        # Benchmark errors should not be caught by BackendError
        with pytest.raises(BenchmarkError):
            try:
                raise BenchmarkNotFoundError("test")
            except BackendError:
                pytest.fail("BenchmarkError caught by BackendError")

        # Config errors should not be caught by BackendError or BenchmarkError
        with pytest.raises(ConfigError):
            try:
                raise ConfigNotFoundError("test")
            except (BackendError, BenchmarkError):
                pytest.fail("ConfigError caught by wrong category")


class TestErrorUsagePatterns:
    """Tests for common error usage patterns."""

    def test_error_with_formatted_message(self):
        """Test errors with formatted messages."""
        backend_name = "my_backend"
        msg = f"Backend '{backend_name}' not found in registry"

        with pytest.raises(BackendNotFoundError, match=msg):
            raise BackendNotFoundError(msg)

    def test_error_with_multiline_message(self):
        """Test errors with multiline messages."""
        msg = """Backend validation failed:
        - Missing required method: run()
        - Class does not inherit from AbstractBackend"""

        with pytest.raises(BackendValidationError):
            raise BackendValidationError(msg)

    def test_error_chaining(self):
        """Test that errors can be chained with 'from'."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise BackendError("Wrapped error") from e
        except BackendError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)

    def test_catching_by_category(self):
        """Test practical pattern of catching by category."""

        def operation_that_fails():
            """Simulated operation that raises specific error."""
            raise BackendConnectionError("Connection timeout")

        # Should be able to catch by specific type
        with pytest.raises(BackendConnectionError):
            operation_that_fails()

        # Should be able to catch by category
        with pytest.raises(BackendError):
            operation_that_fails()

        # Should be able to catch by base
        with pytest.raises(AppsBenchmarkError):
            operation_that_fails()
