# Testing Strategy

This document defines the testing approach for the apps-benchmark project.

## Overview

apps-benchmark uses pytest for testing with a focus on:
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Mock backends for testing without real quantum hardware
- Clear test organization mirroring source structure

---

## Test Structure

### Directory Layout

```
tests/
├── __init__.py
├── conftest.py                      # Shared fixtures and configuration
├── test_core/
│   ├── __init__.py
│   ├── test_backend_interface.py   # Backend ABC tests
│   ├── test_benchmark_interface.py # Benchmark runner ABC tests
│   ├── test_registry.py            # Registry operations
│   └── test_config.py              # Config file handling
├── test_backends/
│   ├── __init__.py
│   ├── test_mock_backend.py        # Mock backend tests
│   ├── test_ionq_backend.py        # IonQ backend (requires API key)
│   └── test_qiskit_aer_sim_backend.py      # Qiskit Aer backend
├── test_benchmarks/
│   ├── __init__.py
│   ├── test_chemistry/
│   │   ├── test_vqe_runner.py
│   │   └── test_benchmark_cases.py
│   └── test_optimization/
│       └── test_qaoa_runner.py
├── test_cli.py                      # CLI commands
├── test_utils/
│   ├── test_validation.py
│   └── test_errors.py
└── fixtures/
    ├── backends/                    # Sample backend files for testing
    │   └── test_backend.py
    ├── benchmarks/                  # Sample benchmark files
    │   └── test_runner.py
    └── benchmark_cases/           # Sample JSON files
        └── test_problem.json
```

---


### Unit Tests

Test individual functions and classes in isolation.

**Examples:**
- `test_backend_interface.py` - Test AbstractBackend methods
- `test_benchmark_case.py` - Test BenchmarkCase loading and validation
- `test_config.py` - Test config file save/load
- `test_errors.py` - Test custom exception hierarchy

**Characteristics:**
- Fast (< 1 second each)
- No external dependencies (mock everything)
- High coverage of edge cases
- Run on every commit

### Integration Tests

Test complete workflows end-to-end.

**Examples:**
- `test_cli.py` - Test full CLI commands
- `test_backend_registration.py` - Test adding/loading backends
- `test_benchmark_execution.py` - Test running benchmarks

**Characteristics:**
- Slower (1-10 seconds each)
- Use mock backends (no real API calls)
- Test component interactions
- Run before merge

### Smoke Tests

Quick sanity checks for critical functionality.

**Examples:**
- Import tests (can package be imported?)
- CLI help text (does `--help` work?)
- Registry initialization (can registries be created?)

**Characteristics:**
- Very fast (< 0.1 second each)
- Minimal coverage
- Run first in CI pipeline


## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core/test_backend_interface.py

# Run specific test
pytest tests/test_core/test_backend_interface.py::test_name_method

# Run tests by marker
pytest -m unit           # Only unit tests
pytest -m integration    # Only integration tests
pytest -m "not slow"     # Skip slow tests

# Parallel execution
pytest -n auto           # Use all CPU cores
pytest -n 4              # Use 4 workers

# Verbose output
pytest -v
pytest -vv               # Extra verbose

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Coverage report
pytest --cov=apps_benchmark --cov-report=html
open htmlcov/index.html  # View coverage
```

### Watch Mode (for development)

```bash
# Install pytest-watch
pip install pytest-watch

# Auto-run tests on file changes
ptw
```
---

## Test Coverage Goals

### Coverage Targets

- **Core framework**: 90%+ coverage
- **CLI commands**: 80%+ coverage
- **Backends**: 70%+ coverage (due to API mocking complexity)
- **Benchmarks**: 70%+ coverage (due to algorithm complexity)
- **Overall**: 80%+ coverage

### Checking Coverage

```bash
# Generate coverage report
pytest --cov=apps_benchmark --cov-report=term-missing

# HTML report
pytest --cov=apps_benchmark --cov-report=html
open htmlcov/index.html

# Fail if coverage below threshold
pytest --cov=apps_benchmark --cov-fail-under=80
```

---

## Summary

### Test Organization

- **Unit tests**: Fast, isolated, comprehensive
- **Integration tests**: End-to-end workflows
- **Mock backend**: Deterministic results for testing
- **Fixtures**: Reusable test setup

### Running Tests

```bash
pytest                  # All tests
pytest -m unit          # Unit tests only
pytest -n auto          # Parallel execution
pytest --cov=apps_benchmark # With coverage
```

### Coverage Goals

- Core: 90%+
- CLI: 80%+
- Backends/Benchmarks: 70%+
- Overall: 80%+

### CI/CD

- Run on every push/PR
- Test matrix: Python 3.12
- Multiple OS: Linux, macOS, Windows
- Real backend tests on main branch only
