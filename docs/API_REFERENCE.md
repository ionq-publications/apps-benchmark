# API Reference

This document provides an overview of the apps-benchmark API for developers who want to programmatically use the framework.

## Module Structure

```
apps_benchmark/
├── core/           # Core benchmarking interfaces and abstractions
├── backends/       # Quantum backend implementations
├── benchmarks/     # Built-in benchmark algorithms
├── primitives/     # Quantum primitives and operations
├── utils/          # Utility functions and helpers
├── cli.py          # Command-line interface
└── errors.py       # Custom exception classes
```

## Core Modules

### `apps_benchmark.core`

The core module contains the fundamental abstractions for the benchmarking framework:

- **Interfaces**: Standard interfaces for backends, benchmarks, and circuits
- **Engine**: Main benchmarking execution engine
- **Results**: Result handling and storage
- **Validation**: Schema validation using Pydantic

### `apps_benchmark.backends`

Backend implementations for different quantum computing platforms:

- **IonQ Cloud Backend**: Direct integration with IonQ quantum hardware
- **Qiskit Aer Backend**: Local simulation using Qiskit Aer
- **Qiskit IonQ Backend**: IonQ access via Qiskit provider

#### Usage Example

```python
import os

from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider
from apps_benchmark.backends.ionq_cloud_backend import IonqCloudBackend

# Create provider with your API key
os.environ["IONQ_API_KEY"] = "your-api-key"
provider = IonQProvider(token="your-api-key")

# Get IonQ backend (simulator or QPU)
ionq_target = provider.get_backend("ionq_simulator")

# Create backend wrapper
backend = IonqCloudBackend(ionq_target, optimization_level=1)

# Create a simple circuit
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Submit a circuit
job_id, job_data = backend.submit([circuit], shots=1000)
```

### `apps_benchmark.benchmarks`

Built-in benchmark algorithm implementations:

- **Image Loading**: Image-loading benchmark with best-over-depth MSE scoring over shipped precompiled circuits
- **QAOA**: Quantum Approximate Optimization Algorithm
- **LR-QAOA**: Long-Range QAOA variant
- **VQE**: Variational Quantum Eigensolver
- **FAA**: Fixed Point Amplitude Amplification - A quantum search algorithm using Quantum Singular Value Transforms (QSVT) for unstructured search problems

#### Usage Example

```python
from apps_benchmark.benchmarks.qaoa import QAOABenchmark

# Create benchmark instance
benchmark = QAOABenchmark(
    num_qubits=4,
    num_layers=2,
    problem_type="maxcut"
)

# Run benchmark
results = benchmark.run(backend=backend, shots=1000)
```

### `apps_benchmark.primitives`

Low-level quantum primitives and operations:

- **Circuit builders**: Helper functions for circuit construction
- **Gates**: Common quantum gate definitions
- **Measurements**: Measurement operations and utilities

### `apps_benchmark.utils`

Utility functions for common tasks:

- **Data processing**: Result parsing and formatting
- **Visualization**: Plotting and visualization helpers
- **Serialization**: JSON/QASM import/export
- **Validation**: Input validation and error checking

## Error Handling

### `apps_benchmark.errors`

Custom exception classes for error handling:

```python
from apps_benchmark.errors import (
    BenchmarkError,      # Base benchmark exception
    BackendError,        # Backend-related errors
    ValidationError,     # Input validation errors
    CircuitError,        # Circuit construction errors
)
```

#### Usage Example

```python
try:
    result = benchmark.run(backend=backend)
except ValidationError as e:
    print(f"Invalid input: {e}")
except BackendError as e:
    print(f"Backend error: {e}")
```

## CLI Interface

### `apps_benchmark.cli`

The CLI module provides the command-line interface built with Click:

```python
from apps_benchmark.cli import main

# Programmatically invoke CLI
if __name__ == "__main__":
    main()
```

### Available Commands

```bash
# Run a benchmark
apps-benchmark run <benchmark_name> [options]

# List available benchmarks
apps-benchmark list

# Show benchmark details
apps-benchmark info <benchmark_name>

# Validate configuration
apps-benchmark validate <config_file>
```

## Configuration

### Pydantic Models

The framework uses Pydantic for configuration and validation:

```python
from pydantic import BaseModel
from apps_benchmark.core import BenchmarkConfig

class MyBenchmarkConfig(BenchmarkConfig):
    num_qubits: int
    num_layers: int
    shots: int = 1000
```

## Type Hints

The codebase uses type annotations throughout. All public APIs have type hints:

```python
from typing import Dict, List, Optional
from apps_benchmark.core import Circuit, BackendResult

def process_results(
    results: List[BackendResult],
    metadata: Optional[Dict[str, str]] = None
) -> Dict[str, float]:
    # Implementation
    pass
```

## Best Practices

### 1. Use Type Hints
Always use type hints when working with the API for better IDE support and type checking.

### 2. Handle Errors Gracefully
Catch specific exceptions rather than bare `except` clauses:

```python
try:
    result = benchmark.run(backend)
except BackendError as e:
    logger.error(f"Backend failed: {e}")
    # Handle error appropriately
```

### 3. Validate Inputs
Use Pydantic models for input validation:

```python
from apps_benchmark.core import BenchmarkConfig

config = BenchmarkConfig(**user_input)  # Validates automatically
```

### 4. Clean Up Resources
Always close connections and clean up resources:

```python
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider
from apps_benchmark.backends.ionq_cloud_backend import IonqCloudBackend

# Set up backend
provider = IonQProvider(token="your-api-key")
ionq_target = provider.get_backend("ionq_simulator")
backend = IonqCloudBackend(ionq_target)

# Create and run circuit
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

results, job_id, job_data = backend.run([circuit], shots=1000)
```

## Examples

See the [DIY_BENCHMARK.md](DIY_BENCHMARK.md) and [DIY_BACKEND.md](DIY_BACKEND.md) guides for comprehensive usage examples and tutorials.

## Further Documentation

- [DIY Guide](DIY_GUIDE.md) - Quick start for creating custom components
- [DIY Benchmark Guide](DIY_BENCHMARK.md) - Create custom benchmarks
- [DIY Backend Guide](DIY_BACKEND.md) - Create custom backends
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute
- [Testing Guide](TESTING.md) - Testing best practices
- [Main README](../README.md) - Project overview

## API Stability

This is version 1.0.0. The public API is considered stable, but:
- Internal APIs (prefixed with `_`) may change without notice
- Experimental features are marked as such in docstrings
- Deprecated features will be marked for at least one minor version before removal

## Getting Help

For API questions or issues:
- Check the inline docstrings (`help(function_name)`)
- Review the source code in the repository
- Open an issue on GitHub
- Contact App Benchmark Support at apps-benchmark-support@ionq.co
