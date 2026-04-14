# Apps-Benchmark - IonQ Quantum Application Benchmarking Framework

A framework for running quantum benchmarks on various backends and managing custom backends and benchmarks. The goal of
this, as compared to other quantum benchmarking tools, is to zoom out to compare overall algorithm performance.  

If you want to run these benchmarks against other hardware, you can use the DIY_BACKEND.md guide to make your 
own backend connector for other QPU.  If you want to add other Benchmarks to run, you can use DIY_BENCHMARK.md 
guide to make other benchmark tests to run on any QPU with a backend available. 

## Fastest First

Approximate runtime for a fresh local setup: 5-15 minutes, mostly spent installing dependencies. Once installed, the
verification commands below finish in seconds to under a minute each.

1. Create and activate a Python 3.12 environment. We recommend using `micromamba`.

   ```bash
   micromamba create -n apps-benchmark python=3.12
   micromamba activate apps-benchmark
   ```

2. Install the package with development dependencies.

   ```bash
   pip install -e ".[dev]"
   ```

3. Confirm the CLI is available and the backend can connect.

   ```bash
   apps-benchmark run --self-test --backend=mock_backend
   ```

4. Run the built-in 2-qubit H2 chemistry case (`h002_chain_0_75`) end-to-end.

   ```bash
   apps-benchmark run --backend=mock_backend --case-uuid=610cfb55
   ```

## Quick Start

The first `apps-benchmark` command that needs local-dev state reads `~/config_local_dev.json` and
creates it if it is missing.
The default `local_dev_dir` written there is `~/local_dev`, and commands that need DIY storage will
create that directory tree on demand.

```bash
# check base install is OK, list all commands
apps-benchmark --help

# List available backends and benchmarks
apps-benchmark list
apps-benchmark list --backends

# Display the bootstrap config and local development paths
apps-benchmark local-dev

# Run all benchmarks in a category (default: --qbit-max=10 filters out larger cases)
apps-benchmark run --backend=qiskit_aer_sim_backend --category=chemistry

# Override the default category filter
apps-benchmark run --backend=qiskit_aer_sim_backend --category=chemistry --qbit-max=20

# Run a specific benchmark by UUID. Note case-uuid takes precedence over qbit-max
apps-benchmark run --backend=qiskit_aer_sim_backend --case-uuid=610cfb55

# Run with a specific solution algorithm (when multiple algorithms are available)
apps-benchmark run --backend=qiskit_aer_sim_backend --case-uuid=f75ae75f --algorithm=qft

# Test backend connectivity
apps-benchmark run --self-test --backend=qiskit_aer_sim_backend
```

For detailed DIY plugin instructions, see [DIY Benchmark](docs/DIY_BENCHMARK.md) and [DIY Backend](docs/DIY_BACKEND.md) guides.

## Open vs Closed Benchmarks

Shipped benchmark cases may optionally declare `open_solution_algorithms`.
Those entries are open benchmarks: apps-benchmark ships the problem instance
and scoring metadata, but not the solver. Closed benchmarks only advertise
solvers that ship in this repository. The CLI lists open benchmark solvers and
fails with a clear bring-your-own-solver message instead of crashing at import
time.

## Backend setup for IonQ cloud
For using IonQ Cloud Backend, simulator or live, you will need to set the
environment variable IONQ_API_KEY to your own IonQ API key. You can do that
with a command like `export IONQ_API_KEY=foo` where your key is `foo`.


## Architecture

### Core Components

- **`apps_benchmark/core/backend.py`** - Abstract backend interfaces
  - `AbstractBackend` - Base class for all backends
  - `AbstractAsyncBackend` - For cloud/async backends
  - `JobStatus` - Job status enumeration

- **`apps_benchmark/core/benchmark_runner.py`** - Benchmark runner interfaces
  - `AbstractBenchmarkRunner` - Base class for benchmark runners
  - `BenchmarkSubmissionRecord` - Results dataclass

- **`apps_benchmark/primitives/benchmark_case.py`** - Problem definitions
  - `BenchmarkCase` - Standard format for benchmark problems

- **`apps_benchmark/core/registry.py`** - Component management
  - Auto-discovery of built-in components
  - Registration of DIY components
  - Atomic file operations

- **`apps_benchmark/cli.py`** - Command-line interface
  - `run` - Execute benchmarks
  - `list` - Show available components
  - `add` - Register DIY components


**Plugin Directory Structure:**
Note that custom components (backends or benchmarks) go in a `local_dev` directory as configured by the developer. See [docs/DIY_GUIDE.md](docs/DIY_GUIDE.md) for step-by-step instructions,
complete examples, and troubleshooting.

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[README.md](docs/README.md)** - Documentation index and overview
- **[DIY_GUIDE.md](docs/DIY_GUIDE.md)** - Quick start guide for creating custom components
- **[DIY_BENCHMARK.md](docs/DIY_BENCHMARK.md)** - Create custom benchmark algorithms
- **[DIY_BACKEND.md](docs/DIY_BACKEND.md)** - Create custom quantum backends
- **[DIY_REGISTRY.md](docs/DIY_REGISTRY.md)** - Registry system for custom components
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - API documentation for developers
- **[TESTING.md](docs/TESTING.md)** - Testing strategy and best practices
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** - Contribution guidelines
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Version history

See [docs/README.md](docs/README.md) for a complete documentation index.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=apps-benchmark

# Run specific test file
pytest tests/test_core/test_registry.py
```

### Code Quality

```bash
# Format code
ruff format apps-benchmark tests

# Lint code
ruff check apps-benchmark tests

# Type check package
mypy apps_benchmark

# Install git hooks
pre-commit install

# Run all configured hooks
pre-commit run --all-files
```

## Contributing

For contribution guidelines, see:
1. [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) - Complete contribution guide
2. [docs/DIY_BENCHMARK.md](docs/DIY_BENCHMARK.md) - Creating custom benchmarks
3. [docs/DIY_BACKEND.md](docs/DIY_BACKEND.md) - Creating custom backends
4. [docs/TESTING.md](docs/TESTING.md) - Testing requirements

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).

- **License Deed**: https://creativecommons.org/licenses/by-nc-nd/4.0/
- **Legal Code**: https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en

### What This Means

You are free to:
- **Share**: Copy and redistribute the material in any medium or format

Under the following terms:
- **Attribution**: You must give appropriate credit to IonQ, Inc., provide a link to the license, and indicate if changes were made
- **NonCommercial**: You may not use the material for commercial purposes
- **NoDerivatives**: If you remix, transform, or build upon the material, you may not distribute the modified material

For details on third-party dependencies and their licenses, see [NOTICE](NOTICE).
