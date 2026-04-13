# Changelog

All notable changes to the IonQ Quantum Application Benchmarking Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-10

### Added
- Initial release of apps-benchmark framework
- Core benchmarking engine with standardized interfaces
- Support for multiple quantum backends:
  - IonQ Cloud backend
  - Qiskit Aer simulator
- Built-in benchmark algorithms:
  - QAOA (Quantum Approximate Optimization Algorithm)
  - LR-QAOA (Long-Range QAOA)
  - VQE (Variational Quantum Eigensolver)
- CLI interface with `apps-benchmark` command
- Comprehensive test suite with pytest
- Pre-commit hooks for code quality
- Type checking with mypy
- Code formatting with ruff

### Dependencies
- Python 3.12+ support
- Qiskit 2.3.1
- Qiskit Aer 0.17.2
- NetworkX for graph algorithms
- Pydantic for data validation
- Click for CLI framework

## [Unreleased]

### Planned
- Additional benchmark algorithms
- Enhanced performance metrics
- Improved visualization tools
- Extended backend support

---

## Version History Format

### Added
New features and capabilities

### Changed
Changes to existing functionality

### Deprecated
Features that will be removed in upcoming releases

### Removed
Removed features

### Fixed
Bug fixes

### Security
Security-related changes
