# Contributing to apps-benchmark

Thank you for your interest in contributing to the IonQ Quantum Application Benchmarking Framework!
This guide covers development workflows, tools, and best practices for contributing to apps-benchmark.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to:

- Be respectful and considerate in communication
- Accept constructive criticism gracefully
- Focus on what's best for the project
- Show empathy towards other contributors

Unacceptable behavior includes harassment, trolling, or personal attacks. Report issues to App Benchmark Support at apps-benchmark-support@ionq.co.


## Licensing

By contributing to this project, you agree that your contributions will be licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) License. This means:

- **Attribution**: All contributions must properly attribute IonQ, Inc.
- **NonCommercial**: Contributions cannot be used for commercial purposes
- **NoDerivatives**: Modified versions require permission from IonQ

Your contributions must be compatible with this license. If you're contributing code that depends on third-party libraries, ensure those libraries have compatible licenses (see [NOTICE](../NOTICE) for acceptable licenses).

For questions about licensing or to request special permissions, contact App Benchmark Support at apps-benchmark-support@ionq.co.

For complete license details, see:
- [LICENSE](../LICENSE) - Full license text
- [docs/LICENSE.md](LICENSE.md) - License summary and terms


## Development Setup

### Prerequisites
- Python 3.12 or higher
- Git
- Micromamba (recommended)

### Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/ionq/apps-benchmark.git
cd apps-benchmark
```
Note that based on release version and scope, the URL may change.

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

### About Pre-commit Hooks

Pre-commit hooks automatically run code quality checks (ruff formatting, linting, and mypy type checking) before each git commit. If any check fails, the commit is blocked and you'll need to fix the issues before committing. This ensures all committed code meets project standards without manual intervention.

## Code Standards

### Code Style
We use `ruff` for linting and formatting:
- Line length: 100 characters
- Target: Python 3.12
- Quote style: double quotes
- Indentation: spaces

Format your code:
```bash
ruff format .
```

Check for issues:
```bash
ruff check .
```

### Type Checking
We use `mypy` for static type checking. All code must be fully typed:
```bash
mypy apps_benchmark
```

## Testing

### Running Tests
```bash
pytest
```

Run tests in parallel:
```bash
pytest -n auto
```

Run with coverage:
```bash
pytest --cov=apps_benchmark --cov-report=term-missing
```

### Writing Tests
- Place tests in the `tests/` directory
- Use the naming convention `test_*.py`
- Test classes should be named `Test*`
- Test functions should be named `test_*`

## Git Workflow

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Description of changes"
```

3. Push to your branch:
```bash
git push origin feature/your-feature-name
```

4. Open a Pull Request against `main`

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters

## Pre-commit Hooks

Pre-commit hooks automatically run on every commit to ensure code quality:
- `ruff` formatting and linting
- `mypy` type checking
- Trailing whitespace removal
- YAML/JSON validation

If hooks fail, fix the issues and re-commit.

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add your changes to CHANGELOG.md (if applicable)
4. Request review from maintainers
5. Address review feedback
6. Once approved, a maintainer will merge your PR

## Documentation

### Building HTML Docs

Install pandoc, then generate HTML from any Markdown file:
```bash
# macOS
brew install pandoc

# Ubuntu/Debian
sudo apt-get install pandoc

# Generate HTML
pandoc docs/CONTRIBUTING.md --from=gfm --to=html5 --standalone -o CONTRIBUTING.html
```

HTML is also auto-generated via GitHub Actions on every push to `main`. See [README.md](README.md) for details.

## Troubleshooting

### Import Errors
Ensure package is installed in editable mode:
```bash
pip install -e .
```

### Test Failures
Run specific test with verbose output:
```bash
pytest -vv tests/test_file.py::test_name
```

### Type Checking Issues
Check specific file with verbose output:
```bash
mypy --show-error-codes apps_benchmark/file.py
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [mypy documentation](https://mypy.readthedocs.io/)
- [ruff documentation](https://docs.astral.sh/ruff/)
- [pre-commit hooks](https://pre-commit.com/)

## Questions or Issues?

If you have questions or encounter issues, please:
- Check existing issues on GitHub
- Open a new issue with detailed information
- Contact App Benchmark Support at apps-benchmark-support@ionq.co
