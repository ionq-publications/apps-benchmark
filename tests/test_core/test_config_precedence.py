"""
Tests for configuration precedence rules.

This module tests that configuration values are applied in the correct order:
CLI args > loaded config > defaults

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import json
from pathlib import Path

import pytest
from apps_benchmark.cli import main
from click.testing import CliRunner


@pytest.fixture
def cli_runner():
    """Fixture for Click CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_home(tmp_path, monkeypatch):
    """Mock home directory for testing."""

    # Mock Path.home() for all relevant modules - use consistent approach
    def mock_home_func():
        return tmp_path

    monkeypatch.setattr(Path, "home", mock_home_func)

    # Initialize local_dev directory
    local_dev = tmp_path / "local_dev"
    local_dev.mkdir()
    (local_dev / "backends").mkdir()
    (local_dev / "benchmarks").mkdir()

    # Create config pointing to local_dev
    config_file = tmp_path / "config_local_dev.json"
    with open(config_file, "w") as f:
        json.dump({"local_dev_dir": str(local_dev), "version": "1.0"}, f)

    # Create registries with correct schema
    backends_registry = {
        "version": "1.0",
        "backends": {
            "mock_backend": {
                "module": "apps_benchmark.backends.mock_backend",
                "class": "MockBackend",
                "builtin": True,
                "location": "built-in",
                "registered_at": "2024-01-01T00:00:00Z",
            }
        },
    }
    benchmarks_registry = {
        "version": "1.0",
        "builtin_benchmarks": {
            "chemistry": {
                "location": "/benchmarks",
                "runners": ["vqe_puccd"],
                "benchmark_cases": [
                    {
                        "uuid": "610cfb55",
                        "name": "h002_chain_0_75",
                        "file": str(
                            Path(__file__).parent.parent.parent
                            / "apps_benchmark/benchmarks/chemistry/benchmark_cases/h002_chain_0_75.json"
                        ),
                        "builtin": True,
                    }
                ],
            }
        },
        "diy_benchmarks": {},
    }

    with open(local_dev / "backends.json", "w") as f:
        json.dump(backends_registry, f)
    with open(local_dev / "benchmarks.json", "w") as f:
        json.dump(benchmarks_registry, f)

    return tmp_path


class TestConfigPrecedence:
    """Tests for configuration precedence rules."""

    def test_cli_overrides_config_backend(self, cli_runner, mock_home):
        """Test that CLI backend arg overrides config backend."""
        # Create config with backend=mock_backend
        config_dir = mock_home / ".apps-benchmark"
        config_dir.mkdir()
        config_file = config_dir / "apps-benchmark-config-test.json"
        config_data = {
            "version": "1.0",
            "backend": "should_be_overridden",
            "qbit_max": 10,
            "shots": 1000,
            "category": "chemistry",
            "case_uuid": None,
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Run with load-config but provide explicit backend
        result = cli_runner.invoke(main, ["run", "--load-config=test", "--backend=mock_backend"])

        # Should use CLI-provided backend, not config backend
        assert result.exit_code == 0, f"Command failed. Output:\n{result.output}"
        assert "Loading configuration 'test'" in result.output

    def test_cli_overrides_config_shots(self, cli_runner, mock_home):
        """Test that CLI shots arg overrides config shots."""
        # Create config with shots=1000
        config_dir = mock_home / ".apps-benchmark"
        config_dir.mkdir()
        config_file = config_dir / "apps-benchmark-config-test.json"
        config_data = {
            "version": "1.0",
            "backend": "mock_backend",
            "qbit_max": 10,
            "shots": 1000,
            "category": "chemistry",
            "case_uuid": None,
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Run with load-config but provide explicit shots
        result = cli_runner.invoke(main, ["run", "--load-config=test", "--shots=5000"])

        # Should use CLI-provided shots (5000), not config shots (1000)
        assert result.exit_code == 0
        assert "Loading configuration 'test'" in result.output

    def test_cli_overrides_config_qbit_max(self, cli_runner, mock_home):
        """Test that CLI qbit-max arg overrides config qbit_max."""
        # Create config with qbit_max=10
        config_dir = mock_home / ".apps-benchmark"
        config_dir.mkdir()
        config_file = config_dir / "apps-benchmark-config-test.json"
        config_data = {
            "version": "1.0",
            "backend": "mock_backend",
            "qbit_max": 10,
            "shots": 1000,
            "category": "chemistry",
            "case_uuid": None,
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Run with load-config but provide explicit qbit-max
        result = cli_runner.invoke(main, ["run", "--load-config=test", "--qbit-max=20"])

        # Should use CLI-provided qbit-max (20), not config qbit_max (10)
        assert result.exit_code == 0
        assert "Loading configuration 'test'" in result.output

    def test_config_overrides_defaults(self, cli_runner, mock_home):
        """Test that loaded config overrides default values."""
        # Create config with custom qbit_max and shots
        config_dir = mock_home / ".apps-benchmark"
        config_dir.mkdir()
        config_file = config_dir / "apps-benchmark-config-custom.json"
        config_data = {
            "version": "1.0",
            "backend": "mock_backend",
            "qbit_max": 15,  # Default is 10
            "shots": 3000,  # Default is 1000
            "category": "chemistry",
            "case_uuid": None,
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Run with just load-config, no explicit args
        result = cli_runner.invoke(main, ["run", "--load-config=custom"])

        # Should use config values, not defaults
        assert result.exit_code == 0
        assert "Loading configuration 'custom'" in result.output

    def test_full_precedence_chain(self, cli_runner, mock_home):
        """Test full precedence: CLI > config > defaults."""
        # Create config with some custom values
        config_dir = mock_home / ".apps-benchmark"
        config_dir.mkdir()
        config_file = config_dir / "apps-benchmark-config-chain.json"
        config_data = {
            "version": "1.0",
            "backend": "mock_backend",
            "qbit_max": 15,  # Config sets this
            "shots": 1000,  # Config uses default
            "category": "chemistry",
            "case_uuid": None,
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Run with load-config and explicit shots (overrides default)
        result = cli_runner.invoke(main, ["run", "--load-config=chain", "--shots=5000"])

        # qbit_max=15 from config (overrides default of 10)
        # shots=5000 from CLI (overrides config's 1000)
        # backend=mock_backend from config
        # category=chemistry from config
        assert result.exit_code == 0
        assert "Loading configuration 'chain'" in result.output

    def test_partial_config(self, cli_runner, mock_home):
        """Test config with only some values set."""
        # Create config with only backend and category
        config_dir = mock_home / ".apps-benchmark"
        config_dir.mkdir()
        config_file = config_dir / "apps-benchmark-config-partial.json"
        config_data = {
            "version": "1.0",
            "backend": "mock_backend",
            "qbit_max": 10,  # Use default
            "shots": 1000,  # Use default
            "category": "chemistry",
            "case_uuid": None,
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Run with just load-config
        result = cli_runner.invoke(main, ["run", "--load-config=partial"])

        # Should use config backend and category, defaults for others
        assert result.exit_code == 0
        assert "Loading configuration 'partial'" in result.output

    def test_cli_category_overrides_config_category(self, cli_runner, mock_home):
        """Test that CLI category overrides config category."""
        # Create config with category=chemistry
        config_dir = mock_home / ".apps-benchmark"
        config_dir.mkdir()
        config_file = config_dir / "apps-benchmark-config-cat.json"
        config_data = {
            "version": "1.0",
            "backend": "mock_backend",
            "qbit_max": 10,
            "shots": 1000,
            "category": "should_be_overridden",
            "case_uuid": None,
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Run with load-config but provide explicit category
        result = cli_runner.invoke(main, ["run", "--load-config=cat", "--category=chemistry"])

        # Should use CLI-provided category
        assert result.exit_code == 0
        assert "Loading configuration 'cat'" in result.output
        assert "Running benchmarks in category 'chemistry'" in result.output

    def test_cli_case_uuid_overrides_config(self, cli_runner, mock_home):
        """Test that CLI case-uuid overrides config category."""
        # Create config with category
        config_dir = mock_home / ".apps-benchmark"
        config_dir.mkdir()
        config_file = config_dir / "apps-benchmark-config-uuid.json"
        config_data = {
            "version": "1.0",
            "backend": "mock_backend",
            "qbit_max": 10,
            "shots": 1000,
            "category": "chemistry",
            "case_uuid": None,
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Run with load-config but provide explicit case-uuid
        result = cli_runner.invoke(main, ["run", "--load-config=uuid", "--case-uuid=610cfb55"])

        # Should use CLI-provided case-uuid (and ignore category)
        assert result.exit_code == 0
        assert "Loading configuration 'uuid'" in result.output

    def test_no_config_uses_defaults(self, cli_runner, mock_home):
        """Test that defaults are used when no config is loaded."""
        # Run without load-config
        result = cli_runner.invoke(main, ["run", "--backend=mock_backend", "--category=chemistry"])

        # Should use default qbit_max=10 and shots=1000
        assert result.exit_code == 0
        assert "Running benchmarks in category 'chemistry'" in result.output
