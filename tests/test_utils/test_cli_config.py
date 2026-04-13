"""
Tests for CLI configuration utilities.

This module tests the cli_config.py utilities for managing saved CLI configurations.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import json
from pathlib import Path

import pytest
from apps_benchmark.errors import ConfigNotFoundError, ConfigValidationError
from apps_benchmark.utils.cli_config import (
    delete_cli_config,
    get_cli_config_dir,
    get_cli_config_path,
    list_saved_configs,
    load_cli_config,
    save_cli_config,
)


class TestGetCliConfigDir:
    """Tests for CLI config directory."""

    def test_get_cli_config_dir_returns_path(self):
        """Test that config directory path is returned."""
        path = get_cli_config_dir()
        assert isinstance(path, Path)
        assert path.name == ".apps-benchmark"
        # Should be in user's home directory
        assert Path.home() in path.parents or path.parent == Path.home()


class TestGetCliConfigPath:
    """Tests for CLI config file path."""

    def test_get_cli_config_path_with_name(self, tmp_path, monkeypatch):
        """Test that config file path is generated correctly."""
        monkeypatch.setattr("apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: tmp_path)

        path = get_cli_config_path("test_config")
        assert isinstance(path, Path)
        assert path.name == "apps-benchmark-config-test_config.json"
        assert path.parent == tmp_path


class TestSaveCliConfig:
    """Tests for saving CLI configurations."""

    def test_save_creates_directory_if_not_exists(self, tmp_path, monkeypatch):
        """Test that config directory is created if it doesn't exist."""
        config_dir = tmp_path / ".apps-benchmark"
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        assert not config_dir.exists()

        config_data = {
            "version": "1.0",
            "backend": "mock_backend",
            "qbit_max": 10,
        }
        save_cli_config("test", config_data)

        # Directory should be created
        assert config_dir.exists()
        assert config_dir.is_dir()

    def test_save_writes_config_file(self, tmp_path, monkeypatch):
        """Test that config is written to file."""
        config_dir = tmp_path / ".apps-benchmark"
        config_dir.mkdir()
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        config_data = {
            "version": "1.0",
            "backend": "qiskit_aer_sim_backend",
            "qbit_max": 15,
            "shots": 2000,
        }
        save_cli_config("production", config_data)

        # File should exist
        config_file = config_dir / "apps-benchmark-config-production.json"
        assert config_file.exists()

        # Content should match
        with open(config_file) as f:
            loaded = json.load(f)
        assert loaded == config_data

    def test_save_without_version_raises_error(self, tmp_path, monkeypatch):
        """Test that config without version field raises error."""
        config_dir = tmp_path / ".apps-benchmark"
        config_dir.mkdir()
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        config_data = {
            "backend": "mock_backend",  # Missing version
        }

        with pytest.raises(ConfigValidationError, match="must have 'version'"):
            save_cli_config("test", config_data)

    def test_save_overwrites_existing_config(self, tmp_path, monkeypatch):
        """Test that saving overwrites existing config."""
        config_dir = tmp_path / ".apps-benchmark"
        config_dir.mkdir()
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        # Save initial config
        config1 = {"version": "1.0", "backend": "backend1"}
        save_cli_config("test", config1)

        # Save updated config
        config2 = {"version": "1.0", "backend": "backend2", "shots": 5000}
        save_cli_config("test", config2)

        # Should have new content
        config_file = config_dir / "apps-benchmark-config-test.json"
        with open(config_file) as f:
            loaded = json.load(f)
        assert loaded == config2


class TestLoadCliConfig:
    """Tests for loading CLI configurations."""

    def test_load_reads_config_file(self, tmp_path, monkeypatch):
        """Test that config is loaded from file."""
        config_dir = tmp_path / ".apps-benchmark"
        config_dir.mkdir()
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        # Create config file
        config_data = {
            "version": "1.0",
            "backend": "ionq.forte",
            "qbit_max": 11,
            "shots": 3000,
        }
        config_file = config_dir / "apps-benchmark-config-prod.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        # Load config
        loaded = load_cli_config("prod")
        assert loaded == config_data

    def test_load_missing_config_raises_error(self, tmp_path, monkeypatch):
        """Test that loading missing config raises error."""
        config_dir = tmp_path / ".apps-benchmark"
        config_dir.mkdir()
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        with pytest.raises(ConfigNotFoundError, match="not found"):
            load_cli_config("nonexistent")

    def test_load_corrupted_config_raises_error(self, tmp_path, monkeypatch):
        """Test that corrupted JSON raises error."""
        config_dir = tmp_path / ".apps-benchmark"
        config_dir.mkdir()
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        # Create invalid JSON file
        config_file = config_dir / "apps-benchmark-config-bad.json"
        with open(config_file, "w") as f:
            f.write("{invalid json")

        with pytest.raises(ConfigValidationError, match="corrupted"):
            load_cli_config("bad")

    def test_load_wrong_version_raises_error(self, tmp_path, monkeypatch):
        """Test that wrong version raises error."""
        config_dir = tmp_path / ".apps-benchmark"
        config_dir.mkdir()
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        # Create config with wrong version
        config_data = {"version": "2.0", "backend": "mock_backend"}
        config_file = config_dir / "apps-benchmark-config-future.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ConfigValidationError, match="unsupported version"):
            load_cli_config("future")


class TestListSavedConfigs:
    """Tests for listing saved configurations."""

    def test_list_empty_directory(self, tmp_path, monkeypatch):
        """Test that empty directory returns empty list."""
        config_dir = tmp_path / ".apps-benchmark"
        # Don't create directory
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        configs = list_saved_configs()
        assert configs == []

    def test_list_returns_config_names(self, tmp_path, monkeypatch):
        """Test that config names are returned."""
        config_dir = tmp_path / ".apps-benchmark"
        config_dir.mkdir()
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        # Create some config files
        for name in ["prod", "dev", "test"]:
            config_file = config_dir / f"apps-benchmark-config-{name}.json"
            config_file.write_text('{"version": "1.0"}')

        configs = list_saved_configs()
        assert sorted(configs) == ["dev", "prod", "test"]

    def test_list_ignores_non_config_files(self, tmp_path, monkeypatch):
        """Test that non-config files are ignored."""
        config_dir = tmp_path / ".apps-benchmark"
        config_dir.mkdir()
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        # Create config file
        (config_dir / "apps-benchmark-config-valid.json").write_text('{"version": "1.0"}')

        # Create non-config files
        (config_dir / "other-file.json").write_text("{}")
        (config_dir / "README.txt").write_text("readme")

        configs = list_saved_configs()
        assert configs == ["valid"]


class TestDeleteCliConfig:
    """Tests for deleting CLI configurations."""

    def test_delete_removes_config_file(self, tmp_path, monkeypatch):
        """Test that config file is deleted."""
        config_dir = tmp_path / ".apps-benchmark"
        config_dir.mkdir()
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        # Create config file
        config_file = config_dir / "apps-benchmark-config-todel.json"
        config_file.write_text('{"version": "1.0"}')
        assert config_file.exists()

        # Delete config
        delete_cli_config("todel")
        assert not config_file.exists()

    def test_delete_missing_config_raises_error(self, tmp_path, monkeypatch):
        """Test that deleting missing config raises error."""
        config_dir = tmp_path / ".apps-benchmark"
        config_dir.mkdir()
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        with pytest.raises(ConfigNotFoundError, match="not found"):
            delete_cli_config("nonexistent")


class TestSaveLoadRoundTrip:
    """Tests for save/load round trip."""

    def test_save_and_load_preserves_data(self, tmp_path, monkeypatch):
        """Test that saving and loading preserves data."""
        config_dir = tmp_path / ".apps-benchmark"
        config_dir.mkdir()
        monkeypatch.setattr(
            "apps_benchmark.utils.cli_config.get_cli_config_dir", lambda: config_dir
        )

        # Save config
        config_data = {
            "version": "1.0",
            "backend": "qiskit_aer_sim_backend",
            "qbit_max": 20,
            "shots": 10000,
            "category": "chemistry",
            "case_uuid": None,
        }
        save_cli_config("roundtrip", config_data)

        # Load config
        loaded = load_cli_config("roundtrip")

        # Should match exactly
        assert loaded == config_data
