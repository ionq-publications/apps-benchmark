"""
Tests for configuration utilities.

This module tests the config.py utilities for managing local_dev configuration.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import json
from pathlib import Path

import pytest
from apps_benchmark.errors import ConfigError, ConfigValidationError
from apps_benchmark.utils.config import (
    get_config_file_path,
    get_local_dev_dir_from_config,
    load_local_dev_config,
    save_local_dev_config,
)


class TestConfigFilePath:
    """Tests for config file path."""

    def test_get_config_file_path_returns_path(self):
        """Test that config file path is returned."""
        path = get_config_file_path()
        assert isinstance(path, Path)
        assert path.name == "config_local_dev.json"
        assert path.parent == Path.home()


class TestLoadLocalDevConfig:
    """Tests for loading local_dev configuration."""

    def test_load_creates_default_config_if_not_exists(self, tmp_path, monkeypatch):
        """Test that default config is created if file doesn't exist."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        config = load_local_dev_config()

        # Should create the file
        assert config_file.exists()

        # Should have default values
        assert "local_dev_dir" in config
        assert "version" in config
        assert config["version"] == "1.0"
        assert "local_dev" in config["local_dev_dir"]

    def test_load_reads_existing_config(self, tmp_path, monkeypatch):
        """Test that existing config is loaded."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        # Create config file
        test_config = {
            "local_dev_dir": "/custom/path/to/local_dev",
            "version": "1.0",
        }
        with open(config_file, "w") as f:
            json.dump(test_config, f)

        # Load config
        config = load_local_dev_config()

        assert config["local_dev_dir"] == "/custom/path/to/local_dev"
        assert config["version"] == "1.0"

    def test_load_raises_error_on_invalid_json(self, tmp_path, monkeypatch):
        """Test that invalid JSON raises error."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        # Create invalid JSON file
        with open(config_file, "w") as f:
            f.write("{ invalid json }")

        with pytest.raises(ConfigValidationError, match="not valid JSON"):
            load_local_dev_config()

    def test_load_raises_error_on_missing_key(self, tmp_path, monkeypatch):
        """Test that missing local_dev_dir key raises error."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        # Create config without required key
        with open(config_file, "w") as f:
            json.dump({"version": "1.0"}, f)

        with pytest.raises(ConfigValidationError, match="missing required key"):
            load_local_dev_config()


class TestSaveLocalDevConfig:
    """Tests for saving local_dev configuration."""

    def test_save_creates_config_file(self, tmp_path, monkeypatch):
        """Test that save creates config file."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        save_local_dev_config("/custom/path")

        assert config_file.exists()

        with open(config_file) as f:
            config = json.load(f)

        assert config["local_dev_dir"] == "/custom/path"
        assert config["version"] == "1.0"

    def test_save_overwrites_existing_config(self, tmp_path, monkeypatch):
        """Test that save overwrites existing config."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        # Create initial config
        save_local_dev_config("/path/one")

        # Overwrite with new config
        save_local_dev_config("/path/two")

        with open(config_file) as f:
            config = json.load(f)

        assert config["local_dev_dir"] == "/path/two"


class TestGetLocalDevDirFromConfig:
    """Tests for getting local_dev dir from config."""

    def test_get_returns_default_path_on_first_call(self, tmp_path, monkeypatch):
        """Test that default path is returned on first call."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        path = get_local_dev_dir_from_config()

        assert isinstance(path, Path)
        assert "local_dev" in str(path)

    def test_get_returns_configured_path(self, tmp_path, monkeypatch):
        """Test that configured path is returned."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        # Save custom path
        custom_path = tmp_path / "my_custom_local_dev"
        save_local_dev_config(str(custom_path))

        # Get path from config
        path = get_local_dev_dir_from_config()

        assert path == custom_path

    def test_get_expands_user_paths(self, tmp_path, monkeypatch):
        """Test that ~ is expanded in paths."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        # Save path with ~
        save_local_dev_config("~/my_local_dev")

        # Get path (should be expanded)
        path = get_local_dev_dir_from_config()

        assert "~" not in str(path)
        assert path.is_absolute()

    def test_get_resolves_relative_paths(self, tmp_path, monkeypatch):
        """Test that relative paths are resolved."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        # Save relative path
        save_local_dev_config("./local_dev")

        # Get path (should be absolute)
        path = get_local_dev_dir_from_config()

        assert path.is_absolute()


class TestConfigIntegration:
    """Integration tests for config system."""

    def test_full_workflow(self, tmp_path, monkeypatch):
        """Test full workflow: load defaults, save new config, reload."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        # 1. First load creates default
        path1 = get_local_dev_dir_from_config()
        assert "local_dev" in str(path1)

        # 2. Save custom path
        custom_path = tmp_path / "custom_location"
        save_local_dev_config(str(custom_path))

        # 3. Reload and verify
        path2 = get_local_dev_dir_from_config()
        assert path2 == custom_path

        # 4. Save another path
        another_path = tmp_path / "another_location"
        save_local_dev_config(str(another_path))

        # 5. Verify final state
        path3 = get_local_dev_dir_from_config()
        assert path3 == another_path

    def test_config_persists_across_loads(self, tmp_path, monkeypatch):
        """Test that config persists across multiple loads."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        # Set custom path
        custom_path = tmp_path / "persistent_location"
        save_local_dev_config(str(custom_path))

        # Load multiple times
        path1 = get_local_dev_dir_from_config()
        path2 = get_local_dev_dir_from_config()
        path3 = get_local_dev_dir_from_config()

        # All should be the same
        assert path1 == path2 == path3 == custom_path


class TestConfigErrorHandling:
    """Tests for error handling in config system."""

    def test_handles_permission_error_on_create(self, tmp_path, monkeypatch):
        """Test handling of permission errors when creating config."""
        # Create a read-only directory
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        read_only_dir.chmod(0o444)

        config_file = read_only_dir / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        try:
            # Permission errors can happen at various points (exists check or write)
            with pytest.raises((ConfigError, PermissionError)):
                load_local_dev_config()
        finally:
            # Cleanup: restore permissions
            read_only_dir.chmod(0o755)

    def test_empty_config_file_raises_error(self, tmp_path, monkeypatch):
        """Test that empty config file raises error."""
        config_file = tmp_path / "config_local_dev.json"
        monkeypatch.setattr("apps_benchmark.utils.config.get_config_file_path", lambda: config_file)

        # Create empty file
        config_file.touch()

        with pytest.raises(ConfigValidationError):
            load_local_dev_config()
