"""
Configuration utilities for apps-benchmark.

This module handles configuration file management, including the local_dev
directory location configuration.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import json
from pathlib import Path
from typing import Any, cast

from apps_benchmark.errors import ConfigError, ConfigValidationError


def get_config_file_path() -> Path:
    """
    Get the path to the local_dev configuration file.

    Returns:
        Path to ~/config_local_dev.json
    """
    return Path.home() / "config_local_dev.json"


def load_local_dev_config() -> dict[str, Any]:
    """
    Load local_dev configuration from config file.

    If the config file doesn't exist, creates it with default values.

    Returns:
        dict: Configuration dictionary with 'local_dev_dir' key

    Raises:
        ConfigError: If config file is malformed
    """
    config_file = get_config_file_path()

    # Create config file with defaults if it doesn't exist
    if not config_file.exists():
        default_config = {
            "local_dev_dir": str(Path.home() / "local_dev"),
            "version": "1.0",
        }
        try:
            with open(config_file, "w") as f:
                json.dump(default_config, f, indent=2)
        except Exception as exc:
            raise ConfigError(f"Failed to create config file {config_file}: {exc}") from exc

        return default_config

    # Load existing config file
    try:
        with open(config_file) as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"Config file {config_file} is not valid JSON: {exc}") from exc
    except Exception as exc:
        raise ConfigError(f"Failed to read config file {config_file}: {exc}") from exc

    # Validate config structure
    if not isinstance(config, dict):
        raise ConfigValidationError(f"Config file {config_file} must contain a JSON object")

    if "local_dev_dir" not in config:
        raise ConfigValidationError(
            f"Config file {config_file} missing required key 'local_dev_dir'"
        )

    return cast(dict[str, Any], config)


def save_local_dev_config(local_dev_dir: str) -> None:
    """
    Save local_dev directory location to config file.

    Args:
        local_dev_dir: Path to local_dev directory

    Raises:
        ConfigError: If unable to save config file
    """
    config_file = get_config_file_path()

    config = {
        "local_dev_dir": str(local_dev_dir),
        "version": "1.0",
    }

    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as exc:
        raise ConfigError(f"Failed to save config file {config_file}: {exc}") from exc


def get_local_dev_dir_from_config() -> Path:
    """
    Get local_dev directory path from configuration.

    Loads the configuration file (creating it with defaults if needed)
    and returns the configured local_dev directory path.

    To change the local_dev location, edit ~/config_local_dev.json directly.

    Returns:
        Path: Path to local_dev directory

    Raises:
        ConfigError: If config cannot be loaded
    """
    config = load_local_dev_config()
    return Path(config["local_dev_dir"]).expanduser().resolve()
