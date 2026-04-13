"""
CLI configuration management for apps-benchmark.

This module handles saving and loading command-line configurations,
allowing users to persist their commonly-used CLI arguments.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

from apps_benchmark.errors import ConfigNotFoundError, ConfigValidationError


def get_cli_config_dir() -> Path:
    """
    Get the directory for CLI configuration files.

    Returns:
        Path: ~/.apps-benchmark/ directory (expanded and resolved)
    """
    config_dir = Path.home() / ".apps-benchmark"
    return config_dir.expanduser().resolve()


def get_cli_config_path(config_name: str) -> Path:
    """
    Get the path for a named CLI configuration file.

    Args:
        config_name: Name of the configuration

    Returns:
        Path: Full path to config file (~/.apps-benchmark/apps-benchmark-config-{name}.json)
    """
    config_dir = get_cli_config_dir()
    config_file = config_dir / f"apps-benchmark-config-{config_name}.json"
    return config_file


def save_cli_config(config_name: str, config_dict: dict[str, Any]) -> None:
    """
    Save CLI configuration to a named file atomically.

    Uses temp file + rename pattern to prevent corruption.

    Args:
        config_name: Name of the configuration
        config_dict: Configuration dictionary to save

    Raises:
        ConfigValidationError: If config_dict is invalid
    """
    # Ensure config directory exists
    config_dir = get_cli_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = get_cli_config_path(config_name)

    # Validate config has version field
    if "version" not in config_dict:
        raise ConfigValidationError("Config must have 'version' field")

    # Write to temporary file first
    temp_fd, temp_path = tempfile.mkstemp(dir=config_dir, prefix=".config_", suffix=".tmp")

    try:
        with os.fdopen(temp_fd, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Atomic rename
        temp_path_obj = Path(temp_path)
        temp_path_obj.replace(config_path)

    except Exception as exc:
        # Clean up temp file on error
        Path(temp_path).unlink(missing_ok=True)
        raise ConfigValidationError(f"Failed to save config '{config_name}': {exc}") from exc


def load_cli_config(config_name: str) -> dict[str, Any]:
    """
    Load CLI configuration from a named file.

    Args:
        config_name: Name of the configuration

    Returns:
        dict: Configuration dictionary

    Raises:
        ConfigNotFoundError: If config file doesn't exist
        ConfigValidationError: If config file is corrupted or invalid
    """
    config_path = get_cli_config_path(config_name)

    if not config_path.exists():
        raise ConfigNotFoundError(f"Configuration '{config_name}' not found at {config_path}")

    try:
        with open(config_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"Configuration '{config_name}' is corrupted: {exc}") from exc
    except Exception as exc:
        raise ConfigValidationError(f"Failed to load configuration '{config_name}': {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigValidationError(f"Configuration '{config_name}' must contain a JSON object")

    # Validate version
    if data.get("version") != "1.0":
        raise ConfigValidationError(
            f"Configuration '{config_name}' has unsupported version: {data.get('version')}"
        )

    return cast(dict[str, Any], data)


def list_saved_configs() -> list[str]:
    """
    List all saved CLI configurations.

    Returns:
        list[str]: List of configuration names (without path or extension)
    """
    config_dir = get_cli_config_dir()

    if not config_dir.exists():
        return []

    config_files = config_dir.glob("apps-benchmark-config-*.json")
    config_names = []

    for config_file in config_files:
        # Extract name from "apps-benchmark-config-{name}.json"
        name = config_file.stem.replace("apps-benchmark-config-", "")
        config_names.append(name)

    return sorted(config_names)


def delete_cli_config(config_name: str) -> None:
    """
    Delete a named CLI configuration.

    Args:
        config_name: Name of the configuration to delete

    Raises:
        ConfigNotFoundError: If config file doesn't exist
    """
    config_path = get_cli_config_path(config_name)

    if not config_path.exists():
        raise ConfigNotFoundError(f"Configuration '{config_name}' not found at {config_path}")

    try:
        config_path.unlink()
    except Exception as exc:
        raise ConfigValidationError(
            f"Failed to delete configuration '{config_name}': {exc}"
        ) from exc
