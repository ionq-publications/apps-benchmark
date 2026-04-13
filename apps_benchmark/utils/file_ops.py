"""
File operations utilities for apps-benchmark.

This module provides atomic file writing and registry loading utilities.

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

from apps_benchmark.errors import RegistryError


def snake_to_camel(snake_str: str) -> str:
    """
    Convert snake_case to CamelCase.

    Args:
        snake_str: String in snake_case

    Returns:
        String in CamelCase
    """
    components = snake_str.split("_")
    return "".join(x.title() for x in components)


def write_registry_atomic(registry_path: Path, registry_data: dict[str, Any]) -> None:
    """
    Write registry file atomically (temp file + rename).

    This prevents corruption by writing to a temporary file first,
    then atomically renaming it to the target path.

    Args:
        registry_path: Path to registry file
        registry_data: Registry data to write

    Raises:
        RegistryError: If write fails
    """
    # Write to temporary file first
    temp_fd, temp_path = tempfile.mkstemp(
        dir=registry_path.parent, prefix=".registry_", suffix=".tmp"
    )

    try:
        with os.fdopen(temp_fd, "w") as f:
            json.dump(registry_data, f, indent=2)

        # Atomic rename
        temp_path_obj = Path(temp_path)
        temp_path_obj.replace(registry_path)

    except Exception as exc:
        # Clean up temp file on error
        Path(temp_path).unlink(missing_ok=True)
        raise RegistryError(f"Failed to write registry: {exc}") from exc


def load_registry(registry_path: Path) -> dict[str, Any]:
    """
    Load registry from JSON file.

    Args:
        registry_path: Path to registry file

    Returns:
        Registry data dict

    Raises:
        RegistryError: If file is corrupted or invalid
    """
    try:
        with open(registry_path) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise RegistryError(f"Registry file {registry_path} must contain a JSON object")

        # Validate version
        if data.get("version") != "1.0":
            raise RegistryError(f"Unsupported registry version: {data.get('version')}")

        return cast(dict[str, Any], data)

    except FileNotFoundError:
        # Return empty registry structure
        if "backends" in registry_path.name:
            return {"version": "1.0", "backends": {}}
        else:
            return {
                "version": "1.0",
                "builtin_benchmarks": {},
                "diy_benchmarks": {},
            }

    except json.JSONDecodeError as exc:
        raise RegistryError(f"Corrupted registry file {registry_path}: {exc}") from exc
