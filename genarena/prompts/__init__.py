"""Prompt module loader and validator."""

import importlib
import importlib.util
import os
from types import ModuleType
from typing import Optional


# Required attributes for a valid prompt module
REQUIRED_ATTRIBUTES = ["PROMPT_TEXT", "ALLOW_TIE", "build_prompt", "parse_response"]


def load_prompt(name: str) -> ModuleType:
    """
    Load a prompt module by name.

    First tries to load from the genarena.prompts package, then attempts
    to load from a file path if the name looks like a path.

    Args:
        name: Prompt module name (e.g., 'mmrb2') or path to a .py file

    Returns:
        Loaded module

    Raises:
        ImportError: If module cannot be found
        ValueError: If module is invalid
    """
    module = None

    # Try loading from genarena.prompts package
    try:
        module = importlib.import_module(f"genarena.prompts.{name}")
    except ImportError:
        pass

    # If not found and name looks like a path, try loading from file
    if module is None and (name.endswith('.py') or os.path.sep in name):
        if os.path.isfile(name):
            spec = importlib.util.spec_from_file_location("custom_prompt", name)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

    if module is None:
        raise ImportError(
            f"Could not load prompt module '{name}'. "
            f"Make sure it exists in genarena/prompts/ or provide a valid file path."
        )

    # Validate the module
    if not validate_prompt(module):
        missing = get_missing_attributes(module)
        raise ValueError(
            f"Invalid prompt module '{name}'. "
            f"Missing required attributes: {missing}"
        )

    return module


def validate_prompt(module: ModuleType) -> bool:
    """
    Validate that a module contains all required prompt attributes.

    Required attributes:
    - PROMPT_TEXT: str - The evaluation prompt text
    - ALLOW_TIE: bool - Whether single-round ties are allowed
    - build_prompt: callable - Function to build VLM messages
    - parse_response: callable - Function to parse VLM response

    Args:
        module: Module to validate

    Returns:
        True if valid, False otherwise
    """
    for attr in REQUIRED_ATTRIBUTES:
        if not hasattr(module, attr):
            return False

        # Check callable attributes
        if attr in ("build_prompt", "parse_response"):
            if not callable(getattr(module, attr)):
                return False

    return True


def get_missing_attributes(module: ModuleType) -> list[str]:
    """
    Get list of missing required attributes from a module.

    Args:
        module: Module to check

    Returns:
        List of missing attribute names
    """
    missing = []
    for attr in REQUIRED_ATTRIBUTES:
        if not hasattr(module, attr):
            missing.append(attr)
        elif attr in ("build_prompt", "parse_response"):
            if not callable(getattr(module, attr)):
                missing.append(f"{attr} (not callable)")
    return missing


def list_available_prompts() -> list[str]:
    """
    List all available prompt modules in the prompts directory.

    Returns:
        List of prompt module names
    """
    prompts_dir = os.path.dirname(__file__)
    available = []

    for filename in os.listdir(prompts_dir):
        if filename.endswith('.py') and not filename.startswith('_'):
            name = filename[:-3]  # Remove .py extension
            available.append(name)

    return sorted(available)
