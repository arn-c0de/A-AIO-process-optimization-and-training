"""Component profile hashing and loading utilities.

This module provides functions for:
- Loading component profiles from YAML files
- Canonicalizing profiles for stable hashing
- Computing SHA256 hashes of profiles for validation
"""

from __future__ import annotations

import hashlib
import yaml
from pathlib import Path
from typing import Any
from simple_sim.io_utils import load_yaml


def canonicalize_profile(profile_dict: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize profile for stable hashing.

    Rules:
    - Remove metadata fields (created_at, description) from profile section
    - Sort all dict keys recursively
    - Normalize float precision to 10 decimals
    - Preserve list order (important for defect_set)

    Args:
        profile_dict: Raw profile dictionary from YAML

    Returns:
        Canonicalized profile dictionary
    """
    def _canonicalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            # Sort keys and recursively canonicalize values
            result = {}
            for key in sorted(obj.keys()):
                # Skip metadata fields in profile section
                if key in ('created_at', 'description'):
                    continue
                result[key] = _canonicalize(obj[key])
            return result
        elif isinstance(obj, list):
            # Preserve list order but canonicalize elements
            return [_canonicalize(item) for item in obj]
        elif isinstance(obj, float):
            # Normalize float precision
            return round(obj, 10)
        else:
            return obj

    return _canonicalize(profile_dict)


def hash_profile(profile_path: Path) -> str:
    """Compute SHA256 hash of component profile.

    The hash is computed on the canonical form of the profile,
    making it stable across formatting changes but sensitive to
    semantic changes in geometry, tolerances, etc.

    Args:
        profile_path: Path to profile YAML file

    Returns:
        Hash string in format "sha256:hexdigest"

    Raises:
        FileNotFoundError: If profile file doesn't exist
        yaml.YAMLError: If profile YAML is invalid
    """
    profile_path = Path(profile_path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    profile_dict = load_yaml(profile_path)

    # Canonicalize: remove metadata, sort keys, normalize floats
    canonical = canonicalize_profile(profile_dict)

    # Convert to deterministic YAML string
    canonical_yaml = yaml.dump(canonical, sort_keys=True, default_flow_style=False)

    # Compute SHA256 hash
    hash_obj = hashlib.sha256(canonical_yaml.encode('utf-8'))
    return f"sha256:{hash_obj.hexdigest()}"


def load_profile(profile_id: str, profiles_dir: Path) -> dict[str, Any]:
    """Load component profile from YAML file.

    Args:
        profile_id: Profile identifier (e.g., "chip_0603_resistor@1")
        profiles_dir: Directory containing profile YAML files

    Returns:
        Profile dictionary with all sections

    Raises:
        FileNotFoundError: If profile file doesn't exist
        yaml.YAMLError: If profile YAML is invalid
        ValueError: If profile_id in file doesn't match filename
    """
    profile_path = Path(profiles_dir) / f"{profile_id}.yaml"

    if not profile_path.exists():
        raise FileNotFoundError(
            f"Profile '{profile_id}' not found at: {profile_path}\n"
            f"Available profiles: {list(profiles_dir.glob('*.yaml'))}"
        )

    profile_dict = load_yaml(profile_path)

    # Validate profile structure
    if 'profile' not in profile_dict:
        raise ValueError(f"Invalid profile: missing 'profile' section in {profile_path}")

    if 'component' not in profile_dict:
        raise ValueError(f"Invalid profile: missing 'component' section in {profile_path}")

    # Validate profile_id matches filename
    file_profile_id = profile_dict['profile'].get('profile_id')
    if file_profile_id != profile_id:
        raise ValueError(
            f"Profile ID mismatch:\n"
            f"  Filename:  {profile_id}\n"
            f"  File says: {file_profile_id}\n"
            f"Update profile_id in {profile_path}"
        )

    return profile_dict
