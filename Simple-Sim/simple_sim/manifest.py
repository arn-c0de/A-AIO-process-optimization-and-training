"""Dataset manifest management.

This module provides functions for:
- Writing dataset manifests with component profile metadata
- Reading and validating dataset manifests
- Computing file hashes for integrity checking
"""

import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from simple_sim import __version__ as _VERSION


def hash_file(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to file to hash

    Returns:
        Hash string in format "sha256:hexdigest"

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hash_obj = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks for large files
        while chunk := f.read(8192):
            hash_obj.update(chunk)

    return f"sha256:{hash_obj.hexdigest()}"


def _get_git_commit() -> str | None:
    """Get current git commit hash (short).

    Returns:
        Short commit hash or None if not in git repo or error occurs
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _compute_class_counts(label_rows: list[Any]) -> dict[str, int]:
    """Return class_name → count mapping for a list of LabelRow instances."""
    counts: dict[str, int] = {}
    for row in label_rows:
        counts[row.class_name] = counts.get(row.class_name, 0) + 1
    return counts


def _atomic_json_write(path: Path, obj: Any) -> None:
    """Write JSON to *path* atomically via a .tmp rename."""
    temp_path = path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(obj, f, indent=2)
    temp_path.replace(path)


def write_dataset_manifest(
    output_dir: Path,
    run_id: str,
    profile_id: str,
    profile_hash: str,
    profile_path: str,
    meta_rows: list[Any],
    label_rows: list[Any],
    splits: dict[str, list[str]],
    extend: bool = False
) -> None:
    """Write or update dataset manifest.

    Args:
        output_dir: Dataset output directory
        run_id: Run identifier
        profile_id: Component profile ID
        profile_hash: SHA256 hash of profile
        profile_path: Relative path to profile YAML
        meta_rows: List of MetaRow instances
        label_rows: List of LabelRow instances
        splits: Dictionary mapping split names to sample IDs
        extend: If True, update existing manifest; if False, create new

    Raises:
        ValueError: If extend=True but manifest doesn't exist
        ValueError: If extend=True and profile mismatch detected
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / "dataset_manifest.json"

    git_commit = _get_git_commit()
    class_counts = _compute_class_counts(label_rows)
    split_counts = {k: len(v) for k, v in splits.items()}

    if extend and manifest_path.exists():
        # Update existing manifest
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Validate profile match (should have been caught earlier, but double-check)
        existing_profile_id = manifest['component_profile']['profile_id']
        existing_profile_hash = manifest['component_profile']['profile_hash']

        if existing_profile_id != profile_id:
            raise ValueError(
                f"Profile ID mismatch during manifest update:\n"
                f"  Existing: {existing_profile_id}\n"
                f"  Current:  {profile_id}"
            )

        if existing_profile_hash != profile_hash:
            raise ValueError(
                f"Profile hash mismatch during manifest update:\n"
                f"  Existing: {existing_profile_hash}\n"
                f"  Current:  {profile_hash}"
            )

        # Update extend history
        manifest['extend_history'].append({
            'timestamp': datetime.now().isoformat(),
            'samples_added': len(meta_rows),
            'git_commit': git_commit
        })

        # Update stats (add to existing counts)
        manifest['dataset_stats']['total_samples'] += len(meta_rows)

        # `splits` is the complete (already merged) split map, so overwrite instead of adding.
        manifest['dataset_stats']['splits'] = dict(split_counts)

        for class_name, count in class_counts.items():
            manifest['dataset_stats']['classes'][class_name] = \
                manifest['dataset_stats']['classes'].get(class_name, 0) + count

    else:
        # Create new manifest
        if extend:
            raise ValueError(
                f"Cannot extend: manifest not found at {manifest_path}\n"
                f"Remove --extend flag to create new dataset"
            )

        manifest = {
            'manifest_version': 1,
            'created_at': datetime.now().isoformat(),
            'run_id': run_id,

            'component_profile': {
                'profile_id': profile_id,
                'profile_hash': profile_hash,
                'profile_path': profile_path
            },

            'generator': {
                'version': _VERSION,
                'git_commit': git_commit,
                'script': 'scripts/generate.py'
            },

            'dataset_stats': {
                'total_samples': len(meta_rows),
                'splits': split_counts,
                'classes': class_counts
            },

            'extend_history': [{
                'timestamp': datetime.now().isoformat(),
                'samples_added': len(meta_rows),
                'git_commit': git_commit
            }]
        }

    _atomic_json_write(manifest_path, manifest)
    print(f"✓ Dataset manifest written: {manifest_path}")


def resolve_dataset_profile(manifest: dict[str, Any]) -> tuple[str, str]:
    """Return (profile_id, profile_hash) for a dataset manifest.

    v1 manifests carry a single ``component_profile``; v2 (multi-profile)
    manifests are mapped to the logical profile ``"multi"``.
    """
    mver = int(manifest.get('manifest_version', 1) or 1)
    if mver == 1:
        cp = manifest['component_profile']
        return cp['profile_id'], cp['profile_hash']
    if mver == 2:
        return "multi", "multi"
    raise ValueError(f"Unsupported dataset manifest_version: {mver}")


def read_dataset_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read and validate dataset manifest.

    Supports manifest_version 1 (single profile) and manifest_version 2 (multi-profile).

    Args:
        manifest_path: Path to dataset_manifest.json

    Returns:
        Manifest dictionary

    Raises:
        FileNotFoundError: If manifest doesn't exist
        ValueError: If manifest is invalid
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Validate manifest structure
    required_fields = [
        'manifest_version',
        'created_at',
        'run_id',
        'generator',
        'dataset_stats',
        'extend_history'
    ]

    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Invalid manifest: missing field '{field}'")

    version = manifest.get('manifest_version', 1)

    if version == 1:
        # Single-profile manifest
        if 'component_profile' not in manifest:
            raise ValueError("Invalid manifest: missing 'component_profile'")
        profile_fields = ['profile_id', 'profile_hash', 'profile_path']
        for field in profile_fields:
            if field not in manifest['component_profile']:
                raise ValueError(
                    f"Invalid manifest: missing component_profile.{field}"
                )
    elif version == 2:
        # Multi-profile manifest
        if 'component_profiles' not in manifest:
            raise ValueError("Invalid manifest: missing 'component_profiles'")
        profiles = manifest['component_profiles']
        if not isinstance(profiles, list) or len(profiles) < 2:
            raise ValueError("manifest_version=2 requires component_profiles list with 2+ entries")
        for i, p in enumerate(profiles):
            for field in ['profile_id', 'profile_hash', 'profile_path']:
                if field not in p:
                    raise ValueError(f"Invalid manifest: missing component_profiles[{i}].{field}")
    else:
        raise ValueError(f"Unsupported manifest_version: {version}")

    return manifest


def check_checkpoint_profile(
    checkpoint: dict[str, Any],
    dataset_profile_id: str,
    dataset_profile_hash: str,
    *,
    action: str,
    allow_mismatch: bool | None = None,
    override_verb: str = "Continuing",
    ckpt_label: str = "Model trained on",
) -> str | None:
    """Guard a checkpoint against being used on a dataset of another profile.

    Profile ID mismatch is a hard failure. ``allow_mismatch=None`` means the
    caller has no override flag; ``True``/``False`` reflect an
    ``--allow-profile-mismatch`` flag (True downgrades the failure to a warning).
    A profile hash mismatch always only warns.

    Args:
        checkpoint: Loaded checkpoint dict
        dataset_profile_id: Profile ID from resolve_dataset_profile()
        dataset_profile_hash: Profile hash from resolve_dataset_profile()
        action: Phrase for the error, e.g. "evaluate model on"
        allow_mismatch: Override flag value, or None if the caller has none
        override_verb: Leading verb of the override warning ("Resuming", ...)
        ckpt_label: Label for the checkpoint profile line

    Returns:
        The checkpoint profile ID, or None for legacy checkpoints without metadata.
    """
    ckpt_profile = checkpoint.get('component_profile')
    if not ckpt_profile:
        print("WARNING: Checkpoint lacks profile metadata (legacy checkpoint)")
        return None

    ckpt_profile_id = ckpt_profile.get('profile_id')
    ckpt_profile_hash = ckpt_profile.get('profile_hash')
    label = f"{ckpt_label}:"
    width = max(len(label), len("Dataset profile:")) + 1

    # HARD FAIL by default: Profile ID mismatch
    if ckpt_profile_id and ckpt_profile_id != dataset_profile_id:
        if not allow_mismatch:
            hint = (
                "\n\nTo override (NOT recommended): pass --allow-profile-mismatch"
                if allow_mismatch is not None else ""
            )
            raise ValueError(
                f"Component profile mismatch:\n"
                f"  {label:<{width}}{ckpt_profile_id}\n"
                f"  {'Dataset profile:':<{width}}{dataset_profile_id}\n"
                f"Cannot {action} different component type.{hint}"
            )
        print(f"WARNING: {override_verb} with component profile mismatch (--allow-profile-mismatch).")
        print(f"  {label:<{width}}{ckpt_profile_id}")
        print(f"  {'Dataset profile:':<{width}}{dataset_profile_id}")

    # WARNING: Profile hash mismatch
    if ckpt_profile_hash and ckpt_profile_hash != dataset_profile_hash:
        print("WARNING: Profile hash mismatch (different tolerance/geometry)")
        print(f"  Checkpoint hash: {ckpt_profile_hash[:72]}...")
        print(f"  Dataset hash:    {dataset_profile_hash[:72]}...")

    return ckpt_profile_id


def validate_manifest_profile(
    manifest: dict[str, Any],
    expected_profile_id: str,
    expected_profile_hash: str,
    *,
    strict_hash: bool = True
) -> None:
    """Validate that manifest profile matches expected values.

    Args:
        manifest: Manifest dictionary from read_dataset_manifest()
        expected_profile_id: Expected component profile ID
        expected_profile_hash: Expected profile hash
        strict_hash: If False, only warn on hash mismatch instead of raising

    Raises:
        ValueError: If profile ID mismatch or (if strict_hash) hash mismatch
    """
    manifest_profile_id = manifest['component_profile']['profile_id']
    manifest_profile_hash = manifest['component_profile']['profile_hash']

    # Profile ID must match
    if manifest_profile_id != expected_profile_id:
        raise ValueError(
            f"Component profile mismatch:\n"
            f"  Dataset profile: {manifest_profile_id}\n"
            f"  Expected profile: {expected_profile_id}\n"
            f"Cannot use dataset with different component type."
        )

    # Profile hash mismatch
    if manifest_profile_hash != expected_profile_hash:
        message = (
            f"Profile hash mismatch:\n"
            f"  Dataset hash: {manifest_profile_hash[:72]}...\n"
            f"  Expected hash: {expected_profile_hash[:72]}...\n"
            f"Profile '{expected_profile_id}' has changed since dataset creation."
        )
        if strict_hash:
            raise ValueError(message)
        else:
            print(f"WARNING: {message}")


def write_multi_profile_manifest(
    output_dir: Path,
    run_id: str,
    profiles: list[dict[str, str]],
    meta_rows: list[Any],
    label_rows: list[Any],
    splits: dict[str, list[str]],
    script_name: str = "scripts/generate_profile_dataset.py",
) -> None:
    """Write a multi-profile dataset manifest (manifest_version 2).

    Args:
        output_dir: Dataset output directory
        run_id: Run identifier
        profiles: List of dicts with keys: profile_id, profile_hash, profile_path
        meta_rows: List of MetaRow instances
        label_rows: List of LabelRow instances
        splits: Dictionary mapping split names to sample IDs
        script_name: Generator script name for metadata
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / "dataset_manifest.json"

    git_commit = _get_git_commit()
    class_counts = _compute_class_counts(label_rows)
    split_counts = {k: len(v) for k, v in splits.items()}

    # Profile sample counts
    profile_counts: dict[str, int] = {}
    for label_row in label_rows:
        pid = getattr(label_row, 'profile_id', '')
        if pid:
            profile_counts[pid] = profile_counts.get(pid, 0) + 1

    manifest = {
        'manifest_version': 2,
        'created_at': datetime.now().isoformat(),
        'run_id': run_id,

        'component_profiles': profiles,

        'generator': {
            'version': _VERSION,
            'git_commit': git_commit,
            'script': script_name,
        },

        'dataset_stats': {
            'total_samples': len(meta_rows),
            'splits': split_counts,
            'classes': class_counts,
            'profile_counts': profile_counts,
        },

        'extend_history': [{
            'timestamp': datetime.now().isoformat(),
            'samples_added': len(meta_rows),
            'git_commit': git_commit,
        }],
    }

    _atomic_json_write(manifest_path, manifest)
    print(f"✓ Multi-profile dataset manifest written: {manifest_path}")
