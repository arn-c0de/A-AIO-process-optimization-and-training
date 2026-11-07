"""Dataset manifest management.

This module provides functions for:
- Writing dataset manifests with component profile metadata
- Reading and validating dataset manifests
- Computing file hashes for integrity checking
"""

from __future__ import annotations

import logging
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from simple_sim import __version__ as _VERSION
from simple_sim.dataset_files import DATASET_MANIFEST
from simple_sim.io_utils import atomic_write_json, load_json, now_iso, sha256_tag


log = logging.getLogger(__name__)


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

    return sha256_tag(hash_obj)


@lru_cache(maxsize=1)
def _get_git_commit() -> str | None:
    """Get current git commit hash (short).

    Resolved once per process (plan S2: a manifest write used to spawn ``git rev-parse`` every
    time); a commit made while a long-lived process (GUI) runs is picked up after a restart or
    ``_get_git_commit.cache_clear()``. Writers also accept an explicit ``git_commit``.

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
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _compute_class_counts(label_rows: list[Any]) -> dict[str, int]:
    """Return class_name → count mapping for a list of LabelRow instances."""
    counts: dict[str, int] = {}
    for row in label_rows:
        counts[row.class_name] = counts.get(row.class_name, 0) + 1
    return counts


def dataset_stats(meta_rows: list[Any], label_rows: list[Any], splits: dict[str, list[str]]) -> dict[str, Any]:
    """``dataset_stats`` block of a manifest: total samples, split sizes and class counts."""
    return {
        'total_samples': len(meta_rows),
        'splits': {k: len(v) for k, v in splits.items()},
        'classes': _compute_class_counts(label_rows),
    }


def _history_entry(samples_added: int, git_commit: str | None, renderer_version: str | None) -> dict[str, Any]:
    """One ``extend_history`` entry: ``timestamp``, ``samples_added``, ``git_commit`` [, ``renderer_version``]."""
    entry: dict[str, Any] = {
        'timestamp': now_iso(),
        'samples_added': samples_added,
        'git_commit': git_commit,
    }
    if renderer_version is not None:
        entry['renderer_version'] = renderer_version
    return entry


def _generator_block(script_name: str, git_commit: str | None, renderer_version: str | None) -> dict[str, Any]:
    """The ``generator`` block: ``version``, ``git_commit``, ``script`` [, ``renderer_version``]."""
    generator: dict[str, Any] = {
        'version': _VERSION,
        'git_commit': git_commit,
        'script': script_name,
    }
    if renderer_version is not None:
        generator['renderer_version'] = renderer_version
    return generator


def write_dataset_manifest(
    output_dir: Path,
    run_id: str,
    profile_id: str,
    profile_hash: str,
    profile_path: str,
    meta_rows: list[Any],
    label_rows: list[Any],
    splits: dict[str, list[str]],
    extend: bool = False,
    script_name: str = "scripts/generate.py",
    renderer_version: str | None = None,
    git_commit: str | None = None,
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
        script_name: Generator script name for metadata (new manifests only)
        renderer_version: ``simple_sim.generators.version.RENDERER_VERSION`` of the images written
            (``generator.renderer_version``; also recorded per extend history entry). None for
            tools that do not render (merges).
        git_commit: commit recorded in ``generator``/history; default (``None``): the checkout's
            short HEAD, cached per process (:func:`_get_git_commit`).

    Raises:
        ValueError: If extend=True but manifest doesn't exist
        ValueError: If extend=True and profile mismatch detected
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / DATASET_MANIFEST

    git_commit = git_commit if git_commit is not None else _get_git_commit()
    if extend and manifest_path.exists():
        # Update existing manifest
        manifest = load_json(manifest_path)

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
        if renderer_version is not None:
            manifest.setdefault('generator', {})['renderer_version'] = renderer_version
        manifest['extend_history'].append(_history_entry(len(meta_rows), git_commit, renderer_version))

        # Update stats (add to existing counts)
        manifest['dataset_stats']['total_samples'] += len(meta_rows)

        # `splits` is the complete (already merged) split map, so overwrite instead of adding.
        manifest['dataset_stats']['splits'] = {k: len(v) for k, v in splits.items()}

        for class_name, count in _compute_class_counts(label_rows).items():
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
            'created_at': now_iso(),
            'run_id': run_id,

            'component_profile': {
                'profile_id': profile_id,
                'profile_hash': profile_hash,
                'profile_path': profile_path
            },

            'generator': _generator_block(script_name, git_commit, renderer_version),

            'dataset_stats': dataset_stats(meta_rows, label_rows, splits),

            'extend_history': [_history_entry(len(meta_rows), git_commit, renderer_version)]
        }

    atomic_write_json(manifest_path, manifest)
    log.info(f"Dataset manifest written: {manifest_path}")


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

    manifest = load_json(manifest_path)

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


class ProfileMismatchError(ValueError):
    """A checkpoint's component profile differs from the dataset's (see :func:`check_checkpoint_profile`)."""

    def __init__(self, message: str, *, checkpoint_profile: str, dataset_profile: str) -> None:
        super().__init__(message)
        self.checkpoint_profile = checkpoint_profile
        self.dataset_profile = dataset_profile


class RendererMismatchError(ValueError):
    """Extending a dataset whose images come from another renderer version (see :func:`check_renderer_version`)."""

    def __init__(self, message: str, *, dataset_version: str | None, current_version: str) -> None:
        super().__init__(message)
        self.dataset_version = dataset_version
        self.current_version = current_version


def check_checkpoint_profile(
    checkpoint: dict[str, Any],
    dataset_profile_id: str,
    dataset_profile_hash: str,
    *,
    action: str,
    allow_mismatch: bool = False,
    override_verb: str = "Continuing",
    ckpt_label: str = "Model trained on",
) -> str | None:
    """Guard a checkpoint against being used on a dataset of another profile.

    A profile ID mismatch raises :class:`ProfileMismatchError` unless *allow_mismatch*, which
    downgrades it to a logged warning. A profile hash mismatch and a checkpoint without profile
    metadata (legacy) are only logged as warnings.

    Args:
        checkpoint: Loaded checkpoint dict
        dataset_profile_id: Profile ID from resolve_dataset_profile()
        dataset_profile_hash: Profile hash from resolve_dataset_profile()
        action: Phrase for the error, e.g. "evaluate model on"
        allow_mismatch: Accept a profile ID mismatch (warning only)
        override_verb: Leading verb of the override warning ("Resuming", ...)
        ckpt_label: Label for the checkpoint profile line

    Returns:
        The checkpoint profile ID, or None for legacy checkpoints without metadata.
    """
    ckpt_profile = checkpoint.get('component_profile')
    if not ckpt_profile:
        log.warning("Checkpoint lacks profile metadata (legacy checkpoint)")
        return None

    ckpt_profile_id = ckpt_profile.get('profile_id')
    ckpt_profile_hash = ckpt_profile.get('profile_hash')
    label = f"{ckpt_label}:"
    width = max(len(label), len("Dataset profile:")) + 1
    lines = (f"  {label:<{width}}{ckpt_profile_id}\n"
             f"  {'Dataset profile:':<{width}}{dataset_profile_id}")

    if ckpt_profile_id and ckpt_profile_id != dataset_profile_id:
        if not allow_mismatch:
            raise ProfileMismatchError(
                f"Component profile mismatch:\n{lines}\nCannot {action} different component type.",
                checkpoint_profile=str(ckpt_profile_id), dataset_profile=str(dataset_profile_id),
            )
        log.warning(f"{override_verb} with component profile mismatch (mismatch allowed).\n{lines}")

    if ckpt_profile_hash and ckpt_profile_hash != dataset_profile_hash:
        log.warning("Profile hash mismatch (different tolerance/geometry)\n"
                    f"  Checkpoint hash: {ckpt_profile_hash[:72]}...\n"
                    f"  Dataset hash:    {dataset_profile_hash[:72]}...")

    return ckpt_profile_id


def write_multi_profile_manifest(
    output_dir: Path,
    run_id: str,
    profiles: list[dict[str, str]],
    meta_rows: list[Any],
    label_rows: list[Any],
    splits: dict[str, list[str]],
    script_name: str = "scripts/generate_profile_dataset.py",
    existing: dict[str, Any] | None = None,
    renderer_version: str | None = None,
    git_commit: str | None = None,
) -> None:
    """Write a multi-profile dataset manifest (manifest_version 2).

    Args:
        output_dir: Dataset output directory
        run_id: Run identifier
        profiles: List of dicts with keys: profile_id, profile_hash, profile_path
        meta_rows: List of MetaRow instances (the complete dataset)
        label_rows: List of LabelRow instances (the complete dataset)
        splits: Dictionary mapping split names to sample IDs
        script_name: Generator script name for metadata
        existing: Previous manifest when extending; its ``created_at`` and
            ``extend_history`` are preserved and a new history entry is appended
            (``samples_added`` = growth of ``total_samples``).
        renderer_version: renderer of the images written now (see :func:`write_dataset_manifest`);
            None keeps the existing manifest's value (merges do not render).
        git_commit: see :func:`write_dataset_manifest`.
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / DATASET_MANIFEST

    git_commit = git_commit if git_commit is not None else _get_git_commit()

    # Profile sample counts
    profile_counts: dict[str, int] = {}
    for label_row in label_rows:
        pid = getattr(label_row, 'profile_id', '')
        if pid:
            profile_counts[pid] = profile_counts.get(pid, 0) + 1

    samples_added = len(meta_rows)
    if existing:
        prev_total = int((existing.get('dataset_stats') or {}).get('total_samples') or 0)
        samples_added = max(0, len(meta_rows) - prev_total)
    if renderer_version is None and existing:
        renderer_version = (existing.get('generator') or {}).get('renderer_version')
    history_entry = _history_entry(samples_added, git_commit, renderer_version)
    created_at = now_iso()
    extend_history = [history_entry]
    if existing:
        created_at = existing.get('created_at') or created_at
        extend_history = list(existing.get('extend_history') or []) + [history_entry]

    generator = _generator_block(script_name, git_commit, renderer_version)

    manifest = {
        'manifest_version': 2,
        'created_at': created_at,
        'run_id': run_id,

        'component_profiles': profiles,

        'generator': generator,

        'dataset_stats': {**dataset_stats(meta_rows, label_rows, splits), 'profile_counts': profile_counts},

        'extend_history': extend_history,
    }

    atomic_write_json(manifest_path, manifest)
    log.info(f"Multi-profile dataset manifest written: {manifest_path}")


def check_renderer_version(manifest: dict[str, Any], current: str, *, allow_mismatch: bool = False) -> None:
    """Refuse to extend a dataset whose images come from another renderer version.

    A manifest without ``generator.renderer_version`` (v1.0.3 and older, merged datasets) counts as
    a mismatch and raises :class:`RendererMismatchError`. With *allow_mismatch* only a warning is
    logged (the dataset then mixes two looks).
    """
    existing = (manifest.get('generator') or {}).get('renderer_version')
    if existing is not None and str(existing) == str(current):
        return
    shown = existing if existing is not None else "unknown (v1.0.3 or older, or a merged dataset)"
    msg = (f"Renderer version mismatch when extending a dataset:\n"
           f"  Dataset renderer: {shown}\n"
           f"  Current renderer: {current}\n"
           f"New images would look different from the existing ones (same config and seeds do not\n"
           f"reproduce them). Generate a new dataset, or allow the mismatch to mix them.")
    if not allow_mismatch:
        raise RendererMismatchError(msg, dataset_version=None if existing is None else str(existing), current_version=str(current))
    log.warning(msg)


def record_added_samples(manifest_path: Path, *, split: str, classes: dict[str, int],
                         log_key: str, log_entry: dict[str, Any]) -> bool:
    """Count samples appended to a dataset in place (no re-scan of meta/labels).

    Adds ``sum(classes)`` to ``dataset_stats.total_samples`` and ``splits[split]``, each count
    to ``classes[name]``, and appends *log_entry* to the list ``manifest[log_key]`` (e.g.
    ``real_crops``). Returns False (and writes nothing) when *manifest_path* does not exist.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return False
    manifest = load_json(manifest_path)
    stats = manifest.setdefault('dataset_stats', {})
    total = sum(classes.values())
    stats['total_samples'] = int(stats.get('total_samples', 0)) + total
    split_counts = stats.setdefault('splits', {})
    split_counts[split] = int(split_counts.get(split, 0)) + total
    class_counts = stats.setdefault('classes', {})
    for cls, n in classes.items():
        class_counts[cls] = int(class_counts.get(cls, 0)) + n
    manifest.setdefault(log_key, []).append(log_entry)
    atomic_write_json(manifest_path, manifest)
    return True
