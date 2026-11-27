"""Guards that refuse to mix things a dataset manifest says do not belong together.

* :func:`check_checkpoint_profile`: a checkpoint used on a dataset of another component profile.
* :func:`check_renderer_version`: extending a dataset with images of another renderer version.

Re-exported by :mod:`simple_sim.manifest`; the warnings keep that module's logger name.
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = ["ProfileMismatchError", "RendererMismatchError", "check_checkpoint_profile", "check_renderer_version"]

log = logging.getLogger("simple_sim.manifest")


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
