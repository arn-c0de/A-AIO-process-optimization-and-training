"""Torch device selection shared by the scripts and the library (plan D1).

Importing this module does not import torch; :func:`default_device_name` and
:func:`resolve_device` import it on first call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["default_device_name", "resolve_device"]


def default_device_name() -> str:
    """``"cuda"`` when a CUDA device is available, else ``"cpu"`` (the scripts' ``--device`` default)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device(spec: str | torch.device | None = None) -> torch.device:
    """``torch.device`` for *spec* (a name such as ``"cpu"``/``"cuda:1"`` or a device);
    ``None`` picks :func:`default_device_name`. Invalid names raise torch's ``RuntimeError``."""
    import torch

    if spec is None:
        spec = default_device_name()
    return torch.device(spec)
