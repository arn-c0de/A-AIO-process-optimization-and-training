"""Shared model construction and checkpoint loading (ResNet18 classifiers + ensembles)."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

ENSEMBLE_FORMAT = "simple_sim_ensemble_v1"


def build_resnet18(num_classes: int, pretrained: bool = False) -> nn.Module:
    """Create a ResNet18 with a ``num_classes``-way classification head.

    With ``pretrained=True`` ImageNet weights are used; if they cannot be loaded
    (e.g. no network on first run) a warning is printed and random init is used.
    """
    model = None
    if pretrained:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"WARNING: Failed to load pretrained weights ({e}). Falling back to random init.")
    if model is None:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def model_from_checkpoint(checkpoint: dict[str, Any], device: str | torch.device) -> nn.Module:
    """Build a ResNet18 from a single-model checkpoint dict and put it in eval mode on ``device``."""
    model = build_resnet18(len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


class EnsembleModel(nn.Module):
    """Average the logits of several sub-models (softmax is applied by the caller)."""

    def __init__(self, parts: list[nn.Module]) -> None:
        super().__init__()
        self.parts = nn.ModuleList(parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for m in self.parts:
            y = m(x)
            out = y if out is None else (out + y)
        assert out is not None
        return out / float(len(self.parts))


def load_checkpoint_model(
    model_path: Path, device: str | torch.device
) -> tuple[nn.Module, list[str], dict[str, Any]]:
    """Load a single-model or ensemble checkpoint.

    Returns ``(model, class_names, checkpoint)`` with the model in eval mode on ``device``.
    """
    # Always load to CPU first to avoid accidentally loading a large ensemble onto GPU.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and checkpoint.get("format") == ENSEMBLE_FORMAT:
        class_names = list(checkpoint.get("class_names") or [])
        items = checkpoint.get("models") or []
        if not class_names or not isinstance(items, list) or not items:
            raise ValueError(f"Invalid ensemble checkpoint (missing class_names/models): {model_path}")

        sub_models: list[nn.Module] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            ckpt = it.get("checkpoint") or {}
            if not isinstance(ckpt, dict):
                continue
            if list(ckpt.get("class_names") or []) != class_names:
                raise ValueError(
                    "Ensemble class_names mismatch between sub-models. "
                    "All merged models must share the same defect class list."
                )
            sub_models.append(model_from_checkpoint(ckpt, device))

        if not sub_models:
            raise ValueError(f"Invalid ensemble checkpoint (no valid sub-models): {model_path}")

        model = EnsembleModel(sub_models).to(device)
        model.eval()
        return model, class_names, checkpoint

    class_names = list(checkpoint["class_names"])
    return model_from_checkpoint(checkpoint, device), class_names, checkpoint
