"""File names inside a dataset directory (plan D7). Dependency-free (no torch/cv2/yaml).

A dataset directory holds ``meta.jsonl`` and ``labels.jsonl`` (one row per sample),
``config.yaml`` (the run config it was generated with), ``dataset_manifest.json``
(profile, generator and stats) and ``splits/<split>.txt`` (sample ids per split).
"""

from __future__ import annotations

__all__ = ["CONFIG_YAML", "DATASET_MANIFEST", "LABELS_JSONL", "META_JSONL", "SPLITS_DIR"]

DATASET_MANIFEST = "dataset_manifest.json"
META_JSONL = "meta.jsonl"
LABELS_JSONL = "labels.jsonl"
CONFIG_YAML = "config.yaml"
SPLITS_DIR = "splits"
