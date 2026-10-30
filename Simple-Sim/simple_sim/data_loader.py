"""PyTorch Dataset for ROI images."""

import os
import warnings
import cv2
import numpy as np
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from simple_sim.schema import read_jsonl, MetaRow, LabelRow
from simple_sim.splits import read_split

# ImageNet normalization constants — single source of truth for the whole package.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Shared inference/training transform (RGB uint8 HWC -> normalised CHW tensor).
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
_IMAGENET_TRANSFORM = IMAGENET_TRANSFORM  # backward-compatible alias


def preprocess_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR image (cv2.imread) to a normalised model input tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return IMAGENET_TRANSFORM(img_rgb)


def make_loader_kwargs(num_workers: int, device: torch.device, *, verbose: bool = True) -> dict[str, Any]:
    """Build shared DataLoader kwargs, auto-tuning workers for CUDA runs.

    When the config leaves num_workers at 0 on CUDA, pick a worker count from
    the CPU count. num_workers>0 can fail in restricted environments
    (semaphores / shared memory perms), so CPU runs keep the configured value.
    """
    num_workers = int(num_workers or 0)
    effective = num_workers
    if device.type == "cuda" and effective <= 0:
        cpu_count = max(1, int(os.cpu_count() or 1))
        effective = max(2, min(8, cpu_count - 1 if cpu_count > 1 else 1))
        if verbose:
            print(f"  Num workers: {num_workers} (auto-tuned to {effective} for CUDA)")
    elif verbose:
        print(f"  Num workers: {effective}")

    kwargs: dict[str, Any] = {
        "num_workers": effective,
        "pin_memory": (device.type == "cuda"),
    }
    if effective > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _BaseROIDataset(Dataset):
    """Base class with shared image loading and metadata handling."""

    def __init__(self, data_dir: Path, split: str, return_id: bool = False) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.return_id = return_id
        self.transform = IMAGENET_TRANSFORM

        split_file = self.data_dir / 'splits' / f'{split}.txt'
        self.sample_ids = read_split(split_file)

        meta_rows = read_jsonl(self.data_dir / 'meta.jsonl', MetaRow)
        label_rows = read_jsonl(self.data_dir / 'labels.jsonl', LabelRow)

        self.id_to_meta = {row.id: row for row in meta_rows}
        self._init_label_mapping(label_rows)

        self.samples = [
            sid for sid in self.sample_ids
            if self._sample_is_valid(sid)
        ]

        missing = len(self.sample_ids) - len(self.samples)
        if missing:
            warnings.warn(
                f"{missing} sample(s) excluded from '{split}' split "
                f"(missing metadata or unknown label).",
                stacklevel=3,
            )

    # Subclasses implement these two hooks:

    def _init_label_mapping(self, label_rows: list[LabelRow]) -> None:
        raise NotImplementedError

    def _sample_is_valid(self, sample_id: str) -> bool:
        raise NotImplementedError

    def _get_label_index(self, sample_id: str) -> int:
        raise NotImplementedError

    # Shared implementation:

    def __len__(self) -> int:
        return len(self.samples)

    def _load_tensor(self, sample_id: str) -> torch.Tensor:
        meta = self.id_to_meta[sample_id]
        image_path = self.data_dir / meta.image_path
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise OSError(f"Failed to load image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, str, str]:
        sample_id = self.samples[idx]
        img_tensor = self._load_tensor(sample_id)
        label_idx = self._get_label_index(sample_id)
        if self.return_id:
            return img_tensor, label_idx, sample_id, self.id_to_meta[sample_id].image_path
        return img_tensor, label_idx


# ---------------------------------------------------------------------------
# Concrete datasets
# ---------------------------------------------------------------------------

class ROIDataset(_BaseROIDataset):
    """PyTorch Dataset for PCB ROI images, labelled by defect class."""

    def __init__(self, data_dir: Path, split: str, class_names: list[str], return_id: bool = False):
        """Initialize dataset.

        Args:
            data_dir: Root directory of dataset
            split: Split name ('train', 'val', 'test')
            class_names: Ordered list of class names
            return_id: If True, __getitem__ returns (image, label, sample_id, image_rel_path)
        """
        self.class_names = class_names
        super().__init__(data_dir, split, return_id)

    def _init_label_mapping(self, label_rows: list[LabelRow]) -> None:
        self.id_to_label: dict[str, str] = {row.id: row.class_name for row in label_rows}

    def _sample_is_valid(self, sample_id: str) -> bool:
        return (
            sample_id in self.id_to_meta
            and sample_id in self.id_to_label
            and self.id_to_label[sample_id] in self.class_names
        )

    def _get_label_index(self, sample_id: str) -> int:
        return self.class_names.index(self.id_to_label[sample_id])

    def get_class_distribution(self) -> dict[str, int]:
        """Return class name → sample count for this split."""
        dist = {name: 0 for name in self.class_names}
        for sid in self.samples:
            dist[self.id_to_label[sid]] += 1
        return dist


class ProfileDataset(_BaseROIDataset):
    """PyTorch Dataset for component-profile classification.

    Labels are profile indices rather than defect classes.
    Requires v2 labels with profile_id set.
    """

    def __init__(self, data_dir: Path, split: str, profile_names: list[str], return_id: bool = False):
        """Initialize profile dataset.

        Args:
            data_dir: Root directory of dataset
            split: Split name ('train', 'val', 'test')
            profile_names: Ordered list of profile ID strings; the order defines
                label indices and is kept as given (callers sort when training)
            return_id: If True, __getitem__ returns (image, label, sample_id, image_rel_path)
        """
        self.profile_names = list(profile_names)
        super().__init__(data_dir, split, return_id)

    def _init_label_mapping(self, label_rows: list[LabelRow]) -> None:
        self.id_to_profile: dict[str, str] = {row.id: row.profile_id for row in label_rows}

    def _sample_is_valid(self, sample_id: str) -> bool:
        return (
            sample_id in self.id_to_meta
            and sample_id in self.id_to_profile
            and self.id_to_profile[sample_id] in self.profile_names
        )

    def _get_label_index(self, sample_id: str) -> int:
        return self.profile_names.index(self.id_to_profile[sample_id])

    def get_class_distribution(self) -> dict[str, int]:
        """Return profile name → sample count for this split."""
        dist = {name: 0 for name in self.profile_names}
        for sid in self.samples:
            dist[self.id_to_profile[sid]] += 1
        return dist
