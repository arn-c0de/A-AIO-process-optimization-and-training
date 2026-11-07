"""PyTorch Dataset for ROI images."""

import contextlib
import logging
import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from simple_sim.schema import read_jsonl, MetaRow, LabelRow
from simple_sim.splits import read_split

log = logging.getLogger(__name__)

# ImageNet normalization constants — single source of truth for the whole package.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Shared inference/training transform (RGB uint8 HWC -> normalised CHW tensor).
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class _RandomRot90:
    """Rotate the CHW tensor by a random multiple of 90 degrees (torch RNG: per-worker seeded).

    Non-square images only get half turns, so every tensor of a batch keeps the same shape.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(0, 4, (1,)).item())
        if x.shape[-1] != x.shape[-2]:
            k = 2 * (k % 2)
        return torch.rot90(x, k, dims=(1, 2)) if k else x


def build_train_transform(augment: dict[str, Any] | None) -> transforms.Compose:
    """Training-time transform from ``train.augment`` (see ``config._validate_train_extras``).

    ``hflip`` / ``vflip`` / ``rot90`` (booleans) are label-neutral for the defect classes (only the
    magnitude of a misplacement matters, the ROI is centred on the part); ``brightness``, ``contrast``,
    ``saturation``, ``hue`` are ``torchvision.transforms.ColorJitter`` strengths. An empty or missing
    mapping gives :data:`IMAGENET_TRANSFORM` (no augmentation), the historic behaviour. Augmentation
    runs on the ``[0, 1]`` tensor before normalisation and draws from torch's RNG, so it is seeded by
    ``seed_everything`` and per DataLoader worker.
    """
    aug = dict(augment or {})
    steps: list[Any] = [transforms.ToTensor()]
    if aug.get("hflip"):
        steps.append(transforms.RandomHorizontalFlip())
    if aug.get("vflip"):
        steps.append(transforms.RandomVerticalFlip())
    if aug.get("rot90"):
        steps.append(_RandomRot90())
    jitter = {k: float(aug.get(k, 0.0) or 0.0) for k in ("brightness", "contrast", "saturation", "hue")}
    if any(jitter.values()):
        steps.append(transforms.ColorJitter(**jitter))
    steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)


@contextlib.contextmanager
def single_threaded_torch() -> Iterator[None]:
    """Run the enclosed per-image tensor ops with one intra-op thread (results unchanged).

    The transforms work on one 256 px image at a time, and such small element-wise ops are
    dominated by the fork/join of torch's thread pool: with the default pool of 16 threads
    ``ToTensor`` + ``Normalize`` take 27 ms instead of 0.3 ms per image and ``ColorJitter``
    600 ms instead of 10 ms (i9-12900K). DataLoader workers already run with one thread; this
    gives the main process (``num_workers: 0``, the config default) the same behaviour.
    """
    threads = torch.get_num_threads()
    if threads <= 1:
        yield
        return
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def apply_transform(transform: Any, img_rgb: np.ndarray) -> torch.Tensor:
    """``transform(img_rgb)`` (RGB uint8 HWC -> CHW tensor) inside :func:`single_threaded_torch`."""
    with single_threaded_torch():
        return transform(img_rgb)


def load_image_bgr(image_path: Path) -> np.ndarray:
    """Read an image with OpenCV (BGR); raise OSError if it cannot be loaded."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise OSError(f"Failed to load image: {image_path}")
    return img_bgr


def preprocess_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR image (cv2.imread) to a normalised model input tensor."""
    return apply_transform(IMAGENET_TRANSFORM, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def make_loader_kwargs(num_workers: int, device: torch.device) -> dict[str, Any]:
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
        log.info(f"  Num workers: {num_workers} (auto-tuned to {effective} for CUDA)")
    else:
        log.info(f"  Num workers: {effective}")

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

    def __init__(self, data_dir: Path, split: str, return_id: bool = False,
                 transform: Any | None = None, cache_images: bool = False) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.return_id = return_id
        self.transform = IMAGENET_TRANSFORM if transform is None else transform
        self.cache_images = bool(cache_images)
        self._rgb_cache: dict[str, np.ndarray] = {}

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

    def _load_rgb(self, sample_id: str) -> np.ndarray:
        """Decoded RGB uint8 image of *sample_id*; kept in RAM after the first read with ``cache_images``."""
        rgb = self._rgb_cache.get(sample_id)
        if rgb is None:
            img_bgr = load_image_bgr(self.data_dir / self.id_to_meta[sample_id].image_path)
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if self.cache_images:
                self._rgb_cache[sample_id] = rgb
        return rgb

    def _load_tensor(self, sample_id: str) -> torch.Tensor:
        return apply_transform(self.transform, self._load_rgb(sample_id))

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

    def __init__(self, data_dir: Path, split: str, class_names: list[str], return_id: bool = False,
                 transform: Any | None = None, cache_images: bool = False):
        """Initialize dataset.

        Args:
            data_dir: Root directory of dataset
            split: Split name ('train', 'val', 'test')
            class_names: Ordered list of class names
            return_id: If True, __getitem__ returns (image, label, sample_id, image_rel_path)
            transform: Image transform (RGB uint8 HWC -> tensor); default :data:`IMAGENET_TRANSFORM`,
                training passes :func:`build_train_transform`
            cache_images: Keep the decoded images in RAM after the first epoch (``train.cache_images``;
                about 200 KB per 256 px image, each DataLoader worker holds its own copy)
        """
        self.class_names = class_names
        super().__init__(data_dir, split, return_id, transform, cache_images)

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
