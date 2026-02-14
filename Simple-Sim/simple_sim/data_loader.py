"""PyTorch Dataset for ROI images."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from simple_sim.schema import read_jsonl, MetaRow, LabelRow
from simple_sim.splits import read_split


class ROIDataset(Dataset):
    """PyTorch Dataset for PCB ROI images."""

    def __init__(self, data_dir: Path, split: str, class_names: List[str], return_id: bool = False):
        """Initialize dataset.

        Args:
            data_dir: Root directory of dataset
            split: Split name ('train', 'val', 'test')
            class_names: Ordered list of class names
            return_id: If True, __getitem__ returns (image, label, sample_id, image_rel_path)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.class_names = class_names
        self.return_id = return_id

        # Load split IDs
        split_file = self.data_dir / 'splits' / f'{split}.txt'
        self.sample_ids = read_split(split_file)

        # Load metadata and labels
        meta_rows = read_jsonl(self.data_dir / 'meta.jsonl', MetaRow)
        label_rows = read_jsonl(self.data_dir / 'labels.jsonl', LabelRow)

        # Create ID mappings
        self.id_to_meta = {row.id: row for row in meta_rows}
        self.id_to_label = {row.id: row.class_name for row in label_rows}

        # Filter to current split
        self.samples = []
        for sample_id in self.sample_ids:
            if sample_id in self.id_to_meta and sample_id in self.id_to_label:
                self.samples.append(sample_id)

        if len(self.samples) != len(self.sample_ids):
            print(f"WARNING: {len(self.sample_ids) - len(self.samples)} samples missing from metadata/labels")

        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, str, str]]:
        """Get sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label_index)
        """
        sample_id = self.samples[idx]

        # Load image
        meta = self.id_to_meta[sample_id]
        image_path = self.data_dir / meta.image_path

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise IOError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply transforms
        img_tensor = self.transform(img_rgb)

        # Get label
        class_name = self.id_to_label[sample_id]
        label_idx = self.class_names.index(class_name)

        if self.return_id:
            return img_tensor, label_idx, sample_id, meta.image_path
        return img_tensor, label_idx

    def get_class_distribution(self) -> dict:
        """Get class distribution in this dataset.

        Returns:
            Dictionary mapping class name to count
        """
        distribution = {class_name: 0 for class_name in self.class_names}
        for sample_id in self.samples:
            class_name = self.id_to_label[sample_id]
            distribution[class_name] += 1
        return distribution


class ProfileDataset(Dataset):
    """PyTorch Dataset for profile (component type) classification.

    Labels are profile indices rather than defect classes.
    Requires v2 labels with profile_id set.
    """

    def __init__(self, data_dir: Path, split: str, profile_names: List[str], return_id: bool = False):
        """Initialize profile dataset.

        Args:
            data_dir: Root directory of dataset
            split: Split name ('train', 'val', 'test')
            profile_names: Sorted list of profile ID strings
            return_id: If True, __getitem__ returns (image, label, sample_id, image_rel_path)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.profile_names = sorted(profile_names)
        self.return_id = return_id

        # Load split IDs
        split_file = self.data_dir / 'splits' / f'{split}.txt'
        self.sample_ids = read_split(split_file)

        # Load metadata and labels
        meta_rows = read_jsonl(self.data_dir / 'meta.jsonl', MetaRow)
        label_rows = read_jsonl(self.data_dir / 'labels.jsonl', LabelRow)

        self.id_to_meta = {row.id: row for row in meta_rows}
        self.id_to_profile = {row.id: row.profile_id for row in label_rows}

        # Filter to current split and ensure profile_id is present
        self.samples = []
        for sample_id in self.sample_ids:
            if sample_id in self.id_to_meta and sample_id in self.id_to_profile:
                pid = self.id_to_profile[sample_id]
                if pid in self.profile_names:
                    self.samples.append(sample_id)

        if len(self.samples) != len(self.sample_ids):
            print(f"WARNING: {len(self.sample_ids) - len(self.samples)} samples excluded (missing metadata or unknown profile)")

        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, str, str]]:
        sample_id = self.samples[idx]

        meta = self.id_to_meta[sample_id]
        image_path = self.data_dir / meta.image_path

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise IOError(f"Failed to load image: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb)

        profile_id = self.id_to_profile[sample_id]
        label_idx = self.profile_names.index(profile_id)

        if self.return_id:
            return img_tensor, label_idx, sample_id, meta.image_path
        return img_tensor, label_idx

    def get_class_distribution(self) -> dict:
        """Get profile distribution in this dataset."""
        distribution = {name: 0 for name in self.profile_names}
        for sample_id in self.samples:
            profile_id = self.id_to_profile[sample_id]
            distribution[profile_id] += 1
        return distribution
