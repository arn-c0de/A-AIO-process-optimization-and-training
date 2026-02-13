"""LRU cache for image loading to improve performance."""

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image, ImageTk


class ImageCache:
    """LRU cache for images with separate caches for thumbnails and full images."""

    def __init__(self, thumb_size: Tuple[int, int] = (150, 150),
                 full_size: Tuple[int, int] = (512, 512),
                 thumb_capacity: int = 200,
                 full_capacity: int = 20) -> None:
        """Initialize image cache.

        Args:
            thumb_size: Thumbnail size (width, height)
            full_size: Full image size (width, height)
            thumb_capacity: Maximum thumbnails to cache
            full_capacity: Maximum full images to cache
        """
        self.thumb_size = thumb_size
        self.full_size = full_size
        self.thumb_capacity = thumb_capacity
        self.full_capacity = full_capacity

        # OrderedDict provides O(1) access and maintains insertion order for LRU
        self._thumb_cache: OrderedDict[str, ImageTk.PhotoImage] = OrderedDict()
        self._full_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get_thumbnail(self, image_path: Path) -> Optional[ImageTk.PhotoImage]:
        """Get thumbnail from cache or load and cache it.

        Args:
            image_path: Path to image file

        Returns:
            PIL ImageTk.PhotoImage or None if failed to load
        """
        key = str(image_path)

        # Cache hit - move to end (most recently used)
        if key in self._thumb_cache:
            self._thumb_cache.move_to_end(key)
            return self._thumb_cache[key]

        # Cache miss - load image
        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail(self.thumb_size)
            tkimg = ImageTk.PhotoImage(img)

            # Add to cache
            self._thumb_cache[key] = tkimg

            # Evict oldest if over capacity
            if len(self._thumb_cache) > self.thumb_capacity:
                self._thumb_cache.popitem(last=False)

            return tkimg
        except Exception as e:
            print(f"Failed to load thumbnail {image_path}: {e}")
            return None

    def get_full_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Get full-size image from cache or load and cache it.

        Args:
            image_path: Path to image file

        Returns:
            Numpy array (BGR format) or None if failed to load
        """
        key = str(image_path)

        # Cache hit - move to end (most recently used)
        if key in self._full_cache:
            self._full_cache.move_to_end(key)
            return self._full_cache[key]

        # Cache miss - load image
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            # Resize if needed
            h, w = img.shape[:2]
            if h > self.full_size[1] or w > self.full_size[0]:
                img = cv2.resize(img, self.full_size, interpolation=cv2.INTER_AREA)

            # Add to cache
            self._full_cache[key] = img

            # Evict oldest if over capacity
            if len(self._full_cache) > self.full_capacity:
                self._full_cache.popitem(last=False)

            return img
        except Exception as e:
            print(f"Failed to load full image {image_path}: {e}")
            return None

    def clear(self) -> None:
        """Clear all caches."""
        self._thumb_cache.clear()
        self._full_cache.clear()

    def clear_thumbnails(self) -> None:
        """Clear thumbnail cache only."""
        self._thumb_cache.clear()

    def clear_full_images(self) -> None:
        """Clear full image cache only."""
        self._full_cache.clear()
