"""Image-file discovery shared by the sim-to-real tools and the texture substrate.

Dependency-free (no cv2/numpy) so listing a folder never loads the renderer.
"""

from __future__ import annotations

from pathlib import Path

#: File suffixes (lower case) treated as images.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image(path: Path) -> bool:
    """True if *path* has an image suffix (case-insensitive)."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(folder: Path | str, recursive: bool = True) -> list[Path]:
    """Sorted image paths in *folder* (all sub-folders with *recursive*, else direct children only)."""
    folder = Path(folder)
    entries = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in entries if is_image(p))
