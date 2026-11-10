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


def list_synthetic_images(folder: Path | str) -> list[Path]:
    """:func:`list_images` for the rendered images of a dataset: *folder* is a dataset directory or
    its ``images/`` folder, and the real crops ``mix-real`` put into ``images/real/`` are left out
    (they would count as synthetic in a sim-to-real comparison). Any other folder is listed whole.
    """
    folder = Path(folder)
    if (folder / "meta.jsonl").exists() and (folder / "images").is_dir():
        folder = folder / "images"
    files = list_images(folder)
    real = folder / "real"
    if (folder.parent / "meta.jsonl").exists() and real.is_dir():
        files = [p for p in files if real not in p.parents]
    return files
