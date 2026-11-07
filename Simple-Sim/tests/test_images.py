"""Tests for simple_sim.images (shared image-folder listing)."""

from __future__ import annotations

from simple_sim.generators.opencv2d.photoreal import IMAGE_EXTENSIONS as PHOTOREAL_EXTENSIONS
from simple_sim.images import IMAGE_EXTENSIONS, is_image, list_images


def test_extensions_match_photoreal_reexport():
    assert IMAGE_EXTENSIONS == PHOTOREAL_EXTENSIONS


def test_list_images_matches_legacy_order(tmp_path):
    for rel in ("b.PNG", "a.jpg", "sub/c.tif", "sub/deeper/d.bmp", "notes.txt", "sub/e.json"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    legacy = sorted(p for p in tmp_path.rglob("*") if p.suffix.lower() in PHOTOREAL_EXTENSIONS)
    assert list_images(tmp_path) == legacy
    assert [p.relative_to(tmp_path).as_posix() for p in legacy] == ["a.jpg", "b.PNG", "sub/c.tif", "sub/deeper/d.bmp"]
    assert list_images(str(tmp_path), recursive=False) == [tmp_path / "a.jpg", tmp_path / "b.PNG"]
    assert is_image(tmp_path / "x.JPEG") and not is_image(tmp_path / "x.txt")
