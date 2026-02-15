#!/usr/bin/env python3
"""Generate a sample gallery from all available datasets.

For each dataset, picks one random image per defect class, renders the
defect overlay (same as the GUI pipeline preview), copies it into
images/samples/<dataset_name>/, and writes SAMPLE_GALLERY.md with inline
references so the images are visible on GitHub.

Re-running the script picks fresh random samples and overwrites the gallery.

Usage:
    .venv/bin/python tools/update_sample_gallery.py
"""

import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Ensure project imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gui.components.overlay_renderer import draw_defect_overlay

# Standard defect classes for PCB components
STANDARD_CLASSES = {"OK", "MISSING", "MISALIGNED", "TOMBSTONE"}


def discover_datasets(runs_dir: Path) -> list[Path]:
    """Return all dataset directories that contain labels.jsonl."""
    if not runs_dir.exists():
        return []
    return sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "labels.jsonl").exists()
    )


def create_placeholder_image(width: int, height: int, class_name: str) -> np.ndarray:
    """Create a placeholder image for missing samples.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        class_name: Defect class name to display

    Returns:
        BGR image array with placeholder content
    """
    # Create blank image with light background
    img = np.ones((height, width, 3), dtype=np.uint8) * 240

    # Color coding for different classes
    colors = {
        "OK": (0, 200, 0),           # Green
        "MISSING": (0, 0, 200),      # Red
        "MISALIGNED": (0, 165, 200), # Orange
        "TOMBSTONE": (200, 0, 200),  # Magenta
    }
    color = colors.get(class_name, (128, 128, 128))

    # Draw colored border
    border_thickness = 8
    cv2.rectangle(img, (border_thickness, border_thickness),
                  (width - border_thickness, height - border_thickness),
                  color, border_thickness)

    # Add class name text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2
    text = class_name

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, font_thickness)

    # Add "NO SAMPLES" label at bottom
    label_text = "NO SAMPLES"
    label_font_scale = 0.7
    label_font_thickness = 1
    label_size = cv2.getTextSize(label_text, font, label_font_scale, label_font_thickness)[0]
    label_x = (width - label_size[0]) // 2
    label_y = height - 30

    cv2.putText(img, label_text, (label_x, label_y), font, label_font_scale,
                (100, 100, 100), label_font_thickness)

    return img


def read_profile_id(ds: Path) -> str:
    """Read profile_id from the dataset manifest (fall back to dataset name)."""
    manifest = ds / "dataset_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            pid = data.get("component_profile", {}).get("profile_id")
            if pid:
                return pid
        except Exception:
            pass
    return ds.name


def load_meta_and_labels(ds: Path) -> tuple[dict[str, list[dict]], str]:
    """Load meta.jsonl + labels.jsonl and group sample info by class_name.

    Returns:
        tuple of (class_dict, render_backend) where:
            class_dict maps class_name -> list of dicts with keys:
                image_path, defect, nominal, render_backend
            render_backend is the rendering engine used (e.g., 'opencv_2d' or 'blender_3d')
    """
    meta_path = ds / "meta.jsonl"
    labels_path = ds / "labels.jsonl"

    # Build id -> meta row lookup
    id_to_meta: dict[str, dict] = {}
    render_backend = "unknown"
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            id_to_meta[row["id"]] = row
            # Capture render_backend from first row
            if render_backend == "unknown":
                render_backend = row.get("render_backend", "unknown")

    # Group by class with full metadata
    by_class: dict[str, list[dict]] = defaultdict(list)
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row["id"]
            class_name = row["class_name"]
            meta = id_to_meta.get(sample_id)
            if meta:
                by_class[class_name].append({
                    "image_path": meta["image_path"],
                    "defect": meta["defect"],
                    "nominal": meta["nominal"],
                    "render_backend": meta.get("render_backend", "unknown"),
                })

    return dict(by_class), render_backend


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    runs_dir = project_root / "outputs" / "sim_data" / "runs"
    gallery_dir = project_root / "images" / "samples"
    gallery_md = project_root / "SAMPLE_GALLERY.md"

    datasets = discover_datasets(runs_dir)
    if not datasets:
        print(f"No datasets found in {runs_dir}")
        sys.exit(1)

    print(f"Found {len(datasets)} dataset(s): {', '.join(d.name for d in datasets)}")

    # Clean previous gallery
    if gallery_dir.exists():
        shutil.rmtree(gallery_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # Collect gallery entries
    entries: list[tuple[str, str, str, str, str]] = []

    for ds in datasets:
        ds_name = ds.name
        profile_id = read_profile_id(ds)
        by_class, render_backend = load_meta_and_labels(ds)

        if not by_class:
            print(f"  {ds_name}: no labels found, skipping")
            continue

        ds_gallery = gallery_dir / ds_name
        ds_gallery.mkdir(parents=True, exist_ok=True)

        # Detect missing classes
        available_classes = set(by_class.keys())
        missing_classes = STANDARD_CLASSES - available_classes

        if missing_classes:
            print(f"  {ds_name}: missing classes {sorted(missing_classes)}, will create placeholders")

        # Process all standard classes
        for class_name in sorted(STANDARD_CLASSES):
            if class_name in by_class:
                # Use real sample
                samples = by_class[class_name]
                sample = random.choice(samples)
                src = ds / sample["image_path"]
                if not src.exists():
                    print(f"  {ds_name}/{class_name}: image not found ({sample['image_path']}), creating placeholder")
                    # Create placeholder for missing image
                    img = create_placeholder_image(256, 256, class_name)
                else:
                    # Load image
                    img = cv2.imread(str(src))
                    if img is None:
                        print(f"  {ds_name}/{class_name}: failed to read image ({src}), creating placeholder")
                        img = create_placeholder_image(256, 256, class_name)
                    else:
                        # Apply defect overlay (same as GUI pipeline preview)
                        img = draw_defect_overlay(img, sample["defect"], sample["nominal"])

                dst = ds_gallery / f"{class_name}.png"
                cv2.imwrite(str(dst), img)
                rel = dst.relative_to(project_root)
                entries.append((ds_name, profile_id, class_name, str(rel), render_backend))
                print(f"  {ds_name}/{class_name} [{render_backend}]: -> {rel}")
            else:
                # Create placeholder for missing class
                print(f"  {ds_name}/{class_name}: no samples, creating placeholder")
                img = create_placeholder_image(256, 256, class_name)
                dst = ds_gallery / f"{class_name}.png"
                cv2.imwrite(str(dst), img)
                rel = dst.relative_to(project_root)
                entries.append((ds_name, profile_id, class_name, str(rel), render_backend))
                print(f"  {ds_name}/{class_name} [placeholder]: -> {rel}")

    # Write SAMPLE_GALLERY.md
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Sample Gallery")
    lines.append("")
    lines.append(f"Auto-generated reference images from all available datasets (2D and 3D). Last updated: {now}")
    lines.append("")
    lines.append("Re-generate by running:")
    lines.append("```bash")
    lines.append(".venv/bin/python tools/update_sample_gallery.py")
    lines.append("```")
    lines.append("")

    # Group entries by dataset and render backend
    datasets_seen: dict[str, dict[str, list[tuple[str, str, str]]]] = defaultdict(lambda: defaultdict(list))
    for ds_name, profile_id, class_name, rel_path, render_backend in entries:
        datasets_seen[ds_name][render_backend].append((profile_id, class_name, rel_path))

    for ds_name in sorted(datasets_seen.keys()):
        backends = datasets_seen[ds_name]

        # Use the first profile_id available for this dataset
        first_profile_id = backends[list(backends.keys())[0]][0][0] if backends else "unknown"

        lines.append(f"## {ds_name}")
        lines.append("")
        lines.append(f"Profile: `{first_profile_id}`")
        lines.append("")

        # Organize by render backend
        for backend in sorted(backends.keys()):
            items = backends[backend]

            # Add backend label if multiple backends present
            if len(backends) > 1:
                backend_label = "2D (OpenCV)" if backend == "opencv_2d" else "3D (Blender)" if backend == "blender_3d" else backend
                lines.append(f"### {backend_label}")
                lines.append("")

            lines.append("| Class | Sample |")
            lines.append("|-------|--------|")
            for _, class_name, rel_path in sorted(items, key=lambda x: x[1]):
                lines.append(f"| {class_name} | ![{class_name}]({rel_path}) |")
            lines.append("")

    gallery_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nGallery written to {gallery_md}")
    print(f"Images in {gallery_dir}")
    print(f"Total samples: {len(entries)}")


if __name__ == "__main__":
    main()
