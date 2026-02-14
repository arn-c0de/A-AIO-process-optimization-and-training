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


def discover_datasets(runs_dir: Path) -> list[Path]:
    """Return all dataset directories that contain labels.jsonl."""
    if not runs_dir.exists():
        return []
    return sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "labels.jsonl").exists()
    )


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


def load_meta_and_labels(ds: Path) -> dict[str, list[dict]]:
    """Load meta.jsonl + labels.jsonl and group sample info by class_name.

    Returns:
        dict mapping class_name -> list of dicts with keys:
            image_path, defect, nominal
    """
    meta_path = ds / "meta.jsonl"
    labels_path = ds / "labels.jsonl"

    # Build id -> meta row lookup
    id_to_meta: dict[str, dict] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            id_to_meta[row["id"]] = row

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
                })

    return dict(by_class)


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
    entries: list[tuple[str, str, str, str]] = []

    for ds in datasets:
        ds_name = ds.name
        profile_id = read_profile_id(ds)
        by_class = load_meta_and_labels(ds)

        if not by_class:
            print(f"  {ds_name}: no labels found, skipping")
            continue

        ds_gallery = gallery_dir / ds_name
        ds_gallery.mkdir(parents=True, exist_ok=True)

        for class_name in sorted(by_class.keys()):
            samples = by_class[class_name]
            sample = random.choice(samples)
            src = ds / sample["image_path"]
            if not src.exists():
                print(f"  {ds_name}/{class_name}: image not found ({sample['image_path']}), skipping")
                continue

            # Load image
            img = cv2.imread(str(src))
            if img is None:
                print(f"  {ds_name}/{class_name}: failed to read image ({src}), skipping")
                continue

            # Apply defect overlay (same as GUI pipeline preview)
            img_overlay = draw_defect_overlay(img, sample["defect"], sample["nominal"])

            # Save overlaid image
            dst = ds_gallery / f"{class_name}.png"
            cv2.imwrite(str(dst), img_overlay)

            rel = dst.relative_to(project_root)
            entries.append((ds_name, profile_id, class_name, str(rel)))
            print(f"  {ds_name}/{class_name}: {sample['image_path']} -> {rel}")

    # Write SAMPLE_GALLERY.md
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Sample Gallery")
    lines.append("")
    lines.append(f"Auto-generated reference images from all available datasets. Last updated: {now}")
    lines.append("")
    lines.append("Re-generate by running:")
    lines.append("```bash")
    lines.append(".venv/bin/python tools/update_sample_gallery.py")
    lines.append("```")
    lines.append("")

    # Group entries by dataset
    datasets_seen: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for ds_name, profile_id, class_name, rel_path in entries:
        datasets_seen[ds_name].append((profile_id, class_name, rel_path))

    for ds_name in sorted(datasets_seen.keys()):
        items = datasets_seen[ds_name]
        profile_id = items[0][0]
        lines.append(f"## {ds_name}")
        lines.append("")
        lines.append(f"Profile: `{profile_id}`")
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
