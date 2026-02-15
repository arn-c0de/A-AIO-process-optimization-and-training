#!/usr/bin/env python3
"""Generate a sample gallery from all available component profiles.

For each component profile in configs/profiles/, generates one synthetic sample
image per defect class (OK, MISSING, MISALIGNED, TOMBSTONE) and creates a
SAMPLE_GALLERY.md with all profiles and classes represented.

Re-running the script regenerates all images freshly.

Usage:
    .venv/bin/python tools/update_sample_gallery.py
"""

import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

# Ensure project imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gui.components.overlay_renderer import draw_defect_overlay

# Standard defect classes for PCB components
STANDARD_CLASSES = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]


def discover_profiles(profiles_dir: Path) -> list[Path]:
    """Return all profile YAML files sorted by name."""
    if not profiles_dir.exists():
        return []
    return sorted(profiles_dir.glob("*.yaml"))


def read_profile(profile_path: Path) -> dict:
    """Load profile YAML and return its contents."""
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"  Error reading {profile_path}: {e}")
        return {}


def get_profile_id(profile: dict) -> str:
    """Extract profile_id from profile dict."""
    return profile.get("profile", {}).get("profile_id", "unknown")


def get_render_backend(profile: dict) -> str:
    """Determine render backend from profile (2D or 3D)."""
    backends = profile.get("profile", {}).get("supported_render_backends", [])
    if "blender_3d" in backends:
        return "blender_3d"
    elif "opencv_2d" in backends:
        return "opencv_2d"
    # Fallback: check profile name for hints
    profile_id = get_profile_id(profile)
    if "3d" in profile_id.lower():
        return "blender_3d"
    return "opencv_2d"


def create_synthetic_image(width: int, height: int, class_name: str, profile_id: str) -> np.ndarray:
    """Create a synthetic sample image for a defect class.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        class_name: Defect class name (OK, MISSING, MISALIGNED, TOMBSTONE)
        profile_id: Profile identifier for context

    Returns:
        BGR image array
    """
    # Create base image with PCB-like background (dark green)
    img = np.ones((height, width, 3), dtype=np.uint8) * np.uint8([50, 102, 51])  # Dark green BGR

    # Create a component representation in center
    center_x, center_y = width // 2, height // 2
    comp_width, comp_height = 80, 60

    # Color coding for different classes
    class_colors = {
        "OK": (0, 200, 0),           # Green - OK component
        "MISSING": (0, 0, 255),      # Red - missing component
        "MISALIGNED": (0, 165, 255), # Orange - misaligned
        "TOMBSTONE": (255, 0, 255),  # Magenta - tombstone/tilted
    }
    comp_color = class_colors.get(class_name, (128, 128, 128))

    # Draw component body
    pt1 = (center_x - comp_width // 2, center_y - comp_height // 2)
    pt2 = (center_x + comp_width // 2, center_y + comp_height // 2)

    if class_name == "OK":
        # Normal component
        cv2.rectangle(img, pt1, pt2, comp_color, -1)
        cv2.rectangle(img, pt1, pt2, (255, 255, 255), 2)
    elif class_name == "MISSING":
        # Just show empty space with markers
        cv2.circle(img, (center_x, center_y), 40, comp_color, 2)
        cv2.line(img, (center_x - 40, center_y), (center_x + 40, center_y), comp_color, 2)
        cv2.line(img, (center_x, center_y - 40), (center_x, center_y + 40), comp_color, 2)
    elif class_name == "MISALIGNED":
        # Rotated component
        pts = np.array([
            [center_x - comp_width // 2, center_y - comp_height // 2],
            [center_x + comp_width // 2, center_y - comp_height // 3],
            [center_x + comp_width // 2, center_y + comp_height // 2],
            [center_x - comp_width // 2, center_y + comp_height // 3],
        ], np.int32)
        cv2.fillPoly(img, [pts], comp_color)
        cv2.polylines(img, [pts], True, (255, 255, 255), 2)
    elif class_name == "TOMBSTONE":
        # Tilted component
        pts = np.array([
            [center_x - comp_width // 2, center_y - comp_height // 2 - 30],
            [center_x + comp_width // 2, center_y - comp_height // 2],
            [center_x + comp_width // 2, center_y + comp_height // 2],
            [center_x - comp_width // 2, center_y + comp_height // 2 + 30],
        ], np.int32)
        cv2.fillPoly(img, [pts], comp_color)
        cv2.polylines(img, [pts], True, (255, 255, 255), 2)

    # Add text label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text = class_name
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height - 30
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Add profile name at top
    font_small = 0.6
    profile_text = f"Profile: {profile_id}"
    profile_size = cv2.getTextSize(profile_text, font, font_small, 1)[0]
    profile_x = (width - profile_size[0]) // 2
    cv2.putText(img, profile_text, (profile_x, 25), font, font_small, (200, 200, 200), 1)

    return img


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    profiles_dir = project_root / "configs" / "profiles"
    gallery_dir = project_root / "images" / "samples"
    gallery_md = project_root / "SAMPLE_GALLERY.md"

    profiles = discover_profiles(profiles_dir)
    if not profiles:
        print(f"No profiles found in {profiles_dir}")
        sys.exit(1)

    print(f"Found {len(profiles)} profile(s)")

    # Clean previous gallery
    if gallery_dir.exists():
        shutil.rmtree(gallery_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # Collect gallery entries
    entries: list[tuple[str, str, str, str, str]] = []

    for profile_path in profiles:
        profile_data = read_profile(profile_path)
        if not profile_data:
            continue

        profile_id = get_profile_id(profile_data)
        render_backend = get_render_backend(profile_data)

        print(f"\nProcessing profile: {profile_id} [{render_backend}]")

        profile_gallery = gallery_dir / profile_id
        profile_gallery.mkdir(parents=True, exist_ok=True)

        # Generate one image per class
        for class_name in STANDARD_CLASSES:
            img = create_synthetic_image(256, 256, class_name, profile_id)

            dst = profile_gallery / f"{class_name}.png"
            cv2.imwrite(str(dst), img)

            rel = dst.relative_to(project_root)
            entries.append((profile_id, profile_id, class_name, str(rel), render_backend))
            print(f"  {class_name:12} [{render_backend}]: -> {rel}")

    # Write SAMPLE_GALLERY.md
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Sample Gallery")
    lines.append("")
    lines.append(f"Auto-generated reference images for all component profiles (2D and 3D). Last updated: {now}")
    lines.append("")
    lines.append("Re-generate by running:")
    lines.append("```bash")
    lines.append(".venv/bin/python tools/update_sample_gallery.py")
    lines.append("```")
    lines.append("")

    # Group entries by profile and render backend
    profiles_seen: dict[str, dict[str, list[tuple[str, str, str]]]] = defaultdict(lambda: defaultdict(list))
    for profile_id, _, class_name, rel_path, render_backend in entries:
        profiles_seen[profile_id][render_backend].append((profile_id, class_name, rel_path))

    for profile_id in sorted(profiles_seen.keys()):
        backends = profiles_seen[profile_id]

        lines.append(f"## {profile_id}")
        lines.append("")

        # Organize by render backend
        for backend in sorted(backends.keys()):
            items = backends[backend]

            # Add backend label if multiple backends present for this profile
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
    print(f"\n✓ Gallery written to {gallery_md}")
    print(f"✓ Images in {gallery_dir}")
    print(f"✓ Total samples: {len(entries)}")


if __name__ == "__main__":
    main()
