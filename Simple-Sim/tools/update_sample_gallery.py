#!/usr/bin/env python3
"""Generate a sample gallery from all available component profiles.

For each component profile in configs/profiles/, generates one synthetic sample
image per defect class (OK, MISSING, MISALIGNED, TOMBSTONE) and creates a
SAMPLE_GALLERY.md with all profiles and classes represented.

Re-running the script regenerates all images freshly.

Usage:
    .venv/bin/python tools/update_sample_gallery.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Ensure project imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gui.components.overlay_renderer import draw_defect_overlay
from simple_sim.config import load_config, validate_config
from simple_sim.defects import sample_defect_params
from simple_sim.generator_2d import render_roi, sample_nominal_geometry
from simple_sim.generator_3d import render_blender_batch, write_jobs_jsonl

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


def _stable_int_seed(*parts: str) -> int:
    # Deterministic per (profile_id, class, backend) without depending on Python's hash randomization.
    import hashlib
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def _iter_run_configs(configs_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(Path(configs_dir).glob("run_*.yaml")):
        if p.name.startswith("."):
            continue
        out.append(p)
    return out


def _find_matching_run_cfg(
    *,
    configs_dir: Path,
    profile_id: str,
    backend: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Find a run_*.yaml that uses `run.component_profile == profile_id` and `render.backend == backend`."""
    for cfg_path in _iter_run_configs(configs_dir):
        try:
            cfg = load_config(cfg_path)
            validate_config(cfg)
        except Exception:
            continue
        if str(cfg.get("run", {}).get("component_profile", "")) != str(profile_id):
            continue
        if str(cfg.get("render", {}).get("backend", "")) != str(backend):
            continue
        return cfg, cfg_path
    return None, None


def _profile_defect_set(profile: Dict[str, Any]) -> List[str]:
    ds = profile.get("defect_set") or []
    if isinstance(ds, list) and ds:
        return [str(x) for x in ds]
    return list(STANDARD_CLASSES)


def _merge_component_color(render_cfg: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    # Match the pipeline behavior: schema v2 run configs can omit component_color and pull from the profile.
    out = dict(render_cfg or {})
    if "component_color" not in out:
        prof_render = profile.get("render") or {}
        if "component_color_bgr" in prof_render:
            out["component_color"] = prof_render["component_color_bgr"]
    return out


def _render_one_2d(
    *,
    profile_id: str,
    profile: Dict[str, Any],
    cfg: Dict[str, Any],
    out_path: Path,
    defect_type: str,
    seed_base: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    roi_cfg = cfg.get("roi") or {"width_px": 256, "height_px": 256, "mm_per_px": 0.01}
    roi_w = int(roi_cfg.get("width_px", 256))
    roi_h = int(roi_cfg.get("height_px", 256))

    rng = np.random.default_rng(_stable_int_seed(str(seed_base), profile_id, defect_type, "opencv_2d"))

    footprint = str((profile.get("component") or {}).get("footprint", "chip_2pad"))
    geometry_ranges = profile.get("geometry_ranges") or {}
    tolerances = profile.get("tolerances") or {}

    nominal = sample_nominal_geometry(roi_cfg, rng, geometry_ranges=geometry_ranges)
    defect = sample_defect_params(str(defect_type), rng, tolerances=tolerances)

    render_cfg = _merge_component_color(cfg.get("render") or {}, profile)
    # Gallery should be stable and "clean": no blur/noise/brightness jitter.
    augment = {
        "blur_sigma": 0.0,
        "noise_stddev": 0.0,
        "brightness_factor": 1.0,
        "contrast_factor": 1.0,
        "rotation_deg": 0.0,
    }

    img = render_roi(
        nominal=nominal,
        defect_params=defect,
        augment=augment,
        roi_size=(roi_w, roi_h),
        config=render_cfg,
        tolerances=tolerances,
        rng=rng,
        footprint=footprint,
    )
    img = draw_defect_overlay(img, defect, nominal)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")
    return nominal, defect


def _build_3d_job(
    *,
    project_root: Path,
    profile_id: str,
    profile: Dict[str, Any],
    cfg: Dict[str, Any],
    out_path: Path,
    defect_type: str,
    seed_base: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    roi_cfg = cfg.get("roi") or {"width_px": 256, "height_px": 256, "mm_per_px": 0.01}
    mm_per_px = float(roi_cfg.get("mm_per_px", 0.01))
    roi_w = int(roi_cfg.get("width_px", 256))
    roi_h = int(roi_cfg.get("height_px", 256))

    rng = np.random.default_rng(_stable_int_seed(str(seed_base), profile_id, defect_type, "blender_3d"))

    footprint = str((profile.get("component") or {}).get("footprint", "chip_2pad"))
    geometry_ranges = profile.get("geometry_ranges") or {}
    tolerances = profile.get("tolerances") or {}
    component_height_mm = float(((profile.get("component") or {}).get("nominal_dims_mm") or {}).get("height", 0.45) or 0.45)
    render_3d = profile.get("render_3d") or {}

    nominal = sample_nominal_geometry(roi_cfg, rng, geometry_ranges=geometry_ranges)
    defect = sample_defect_params(str(defect_type), rng, tolerances=tolerances)

    rel_image_path = str(out_path.relative_to(project_root))
    job = {
        "image_path": rel_image_path,
        "seed": int(_stable_int_seed(str(seed_base), profile_id, defect_type, "blender_3d", "job_seed")),
        "mm_per_px": float(mm_per_px),
        "roi_width_px": int(roi_w),
        "roi_height_px": int(roi_h),
        "footprint": str(footprint),
        "component_height_mm": float(component_height_mm),
        "nominal": nominal,
        "defect": defect,
        "render_3d": render_3d,
    }
    return job, nominal, defect


def _overlay_in_place(path: Path, defect: Dict[str, Any], nominal: Dict[str, Any]) -> None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read rendered image: {path}")
    out = draw_defect_overlay(img, defect, nominal)
    ok = cv2.imwrite(str(path), out)
    if not ok:
        raise RuntimeError(f"Failed to write overlay image: {path}")

import re
def generate_slug(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r'[^\w\s-]', '', s) # Remove all non-word chars (except whitespace and hyphen)
    s = re.sub(r'[\s_-]+', '-', s) # Replace all whitespace and underscore with a single hyphen
    return s.strip('-') # Remove leading/trailing hyphens


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SAMPLE_GALLERY.md images from real 2D/3D renderers.")
    parser.add_argument("--profiles-dir", default="configs/profiles", help="Directory with component profile YAMLs")
    parser.add_argument("--configs-dir", default="configs", help="Directory with run_*.yaml configs (for per-profile settings)")
    parser.add_argument("--out-images-dir", default="images/samples", help="Output directory for gallery images")
    parser.add_argument("--out-md", default="SAMPLE_GALLERY.md", help="Output markdown file path")

    parser.add_argument("--seed", type=int, default=2026, help="Base seed for deterministic gallery images")
    parser.add_argument("--roi", default="", help="Override ROI as WIDTHxHEIGHT@MM_PER_PX, e.g. 256x256@0.01")

    parser.add_argument("--blender", default="", help="Override blender executable for 3D profiles")
    parser.add_argument("--samples", type=int, default=0, help="Override blender samples for 3D profiles")
    parser.add_argument("--device", default="", help="Override blender device (CPU/GPU) for 3D profiles")

    parser.add_argument("--skip-2d", action="store_true", help="Do not generate 2D gallery images")
    parser.add_argument("--skip-3d", action="store_true", help="Do not generate 3D gallery images")
    parser.add_argument("--dry-run", action="store_true", help="Only write SAMPLE_GALLERY.md entries, do not render images")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    profiles_dir = (project_root / args.profiles_dir).resolve()
    configs_dir = (project_root / args.configs_dir).resolve()
    gallery_dir = (project_root / args.out_images_dir).resolve()
    gallery_md = (project_root / args.out_md).resolve()

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

    # Optional ROI override parsing
    roi_override: Optional[Tuple[int, int, float]] = None
    if args.roi:
        import re
        m = re.match(r"^\\s*(\\d+)x(\\d+)@([0-9]*\\.?[0-9]+)\\s*$", str(args.roi))
        if not m:
            raise SystemExit(f"Invalid --roi format: {args.roi!r} (expected WIDTHxHEIGHT@MM_PER_PX)")
        roi_override = (int(m.group(1)), int(m.group(2)), float(m.group(3)))

    # Stage 3D jobs and run Blender in batches grouped by (exe, samples, device).
    # We can't mix sample counts in a single Blender invocation because render_batch.py
    # takes samples via a single CLI flag.
    staged_3d_groups: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    for profile_path in profiles:
        profile_data = read_profile(profile_path)
        if not profile_data:
            continue

        profile_id = get_profile_id(profile_data)
        render_backend = get_render_backend(profile_data)

        print(f"\nProcessing profile: {profile_id} [{render_backend}]")

        profile_gallery = gallery_dir / profile_id
        profile_gallery.mkdir(parents=True, exist_ok=True)

        # Resolve render settings from a matching run_*.yaml if available.
        cfg, cfg_path = _find_matching_run_cfg(configs_dir=configs_dir, profile_id=profile_id, backend=render_backend)
        if cfg is None:
            # Reasonable defaults for gallery generation; explicit per-profile run configs are preferred.
            cfg = {
                "roi": {"width_px": 256, "height_px": 256, "mm_per_px": 0.01},
                "render": {},
            }
            if render_backend == "opencv_2d":
                cfg["render"] = {
                    "backend": "opencv_2d",
                    "substrate_color": [40, 90, 40],
                    "copper_color": [60, 120, 180],
                    "solder_mask_alpha": 0.3,
                }
            else:
                cfg["render"] = {
                    "backend": "blender_3d",
                    "blender": {"executable": "blender", "engine": "CYCLES", "samples": 64, "device": "CPU"},
                }

        # Apply ROI override if requested.
        if roi_override is not None:
            w, h, mpp = roi_override
            cfg = dict(cfg)
            cfg["roi"] = {"width_px": int(w), "height_px": int(h), "mm_per_px": float(mpp)}

        defect_types = _profile_defect_set(profile_data)

        if render_backend == "opencv_2d":
            if args.skip_2d:
                print("  [skip] 2D generation disabled via --skip-2d")
                continue
            for class_name in defect_types:
                dst = profile_gallery / f"{class_name}.png"
                if not args.dry_run:
                    _render_one_2d(
                        profile_id=profile_id,
                        profile=profile_data,
                        cfg=cfg,
                        out_path=dst,
                        defect_type=class_name,
                        seed_base=int(args.seed),
                    )
                rel = dst.relative_to(project_root)
                entries.append((profile_id, profile_id, class_name, str(rel), render_backend))
                print(f"  {class_name:12} [{render_backend}]: -> {rel}" + ("" if cfg_path is None else f" (cfg: {cfg_path.name})"))

        elif render_backend == "blender_3d":
            if args.skip_3d:
                print("  [skip] 3D generation disabled via --skip-3d")
                continue

            blender_cfg = (cfg.get("render") or {}).get("blender") or {}
            this_exe = str(args.blender or blender_cfg.get("executable", "blender"))
            this_samples = int(args.samples or blender_cfg.get("samples", 64))
            this_device = str(args.device or blender_cfg.get("device", "CPU"))
            group_key = (this_exe, this_samples, this_device)
            if group_key not in staged_3d_groups:
                staged_3d_groups[group_key] = {"jobs": [], "overlays": []}

            for class_name in defect_types:
                dst = profile_gallery / f"{class_name}.png"
                if not args.dry_run:
                    job, nominal, defect = _build_3d_job(
                        project_root=project_root,
                        profile_id=profile_id,
                        profile=profile_data,
                        cfg=cfg,
                        out_path=dst,
                        defect_type=class_name,
                        seed_base=int(args.seed),
                    )
                    staged_3d_groups[group_key]["jobs"].append(job)
                    staged_3d_groups[group_key]["overlays"].append((dst, defect, nominal))
                rel = dst.relative_to(project_root)
                entries.append((profile_id, profile_id, class_name, str(rel), render_backend))
                print(f"  {class_name:12} [{render_backend}]: -> {rel}" + ("" if cfg_path is None else f" (cfg: {cfg_path.name})"))
        else:
            print(f"  [skip] Unsupported render backend: {render_backend}")
            continue

    # Render all 3D jobs (grouped) via Blender, then apply overlays.
    if staged_3d_groups and not args.dry_run:
        out_jobs_dir = (project_root / "outputs").resolve()
        out_jobs_dir.mkdir(parents=True, exist_ok=True)

        for gi, (group_key, group) in enumerate(sorted(staged_3d_groups.items(), key=lambda kv: kv[0]), 1):
            exe, samples, device = group_key
            jobs: List[Dict[str, Any]] = group["jobs"]
            overlays: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]] = group["overlays"]
            if not jobs:
                continue

            jobs_path = (out_jobs_dir / f"sample_gallery_jobs_{gi:02d}.jsonl").resolve()
            write_jobs_jsonl(jobs_path, jobs)

            print(f"\nRendering {len(jobs)} 3D sample(s) via Blender (exe={exe!r} samples={samples} device={device!r})...")
            try:
                render_blender_batch(
                    sim_root=project_root,
                    jobs_path=jobs_path,
                    output_root=project_root,
                    blender_executable=str(exe),
                    cycles_samples=int(samples),
                    device=str(device),
                )
            except Exception as e:
                raise SystemExit(
                    "Blender batch render failed.\n"
                    f"- blender executable: {exe!r}\n"
                    f"- samples: {samples}\n"
                    f"- device: {device!r}\n"
                    f"- jobs: {jobs_path}\n"
                    f"- error: {e}"
                )

            print("Applying overlays to 3D renders...")
            for path, defect, nominal in overlays:
                _overlay_in_place(path, defect, nominal)

    # Write SAMPLE_GALLERY.md
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_lines: list[str] = []
    header_lines.append("# Sample Gallery")
    header_lines.append("")
    header_lines.append(f"Auto-generated reference images for all component profiles (2D and 3D). Last updated: {now}")
    header_lines.append("")
    header_lines.append("Re-generate by running:")
    header_lines.append("```bash")
    header_lines.append(".venv/bin/python tools/update_sample_gallery.py")
    header_lines.append("```")
    header_lines.append("")

    # Group entries by render backend and then by profile_id
    profiles_by_backend: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for profile_id, _, class_name, rel_path, render_backend in entries:
        profiles_by_backend[render_backend][profile_id].append((class_name, rel_path))

    toc_lines: list[str] = []
    content_lines: list[str] = []

    # Generate 2D Profiles Section
    if "opencv_2d" in profiles_by_backend:
        toc_lines.append(f"- [2D Profiles](#{generate_slug('2D Profiles')})")
        content_lines.append("## 2D Profiles")
        content_lines.append("")
        for profile_id in sorted(profiles_by_backend["opencv_2d"].keys()):
            toc_lines.append(f"  - [{profile_id}](#{generate_slug(profile_id)})")
            content_lines.append(f"### {profile_id}")
            content_lines.append("")
            content_lines.append("| Class | Sample |")
            content_lines.append("|-------|--------|")
            for class_name, rel_path in sorted(profiles_by_backend["opencv_2d"][profile_id], key=lambda x: x[0]):
                content_lines.append(f"| {class_name} | ![{class_name}]({rel_path}) |")
            content_lines.append("")

    # Generate 3D Profiles Section
    if "blender_3d" in profiles_by_backend:
        toc_lines.append(f"- [3D Profiles](#{generate_slug('3D Profiles')})")
        content_lines.append("## 3D Profiles")
        content_lines.append("")
        for profile_id in sorted(profiles_by_backend["blender_3d"].keys()):
            toc_lines.append(f"  - [{profile_id}](#{generate_slug(profile_id)})")
            content_lines.append(f"### {profile_id}")
            content_lines.append("")
            content_lines.append("| Class | Sample |")
            content_lines.append("|-------|--------|")
            for class_name, rel_path in sorted(profiles_by_backend["blender_3d"][profile_id], key=lambda x: x[0]):
                content_lines.append(f"| {class_name} | ![{class_name}]({rel_path}) |")
            content_lines.append("")

    final_lines = header_lines + ["## Table of Contents", ""] + toc_lines + [""] + content_lines
    gallery_md.write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    print(f"\n✓ Gallery written to {gallery_md}")
    print(f"✓ Images in {gallery_dir}")
    print(f"✓ Total samples: {len(entries)}")


if __name__ == "__main__":
    main()
