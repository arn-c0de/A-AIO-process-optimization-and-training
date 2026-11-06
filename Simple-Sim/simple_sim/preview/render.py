"""Per-profile debug preview rendering (OpenCV 2D and Blender 3D), tk-free.

One image per defect state is written to ``<out>/previews/<profile_id>/<backend>/<STATE>.png``
plus an ``<out>/index.html`` gallery. Used by ``scripts/render_debug_previews.py`` and its
Tk viewer (:mod:`simple_sim.preview.viewer_tk`).
"""

from __future__ import annotations

import hashlib
import html
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from simple_sim.config import load_config, merge_profile_render, validate_config
from simple_sim.defects import sample_defect_params
from simple_sim.generation.blender import blender_job, blender_settings, run_blender_jobs
from simple_sim.generators.filter_settings import normalize_image_filters
from simple_sim.generators.opencv2d.geometry import sample_nominal_geometry
from simple_sim.profile_hash import load_profile

BACKEND_2D = "opencv_2d"
BACKEND_3D = "blender_3d"
BACKEND_CHOICES = [BACKEND_2D, BACKEND_3D, "both", "auto"]
DEFAULT_STATES = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]

# (profile_id, backend, defect, rel_image_path)
PreviewRec = tuple[str, str, str, str]
# (profile_id, profile dict, supported backends)
ProfileEntry = tuple[str, dict[str, Any], set[str]]
RoiOverride = tuple[int, int, float]


@dataclass(frozen=True)
class RenderSettings:
    roi_width_px: int
    roi_height_px: int
    mm_per_px: float
    blender_executable: str
    cycles_samples: int
    device: str
    engine: str = "BLENDER_EEVEE"


@dataclass
class PreviewOptions:
    """Render inputs shared by the CLI run and the Tk viewer's re-renders."""

    backend_mode: str = "both"
    forced_cfg: dict[str, Any] | None = None
    forced_label: str = ""
    roi_override: RoiOverride | None = None
    seed_base: int = 2026
    variable_seeds: bool = False
    states: str = ""
    enable_cardinal_rotation_90: bool = True
    blender: str = ""
    samples: int = 0
    device: str = ""


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------


def parse_csv_list(s: str | None) -> list[str]:
    return [chunk.strip() for chunk in str(s or "").split(",") if chunk.strip()]


def parse_selection(selection: str, n: int) -> list[int]:
    """Parse '1,2,4-6' / 'all' (1-based) into unique 0-based indices, preserving order."""
    selection = (selection or "").strip().lower()
    if selection in {"a", "all", "*"}:
        return list(range(n))

    idxs: list[int] = []
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s.strip()), int(hi_s.strip())
            if lo <= 0 or hi <= 0:
                raise ValueError("Selection indices are 1-based and must be positive.")
            idxs.extend(k - 1 for k in range(min(lo, hi), max(lo, hi) + 1))
        else:
            k = int(part)
            if k <= 0:
                raise ValueError("Selection indices are 1-based and must be positive.")
            idxs.append(k - 1)

    final: list[int] = []
    for i in idxs:
        if i < 0 or i >= n:
            raise ValueError(f"Selection index out of range: {i + 1} (valid: 1..{n})")
        if i not in final:
            final.append(i)
    return final


def parse_roi(s: str) -> RoiOverride:
    """Parse ``WIDTHxHEIGHT@MM_PER_PX`` (e.g. ``256x256@0.01``)."""
    m = re.match(r"^\s*(\d+)x(\d+)@([0-9]*\.?[0-9]+)\s*$", str(s))
    if not m:
        raise ValueError("Invalid --roi. Expected WIDTHxHEIGHT@MM_PER_PX, e.g. 256x256@0.01")
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def sanitize_dir_name(s: str) -> str:
    """Stable, filesystem friendly directory name for a profile id."""
    return re.sub(r"[^a-zA-Z0-9._@+-]+", "_", s).strip("_") or "profile"


def preview_rel_path(profile_id: str, backend: str, defect_type: str) -> str:
    return f"previews/{sanitize_dir_name(profile_id)}/{backend}/{defect_type}.png"


def resolve_defect_types(profile: dict[str, Any], states_override: str | None) -> list[str]:
    return parse_csv_list(states_override) or list(profile.get("defect_set") or []) or list(DEFAULT_STATES)


def backends_for_mode(mode: str, supported: set[str]) -> list[str]:
    """Backends to render for *mode* ('both' or a single backend) among *supported* ones."""
    if mode == "both":
        return [b for b in (BACKEND_2D, BACKEND_3D) if b in supported]
    return [mode] if mode in supported else []


def new_run_id(variable_seeds: bool) -> int | None:
    """Per-render salt for variable seeds (``None`` = deterministic)."""
    return int(time.time() * 1000000) if variable_seeds else None


# ---------------------------------------------------------------------------
# Profiles and configs
# ---------------------------------------------------------------------------


def profile_backends(profile: dict[str, Any]) -> set[str]:
    b = (profile.get("profile") or {}).get("supported_render_backends") or []
    return {str(x) for x in b if str(x) in {BACKEND_2D, BACKEND_3D}}


def discover_profiles(profiles_dir: Path, *, backend_mode: str = "both") -> list[ProfileEntry]:
    """Profiles under *profiles_dir* supporting 2D and/or 3D (filtered by a single backend mode)."""
    out: list[ProfileEntry] = []
    for p in sorted(Path(profiles_dir).glob("*.yaml")):
        if p.name.startswith("."):
            continue
        try:
            prof = load_profile(p.stem, profiles_dir)
        except Exception:  # noqa: BLE001 - a broken profile must not hide the others
            continue
        backends = profile_backends(prof)
        if not backends or (backend_mode in {BACKEND_2D, BACKEND_3D} and backend_mode not in backends):
            continue
        out.append((p.stem, prof, backends))
    out.sort(key=lambda t: t[0])
    return out


def find_matching_run_configs(configs_dir: Path, *, profile_id: str, backend: str) -> list[Path]:
    hits: list[Path] = []
    for p in sorted(Path(configs_dir).glob("*.yaml")):
        try:
            cfg = load_config(p)
        except Exception:  # noqa: BLE001 - skip unrelated/invalid YAML files
            continue
        if str((cfg.get("render") or {}).get("backend", "")) != str(backend):
            continue
        if str((cfg.get("run") or {}).get("component_profile", "")) == str(profile_id):
            hits.append(p)
    return hits


def auto_select_profiles_with_missing_previews(
    discovered: list[ProfileEntry],
    *,
    profiles_dir: Path,
    out_root: Path,
    backend_mode: str,
    states_override: str,
) -> list[str]:
    """Profile IDs with at least one missing or outdated (older than the profile YAML) preview image."""
    selected: list[str] = []
    for pid, prof, backends in discovered:
        defect_types = resolve_defect_types(prof, states_override)
        profile_path = (Path(profiles_dir) / f"{pid}.yaml").resolve()
        profile_mtime = profile_path.stat().st_mtime if profile_path.exists() else 0.0
        for backend in backends_for_mode(backend_mode, backends):
            imgs = [(Path(out_root) / preview_rel_path(pid, backend, str(d))).resolve() for d in defect_types]
            if any(not p.exists() or profile_mtime > p.stat().st_mtime for p in imgs):
                selected.append(pid)
                break
    return selected


def settings_from_config(cfg: dict[str, Any]) -> RenderSettings:
    validate_config(cfg)
    roi = cfg["roi"]
    b = blender_settings(cfg)
    return RenderSettings(
        roi_width_px=int(roi["width_px"]),
        roi_height_px=int(roi["height_px"]),
        mm_per_px=float(roi["mm_per_px"]),
        blender_executable=b["executable"],
        cycles_samples=b["samples"],
        device=b["device"],
        engine=b["engine"],
    )


def load_preview_config(
    *,
    forced_cfg: dict[str, Any] | None,
    sim_root: Path,
    configs_dir: Path,
    profile_id: str,
    backend: str,
    forced_label: str = "",
) -> tuple[dict[str, Any], str]:
    """Return ``(config, source description)`` for one profile/backend.

    A forced config (--config) is used only for its own backend; other backends
    (e.g. with --backend=both) fall back to per-profile auto matching, then to the
    default run config of that backend.
    """
    if forced_cfg is not None and str((forced_cfg.get("render") or {}).get("backend", "")) == backend:
        validate_config(forced_cfg)
        return forced_cfg, f"(forced) {forced_label}".rstrip()

    matches = find_matching_run_configs(configs_dir, profile_id=profile_id, backend=backend)
    if matches:
        cfg = load_config(matches[0])
        try:
            rel = str(matches[0].relative_to(sim_root))
        except ValueError:
            rel = str(matches[0])
        cfg_src = f"(auto) {rel}"
    else:
        fallback = "configs/run_0001.yaml" if backend == BACKEND_2D else "configs/run_0001_3d.yaml"
        cfg = load_config(Path(sim_root) / fallback)
        cfg_src = f"(fallback) {fallback}"
    validate_config(cfg)
    return cfg, cfg_src


def apply_3d_overrides(settings: RenderSettings, opts: PreviewOptions) -> RenderSettings:
    if opts.roi_override is not None:
        w, h, mpp = opts.roi_override
        settings = replace(settings, roi_width_px=int(w), roi_height_px=int(h), mm_per_px=float(mpp))
    if opts.blender:
        settings = replace(settings, blender_executable=str(opts.blender))
    if opts.samples and int(opts.samples) > 0:
        settings = replace(settings, cycles_samples=int(opts.samples))
    if opts.device:
        settings = replace(settings, device=str(opts.device))
    return settings


def resolve_2d_roi(cfg: dict[str, Any], roi_override: RoiOverride | None) -> RoiOverride:
    if roi_override is not None:
        return int(roi_override[0]), int(roi_override[1]), float(roi_override[2])
    roi = cfg["roi"]
    return int(roi["width_px"]), int(roi["height_px"]), float(roi["mm_per_px"])


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def rotation_jitter_range(cfg: dict[str, Any] | None) -> tuple[float, float]:
    """``augment.rotation_deg_range`` from config as (lo, hi), fallback ``(0, 0)``."""
    rr = ((cfg or {}).get("augment") or {}).get("rotation_deg_range", [0.0, 0.0])
    if not isinstance(rr, (list, tuple)) or len(rr) != 2:
        return (0.0, 0.0)
    lo, hi = float(rr[0]), float(rr[1])
    return (hi, lo) if lo > hi else (lo, hi)


def sample_preview_augment(
    rng: np.random.Generator, *, rotation_range: tuple[float, float], enable_cardinal_rotation_90: bool = True
) -> dict[str, float]:
    """Preview augment: clean image, but randomized global orientation like dataset generation."""
    base_orientation_deg = float(rng.choice([0.0, 90.0, 180.0, 270.0])) if enable_cardinal_rotation_90 else 0.0
    rotation_jitter_deg = float(rng.uniform(*rotation_range))
    return {
        "blur_sigma": 0.0,
        "noise_stddev": 0.0,
        "brightness_factor": 1.0,
        "contrast_factor": 1.0,
        "rotation_deg": float(base_orientation_deg + rotation_jitter_deg),
    }


def apply_filter_overrides_to_augment(augment: dict[str, float], image_filters: dict[str, Any] | None) -> dict[str, float]:
    """Apply image-filter overrides; enabled filters get a visible baseline when the value is neutral."""
    from simple_sim.generators.opencv_2d import apply_image_filter_overrides

    f = normalize_image_filters(image_filters or {})
    a = dict(augment or {})
    if f.get("enable_blur", True) and float(a.get("blur_sigma", 0.0) or 0.0) <= 1e-6:
        a["blur_sigma"] = 1.0
    if f.get("enable_grain", True) and float(a.get("noise_stddev", 0.0) or 0.0) <= 1e-6:
        a["noise_stddev"] = 8.0
    if f.get("enable_brightness", True) and abs(float(a.get("brightness_factor", 1.0) or 1.0) - 1.0) <= 1e-6:
        a["brightness_factor"] = 1.15
    if f.get("enable_contrast", True) and abs(float(a.get("contrast_factor", 1.0) or 1.0) - 1.0) <= 1e-6:
        a["contrast_factor"] = 1.20
    if f.get("enable_rotation", True) and abs(float(a.get("rotation_deg", 0.0) or 0.0)) <= 1e-6:
        a["rotation_deg"] = 15.0
    return apply_image_filter_overrides(a, f)


def _preview_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12], 16)


def _seed_parts(seed_base: int, profile_id: str, backend: str | None, defect_type: str, run_id: int | None) -> list[object]:
    # 2D seeds include the backend, 3D seeds do not (kept for reproducibility of existing previews).
    parts: list[object] = [int(seed_base), profile_id]
    if backend is not None:
        parts.append(backend)
    parts.append(str(defect_type))
    if run_id is not None:
        parts.append(int(run_id))
    return parts


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_2d_previews(
    *,
    profile_id: str,
    profile: dict[str, Any],
    cfg: dict[str, Any],
    out_root: Path,
    seed_base: int,
    defect_types: Sequence[str],
    run_id: int | None,
    roi_override: RoiOverride | None,
    image_filters: dict[str, Any] | None,
    enable_cardinal_rotation_90: bool,
) -> list[PreviewRec]:
    """Render one OpenCV 2D preview per defect state; return preview records."""
    from simple_sim.dataset_store import write_image
    from simple_sim.generators.opencv_2d import render_roi

    w, h, mpp = resolve_2d_roi(cfg, roi_override)
    roi_cfg = {"width_px": w, "height_px": h, "mm_per_px": mpp}
    render_cfg = merge_profile_render(cfg["render"], profile)
    footprint = str((profile.get("component") or {}).get("footprint", "chip_2pad"))
    geometry_ranges = profile.get("geometry_ranges") or {}
    tolerances = profile.get("tolerances")
    rot_range = rotation_jitter_range(cfg)

    recs: list[PreviewRec] = []
    for defect_type in map(str, defect_types):
        rng = np.random.default_rng(_preview_seed(*_seed_parts(seed_base, profile_id, BACKEND_2D, defect_type, run_id)))
        augment = sample_preview_augment(rng, rotation_range=rot_range, enable_cardinal_rotation_90=bool(enable_cardinal_rotation_90))
        augment = apply_filter_overrides_to_augment(augment, image_filters)
        nominal = sample_nominal_geometry(roi_cfg, rng, geometry_ranges=geometry_ranges)
        defect = sample_defect_params(defect_type, rng, tolerances=tolerances)
        img = render_roi(
            nominal=nominal,
            defect_params=defect,
            augment=augment,
            roi_size=(w, h),
            config=render_cfg,
            tolerances=tolerances,
            rng=rng,
            footprint=footprint,
        )
        rel_path = preview_rel_path(profile_id, BACKEND_2D, defect_type)
        write_image((Path(out_root) / rel_path).resolve(), img)
        recs.append((profile_id, BACKEND_2D, defect_type, rel_path))
    return recs


def build_preview_jobs(
    *,
    profile_id: str,
    profile: dict[str, Any],
    settings: RenderSettings,
    seed_base: int,
    defect_types: Sequence[str],
    run_id: int | None = None,
    backend: str = BACKEND_3D,
    rotation_range: tuple[float, float] = (0.0, 0.0),
    enable_cardinal_rotation_90: bool = True,
    image_filters: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[PreviewRec]]:
    """Return ``(blender_jobs, preview_records)`` for one profile (one job per defect state).

    With *run_id*, seeds vary between runs (like the real 3D pipeline).
    """
    component = profile.get("component") or {}
    footprint = str(component.get("footprint", "chip_2pad"))
    geometry_ranges = profile.get("geometry_ranges") or {}
    tolerances = profile.get("tolerances") or {}
    component_height_mm = float((component.get("nominal_dims_mm") or {}).get("height", 0.45) or 0.45)
    render_3d = profile.get("render_3d") or {}
    roi_cfg = {"width_px": int(settings.roi_width_px), "height_px": int(settings.roi_height_px), "mm_per_px": float(settings.mm_per_px)}

    jobs: list[dict[str, Any]] = []
    previews: list[PreviewRec] = []
    for defect_type in map(str, defect_types):
        seed = _preview_seed(*_seed_parts(seed_base, profile_id, None, defect_type, run_id))
        rng = np.random.default_rng(seed)
        nominal = sample_nominal_geometry(roi_cfg, rng, geometry_ranges=geometry_ranges)
        defect = sample_defect_params(defect_type, rng, tolerances=tolerances)
        augment = sample_preview_augment(rng, rotation_range=rotation_range, enable_cardinal_rotation_90=enable_cardinal_rotation_90)
        augment = apply_filter_overrides_to_augment(augment, image_filters)
        rel_path = preview_rel_path(profile_id, backend, defect_type)
        record = {"image_path": rel_path, "seed": seed, "nominal": nominal, "defect": defect, "augment": augment}
        jobs.append(
            blender_job(
                record,
                mm_per_px=settings.mm_per_px,
                roi_width=settings.roi_width_px,
                roi_height=settings.roi_height_px,
                footprint=footprint,
                component_height_mm=component_height_mm,
                render_3d=render_3d,
            )
        )
        previews.append((profile_id, backend, defect_type, rel_path))
    return jobs, previews


def _settings_dict(s: RenderSettings) -> dict[str, Any]:
    return {"executable": s.blender_executable, "samples": s.cycles_samples, "device": s.device, "engine": s.engine}


def render_profile_backend(
    pid: str,
    prof: dict[str, Any],
    backend: str,
    defect_types: list[str],
    *,
    opts: PreviewOptions,
    sim_root: Path,
    configs_dir: Path,
    out_root: Path,
    seed_base: int,
    run_id: int | None,
    image_filters: dict[str, Any] | None,
    enable_cardinal_rotation_90: bool,
    jobs_path: Path,
) -> list[PreviewRec]:
    """Render one profile/backend immediately (3D: its own Blender batch). Used by the viewer."""
    cfg, _src = load_preview_config(
        forced_cfg=opts.forced_cfg, sim_root=sim_root, configs_dir=configs_dir, profile_id=pid, backend=backend
    )
    if backend == BACKEND_3D:
        settings = apply_3d_overrides(settings_from_config(cfg), opts)
        jobs, recs = build_preview_jobs(
            profile_id=pid,
            profile=prof,
            settings=settings,
            seed_base=seed_base,
            defect_types=defect_types,
            run_id=run_id,
            rotation_range=rotation_jitter_range(cfg),
            enable_cardinal_rotation_90=enable_cardinal_rotation_90,
            image_filters=image_filters,
        )
        run_blender_jobs(jobs, sim_root=sim_root, output_root=out_root, jobs_path=jobs_path, settings=_settings_dict(settings))
        return recs
    return render_2d_previews(
        profile_id=pid,
        profile=prof,
        cfg=cfg,
        out_root=out_root,
        seed_base=seed_base,
        defect_types=defect_types,
        run_id=run_id,
        roi_override=opts.roi_override,
        image_filters=image_filters,
        enable_cardinal_rotation_90=enable_cardinal_rotation_90,
    )


@dataclass
class PreviewRun:
    previews: list[PreviewRec]
    index_path: Path
    jobs_path: Path | None


def render_previews(
    entries: list[ProfileEntry],
    *,
    opts: PreviewOptions,
    sim_root: Path,
    configs_dir: Path,
    out_root: Path,
    image_filters: dict[str, Any] | None,
    dry_run: bool = False,
) -> PreviewRun:
    """Render previews for the selected profiles and write ``index.html``.

    2D previews are rendered directly; all 3D jobs are batched into one Blender run
    (``blender_jobs_previews.jsonl``). With *dry_run*, only the index and jobs file are written.

    Raises:
        ValueError: if a profile does not support the requested backend mode.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    previews: list[PreviewRec] = []
    all_jobs: list[dict[str, Any]] = []
    per_profile_settings: list[RenderSettings] = []

    for pid, prof, backends in entries:
        defect_types = resolve_defect_types(prof, opts.states)
        run_id = new_run_id(opts.variable_seeds)
        backends_to_render = backends_for_mode(opts.backend_mode, backends)
        if not backends_to_render:
            raise ValueError(f"Profile {pid!r} does not support backend={opts.backend_mode!r}")

        for backend in backends_to_render:
            # A forced --config applies to its own backend; others are auto-matched per profile.
            cfg, cfg_src = load_preview_config(
                forced_cfg=opts.forced_cfg,
                sim_root=sim_root,
                configs_dir=configs_dir,
                profile_id=pid,
                backend=backend,
                forced_label=opts.forced_label,
            )
            print(f"\n[preview] Profile:  {pid}")
            print(f"[preview] Backend:  {backend}")
            print(f"[preview] Config:    {cfg_src}")

            if backend == BACKEND_2D:
                w, h, mpp = resolve_2d_roi(cfg, opts.roi_override)
                print(f"[preview] ROI:       {w}x{h} @ {mpp} mm/px")
                print(f"[preview] States:    {defect_types}")
                if dry_run:
                    previews.extend((pid, backend, str(d), preview_rel_path(pid, backend, str(d))) for d in defect_types)
                    continue
                previews.extend(
                    render_2d_previews(
                        profile_id=pid,
                        profile=prof,
                        cfg=cfg,
                        out_root=out_root,
                        seed_base=opts.seed_base,
                        defect_types=defect_types,
                        run_id=run_id,
                        roi_override=opts.roi_override,
                        image_filters=image_filters,
                        enable_cardinal_rotation_90=opts.enable_cardinal_rotation_90,
                    )
                )
            else:
                settings = apply_3d_overrides(settings_from_config(cfg), opts)
                print(f"[preview] ROI:       {settings.roi_width_px}x{settings.roi_height_px} @ {settings.mm_per_px} mm/px")
                print(
                    f"[preview] Blender:   {settings.blender_executable} "
                    f"(samples={settings.cycles_samples}, device={settings.device}, engine={settings.engine})"
                )
                print(f"[preview] States:    {defect_types}")
                per_profile_settings.append(settings)
                jobs, recs = build_preview_jobs(
                    profile_id=pid,
                    profile=prof,
                    settings=settings,
                    seed_base=opts.seed_base,
                    defect_types=defect_types,
                    run_id=run_id,
                    backend=backend,
                    rotation_range=rotation_jitter_range(cfg),
                    enable_cardinal_rotation_90=opts.enable_cardinal_rotation_90,
                    image_filters=image_filters,
                )
                all_jobs.extend(jobs)
                previews.extend(recs)

    jobs_path = out_root / "blender_jobs_previews.jsonl"
    index_path = write_index_html(out_root, previews)
    if dry_run:
        if all_jobs:
            from simple_sim.generators.blender_3d import write_jobs_jsonl

            write_jobs_jsonl(jobs_path, all_jobs)
            print(f"\n[dry-run] Wrote jobs:  {jobs_path}")
        print(f"[dry-run] Wrote index: {out_root / 'index.html'}")
    elif all_jobs:
        print(f"\n[render] Running Blender batch for {len(all_jobs)} previews...")
        run_blender_jobs(
            all_jobs,
            sim_root=sim_root,
            output_root=out_root,
            jobs_path=jobs_path,
            settings=_merge_blender_settings(per_profile_settings, opts),
        )
    if not dry_run:
        print(f"[render] Done. Open: {index_path.resolve()}")
    return PreviewRun(previews=previews, index_path=index_path, jobs_path=jobs_path if all_jobs else None)


def _merge_blender_settings(per_profile: list[RenderSettings], opts: PreviewOptions) -> dict[str, Any]:
    """One Blender setting set for a batch across profiles (CLI overrides win; warn on conflicts)."""
    exes = sorted({s.blender_executable for s in per_profile} or {"blender"})
    devices = sorted({s.device for s in per_profile} or {"CPU"})
    samples_list = sorted({int(s.cycles_samples) for s in per_profile} or {64})
    engines = sorted({s.engine for s in per_profile} or {"BLENDER_EEVEE"})
    if len(exes) > 1 and not opts.blender:
        print(f"\n[warn] Multiple blender executables across auto configs: {exes} (using: {exes[-1]})")
    if len(devices) > 1 and not opts.device:
        print(f"[warn] Multiple devices across auto configs: {devices} (using: {devices[-1]})")
    if len(samples_list) > 1 and not opts.samples:
        print(f"[warn] Multiple sample counts across auto configs: {samples_list} (using max: {max(samples_list)})")
    if len(engines) > 1:
        print(f"[warn] Multiple render engines across auto configs: {engines} (using: {engines[-1]})")
    return {
        "executable": str(opts.blender or exes[-1]),
        "device": str(opts.device or devices[-1]),
        "samples": int(opts.samples or max(samples_list)),
        "engine": str(engines[-1]),
    }


# ---------------------------------------------------------------------------
# Index / existing previews
# ---------------------------------------------------------------------------

_INDEX_CSS = (
    "body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial; margin:24px;}",
    "h1{font-size:20px;margin:0 0 16px 0}",
    "h2{font-size:16px;margin:20px 0 8px 0}",
    ".grid{display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px;}",
    ".card{border:1px solid #ddd; border-radius:10px; padding:10px;}",
    ".cap{font-size:12px;color:#333;margin:0 0 8px 0; display:flex; gap:6px; flex-wrap:wrap; align-items:baseline;}",
    ".pill{font-size:11px; padding:2px 6px; border-radius:999px; border:1px solid #ddd; background:#fafafa;}",
    ".pill.backend{border-color:#ddd; background:#fff;}",
    ".pill.state{border-color:#cfd8ff; background:#f5f7ff;}",
    "img{width:100%; height:auto; border-radius:8px; background:#f6f6f6;}",
)


def write_index_html(out_root: Path, previews: list[PreviewRec]) -> Path:
    """Write ``<out_root>/index.html`` (grouped by profile) and return its path."""
    out_root = Path(out_root)
    index_path = out_root / "index.html"
    by_profile: dict[str, list[tuple[str, str, str]]] = {}
    for pid, backend, defect, rel_path in previews:
        by_profile.setdefault(pid, []).append((backend, defect, rel_path))

    def esc(s: str) -> str:
        return html.escape(str(s), quote=True).replace("&#x27;", "&#39;")

    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Simple-Sim Debug Previews</title>",
        "<style>",
        *_INDEX_CSS,
        "</style></head><body>",
        "<h1>Simple-Sim Debug Previews</h1>",
    ]
    for pid in sorted(by_profile):
        lines.append(f"<h2>{esc(pid)}</h2>")
        lines.append("<div class='grid'>")
        for backend, defect, rel_path in sorted(by_profile[pid], key=lambda t: (t[0], t[1])):
            lines.append("<div class='card'>")
            lines.append("<div class='cap'>")
            lines.append(f"<span class='pill'>{esc(pid)}</span>")
            lines.append(f"<span class='pill backend'>{esc(backend)}</span>")
            lines.append(f"<span class='pill state'>{esc(defect)}</span>")
            lines.append("</div>")
            lines.append(f"<a href='{esc(rel_path)}'><img loading='lazy' src='{esc(rel_path)}'></a>")
            lines.append("</div>")
        lines.append("</div>")
    lines.append("</body></html>")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def collect_existing_previews(out_root: Path) -> list[PreviewRec]:
    """Preview records for existing ``previews/<profile_id>/<backend>/<defect>.png`` files."""
    out_root = Path(out_root)
    recs: list[PreviewRec] = []
    for img_path in sorted((out_root / "previews").glob("*/*/*.png")):
        rel = img_path.relative_to(out_root)
        _, pid, backend, fname = rel.parts
        recs.append((pid, backend, Path(fname).stem, rel.as_posix()))
    return recs
