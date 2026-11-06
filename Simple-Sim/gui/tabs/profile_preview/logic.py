"""Profile preview rendering for the GUI (tk-free).

Renders a few images per defect state of one component profile with user-chosen render
settings (style, supersampling, camera preset, Blender quality) without generating a dataset.
Images go to ``outputs/gui/profile_preview/<profile>/<backend>/<STATE>_v<N>.png``.
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from simple_sim.config import load_config, merge_profile_render, validate_config
from simple_sim.defects import sample_defect_params
from simple_sim.generators.opencv2d.geometry import sample_nominal_geometry
from simple_sim.generators.opencv2d.photoreal import CAMERA_PRESETS, DEFAULT_SUPERSAMPLING, SUPERSAMPLING_CHOICES
from simple_sim.parallel import imap_ordered, render_workers
from simple_sim.preview.render import (
    BACKEND_2D,
    BACKEND_3D,
    ProfileEntry,
    _preview_seed,
    apply_filter_overrides_to_augment,
    discover_profiles,
    find_matching_run_configs,
    resolve_defect_types,
    rotation_jitter_range,
    sample_preview_augment,
    sanitize_dir_name,
)

# Quality presets: photoreal supersampling (2D render and the 3D board bake) and Cycles samples (3D).
QUALITY_PRESETS: dict[str, dict[str, int]] = {
    "Draft": {"supersampling": 1, "samples": 16},
    "Standard": {"supersampling": DEFAULT_SUPERSAMPLING, "samples": 64},
    "Ultra": {"supersampling": max(SUPERSAMPLING_CHOICES), "samples": 256},
}
CAMERA_FROM_CONFIG = "from config"
CAMERA_CHOICES = [CAMERA_FROM_CONFIG, *sorted(CAMERA_PRESETS)]
STYLE_CHOICES = ["photoreal", "flat"]
CONFIG_AUTO = "(auto)"


@dataclass
class PreviewRequest:
    profile_id: str
    backend: str
    config_name: str = CONFIG_AUTO  # file name under configs/ or CONFIG_AUTO
    states: list[str] = field(default_factory=list)
    variants: int = 1
    seed: int = 2026
    style: str = "photoreal"
    supersampling: int = DEFAULT_SUPERSAMPLING
    camera: str = CAMERA_FROM_CONFIG
    use_filters: bool = False  # shared GUI image filters (blur, grain, ...) on top of the camera model
    image_filters: dict[str, Any] | None = None
    cardinal_rotation: bool = True
    blender_samples: int = 64
    blender_engine: str = ""  # "" = from config
    blender_device: str = ""  # "" = from config


@dataclass
class PreviewImage:
    state: str
    variant: int
    path: Path
    seconds: float
    info: str = ""
    defect: dict[str, Any] | None = None
    nominal: dict[str, Any] | None = None


def write_sidecar(im: PreviewImage) -> None:
    """``<image>.json`` with the sampled defect/nominal (for the overlay when reloading old previews)."""
    im.path.with_suffix(".json").write_text(
        json.dumps({"defect": im.defect, "nominal": im.nominal, "info": im.info}, default=float), encoding="utf-8")


def read_sidecar(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


class PreviewCancelled(Exception):
    pass


class ProfilePreviewLogic:
    def __init__(self, sim_root: Path) -> None:
        self.sim_root = Path(sim_root)
        self.profiles_dir = self.sim_root / "configs" / "profiles"
        self.configs_dir = self.sim_root / "configs"
        self.out_root = self.sim_root / "outputs" / "gui" / "profile_preview"

    # ------------------------------------------------------------------ discovery

    def profiles(self) -> list[ProfileEntry]:
        return discover_profiles(self.profiles_dir, backend_mode="both")

    def configs_for(self, profile_id: str, backend: str) -> list[str]:
        """Run configs for *backend*: those naming *profile_id* first, then the others."""
        matching = [p.name for p in find_matching_run_configs(self.configs_dir, profile_id=profile_id, backend=backend)]
        others = []
        for p in sorted(self.configs_dir.glob("*.yaml")):
            if p.name in matching:
                continue
            try:
                cfg = load_config(p)
            except Exception:  # noqa: BLE001 - unrelated YAML in configs/
                continue
            if str((cfg.get("render") or {}).get("backend", "")) == backend:
                others.append(p.name)
        return matching + others

    def resolve_config(self, req: PreviewRequest) -> tuple[dict[str, Any], str]:
        if req.config_name and req.config_name != CONFIG_AUTO:
            name = req.config_name
        else:
            names = self.configs_for(req.profile_id, req.backend)
            name = names[0] if names else ("run_0001.yaml" if req.backend == BACKEND_2D else "run_0001_3d.yaml")
        cfg = load_config(self.configs_dir / name)
        validate_config(cfg)
        return cfg, name

    def output_dir(self, profile_id: str, backend: str) -> Path:
        return self.out_root / sanitize_dir_name(profile_id) / backend

    # ------------------------------------------------------------------ rendering

    def render(
        self,
        req: PreviewRequest,
        *,
        on_image: Callable[[PreviewImage], None],
        on_log: Callable[[str], None],
        should_stop: Callable[[], bool],
    ) -> list[PreviewImage]:
        """Render ``variants`` images per state. Called from a worker thread; callbacks run there too."""
        entry = next((e for e in self.profiles() if e[0] == req.profile_id), None)
        if entry is None:
            raise ValueError(f"Unknown profile: {req.profile_id}")
        _pid, profile, backends = entry
        if req.backend not in backends:
            raise ValueError(f"Profile {req.profile_id} does not support {req.backend}")
        cfg, cfg_name = self.resolve_config(req)
        states = list(req.states) or resolve_defect_types(profile, "")

        out_dir = self.output_dir(req.profile_id, req.backend)
        if out_dir.exists():
            shutil.rmtree(out_dir)  # only our own preview images live here
        out_dir.mkdir(parents=True, exist_ok=True)
        on_log(f"Profile {req.profile_id} | {req.backend} | config {cfg_name}")
        if req.backend == BACKEND_2D:
            return self._render_2d(req, profile, cfg, states, out_dir, on_image, on_log, should_stop)
        return self._render_3d(req, profile, cfg, states, out_dir, on_image, on_log, should_stop)

    def _camera_cfg(self, render_cfg: dict[str, Any], req: PreviewRequest) -> dict[str, Any] | None:
        if req.camera == CAMERA_FROM_CONFIG:
            return render_cfg.get("camera")
        return {"preset": req.camera}

    def _augment(self, rng: np.random.Generator, cfg: dict[str, Any], req: PreviewRequest) -> dict[str, float]:
        augment = sample_preview_augment(
            rng, rotation_range=rotation_jitter_range(cfg), enable_cardinal_rotation_90=req.cardinal_rotation
        )
        if req.use_filters:
            augment = apply_filter_overrides_to_augment(augment, req.image_filters, rng)
        return augment

    def _seed(self, req: PreviewRequest, state: str, variant: int) -> int:
        return _preview_seed(int(req.seed), req.profile_id, req.backend, state, variant)

    def _render_2d(self, req, profile, cfg, states, out_dir, on_image, on_log, should_stop) -> list[PreviewImage]:
        from simple_sim.dataset_store import write_image
        from simple_sim.generators.opencv_2d import render_roi

        roi = cfg["roi"]
        w, h = int(roi["width_px"]), int(roi["height_px"])
        render_cfg = merge_profile_render(copy.deepcopy(cfg["render"]), profile, roi)
        render_cfg["style"] = req.style
        render_cfg["supersampling"] = int(req.supersampling)
        camera = self._camera_cfg(render_cfg, req)
        if camera is None:
            render_cfg.pop("camera", None)
        else:
            render_cfg["camera"] = camera
        footprint = str((profile.get("component") or {}).get("footprint", "chip_2pad"))
        tolerances = profile.get("tolerances")
        on_log(f"ROI {w}x{h} | style {req.style} | supersampling {req.supersampling}x | camera {req.camera}")

        def render_one(item: tuple[str, int]) -> PreviewImage:
            state, v = item
            t0 = time.perf_counter()
            rng = np.random.default_rng(self._seed(req, state, v))
            augment = self._augment(rng, cfg, req)
            nominal = sample_nominal_geometry(roi, rng, geometry_ranges=profile.get("geometry_ranges") or {})
            defect = sample_defect_params(state, rng, tolerances=tolerances)
            img = render_roi(nominal=nominal, defect_params=defect, augment=augment, roi_size=(w, h),
                             config=render_cfg, tolerances=tolerances, rng=rng, footprint=footprint)
            path = out_dir / f"{state}_v{v}.png"
            write_image(path, img)
            rec = PreviewImage(state, v, path, time.perf_counter() - t0, _defect_info(defect), defect, nominal)
            write_sidecar(rec)
            return rec

        # Every image has its own seeded rng, so rendering them in parallel threads gives the same
        # images as a sequential loop; results arrive in order (stable gallery layout).
        items = [(state, v) for state in states for v in range(1, int(req.variants) + 1)]
        workers = render_workers(min(len(items), render_workers()) if items else 1)
        on_log(f"{len(items)} images | {workers} worker thread(s)")
        out: list[PreviewImage] = []
        for rec in imap_ordered(render_one, items, workers=workers, should_stop=should_stop):
            out.append(rec)
            on_image(rec)
        if len(out) < len(items):
            raise PreviewCancelled()
        return out

    def _render_3d(self, req, profile, cfg, states, out_dir, on_image, on_log, should_stop) -> list[PreviewImage]:
        from simple_sim.generation.blender import blender_job, blender_settings, run_prepared_blender_jobs
        from simple_sim.generators.blender3d.bake import bake_fields, prepare_job, profile_job_fields

        roi = cfg["roi"]
        w, h, mpp = int(roi["width_px"]), int(roi["height_px"]), float(roi["mm_per_px"])
        render_cfg = cfg.get("render") or {}
        settings = blender_settings(cfg)
        settings["samples"] = int(req.blender_samples)
        if req.blender_engine:
            settings["engine"] = req.blender_engine
        if req.blender_device:
            settings["device"] = req.blender_device
        if shutil.which(settings["executable"]) is None and not Path(settings["executable"]).exists():
            raise FileNotFoundError(f"Blender not found: {settings['executable']}")
        component = profile.get("component") or {}
        footprint = str(component.get("footprint", "chip_2pad"))
        height_mm = float((component.get("nominal_dims_mm") or {}).get("height", 0.45) or 0.45)
        tolerances = profile.get("tolerances") or {}
        bake = {**profile_job_fields(profile),
                **bake_fields(render_cfg, roi, camera=self._camera_cfg(render_cfg, req), supersampling=int(req.supersampling))}

        jobs: list[dict[str, Any]] = []
        meta: list[tuple[str, int, Path, dict[str, Any], dict[str, Any]]] = []
        for state in states:
            for v in range(1, int(req.variants) + 1):
                seed = self._seed(req, state, v)
                rng = np.random.default_rng(seed)
                nominal = sample_nominal_geometry(roi, rng, geometry_ranges=profile.get("geometry_ranges") or {})
                defect = sample_defect_params(state, rng, tolerances=tolerances)
                augment = self._augment(rng, cfg, req)
                rel = f"{state}_v{v}.png"
                record = {"image_path": rel, "seed": seed, "nominal": nominal, "defect": defect, "augment": augment}
                jobs.append(blender_job(record, mm_per_px=mpp, roi_width=w, roi_height=h, footprint=footprint,
                                        component_height_mm=height_mm, render_3d=profile.get("render_3d") or {},
                                        bake=bake))
                meta.append((state, v, out_dir / rel, defect, nominal))
        on_log(f"Blender: {len(jobs)} images | {settings['engine']} | {settings['samples']} samples | "
               f"{settings['device']} | board supersampling {req.supersampling}x")
        # One Blender run per image: each image shows up as soon as it is done and Stop takes effect
        # between images. The boards are baked in background threads while Blender renders the
        # previous image, so the bake (2D photoreal render of the board) is off the critical path.
        work_dir = out_dir / "_board_textures"
        bake_workers = min(2, max(1, len(jobs)))
        baked = imap_ordered(lambda item: prepare_job(item[1], work_dir, item[0]), list(enumerate(jobs)),
                             workers=bake_workers, should_stop=should_stop)
        out = []
        t0 = time.perf_counter()
        try:
            for n, (prepared, (st, v, path, defect, nominal)) in enumerate(zip(baked, meta), start=1):
                if should_stop():
                    raise PreviewCancelled()
                on_log(f"[{n}/{len(jobs)}] {st} #{v}: Blender render ...")
                run_prepared_blender_jobs([prepared], sim_root=self.sim_root, output_root=out_dir,
                                          jobs_path=work_dir / f"job_{n:06d}.jsonl", settings=settings)
                rec = PreviewImage(st, v, path, time.perf_counter() - t0, _defect_info(defect), defect, nominal)
                on_log(f"[{n}/{len(jobs)}] {st} #{v}: done in {rec.seconds:.1f}s")
                write_sidecar(rec)
                out.append(rec)
                on_image(rec)
                t0 = time.perf_counter()
        finally:
            baked.close()  # cancel queued bakes, wait for running ones before removing their files
            shutil.rmtree(work_dir, ignore_errors=True)
        if len(out) < len(jobs):
            raise PreviewCancelled()
        return out


def _defect_info(defect: dict[str, Any]) -> str:
    """Short caption of the sampled defect parameters."""
    parts = []
    for key, label, fmt in (("shift_x", "dx", "{:+.1f}px"), ("shift_y", "dy", "{:+.1f}px"),
                            ("rotation_deg", "rot", "{:+.1f}°"), ("tilt_deg", "tilt", "{:.0f}°")):
        val = defect.get(key)
        if isinstance(val, (int, float)) and abs(float(val)) > 1e-6:
            parts.append(f"{label} {fmt.format(float(val))}")
    return ", ".join(parts)
