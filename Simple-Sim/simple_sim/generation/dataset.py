"""Single-profile synthetic dataset generation (``scripts/generate.py``)."""

from __future__ import annotations

import copy
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from simple_sim.config import DEFAULT_COMPONENT_PROFILE, load_config, merge_profile_render, validate_config
from simple_sim.dataset_store import atomic_dataset_dir, write_dataset_files
from simple_sim.generation.blender import blender_settings, render_blender_records
from simple_sim.generation.common import (
    SPLIT_NAMES,
    build_rows,
    finalize_splits,
    project_root as default_project_root,
    rel_profile_path,
    render_2d_records,
    sample_domain,
    sample_record,
    summarize,
)
from simple_sim.filter_settings import load_image_filters_env, normalize_image_filters
from simple_sim.generators.backends import Backend, backend_of
from simple_sim.generators.opencv2d.spec import RenderSpec
from simple_sim.generators.version import RENDERER_VERSION
from simple_sim.generation.render_options import look_differences
from simple_sim.manifest import (RendererMismatchError, check_renderer_version, read_dataset_manifest, write_dataset_manifest,
                                 write_multi_profile_manifest)
from simple_sim.profile_hash import hash_profile, load_profile
from simple_sim.schema import LabelRow, MetaRow, append_jsonl_rows, read_jsonl, validate_jsonl_pair
from simple_sim.splits import assert_no_overlap, read_split, write_splits
from simple_sim.telemetry import emit


log = logging.getLogger(__name__)


@dataclass
class ExtendState:
    """What ``--extend`` needs from the existing dataset."""

    meta_rows: list[MetaRow] = field(default_factory=list)
    label_rows: list[LabelRow] = field(default_factory=list)
    splits: dict[str, list[str]] = field(default_factory=lambda: {k: [] for k in SPLIT_NAMES})
    manifest: dict[str, Any] | None = None
    manifest_version: int = 1
    profiles: list[dict[str, str]] = field(default_factory=list)
    run_id: str = ""
    start_index: int = 0


def read_existing_max_index(meta_rows: list[MetaRow]) -> int:
    """Highest sample index in *meta_rows* (from ``.../000042`` ids or ``images/000042.png``), or -1."""
    mx = -1
    for r in meta_rows:
        idx: int | None
        try:
            idx = int(str(r.id).split("/")[-1])
        except ValueError:
            try:
                idx = int(Path(r.image_path).stem)
            except ValueError:
                idx = None
        if idx is not None and idx > mx:
            mx = idx
    return mx


def _existing_profile(
    manifest: dict[str, Any], *, profile_id: str, profile_hash: str, profile_rel_path: str
) -> tuple[str, str, list[dict[str, str]]]:
    """Return ``(existing_profile_id, existing_profile_hash, target_profiles)`` from a dataset manifest.

    A multi-profile (v2) dataset may gain a new profile, which is appended to target_profiles.
    """
    version = int(manifest.get("manifest_version", 1))
    if "component_profile" in manifest:
        cp = manifest.get("component_profile") or {}
        return str(cp.get("profile_id") or "").strip(), str(cp.get("profile_hash") or "").strip(), []
    if "component_profiles" not in manifest:
        raise ValueError(
            f"Invalid dataset manifest (manifest_version={version}): missing component_profile/component_profiles."
        )
    profiles = manifest.get("component_profiles") or []
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(
            f"Invalid dataset manifest (manifest_version={version}): missing usable component profile information."
        )
    target: list[dict[str, str]] = []
    by_id: dict[str, dict[str, str]] = {}
    for p in profiles:
        pobj = p or {}
        row = {
            "profile_id": str(pobj.get("profile_id") or "").strip(),
            "profile_hash": str(pobj.get("profile_hash") or "").strip(),
            "profile_path": str(pobj.get("profile_path") or "").strip(),
        }
        if row["profile_id"] and row["profile_hash"]:
            by_id[row["profile_id"]] = row
            target.append(row)
    if not target:
        raise ValueError(
            f"Invalid dataset manifest (manifest_version={version}): "
            "component_profiles has no valid profile_id/profile_hash entries."
        )
    if profile_id in by_id:
        return profile_id, by_id[profile_id]["profile_hash"], target
    # Allow adding a new single profile to an existing multi-profile dataset.
    target.append({"profile_id": profile_id, "profile_hash": profile_hash, "profile_path": profile_rel_path})
    log.info(f"Extending multi-profile dataset with new profile: {profile_id}")
    return profile_id, profile_hash, target


def check_look_options(output_dir: Path, config: Mapping[str, Any], *, allow_mismatch: bool = False) -> None:
    """Refuse to extend the dataset in *output_dir* with a config whose images would look different
    (:func:`simple_sim.generation.render_options.look_differences` against its ``config.yaml``).
    Raises :class:`~simple_sim.manifest.RendererMismatchError`; *allow_mismatch* only logs a warning."""
    diffs = look_differences(load_config(output_dir / "config.yaml"), config)
    if not diffs:
        return
    msg = ("Render options differ from the dataset being extended:\n  " + "\n  ".join(diffs)
           + "\nNew images would look different from the existing ones. Generate a new dataset, or allow the "
             "mismatch to mix them.")
    if not allow_mismatch:
        raise RendererMismatchError(msg, dataset_version=RENDERER_VERSION, current_version=RENDERER_VERSION)
    log.warning(msg)


def load_extend_state(
    output_dir: Path, *, profile_id: str, profile_hash: str, profile_rel_path: str, run_id: str,
    allow_renderer_mismatch: bool = False, config: Mapping[str, Any] | None = None,
) -> ExtendState:
    """Validate an existing dataset for ``--extend`` (files, manifest, profile id/hash, renderer
    version, and with *config* the look options of :func:`check_look_options`) and load it."""
    meta_path = output_dir / "meta.jsonl"
    labels_path = output_dir / "labels.jsonl"
    manifest_path = output_dir / "dataset_manifest.json"
    if not (meta_path.exists() and labels_path.exists() and (output_dir / "config.yaml").exists()):
        raise ValueError(f"--extend requires an existing valid dataset dir with meta.jsonl/labels.jsonl/config.yaml: {output_dir}")
    if not manifest_path.exists():
        raise ValueError(
            f"--extend requires dataset with manifest: {manifest_path}\n"
            f"Legacy datasets must be regenerated or use backfill tool:\n"
            f"  .venv/bin/python tools/backfill_manifest.py --data {output_dir}"
        )

    manifest = read_dataset_manifest(manifest_path)
    existing_id, existing_hash, target_profiles = _existing_profile(
        manifest, profile_id=profile_id, profile_hash=profile_hash, profile_rel_path=profile_rel_path
    )
    if not existing_id or not existing_hash:
        raise ValueError(
            "Invalid dataset manifest: component profile metadata is incomplete "
            "(profile_id/profile_hash required)."
        )
    if existing_id != profile_id:
        raise ValueError(
            f"Profile ID mismatch for --extend:\n"
            f"  Dataset profile: {existing_id}\n"
            f"  Config profile:  {profile_id}\n"
            f"Cannot extend dataset with different component type."
        )
    if existing_hash != profile_hash:
        raise ValueError(
            f"Profile hash mismatch for --extend:\n"
            f"  Dataset hash: {existing_hash[:72]}...\n"
            f"  Config hash:  {profile_hash[:72]}...\n"
            f"Profile '{profile_id}' has changed.\n"
            f"Create a new profile version (e.g., @2) or regenerate dataset."
        )
    log.info("Profile validation passed for extend mode")
    check_renderer_version(manifest, RENDERER_VERSION, allow_mismatch=allow_renderer_mismatch)
    if config is not None:
        check_look_options(output_dir, config, allow_mismatch=allow_renderer_mismatch)

    validate_jsonl_pair(meta_path, labels_path)
    state = ExtendState(
        meta_rows=read_jsonl(meta_path, MetaRow),
        label_rows=read_jsonl(labels_path, LabelRow),
        manifest=manifest,
        manifest_version=int(manifest.get("manifest_version", 1)),
        profiles=target_profiles,
        run_id=run_id,
    )
    dataset_run_id = str(manifest.get("run_id") or "").strip()
    if not dataset_run_id and state.meta_rows:
        dataset_run_id = str(state.meta_rows[0].run_id or "").strip()
    if dataset_run_id and dataset_run_id != run_id:
        log.info(f"Extend mode: using existing dataset run_id '{dataset_run_id}' (config had '{run_id}')")
        state.run_id = dataset_run_id

    state.start_index = read_existing_max_index(state.meta_rows) + 1
    log.info(f"\nExtending dataset: {output_dir}")
    log.info(f"Existing samples: {len(state.meta_rows)} (next index: {state.start_index})")
    for s in SPLIT_NAMES:
        p = output_dir / "splits" / f"{s}.txt"
        if p.exists():
            state.splits[s] = read_split(p)
    return state


def _dedup_profiles(profiles: list[dict[str, str]]) -> list[dict[str, str]]:
    """Unique (by profile_id), sorted, stripped profile entries for a multi-profile manifest."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for p in sorted(profiles, key=lambda x: str(x.get("profile_id") or "")):
        pid = str(p.get("profile_id") or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(
            {
                "profile_id": pid,
                "profile_hash": str(p.get("profile_hash") or "").strip(),
                "profile_path": str(p.get("profile_path") or "").strip(),
            }
        )
    return out


@dataclass(frozen=True, slots=True)
class RunInputs:
    """A validated run config resolved against its component profile (input of :func:`plan_records`)."""

    config: dict[str, Any]               # the run config, ``render`` merged with the profile defaults
    profile_id: str
    profile: dict[str, Any]
    profile_hash: str
    profile_rel_path: str                # as stored in manifests
    footprint: str
    geometry_ranges: dict[str, Any] | None
    tolerances: dict[str, Any] | None
    component_height_mm: float
    profile_render_3d: dict[str, Any]

    @property
    def backend(self) -> str:
        return backend_of(self.config.get("render")).backend.value

    @property
    def roi_size(self) -> tuple[int, int]:
        return int(self.config["roi"]["width_px"]), int(self.config["roi"]["height_px"])


def resolve_run(config: Mapping[str, Any], *, profiles_dir: Path, root: Path) -> RunInputs:
    """Validate *config* (a copy is taken) and resolve its component profile from *profiles_dir*.

    Schema v1 configs may override the profile's tolerances / geometry ranges; the ``render``
    section gets the profile's defaults (``component_color``, ``component_type``, ``mm_per_px``).
    """
    config = copy.deepcopy(dict(config))
    validate_config(config)
    profile_id = config["run"].get("component_profile", DEFAULT_COMPONENT_PROFILE)
    profile = load_profile(profile_id, profiles_dir)
    profile_path = Path(profiles_dir) / f"{profile_id}.yaml"
    geometry_ranges = profile.get("geometry_ranges")
    tolerances = profile.get("tolerances")
    # Backward compatibility: v1 configs can override from config
    if config["run"]["schema_version"] == 1:
        if "tolerances" in config:
            tolerances = config["tolerances"].get(profile["component"]["package"], tolerances)
        if "geometry_ranges" in config:
            geometry_ranges = config["geometry_ranges"]
    # Fill component_color/component_type from the profile (also persisted in config.yaml).
    config["render"] = merge_profile_render(config["render"], profile, config.get("roi"))
    return RunInputs(
        config=config, profile_id=profile_id, profile=profile, profile_hash=hash_profile(profile_path),
        profile_rel_path=rel_profile_path(profile_path, root), footprint=profile["component"]["footprint"],
        geometry_ranges=geometry_ranges, tolerances=tolerances,
        component_height_mm=float(profile.get("component", {}).get("nominal_dims_mm", {}).get("height", 0.45) or 0.45),
        profile_render_3d=profile.get("render_3d") or {},
    )


#: ``progress(class_name, i, count, record)`` after sample *i* (1-based) of *count* of a class.
SampleProgress = Callable[[str, int, int, dict[str, Any]], None]


def plan_records(run: RunInputs, *, start_index: int = 0, image_filters: dict[str, Any],
                 enable_cardinal_rotation_90: bool = True, labels_profile_id: str = "",
                 progress: SampleProgress | None = None) -> list[dict[str, Any]]:
    """Sample geometry, defect, augmentation and seed of every image of the run (no rendering, no I/O).

    Indices count up from *start_index* over the classes in config order; each sample's domain is
    :func:`~simple_sim.generation.common.sample_domain`. The label of a record is the one derived
    from its geometry (a mismatch with the target class is logged).
    """
    config = run.config
    records: list[dict[str, Any]] = []
    sample_index = int(start_index)
    for class_name, count in config["classes"].items():
        log.info(f"\nGenerating {count} samples for class {class_name}")
        for i in range(count):
            rec, _rng = sample_record(
                sample_index,
                class_name,
                config=config,
                run_seed=config["run"]["seed"],
                geometry_ranges=run.geometry_ranges,
                tolerances=run.tolerances,
                image_filters=image_filters,
                enable_cardinal_rotation_90=enable_cardinal_rotation_90,
                footprint=run.footprint,
                profile_id=labels_profile_id,
                domain=sample_domain(config, sample_index),
            )
            if rec["class_name"] != class_name:
                # The derived label wins to keep labels consistent with geometry.
                log.warning(f"\nLabel mismatch for sample {sample_index}: target={class_name}, derived={rec['class_name']}")
            records.append(rec)
            sample_index += 1
            if progress is not None:
                progress(class_name, i + 1, count, rec)
    return records


def _sampling_progress() -> SampleProgress:
    """tqdm bar per class plus ``gen_progress`` telemetry every 10 samples (the CLI's progress output)."""
    state: dict[str, Any] = {}

    def progress(class_name: str, i: int, count: int, rec: dict[str, Any]) -> None:
        if state.get("class") != class_name:
            if state.get("bar") is not None:
                state["bar"].close()
            state.update({"class": class_name, "bar": tqdm(total=count, desc=class_name), "t0": time.perf_counter()})
        rate = i / max(1e-9, time.perf_counter() - state["t0"])
        bar = state["bar"]
        bar.update(1)
        bar.set_postfix({"idx": rec["index"], "samp/s": f"{rate:.1f}"})
        if i % 10 == 0 or i == count:
            emit("gen_progress", class_name=class_name, class_i=int(i), class_total=int(count),
                 global_idx=int(rec["index"]), samp_per_s=float(rate), last_image_path=rec["image_path"])
        if i == count:
            bar.close()
            state["bar"] = None

    return progress


def generate_dataset(
    config_path: Path | Mapping[str, Any],
    output_dir: Path,
    *,
    extend: bool = False,
    enable_cardinal_rotation_90: bool = True,
    project_root: Path | None = None,
    profiles_dir: Path | None = None,
    image_filters: dict[str, Any] | None = None,
    allow_renderer_mismatch: bool = False,
) -> dict[str, Any]:
    """Generate a complete dataset from a run config into *output_dir*.

    *config_path*: a YAML file or an already loaded config mapping (not modified). With
    ``extend=True`` and an existing dataset, new samples are appended (profile id/hash and the
    renderer version must match; *allow_renderer_mismatch* turns the latter into a warning).
    Profiles come from *profiles_dir* (default ``<project_root>/configs/profiles``); *image_filters*
    (default: ``$IMAGE_FILTERS``, the GUI hand-off) are normalised with
    :func:`simple_sim.filter_settings.normalize_image_filters`.
    Returns a summary (see :func:`simple_sim.generation.common.summarize`).
    """
    output_dir = Path(output_dir)
    root = Path(project_root) if project_root else default_project_root()
    if isinstance(config_path, Mapping):
        raw_config = config_path
    else:
        log.info(f"Loading configuration from {config_path}")
        raw_config = load_config(config_path)
    run = resolve_run(raw_config, profiles_dir=Path(profiles_dir) if profiles_dir else root / "configs" / "profiles",
                      root=root)
    config = run.config
    image_filters = load_image_filters_env() if image_filters is None else normalize_image_filters(image_filters)

    run_id = config["run"]["run_id"]
    run_seed = config["run"]["seed"]
    roi_width, roi_height = run.roi_size
    classes = config["classes"]
    backend = run.backend
    profile_id, profile_hash_str, relative_profile_path = run.profile_id, run.profile_hash, run.profile_rel_path

    log.info(f"Component Profile: {profile_id}")
    log.info(f"Profile Hash: {profile_hash_str[:72]}...")

    ext: ExtendState | None = None
    if extend and output_dir.exists():
        ext = load_extend_state(
            output_dir,
            profile_id=profile_id,
            profile_hash=profile_hash_str,
            profile_rel_path=relative_profile_path,
            run_id=run_id,
            allow_renderer_mismatch=allow_renderer_mismatch,
            config=config,
        )
        run_id = ext.run_id
    start_index = ext.start_index if ext else 0
    # v2 (multi-profile) datasets need profile ids on labels.
    labels_v2 = ext is not None and ext.manifest_version == 2
    labels_schema_version = 2 if labels_v2 else 1
    labels_profile_id = profile_id if labels_v2 else ""

    log.info(f"\nGenerating dataset: {run_id}")
    log.info(f"Image filters: {json.dumps(image_filters, ensure_ascii=True)}")
    log.info(f"Classes: {classes}")
    total_target = sum(classes.values())
    log.info(f"Total samples: {total_target}")
    log.info(f"Render backend: {backend}")
    emit(
        "gen_start",
        run_id=run_id,
        run_seed=run_seed,
        output_dir=str(output_dir),
        total_target=int(total_target),
        classes=classes,
        roi_width=int(roi_width),
        roi_height=int(roi_height),
        extend=bool(extend),
        start_index=int(start_index),
    )

    # Intermediate records keyed by stable index; splits are assigned after stratification.
    t_total0 = time.perf_counter()
    records = plan_records(run, start_index=start_index, image_filters=image_filters,
                           enable_cardinal_rotation_90=enable_cardinal_rotation_90,
                           labels_profile_id=labels_profile_id, progress=_sampling_progress())

    log.info("\nGenerating train/val/test splits")
    extra_render_meta = None
    if backend_of(config["render"]).needs_blender:
        extra_render_meta = {
            "backend": backend,
            "mm_per_px": float(config["roi"]["mm_per_px"]),
            "cycles_samples": int(((config.get("render") or {}).get("blender") or {}).get("samples", 0) or 0),
            "device": blender_settings(config)["device"],
            "profile_render_3d": run.profile_render_3d,
        }
    meta_rows, label_rows, new_splits = build_rows(
        records,
        config=config,
        run_id=run_id,
        backend=backend,
        image_filters=image_filters,
        labels_schema_version=labels_schema_version,
        # Salted seed keeps split assignment deterministic across extend calls.
        split_seed=int(run_seed) + int(start_index),
        extra_render_meta=extra_render_meta,
    )
    def render(out_root: Path, jobs_filename: str, at_least: bool) -> None:
        if backend == Backend.OPENCV_2D:
            # Per-domain camera (domains.<name>.camera) on top of render.camera.
            spec = RenderSpec.from_run_config(config, run.profile, tolerances=run.tolerances)
            bar = tqdm(total=len(records), desc="render")
            render_2d_records([(spec, r, None) for r in records], out_root, progress=lambda done, _total: bar.update(1))
            bar.close()
            return
        render_blender_records(
            records,
            config=config,
            roi_width=roi_width,
            roi_height=roi_height,
            footprint=run.footprint,
            component_height_mm=run.component_height_mm,
            profile_render_3d=run.profile_render_3d,
            tolerances=run.tolerances,
            output_root=out_root,
            jobs_filename=jobs_filename,
            expect_at_least=at_least,
        )

    log.info(f"\nWriting dataset to {output_dir}")
    if ext is not None and ext.meta_rows:
        # Append mode: keep existing dataset, add new images + rows + update split files.
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        render(output_dir, "blender_jobs_extend.jsonl", True)
        append_jsonl_rows(output_dir / "meta.jsonl", meta_rows)
        append_jsonl_rows(output_dir / "labels.jsonl", label_rows)

        final_splits = {k: sorted(set(ext.splits.get(k, [])) | set(new_splits[k])) for k in SPLIT_NAMES}
        assert_no_overlap(final_splits)
        write_splits(output_dir, final_splits)
        all_meta_rows = ext.meta_rows + meta_rows
        all_label_rows = ext.label_rows + label_rows
    else:
        with atomic_dataset_dir(output_dir) as temp_dir:
            render(temp_dir, "blender_jobs.jsonl", False)
            write_dataset_files(temp_dir, meta_rows, label_rows, config)
        final_splits = new_splits
        all_meta_rows = meta_rows
        all_label_rows = label_rows

    finalize_splits(output_dir, final_splits, all_label_rows)

    if ext is not None and (ext.manifest_version == 2 or (ext.manifest and "component_profiles" in ext.manifest)):
        profiles = ext.profiles or [
            {"profile_id": profile_id, "profile_hash": profile_hash_str, "profile_path": relative_profile_path}
        ]
        write_multi_profile_manifest(
            output_dir=output_dir,
            run_id=run_id,
            profiles=_dedup_profiles(profiles),
            meta_rows=all_meta_rows,
            label_rows=all_label_rows,
            splits=final_splits,
            script_name="scripts/generate.py",
            existing=ext.manifest,
            renderer_version=RENDERER_VERSION,
        )
    else:
        write_dataset_manifest(
            output_dir=output_dir,
            run_id=run_id,
            profile_id=profile_id,
            profile_hash=profile_hash_str,
            profile_path=relative_profile_path,
            meta_rows=meta_rows,  # Only new samples for extend history
            label_rows=label_rows,
            splits=final_splits,
            extend=extend,
            renderer_version=RENDERER_VERSION,
        )

    total_s = max(1e-9, time.perf_counter() - t_total0)
    log.info("\n" + "=" * 60)
    log.info("DATASET GENERATION COMPLETE")
    log.info("=" * 60)
    log.info(f"Output directory: {output_dir}")
    log.info(f"Total samples: {len(meta_rows)}")
    log.info(f"Generation rate (overall): {len(meta_rows) / total_s:.1f} samples/s  (time: {total_s:.1f}s)")
    if meta_rows:
        log.info(f"Last sample: {meta_rows[-1].id}")
        log.info(f"Last image:  {meta_rows[-1].image_path}")
        emit(
            "gen_done",
            output_dir=str(output_dir),
            total_samples=int(len(meta_rows)),
            samp_per_s=float(len(meta_rows) / total_s),
            elapsed_s=float(total_s),
            last_id=meta_rows[-1].id,
            last_image_path=meta_rows[-1].image_path,
            splits={k: int(len(v)) for k, v in final_splits.items()},
        )
    for split_name, sample_ids in final_splits.items():
        log.info(f"  {split_name}: {len(sample_ids)} samples")
    log.info("=" * 60)

    result = summarize(output_dir, all_label_rows, final_splits, new_samples=len(meta_rows), elapsed_s=total_s)
    result.update(run_id=run_id, profile_id=profile_id, backend=backend, extend=bool(ext is not None and ext.meta_rows))
    return result
