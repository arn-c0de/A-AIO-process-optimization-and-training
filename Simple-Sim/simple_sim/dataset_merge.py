"""Merge several generated datasets into one (``tools/create_multi_dataset.py``, GUI "merge for training").

:func:`merge_datasets` copies (or hard-links) every labelled sample of the sources into a new dataset
directory with fresh ids ``<run_id>/<domain>/<split>/<n:06d>`` and image names ``images/<split>_<n:06d>.png``,
writes schema-v2 labels carrying each sample's ``profile_id``, the split files, ``config.yaml`` and a v1
(one profile) or v2 (several) manifest (:func:`simple_sim.manifest.write_manifest_for_profiles`).

Options cover both callers: the CLI copies images unchanged and checks the labels against one config's
classes; the GUI additionally requires equal class sets, resizes every image to the smallest ROI of the
sources and hard-links unchanged images, and merges with ``strict=False`` (see :func:`merge_datasets`).
A failed merge removes *out_dir* again.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from simple_sim.dataset_files import CONFIG_YAML, DATASET_MANIFEST, LABELS_JSONL, META_JSONL
from simple_sim.io_utils import atomic_write_text, load_yaml
from simple_sim.manifest import load_dataset_manifest, write_manifest_for_profiles
from simple_sim.rows import LabelRow, MetaRow, read_jsonl, write_jsonl
from simple_sim.splits import SPLIT_NAMES, write_splits

log = logging.getLogger(__name__)

ProfileEntry = dict[str, str]  # {"profile_id", "profile_hash", "profile_path"}


class MergeError(ValueError):
    """The sources cannot be merged (unknown profile, class or split, different class sets, no ROI)."""


@dataclass
class MergeResult:
    """What :func:`merge_datasets` wrote."""

    out_dir: Path
    run_id: str
    meta_rows: list[MetaRow]
    label_rows: list[LabelRow]
    splits: dict[str, list[str]]
    profiles: list[ProfileEntry]          # profiles named in the manifest (sorted by id)
    manifest_version: int                 # 1, 2, or 0 when no manifest was written (no samples)
    roi: tuple[int, int] | None = None    # (width_px, height_px) the images were resized to (resize mode)
    skipped_unlabeled: dict[str, int] = field(default_factory=dict)  # source -> meta rows without a label row

    def summary(self) -> dict[str, Any]:
        """JSON-able summary (``create-multi-dataset --json``)."""
        return {"out": str(self.out_dir), "run_id": self.run_id, "samples": len(self.meta_rows),
                "splits": {k: len(v) for k, v in self.splits.items()},
                "profiles": [p["profile_id"] for p in self.profiles]}


def profile_entry(profile_id: str, profile_hash: str = "", profile_path: str = "") -> ProfileEntry:
    return {"profile_id": str(profile_id), "profile_hash": str(profile_hash or ""), "profile_path": str(profile_path or "")}


def load_dataset_config(ds: Path) -> dict[str, Any]:
    """``<ds>/config.yaml`` as a dict (``{}`` when missing or not a mapping)."""
    cfg_path = Path(ds) / CONFIG_YAML
    if not cfg_path.exists():
        return {}
    data = load_yaml(cfg_path) or {}
    return data if isinstance(data, dict) else {}


def _profile_from_config(ds: Path, profiles_dir: Path) -> ProfileEntry | None:
    """Fallback for a source without a manifest: ``run.component_profile`` of its config.yaml."""
    from simple_sim.profile_hash import hash_profile

    try:
        run = load_dataset_config(ds).get("run")
    except (OSError, yaml.YAMLError):
        return None
    pid = str(run.get("component_profile") or "").strip() if isinstance(run, dict) else ""
    if not pid:
        return None
    profile_path = Path(profiles_dir) / f"{pid}.yaml"
    if not profile_path.exists():
        return profile_entry(pid)
    return profile_entry(pid, hash_profile(profile_path), str(profile_path))


def source_profiles(ds: Path, profiles_dir: Path | None = None) -> tuple[ProfileEntry | None, list[ProfileEntry]]:
    """``(default_profile, profiles)`` declared by the source dataset *ds*.

    *default_profile* names the label rows without a ``profile_id`` (v1 labels); it is None for a
    multi-profile (v2) source. Without ``dataset_manifest.json`` the profile comes from the source's
    ``run.component_profile`` when *profiles_dir* is given (else, or when unset, :class:`MergeError`).
    """
    ds = Path(ds)
    manifest_path = ds / DATASET_MANIFEST
    if manifest_path.exists():
        manifest = load_dataset_manifest(manifest_path)
        profs = [profile_entry(str(p.profile_id), str(p.profile_hash or ""), str(p.profile_path or ""))
                 for p in manifest.profiles if str(p.profile_id or "").strip()]
        if not profs:
            raise MergeError(f"Invalid manifest in {ds}: no component profile")
        if manifest.version == 1:
            return profs[0], profs[:1]
        return None, profs
    prof = _profile_from_config(ds, profiles_dir) if profiles_dir is not None else None
    if prof is None:
        raise MergeError(f"Cannot determine component profile of {ds}: no {DATASET_MANIFEST}"
                         + (" and no run.component_profile in config.yaml (run tools/backfill_manifest.py first)."
                            if profiles_dir is not None else "."))
    return prof, [prof]


def _config_classes_and_roi(cfg: dict[str, Any]) -> tuple[tuple[str, ...], int, int]:
    classes = cfg.get("classes") or {}
    roi = cfg.get("roi") or {}
    return (tuple(sorted(classes.keys() if isinstance(classes, dict) else [])),
            int(roi.get("width_px") or 0), int(roi.get("height_px") or 0))


def _common_roi(sources: Sequence[Path], require_same_classes: bool) -> tuple[int, int]:
    """Smallest ROI (width, height) over the sources' configs; with *require_same_classes* all class sets must match."""
    target_w = target_h = 0
    classes0: tuple[str, ...] | None = None
    for ds in sources:
        if not (ds / CONFIG_YAML).exists():
            raise FileNotFoundError(f"Missing {CONFIG_YAML} in {ds}")
        try:
            cfg = load_dataset_config(ds)
        except (OSError, yaml.YAMLError):
            cfg = {}
        classes, w, h = _config_classes_and_roi(cfg)
        if require_same_classes:
            if classes0 is None:
                classes0 = classes
            elif classes != classes0:
                raise MergeError(f"Cannot merge datasets with different class sets.\n  first: {classes0}\n  {ds.name}: {classes}")
        if w > 0:
            target_w = w if target_w <= 0 else min(target_w, w)
        if h > 0:
            target_h = h if target_h <= 0 else min(target_h, h)
    if target_w <= 0 or target_h <= 0:
        raise MergeError("Cannot determine target ROI size for merged dataset (missing roi.width_px/height_px).")
    return target_w, target_h


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _place_image(src: Path, dst: Path, *, roi: tuple[int, int] | None, link: bool) -> None:
    """Copy (*link*: hard-link, falling back to a copy) *src* to *dst*; with *roi*, an image of another size is
    resized (area when shrinking, linear when enlarging). An image OpenCV cannot read is copied unchanged."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if roi is not None:
        import cv2  # lazy: only the resize needs OpenCV

        try:
            img = cv2.imread(str(src))
            if img is not None:
                h_src, w_src = img.shape[:2]
                if (w_src, h_src) != roi:
                    shrink = w_src > roi[0] or h_src > roi[1]
                    img = cv2.resize(img, roi, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
                    if cv2.imwrite(str(dst), img):
                        return
        except (cv2.error, OSError, ValueError):
            pass  # unreadable/unresizable image: copied unchanged below
    if link:
        _link_or_copy(src, dst)
    else:
        shutil.copy2(src, dst)


def merge_datasets(
    sources: Sequence[Path],
    out_dir: Path,
    *,
    run_id: str | None = None,
    config_path: Path | None = None,
    profiles_dir: Path | None = None,
    require_same_classes: bool = False,
    resize_to_common_roi: bool = False,
    link_images: bool = False,
    include_unused_profiles: bool = False,
    script_name: str = "simple_sim.dataset_merge",
    on_warning: Callable[[str], None] | None = None,
    strict: bool = True,
) -> MergeResult:
    """Merge *sources* into the new directory *out_dir* (must not exist; removed again when the merge fails).

    Args:
        run_id: run id of the merged samples; None: the first source manifest's ``run_id``, else its folder name.
        config_path: config whose ``classes`` every label must use and that becomes ``config.yaml``;
            None: the first source's ``config.yaml``.
        profiles_dir: profile YAMLs for sources without a manifest (their ``run.component_profile``);
            None: every source needs ``dataset_manifest.json``.
        require_same_classes: all sources' ``classes`` key sets must be equal (GUI).
        resize_to_common_roi: resize images to the smallest ROI of the sources and write that ROI into
            ``config.yaml`` (GUI); otherwise images and the config are copied unchanged.
        link_images: hard-link unchanged images instead of copying them (falls back to a copy).
        include_unused_profiles: the manifest names every profile the sources declare, not only those of
            merged samples (GUI).
        script_name: ``generator.script`` of the manifest.
        on_warning: called with warnings (sources with unlabelled samples); default: ``log.warning``.
        strict: validate the rows (CLI). ``strict=False`` (GUI) merges what the sources hold: labels need not
            be in the config classes (nor need the config declare classes), a source without manifest
            (and no *profiles_dir*) raises
            ``FileNotFoundError``, an unreadable first config counts as empty, and a config that cannot be
            dumped as YAML is copied as text.

    Raises:
        MergeError: unknown class / split / profile, different class sets, no ROI; FileNotFoundError for
            missing sources or config files; FileExistsError when *out_dir* exists.
    """
    sources = [Path(d) for d in sources]
    out_dir = Path(out_dir)
    warn = on_warning or log.warning
    if not sources:
        raise MergeError("No source datasets given")
    for ds in sources:
        if not ds.is_dir():
            raise FileNotFoundError(f"Dataset not found: {ds}")
    if config_path is not None and not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    try:
        return _merge_into(sources, out_dir, run_id=run_id, config_path=config_path, profiles_dir=profiles_dir,
                           require_same_classes=require_same_classes, resize_to_common_roi=resize_to_common_roi,
                           link_images=link_images, include_unused_profiles=include_unused_profiles,
                           script_name=script_name, warn=warn, strict=strict)
    except BaseException:
        # A partial output would block the retry ("already exists") and show up as a dataset.
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _merge_config(sources: list[Path], config_path: Path | None, *, strict: bool) -> tuple[Path, dict[str, Any]]:
    """``(path, mapping)`` of the merged dataset's config: *config_path*, else the first source's ``config.yaml``.
    Non-strict: an unreadable config counts as empty."""
    cfg_src = Path(config_path) if config_path is not None else sources[0] / CONFIG_YAML
    try:
        cfg = load_dataset_config(cfg_src.parent) if config_path is None else (load_yaml(cfg_src) or {})
    except (OSError, yaml.YAMLError):
        if strict:
            raise
        cfg = {}
    return cfg_src, cfg if isinstance(cfg, dict) else {}


def _declared_profiles(sources: list[Path], profiles_dir: Path | None, *,
                       strict: bool) -> tuple[dict[str, ProfileEntry], list[str]]:
    """Profiles the sources declare (id -> entry, first declaration wins) and each source's default profile id
    (``""`` for a multi-profile source)."""
    declared: dict[str, ProfileEntry] = {}
    defaults: list[str] = []
    for ds in sources:
        if not strict and profiles_dir is None and not (ds / DATASET_MANIFEST).exists():
            raise FileNotFoundError(f"Missing {DATASET_MANIFEST} in {ds}")
        default, profs = source_profiles(ds, profiles_dir)
        defaults.append(default["profile_id"] if default else "")
        for prof in profs:
            declared.setdefault(prof["profile_id"], prof)
    return declared, defaults


def _write_merged_config(out_dir: Path, cfg: dict[str, Any], cfg_src: Path, roi: tuple[int, int] | None, *,
                         strict: bool) -> None:
    """``<out_dir>/config.yaml``: *cfg* with the common *roi* (resize mode), else a copy of *cfg_src* (if any)."""
    if roi is None:
        if cfg_src.exists():
            shutil.copy2(cfg_src, out_dir / CONFIG_YAML)
        return
    roi_cfg = {**(cfg.get("roi") or {}), "width_px": roi[0], "height_px": roi[1]}
    try:
        text = yaml.safe_dump({**cfg, "roi": roi_cfg}, sort_keys=False)
    except yaml.YAMLError:
        if strict or not cfg_src.exists():
            raise
        text = cfg_src.read_text(encoding="utf-8")  # e.g. a python-tagged value: keep the source config
    atomic_write_text(out_dir / CONFIG_YAML, text)


def _merge_into(sources: list[Path], out_dir: Path, *, run_id: str | None, config_path: Path | None,
                profiles_dir: Path | None, require_same_classes: bool, resize_to_common_roi: bool, link_images: bool,
                include_unused_profiles: bool, script_name: str, warn: Callable[[str], None],
                strict: bool = True) -> MergeResult:
    cfg_src, cfg = _merge_config(sources, config_path, strict=strict)
    classes = cfg.get("classes")
    if strict and not classes:
        raise MergeError(f"No 'classes' defined in {cfg_src}; pass a config with a classes section")
    roi = _common_roi(sources, require_same_classes) if resize_to_common_roi else None
    declared, defaults = _declared_profiles(sources, profiles_dir, strict=strict)
    if run_id is None:
        manifest_path = sources[0] / DATASET_MANIFEST
        first_run = load_dataset_manifest(manifest_path).run_id if manifest_path.exists() else ""
        run_id = str(first_run or sources[0].name)

    images_dir = out_dir / "images"
    images_dir.mkdir()
    meta_out: list[MetaRow] = []
    label_out: list[LabelRow] = []
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    counters: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for ds, default_pid in zip(sources, defaults, strict=True):
        label_map = {row.id: row for row in read_jsonl(ds / LABELS_JSONL, LabelRow)}
        for row in read_jsonl(ds / META_JSONL, MetaRow):
            label = label_map.get(row.id)
            if label is None:
                skipped[str(ds)] = skipped.get(str(ds), 0) + 1
                continue
            if strict and label.class_name not in classes:
                raise MergeError(f"Class {label.class_name} of {ds} is not declared in the config classes ({cfg_src})")
            if row.split not in splits:  # unreachable for rows read by MetaRow (it validates the split)
                raise MergeError(f"Sample {row.id} from {ds} has unknown split {row.split!r} (expected {'/'.join(SPLIT_NAMES)})")
            profile_id = str(label.profile_id or default_pid).strip()
            if not profile_id:  # also non-strict: a schema-2 label row cannot be written without one
                raise MergeError(f"Sample {row.id} in {ds} has no profile_id and the source has no default profile")
            declared.setdefault(profile_id, profile_entry(profile_id))

            idx = counters.get(row.split, 0)
            counters[row.split] = idx + 1
            new_id = f"{run_id}/{row.domain}/{row.split}/{idx:06d}"
            image_name = f"{row.split}_{idx:06d}.png"
            _place_image(ds / row.image_path, images_dir / image_name, roi=roi, link=link_images)
            meta_out.append(MetaRow(
                schema_version=int(row.schema_version), id=new_id, run_id=run_id, domain=row.domain, split=row.split,
                seed=int(row.seed), image_path=str(Path("images") / image_name), render_backend=row.render_backend,
                footprint=row.footprint, nominal=row.nominal, defect=row.defect, augment=row.augment,
                render_meta=dict(row.render_meta or {}),
            ))
            label_out.append(LabelRow(schema_version=2, id=new_id, class_name=label.class_name, profile_id=profile_id))
            splits[row.split].append(new_id)
    for ds, n in skipped.items():
        warn(f"{n} sample(s) of {ds} have no label row and were not merged")

    write_jsonl(out_dir / META_JSONL, meta_out)
    write_jsonl(out_dir / LABELS_JSONL, label_out)
    write_splits(out_dir, splits)
    _write_merged_config(out_dir, cfg, cfg_src, roi, strict=strict)

    used = {row.profile_id for row in label_out}
    profiles = [declared[pid] for pid in sorted(declared) if include_unused_profiles or pid in used]
    version = write_manifest_for_profiles(out_dir, run_id, profiles=profiles, meta_rows=meta_out, label_rows=label_out,
                                          splits=splits, script_name=script_name)
    if version == 0:
        warn("no samples merged; no dataset_manifest.json written")
    return MergeResult(out_dir=out_dir, run_id=run_id, meta_rows=meta_out, label_rows=label_out, splits=splits,
                       profiles=profiles if version else [], manifest_version=version, roi=roi, skipped_unlabeled=skipped)


__all__ = ["MergeError", "MergeResult", "load_dataset_config", "merge_datasets", "profile_entry", "source_profiles"]
