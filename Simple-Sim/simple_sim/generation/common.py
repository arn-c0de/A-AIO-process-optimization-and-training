"""Building blocks shared by the single-profile and mixed-profile dataset generators."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import numpy as np

from simple_sim.config import defect_units, placement_sampling
from simple_sim.dataset_store import write_image
from simple_sim.defects import classify_defect, sample_defect_params
from simple_sim.filter_settings import augment_for_meta
from simple_sim.generators.opencv2d import apply_image_filter_overrides, sample_augment_params, sample_nominal_geometry
from simple_sim.generators.opencv2d.spec import RenderSpec, render_record_with_meta
from simple_sim.parallel import imap_ordered, render_executor, render_processes, render_workers
from simple_sim.rng import derive_sample_seed, make_sample_id
from simple_sim.rows import LabelRow, MetaRow
from simple_sim.splits import SPLIT_NAMES, assert_no_overlap, check_class_coverage, generate_splits, write_splits

log = logging.getLogger(__name__)

DOMAIN = "domain_A"  # default domain name (single-domain configs and the profile-classifier generator)


def sample_domain(config: dict[str, Any], index: int) -> str:
    """Domain of sample *index*: the config's domains in order, cycled (``index % n``), so every class
    is spread evenly over the domains and a single-domain config always yields that domain."""
    names = list(config["domains"])
    return str(names[int(index) % len(names)])


def project_root() -> Path:
    """Simple-Sim root (the directory containing ``simple_sim/`` and ``configs/``)."""
    return Path(__file__).resolve().parents[2]


def rel_profile_path(profile_path: Path, root: Path) -> str:
    """Profile path relative to *root*, or absolute if outside it (as stored in manifests)."""
    try:
        return str(profile_path.relative_to(root))
    except ValueError:
        return str(profile_path.absolute())


def sample_record(
    index: int,
    class_name: str,
    *,
    config: dict[str, Any],
    run_seed: int,
    geometry_ranges: dict[str, Any] | None,
    tolerances: dict[str, Any] | None,
    image_filters: dict[str, Any],
    enable_cardinal_rotation_90: bool,
    footprint: str,
    profile_id: str = "",
    domain: str = DOMAIN,
) -> tuple[dict[str, Any], np.random.Generator]:
    """Sample geometry/defect/augment for sample *index* of target *class_name* in *domain*.

    Returns ``(record, rng)``; the rng is positioned after sampling so a caller may keep
    using it for rendering. ``record["class_name"]`` is the *derived* label, which can
    differ from the target (callers may warn). ``defects.placement_sampling`` of *config*
    selects the OK / MISALIGNED placement sampler, ``defects.units`` the units of the fixed
    tombstone / bridge / corner-lift shift ranges (``mm`` uses ``roi.mm_per_px``).
    """
    seed = derive_sample_seed(run_seed, domain, index)
    rng = np.random.default_rng(seed)
    nominal = sample_nominal_geometry(config["roi"], rng, geometry_ranges=geometry_ranges)
    defect = sample_defect_params(class_name, rng, tolerances=tolerances, placement=placement_sampling(config),
                                  units=defect_units(config), mm_per_px=(config.get("roi") or {}).get("mm_per_px"))
    derived = classify_defect(nominal, defect, tolerances)
    augment = sample_augment_params(
        config["augment"],
        config["domains"][domain],
        rng,
        enable_cardinal_rotation_90=(enable_cardinal_rotation_90 and bool(image_filters.get("cardinal_rotation_90", True))),
    )
    augment = apply_image_filter_overrides(augment, image_filters, rng)
    record = {
        "index": index,
        "seed": seed,
        "image_path": f"images/{index:06d}.png",
        "nominal": nominal,
        "defect": defect,
        "augment": augment,
        "class_name": derived,
        "footprint": footprint,
        "profile_id": profile_id,
        "domain": domain,
    }
    return record, rng


#: One 2D image to render: its spec, the record and the generator to draw from (``None``: a fresh
#: generator seeded with ``record['seed']``).
RenderTask = tuple[RenderSpec, dict[str, Any], "np.random.Generator | None"]


def _render_and_write(task: tuple[RenderSpec, dict[str, Any], str, np.random.Generator | None]) -> dict[str, Any]:
    spec, record, out_root, rng = task
    img, details = render_record_with_meta(spec, record, rng)
    write_image(Path(out_root) / record["image_path"], img)
    return details


def render_2d_records(tasks: Sequence[RenderTask], out_root: Path, *, executor: str | None = None,
                      progress: Callable[[int, int], None] | None = None) -> list[dict[str, Any]]:
    """Render and write the image of every task below *out_root*, in parallel.

    Streamed (images are never all held in memory) and identical to a sequential loop. *executor*
    (``thread`` | ``process``) defaults to :func:`simple_sim.parallel.render_executor`; *progress* is
    called with ``(done, total)`` after every image. Returns the render details of every task in
    task order (:func:`~simple_sim.generators.opencv2d.spec.render_record_with_meta`, e.g.
    ``board_drc``; empty dicts when there are none), for :func:`add_render_meta`.
    """
    choice = render_executor(len(tasks), executor)
    if choice == "process":
        try:
            return _render_tasks(tasks, out_root, processes=True, progress=progress)
        except BrokenProcessPool:
            # Typical cause: the calling script has no ``if __name__ == "__main__":`` guard, so every
            # spawned worker re-runs it on import and dies. Images depend only on their seeds, so
            # rendering again with threads writes exactly the same files.
            log.warning("Render worker processes failed to start or crashed; rendering with threads instead. "
                        "A script that generates datasets must call it under `if __name__ == \"__main__\":` "
                        "(worker processes import the main module), or set SIMPLE_SIM_RENDER_EXECUTOR=thread.")
    return _render_tasks(tasks, out_root, processes=False, progress=progress)


def _render_tasks(tasks: Sequence[RenderTask], out_root: Path, *, processes: bool,
                  progress: Callable[[int, int], None] | None) -> list[dict[str, Any]]:
    workers = render_processes() if processes else render_workers()
    log.info(f"Rendering {len(tasks)} images with {workers} worker {'processes' if processes else 'threads'}")
    items = ((spec, record, str(out_root), rng) for spec, record, rng in tasks)
    details: list[dict[str, Any]] = []
    for done, d in enumerate(imap_ordered(_render_and_write, items, workers=workers, processes=processes), 1):
        details.append(d)
        if progress is not None:
            progress(done, len(tasks))
    return details


def add_render_meta(meta_rows: Sequence[MetaRow], details: Sequence[Mapping[str, Any]]) -> None:
    """Merge the per-sample render details of :func:`render_2d_records` into the rows' ``render_meta``
    (in place). Rows and details are in record order (:func:`build_rows` keeps it), one each.
    """
    if len(meta_rows) != len(details):
        raise ValueError(f"{len(details)} render details for {len(meta_rows)} meta rows")
    for row, d in zip(meta_rows, details):
        if d:
            row.render_meta.update(d)


def _meta_row(
    r: dict[str, Any], *, sample_id: str, split: str, run_id: str, backend: str, render_meta: dict[str, Any]
) -> MetaRow:
    return MetaRow(
        schema_version=2,
        id=sample_id,
        run_id=run_id,
        domain=str(r.get("domain") or DOMAIN),
        split=split,
        seed=r["seed"],
        image_path=r["image_path"],
        render_backend=str(backend),
        footprint=r["footprint"],
        nominal=r["nominal"],
        defect=r["defect"],
        augment=augment_for_meta(r["augment"]),
        render_meta=render_meta,
    )


def build_rows(
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    run_id: str,
    backend: str,
    image_filters: dict[str, Any],
    labels_schema_version: int,
    split_seed: int,
    extra_render_meta: dict[str, Any] | None = None,
) -> tuple[list[MetaRow], list[LabelRow], dict[str, list[str]]]:
    """Assign stratified splits to *records* and build final meta/label rows.

    Returns ``(meta_rows, label_rows, splits)`` with split id lists sorted.
    Label ``profile_id`` comes from each record's ``profile_id``.
    """

    def render_meta(r: dict[str, Any]) -> dict[str, Any]:
        return {"augment_full": dict(r["augment"]), "image_filters": dict(image_filters)}

    def label(r: dict[str, Any], sample_id: str) -> LabelRow:
        return LabelRow(
            schema_version=labels_schema_version, id=sample_id, class_name=r["class_name"], profile_id=r["profile_id"]
        )

    def tmp_id(r: dict[str, Any]) -> str:
        return f"{r.get('domain') or DOMAIN}/{r['index']:06d}"

    # Temporary rows only drive split grouping; the split value is a placeholder.
    tmp_meta = [
        _meta_row(r, sample_id=tmp_id(r), split="train", run_id=run_id, backend=backend, render_meta=render_meta(r))
        for r in records
    ]
    tmp_labels = [label(r, tmp_id(r)) for r in records]
    id_to_split = {sid: name for name, ids in generate_splits(tmp_meta, tmp_labels, config, split_seed).items() for sid in ids}

    meta_rows: list[MetaRow] = []
    label_rows: list[LabelRow] = []
    splits: dict[str, list[str]] = {k: [] for k in SPLIT_NAMES}
    for r in records:
        split_name = id_to_split[tmp_id(r)]
        sample_id = make_sample_id(run_id, str(r.get("domain") or DOMAIN), split_name, r["index"])
        meta = render_meta(r)
        meta.update(extra_render_meta or {})
        meta_rows.append(_meta_row(r, sample_id=sample_id, split=split_name, run_id=run_id, backend=backend, render_meta=meta))
        label_rows.append(label(r, sample_id))
        splits[split_name].append(sample_id)
    for ids in splits.values():
        ids.sort()
    return meta_rows, label_rows, splits


def finalize_splits(output_dir: Path, splits: dict[str, list[str]], label_rows: list[LabelRow]) -> list[str]:
    """Check overlap and class coverage, write the split files, return the coverage warnings (also logged)."""
    assert_no_overlap(splits)
    log.info("No split overlap detected")
    warnings = check_class_coverage(splits, label_rows)
    for warning in warnings:
        log.warning(warning)
    if not warnings:
        log.info("All splits contain all classes")
    write_splits(output_dir, splits)
    return warnings


def summarize(
    output_dir: Path,
    label_rows: list[LabelRow],
    splits: dict[str, list[str]],
    *,
    new_samples: int,
    elapsed_s: float,
) -> dict[str, Any]:
    """Result payload for ``--json``: output dir and counts per split/class/profile of the dataset."""
    return {
        "output_dir": str(output_dir),
        "total_samples": len(label_rows),
        "new_samples": int(new_samples),
        "splits": {k: len(v) for k, v in splits.items()},
        "classes": dict(Counter(r.class_name for r in label_rows)),
        "profiles": dict(Counter(r.profile_id for r in label_rows if r.profile_id)),
        "elapsed_s": round(float(elapsed_s), 3),
    }
