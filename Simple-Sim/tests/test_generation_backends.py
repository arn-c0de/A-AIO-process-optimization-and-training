"""Backend renderer entry points (plan N4) and the streamed ``--extend`` state (plan S4)."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from simple_sim.classes import Backend
from simple_sim.config import load_config
from simple_sim.generators import backends
from simple_sim.generators.backends import BACKENDS, backend_info, load_entry_point

ROOT = Path(__file__).resolve().parent.parent
CALLS: list[tuple[str, int]] = []


def _small_config() -> dict:
    cfg = load_config(ROOT / "configs" / "run_0001.yaml")
    cfg["classes"] = {"OK": 2, "MISSING": 1, "MISALIGNED": 1}
    cfg["roi"].update(width_px=48, height_px=48)
    return cfg


def test_every_backend_names_importable_renderers():
    for info in BACKENDS.values():
        assert callable(load_entry_point(info.renderer)) and callable(load_entry_point(info.previewer))
    assert backend_info("opencv_2d") is BACKENDS[Backend.OPENCV_2D]
    with pytest.raises(ValueError, match="module:attr"):
        load_entry_point("no_colon")
    with pytest.raises(ValueError, match="Unsupported render backend"):
        backend_info("vulkan")


def test_backends_module_loads_no_renderer():
    code = ("import sys; import simple_sim.generators.backends as b; b.backend_info('blender_3d'); "
            "print(sorted(m for m in ('cv2', 'simple_sim.generation.render_backends', 'simple_sim.preview.render') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def fake_renderer(groups, out_root, *, jobs_filename, expect_at_least):
    """Stands in for a third backend: no images, one detail per record."""
    n = sum(len(records) for _run, records in groups)
    CALLS.append((jobs_filename, n))
    return [{"fake": i} for i in range(n)]


def test_generate_dataset_dispatches_through_backend_info(tmp_path, monkeypatch):
    from simple_sim.generation import dataset as dataset_mod

    fake = dataclasses.replace(BACKENDS[Backend.OPENCV_2D], renderer=f"{__name__}:fake_renderer")
    monkeypatch.setitem(backends.BACKENDS, Backend.OPENCV_2D, fake)
    monkeypatch.setattr("simple_sim.dataset_store.validate_dataset_files", lambda _d: None)
    CALLS.clear()
    dataset_mod.generate_dataset(_small_config(), tmp_path / "ds", project_root=ROOT, image_filters={})
    assert CALLS == [("blender_jobs.jsonl", 4)]
    meta = (tmp_path / "ds" / "meta.jsonl").read_text(encoding="utf-8")
    assert '"fake": 3' in meta  # details went into render_meta


def test_extend_state_is_streamed_and_keeps_no_meta_rows(tmp_path, monkeypatch):
    from simple_sim.generation.dataset import LabelKey, generate_dataset, load_extend_state, read_existing_max_index, resolve_run
    from simple_sim.schema import LabelRow, MetaRow, read_jsonl

    monkeypatch.delenv("IMAGE_FILTERS", raising=False)
    ds = tmp_path / "ds"
    generate_dataset(_small_config(), ds, project_root=ROOT)
    generate_dataset(_small_config(), ds, project_root=ROOT, extend=True)
    run = resolve_run(_small_config(), profiles_dir=ROOT / "configs" / "profiles", root=ROOT)
    state = load_extend_state(ds, profile_id=run.profile_id, profile_hash=run.profile_hash,
                              profile_rel_path=run.profile_rel_path, run_id="other", config=run.config)
    meta = read_jsonl(ds / "meta.jsonl", MetaRow)
    labels = read_jsonl(ds / "labels.jsonl", LabelRow)
    assert not hasattr(state, "meta_rows") and not hasattr(state, "label_rows")
    assert state.existing_count == len(meta) == 8
    assert state.labels == [LabelKey(r.id, r.class_name, r.profile_id) for r in labels]
    assert state.start_index == read_existing_max_index(meta) + 1 == 8
    assert state.run_id == meta[0].run_id
    assert sorted(i for ids in state.splits.values() for i in ids) == sorted(r.id for r in meta)


def test_validate_jsonl_pair_streams_rows_to_callbacks(tmp_path):
    from simple_sim.generation.dataset import generate_dataset
    from simple_sim.schema import iter_jsonl, read_jsonl, MetaRow, LabelRow, validate_jsonl_pair

    generate_dataset(_small_config(), tmp_path / "ds", project_root=ROOT, image_filters={})
    seen: list[str] = []
    validate_jsonl_pair(tmp_path / "ds" / "meta.jsonl", tmp_path / "ds" / "labels.jsonl",
                        on_meta=lambda r: seen.append("m" + r.id), on_label=lambda r: seen.append("l" + r.id))
    meta_ids = [r.id for r in read_jsonl(tmp_path / "ds" / "meta.jsonl", MetaRow)]
    assert seen == ["m" + i for i in meta_ids] + ["l" + r.id for r in read_jsonl(tmp_path / "ds" / "labels.jsonl", LabelRow)]
    with pytest.raises(FileNotFoundError):
        iter_jsonl(tmp_path / "missing.jsonl", MetaRow)  # raised on the call, not on next()
