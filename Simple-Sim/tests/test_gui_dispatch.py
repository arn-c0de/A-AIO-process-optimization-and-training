"""Tk-free tests for plan N11/S6: DispatchQueue (push instead of after() polling), the
stat-first FileCache, the profile-editor registry cache and the tab helpers built on them."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gui.core.tasks import DispatchQueue, UiDispatcher, dispatcher_for
from gui.tabs.core.base import BaseTab
from simple_sim.file_cache import FileCache, file_signature


def _bump_mtime(p: Path, delta_s: float = 5.0) -> None:
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + int(delta_s * 1e9)))


# --------------------------------------------------------------------------- UiDispatcher / DispatchQueue


def test_post_next_tick_runs_on_the_following_drain() -> None:
    disp = UiDispatcher()
    seen: list[str] = []

    def first() -> None:
        seen.append("a")
        disp.post_next_tick(lambda: seen.append("deferred"))
        disp.post(lambda: seen.append("b"))

    disp.post(first)
    disp.drain()
    assert seen == ["a", "b"]  # deferred did not run inside the same drain
    disp.post(lambda: seen.append("c"))
    disp.drain()
    assert seen == ["a", "b", "deferred", "c"]  # deferred runs first on the next drain


def test_dispatch_queue_pushes_batches_in_order() -> None:
    disp = UiDispatcher()
    q = DispatchQueue()
    batches: list[list[int]] = []
    q.bind(disp.post, batches.append, max_items=3, post_next=disp.post_next_tick)
    for i in range(7):
        q.put(i)
    assert disp.drain() == 1  # one coalesced drain callback for 7 puts
    assert batches == [[0, 1, 2]]
    disp.drain()
    disp.drain()
    assert batches == [[0, 1, 2], [3, 4, 5], [6]]
    assert disp.drain() == 0 and q.empty()  # nothing scheduled when empty


def test_dispatch_queue_put_nowait_and_items_queued_before_bind() -> None:
    disp = UiDispatcher()
    q = DispatchQueue()
    q.put("early")  # producer started before the tab bound the queue: plain queue
    assert disp.drain() == 0
    got: list[str] = []
    q.bind(disp.post, got.extend)
    q.put_nowait("late")
    disp.drain()
    assert got == ["early", "late"]


def test_dispatch_queue_from_threads_keeps_per_producer_order() -> None:
    disp = UiDispatcher()
    q = DispatchQueue()
    got: list[tuple[int, int]] = []
    q.bind(disp.post, got.extend, max_items=50, post_next=disp.post_next_tick)

    def produce(k: int) -> None:
        for i in range(500):
            q.put((k, i))

    ts = [threading.Thread(target=produce, args=(k,)) for k in range(4)]
    for t in ts:
        t.start()
    deadline = time.time() + 10
    while (any(t.is_alive() for t in ts) or not q.empty() or len(got) < 2000) and time.time() < deadline:
        disp.drain()
    assert len(got) == 2000
    for k in range(4):
        assert [i for kk, i in got if kk == k] == list(range(500))


def test_dispatch_queue_handler_error_does_not_lose_the_rest() -> None:
    disp = UiDispatcher()
    q = DispatchQueue()
    got: list[int] = []

    def handler(items: list[int]) -> None:
        got.extend(items)
        if 0 in items:
            raise RuntimeError("boom")

    q.bind(disp.post, handler, max_items=2, post_next=disp.post_next_tick)
    for i in range(4):
        q.put(i)
    for _ in range(3):
        disp.drain()  # the exception is printed by drain(), the queue reschedules itself
    assert got == [0, 1, 2, 3]


def test_dispatcher_for_prefers_ctx_and_falls_back_per_widget() -> None:
    ctx_disp = UiDispatcher()
    assert dispatcher_for(SimpleNamespace(ctx=SimpleNamespace(dispatcher=ctx_disp)), object()) is ctx_disp

    class _W:
        def __init__(self) -> None:
            self.scheduled: list = []

        def after(self, ms, fn=None):
            self.scheduled.append(fn)
            return f"a{len(self.scheduled)}"

        def after_cancel(self, _aid) -> None:
            pass

    w = _W()
    d1 = dispatcher_for(SimpleNamespace(ctx=None), w)
    assert dispatcher_for(SimpleNamespace(ctx=None), w) is d1 and w.scheduled


def test_base_tab_bind_queue_and_post_use_the_dispatcher() -> None:
    disp = UiDispatcher()
    fake = SimpleNamespace(dispatcher=disp)
    q = DispatchQueue()
    lines: list[str] = []
    BaseTab.bind_queue(fake, q, lambda items: lines.append("".join(items)), max_items=2)
    for s in ("a", "b", "c"):
        q.put(s)
    BaseTab.post(fake, lambda: lines.append("posted"))
    disp.drain()
    disp.drain()
    # first batch, then the callback posted after the puts, then the deferred rest
    assert lines == ["ab", "posted", "c"]


# --------------------------------------------------------------------------- FileCache


def test_file_cache_is_stat_first(tmp_path: Path) -> None:
    f = tmp_path / "m.json"
    f.write_text("1")
    cache = FileCache()
    calls: list[int] = []

    def load(p: Path) -> int:
        calls.append(1)
        return int(p.read_text())

    assert cache.get(f, load) == 1
    assert cache.get(f, load) == 1
    assert len(calls) == 1 and cache.hits == 1
    f.write_text("2")
    _bump_mtime(f)
    assert cache.get(f, load) == 2 and len(calls) == 2
    f.unlink()
    assert cache.get(f, load, missing="gone") == "gone" and len(calls) == 2
    assert file_signature(f) is None


def test_file_cache_multi_file_key_and_lru(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("x")
    cache = FileCache(max_entries=2)
    n = {"k": 0}

    def load() -> int:
        n["k"] += 1
        return n["k"]

    assert cache.cached("k", (a, b), load) == 1
    assert cache.cached("k", (a, b), load) == 1
    b.write_text("appeared")  # a dependency appearing invalidates (None -> signature)
    assert cache.cached("k", (a, b), load) == 2
    cache.cached("k2", (a,), load)
    cache.cached("k3", (a,), load)
    assert len(cache) == 2  # "k" evicted
    assert cache.cached("k", (a, b), load) == 5


def test_file_cache_load_error_is_not_cached(tmp_path: Path) -> None:
    f = tmp_path / "x"
    f.write_text("x")
    cache = FileCache()
    with pytest.raises(ValueError):
        cache.cached("x", (f,), lambda: (_ for _ in ()).throw(ValueError("bad")))
    assert cache.cached("x", (f,), lambda: 7) == 7


# --------------------------------------------------------------------------- profile editor registry


def test_profile_registry_poll_parses_nothing_when_unchanged(tmp_path: Path, monkeypatch) -> None:
    from tools.profile_editor import profile_registry as pr

    (tmp_path / "configs" / "profiles").mkdir(parents=True)
    prof = tmp_path / "configs" / "profiles" / "p.yaml"
    prof.write_text("profile:\n  profile_id: chip_a\n")
    run = tmp_path / "configs" / "run_0001.yaml"
    run.write_text("run:\n  run_id: r1\n  component_profile: chip_a\n")

    parsed: list[str] = []
    real = pr.load_yaml

    def counting(p):
        parsed.append(Path(p).name)
        return real(p)

    monkeypatch.setattr(pr, "load_yaml", counting)
    reg = pr.ProfileRegistry(tmp_path)
    assert reg.changed() is True
    assert sorted(parsed) == ["p.yaml", "run_0001.yaml"]
    assert reg.matching_run_ids("chip_a") == ["r1"]
    n = len(parsed)
    for _ in range(5):
        assert reg.changed() is False
    assert len(parsed) == n  # the 1.4 s poll only stats

    prof.write_text("profile:\n  profile_id: chip_b\n")
    _bump_mtime(prof)
    assert reg.changed() is True
    assert parsed[n:] == ["p.yaml"]  # only the changed file is parsed again
    assert set(reg.snapshot().profile_files) == {"chip_b"}

    (tmp_path / "configs" / "run_0002.yaml").write_text("run:\n  run_id: r2\n  component_profile: chip_b\n")
    assert reg.changed() is True
    assert reg.matching_run_ids("chip_b") == ["r2"]

    bad = tmp_path / "configs" / "profiles" / "bad.yaml"
    bad.write_text("profile: [unclosed\n")
    assert reg.changed() is False  # as before: an unparsable file is not listed, so no visible change
    n = len(parsed)
    assert reg.changed() is False and len(parsed) == n  # ... and it is not re-parsed on every poll
    assert "bad" not in reg.snapshot().profile_files and "bad.yaml" not in reg.snapshot().profile_files


# --------------------------------------------------------------------------- pipeline tab helpers


def _write_dataset(root: Path, n: int = 3) -> Path:
    from simple_sim.schema import LabelRow, MetaRow, write_jsonl

    (root / "images").mkdir(parents=True)
    meta, labels = [], []
    for i in range(n):
        image = f"images/{i:06d}.png"
        cv2.imwrite(str(root / image), np.full((40, 60, 3), (50 + 13 * i) % 256, np.uint8))
        sid = f"r/domain_A/train/{i:06d}"
        meta.append(MetaRow(
            schema_version=2, id=sid, run_id="r", domain="domain_A", split="train", seed=1, image_path=image,
            render_backend="opencv_2d", footprint="0603",
            nominal={"pad_width": 1.0, "pad_height": 1.0, "pad_spacing": 1.0, "component_length": 1.0,
                     "component_width": 1.0},
            defect={"type": "OK", "shift_x": 0.0, "shift_y": 0.0, "rotation_deg": 0.0, "tilt_deg": 0.0},
            augment={"blur_sigma": 0.0, "noise_stddev": 0.0, "brightness_factor": 1.0, "contrast_factor": 1.0,
                     "rotation_deg": 0.0},
        ))
        labels.append(LabelRow(schema_version=2, id=sid, class_name=("OK", "MISSING")[i % 2], profile_id="p@1"))
    write_jsonl(root / "meta.jsonl", meta)
    write_jsonl(root / "labels.jsonl", labels)
    return root


def test_load_dataset_view_matches_direct_reads(tmp_path: Path) -> None:
    from gui.tabs.pipeline.tab import load_dataset_view
    from simple_sim.schema import LabelRow, MetaRow, read_jsonl

    ds = _write_dataset(tmp_path / "ds", n=14)
    view = load_dataset_view(ds)
    assert view["error"] is None and view["has_images"]
    rows = read_jsonl(ds / "meta.jsonl", MetaRow)
    assert view["image_to_sample"] == {r.image_path: r.id for r in rows}
    assert view["meta_by_sample"] == {r.id: r for r in rows}
    assert view["label_dict"] == {r.id: r.class_name for r in read_jsonl(ds / "labels.jsonl", LabelRow)}
    assert view["recent_imgs"] == [str(p) for p in sorted((ds / "images").glob("*.png"))[-12:]]

    (ds / "labels.jsonl").write_text("{not json\n")
    _bump_mtime(ds / "labels.jsonl")
    broken = load_dataset_view(ds)
    assert broken["error"] is not None and broken["meta_by_sample"]  # meta kept, error reported


def test_prepare_thumbnail_cells_is_tk_free(tmp_path: Path) -> None:
    from gui.tabs.pipeline.ui import prepare_thumbnail_cells

    ds = _write_dataset(tmp_path / "ds", n=2)
    view = __import__("gui.tabs.pipeline.tab", fromlist=["load_dataset_view"]).load_dataset_view(ds)
    recent = view["recent_imgs"] + [str(ds / "images" / "missing.png")]
    overlays: list[str] = []

    def overlay(img, defect, nominal):
        overlays.append(defect["type"])
        return img

    cells = prepare_thumbnail_cells(4, recent, ds, view["label_dict"], view["image_to_sample"],
                                    view["meta_by_sample"], view["meta_by_image_rel"], overlay)
    assert [c[1] for c in cells] == ["(failed) missing.png", "000001.png\n[MISSING]", "000000.png\n[OK]",
                                     "(no image)"]
    assert cells[1][0].size == (60, 40) and cells[0][0] is None and cells[3][0] is None
    assert overlays == ["OK", "OK"]


# --------------------------------------------------------------------------- weights tab caches


def test_weights_tab_dataset_text_is_cached_on_manifest_and_labels(tmp_path: Path) -> None:
    from gui.tabs.weights.tab import WeightsTab

    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "labels.jsonl").write_text('{"class_name": "OK"}\n')
    calls: list[str] = []

    def describe(d: Path) -> str:
        calls.append("describe")
        return f"Samples: {len((d / 'labels.jsonl').read_text().splitlines())}"

    def classes(d: Path) -> list[str]:
        calls.append("classes")
        return ["OK"]

    fake = SimpleNamespace(_ds_file_cache=FileCache(), logic=SimpleNamespace(
        describe_dataset_samples=describe, get_dataset_classes=classes))
    assert WeightsTab._describe_dataset_samples(fake, ds) == "Samples: 1"
    assert WeightsTab._describe_dataset_samples(fake, ds) == "Samples: 1"
    got = WeightsTab._dataset_classes(fake, ds)
    got.append("mutated")  # callers get a copy
    assert WeightsTab._dataset_classes(fake, ds) == ["OK"]
    assert calls == ["describe", "classes"]
    (ds / "labels.jsonl").write_text('{"class_name": "OK"}\n{"class_name": "OK"}\n')
    assert WeightsTab._describe_dataset_samples(fake, ds) == "Samples: 2"
    from simple_sim.dataset_files import DATASET_MANIFEST

    (ds / DATASET_MANIFEST).write_text("{}")  # manifest appearing invalidates too
    WeightsTab._describe_dataset_samples(fake, ds)
    assert calls.count("describe") == 3


def test_pipeline_dataset_profile_cached_on_manifest(tmp_path: Path) -> None:
    from gui.tabs.pipeline.tab import PipelineControlTab
    from simple_sim.dataset_files import DATASET_MANIFEST

    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / DATASET_MANIFEST).write_text('{"component_profile": {"profile_id": "a", "profile_hash": "h"}}')
    calls: list[int] = []

    def pid(d: Path):
        calls.append(1)
        return "a"

    fake = SimpleNamespace(_manifest_cache=FileCache(), logic=SimpleNamespace(
        get_dataset_profile_id=pid, get_dataset_profile_hash=lambda d: "h"))
    assert PipelineControlTab._dataset_profile(fake, ds) == ("a", "h")
    assert PipelineControlTab._dataset_profile(fake, ds) == ("a", "h")
    assert len(calls) == 1
    _bump_mtime(ds / DATASET_MANIFEST)
    PipelineControlTab._dataset_profile(fake, ds)
    assert len(calls) == 2


def test_no_poll_loops_left_in_the_converted_tabs() -> None:
    """The queue-draining after() loops are gone (N11); remaining timers only watch state."""
    root = Path(__file__).resolve().parents[1]
    for rel in ("gui/tabs/pipeline/tab.py", "gui/tabs/weights/tab.py", "gui/tabs/predictions/tab.py",
                "gui/tabs/profile_preview/tab.py", "tools/profile_editor/app.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert "get_nowait" not in src, rel
        assert "frame.after(0" not in src, rel  # never after() from a worker thread
