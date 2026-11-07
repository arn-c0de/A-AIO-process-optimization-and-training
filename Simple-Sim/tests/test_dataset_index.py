"""DatasetIndex (plan S1/L11): one parse per file version, same rows/samples as the direct readers;
sample_check split selection from the split files (C1)."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from simple_sim import dataset_index
from simple_sim.data_loader import ProfileDataset, ROIDataset
from simple_sim.dataset_index import DatasetIndex
from simple_sim.schema import LabelRow, MetaRow, read_jsonl, write_jsonl
from simple_sim.splits import write_splits

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = ["OK", "MISSING", "TOMBSTONE"]


def _meta(sid: str, split: str, image: str) -> MetaRow:
    return MetaRow(
        schema_version=2, id=sid, run_id="r", domain="domain_A", split=split, seed=1, image_path=image,
        render_backend="opencv_2d", footprint="0603",
        nominal={"pad_width": 1.0, "pad_height": 1.0, "pad_spacing": 1.0, "component_length": 1.0,
                 "component_width": 1.0},
        defect={"type": "OK", "shift_x": 0.0, "shift_y": 0.0, "rotation_deg": 0.0, "tilt_deg": 0.0},
        augment={"blur_sigma": 0.0, "noise_stddev": 0.0, "brightness_factor": 1.0, "contrast_factor": 1.0,
                 "rotation_deg": 0.0},
    )


def make_dataset(root: Path, n: int = 12, *, resplit: bool = False) -> Path:
    """*n* samples with ids ``r/domain_A/<split>/NNNNNN``; *resplit* moves them to other split files
    (a merged/re-split dataset whose ids no longer name their split)."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    img = np.full((8, 8, 3), 100, np.uint8)
    names = ("train", "val", "test")
    meta, labels, splits = [], [], {s: [] for s in names}
    for i in range(n):
        id_split = names[i % 3]
        sid = f"r/domain_A/{id_split}/{i:06d}"
        image = f"images/{i:06d}.png"
        cv2.imwrite(str(root / image), img)
        meta.append(_meta(sid, id_split, image))
        labels.append(LabelRow(schema_version=2, id=sid, class_name=CLASSES[(i // 3) % len(CLASSES)],
                               profile_id=f"p{i % 2}@1"))
        splits[names[(i + 1) % 3] if resplit else id_split].append(sid)
    write_jsonl(root / "meta.jsonl", meta)
    write_jsonl(root / "labels.jsonl", labels)
    write_splits(root, splits)
    return root


@pytest.fixture(autouse=True)
def _fresh_cache():
    dataset_index.clear_cache()
    yield
    dataset_index.clear_cache()


def test_index_matches_direct_readers(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    idx = DatasetIndex(ds)
    assert [vars(r) for r in idx.meta_rows()] == [vars(r) for r in read_jsonl(ds / "meta.jsonl", MetaRow)]
    assert [vars(r) for r in idx.label_rows()] == [vars(r) for r in read_jsonl(ds / "labels.jsonl", LabelRow)]
    assert set(idx.id_to_meta()) == {r.id for r in idx.meta_rows()}
    assert idx.split_sizes() == {"train": 4, "val": 4, "test": 4}
    assert idx.class_distribution() == {"OK": 6, "MISSING": 3, "TOMBSTONE": 3}
    assert idx.profile_map() == {r.id: r.profile_id for r in idx.label_rows()}


def test_index_parses_each_file_once(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path / "ds")
    calls: list[str] = []
    real = dataset_index.read_jsonl

    def counting(path, cls):
        calls.append(Path(path).name)
        return real(path, cls)

    monkeypatch.setattr(dataset_index, "read_jsonl", counting)
    sets = [ROIDataset(ds, s, CLASSES, return_id=True) for s in ("train", "val", "test")]
    DatasetIndex(ds).label_rows()
    assert sorted(calls) == ["labels.jsonl", "meta.jsonl"]
    assert sets[0].id_to_meta is sets[1].id_to_meta  # shared, read-only


def test_cache_invalidates_on_rewrite_and_append(tmp_path):
    ds = make_dataset(tmp_path / "ds", n=6)
    idx = DatasetIndex(ds)
    first = idx.label_rows()
    assert idx.label_rows() is first
    # Rewrite in place with different content but the same size and mtime: inode/ctime still differ.
    labels = ds / "labels.jsonl"
    st = labels.stat()
    rows = list(first)
    rows[0] = LabelRow(schema_version=2, id=rows[0].id, class_name="MISSING", profile_id=rows[0].profile_id)
    write_jsonl(labels, rows)
    os.utime(labels, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert idx.label_rows()[0].class_name == "MISSING"
    # Append: size changes.
    with open(labels, "a", encoding="utf-8") as fh:
        fh.write('{"schema_version": 2, "id": "extra", "class_name": "OK", "profile_id": "p0@1"}\n')
    assert idx.label_rows()[-1].id == "extra"
    # Split file rewritten.
    (ds / "splits" / "test.txt").write_text("r/domain_A/train/000000\n")
    assert idx.split_ids("test") == ("r/domain_A/train/000000",)


def test_missing_files_raise_like_before(tmp_path):
    idx = DatasetIndex(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        idx.meta_rows()
    with pytest.raises(FileNotFoundError):
        idx.split_ids("train")
    assert idx.split_sizes() == {"train": 0, "val": 0, "test": 0}
    with pytest.raises(FileNotFoundError):
        ROIDataset(tmp_path / "nope", "train", CLASSES)


def test_roi_dataset_items_unchanged(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    # class list with a duplicate: list.index semantics (first occurrence) are kept.
    names = ["OK", "MISSING", "OK", "TOMBSTONE"]
    d = ROIDataset(ds, "val", names, return_id=True)
    ids = (ds / "splits" / "val.txt").read_text().split()
    assert d.samples == ids
    label_of = {r.id: r.class_name for r in read_jsonl(ds / "labels.jsonl", LabelRow)}
    for i, sid in enumerate(d.samples):
        x, y, got_id, rel = d[i]
        assert (got_id, y, rel) == (sid, names.index(label_of[sid]), d.id_to_meta[sid].image_path)
        assert tuple(x.shape) == (3, 8, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        only_ok = ROIDataset(ds, "train", ["OK"])
    assert all(label_of[s] == "OK" for s in only_ok.samples)
    p = ProfileDataset(ds, "test", ["p1@1", "p0@1"])
    assert [p._get_label_index(s) for s in p.samples] == [
        ["p1@1", "p0@1"].index(r.profile_id) for r in read_jsonl(ds / "labels.jsonl", LabelRow)
        if r.id in set(p.samples)
    ]


def test_batch_predict_dataset_info_and_profile_map(tmp_path):
    from scripts.batch_predict import _gt_profile_map, build_dataset, dataset_info

    ds = make_dataset(tmp_path / "ds", n=9)
    info = dataset_info(ds)
    assert info.total_samples == 9
    assert info.split_sizes == {"train": 3, "val": 3, "test": 3}
    assert info.class_distribution == {"OK": 3, "MISSING": 3, "TOMBSTONE": 3}
    assert info.class_list == ["OK", "MISSING", "TOMBSTONE"]
    assert len(_gt_profile_map(ds)) == 9
    assert _gt_profile_map(tmp_path / "missing") == {}
    assert len(build_dataset(ds, "all", CLASSES)) == 9


def _old_choose(data_dir, class_names, split, per_class, seed):
    """The pre-C1 rule (``f"/{split}/" in sample_id``), for comparison."""
    import random

    labels = {r.id: r.class_name for r in read_jsonl(data_dir / "labels.jsonl", LabelRow)}
    cand = {c: [] for c in class_names}
    for sid, cls in labels.items():
        if f"/{split}/" in sid and cls in cand:
            cand[cls].append(sid)
    rng = random.Random(seed)
    out = []
    for cls in class_names:
        rng.shuffle(cand[cls])
        out.extend((sid, cls) for sid in cand[cls][:per_class])
    return sorted(out, key=lambda t: t[0])


def test_sample_check_uses_split_files(tmp_path):
    """C1: in a re-split dataset the ``/<split>/`` part of an id is not its split."""
    from scripts.sample_check import choose_samples

    ds = make_dataset(tmp_path / "ds", n=12, resplit=True)
    want = set((ds / "splits" / "test.txt").read_text().split())
    chosen, id_to_meta = choose_samples(ds, CLASSES, "test", per_class=10, seed=0)
    got = {sid for sid, _ in chosen}
    assert got == want
    assert not any("/test/" in sid for sid in got)  # the old substring rule would pick none of these
    assert chosen == sorted(chosen)
    assert set(id_to_meta) == {r.id for r in read_jsonl(ds / "meta.jsonl", MetaRow)}
    # Regular dataset (ids name their split): exactly the former substring selection.
    plain = make_dataset(tmp_path / "plain", n=30)
    a, _ = choose_samples(plain, CLASSES, "val", per_class=2, seed=3)
    assert a == _old_choose(plain, CLASSES, "val", 2, 3) and len(a) == 6
    (plain / "splits" / "val.txt").unlink()
    assert choose_samples(plain, CLASSES, "val", per_class=2, seed=3)[0] == []
