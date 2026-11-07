"""Plan phase 3 (A2, A4, A6): typed records and API fixes, with the files they touch unchanged.

- ``DatasetManifest``: ``from_dict``/``to_dict`` round-trip byte for byte (key order, unknown
  keys) for every manifest the writers produce and for freshly generated datasets;
  ``load_dataset_manifest``, ``DatasetIndex.manifest()``.
- ``Report``: read-only mapping that still answers ``report["_path"]``; ``ModelRanking``.
- Keyword-only manifest writers (legacy positional form still works silently, same bytes),
  ``stratified_splits``, ``parse_iso_ts`` contract, ``config_schema`` deprecation, ``schema`` ->
  ``rows`` alias.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import pytest
import yaml

from simple_sim import manifest as manifest_mod
from simple_sim.dataset_files import DATASET_MANIFEST
from simple_sim.io_utils import atomic_write_json, parse_iso_ts
from simple_sim.manifest import (DatasetManifest, ProfileRef, load_dataset_manifest, read_dataset_manifest,
                                 record_added_samples, resolve_dataset_profile, write_dataset_manifest,
                                 write_multi_profile_manifest)
from simple_sim.rows import LabelRow

ROOT = Path(__file__).resolve().parent.parent


# -- helpers -----------------------------------------------------------------------------------

def _labels(n: int, profile_id: str = "p@1", start: int = 0) -> list[LabelRow]:
    return [LabelRow(schema_version=2, id=f"r/d/train/{i:06d}", class_name="OK" if i % 2 else "MISSING",
                     profile_id=profile_id) for i in range(start, start + n)]


def _assert_round_trip(path: Path, *, same_bytes: bool = True) -> DatasetManifest:
    """``to_dict`` equals the parsed file, and writing it gives the same bytes (*same_bytes*: for
    files written by the library writers, i.e. ``atomic_write_json``)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    typed = DatasetManifest.from_dict(raw)
    out = typed.to_dict()
    assert out == raw
    assert list(out) == list(raw)
    assert json.dumps(out) == json.dumps(raw)  # nested key order too
    if not same_bytes:
        return typed
    with tempfile.TemporaryDirectory() as tmp:
        copy_path = Path(tmp) / DATASET_MANIFEST
        atomic_write_json(copy_path, out)
        assert copy_path.read_bytes() == path.read_bytes()
    return typed


def _written_manifests(tmp_path: Path) -> list[Path]:
    """Every manifest shape the library writers produce (v1 new/extend, v2 new/extend, real crops)."""
    paths: list[Path] = []
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    v1.mkdir()
    v2.mkdir()
    rows = _labels(3)
    write_dataset_manifest(v1, "run", profile_id="p@1", profile_hash="sha256:x", profile_path="p.yaml",
                           meta_rows=rows, label_rows=rows, splits={"train": [r.id for r in rows]},
                           renderer_version="9", git_commit="abc")
    shutil.copy(v1 / DATASET_MANIFEST, tmp_path / "v1_new.json")
    paths.append(tmp_path / "v1_new.json")
    more = _labels(2, start=3)
    write_dataset_manifest(v1, "run", profile_id="p@1", profile_hash="sha256:x", profile_path="p.yaml",
                           meta_rows=more, label_rows=more, splits={"train": [r.id for r in rows + more]},
                           extend=True, git_commit="def")
    shutil.copy(v1 / DATASET_MANIFEST, tmp_path / "v1_extended.json")
    paths.append(tmp_path / "v1_extended.json")
    record_added_samples(v1 / DATASET_MANIFEST, split="train", classes={"OK": 2},
                         log_key="real_crops", log_entry={"source": "x", "n": 2})  # unknown top-level key
    paths.append(v1 / DATASET_MANIFEST)

    profiles = [{"profile_id": "a", "profile_hash": "h", "profile_path": "a.yaml"},
                {"profile_id": "b", "profile_hash": "h", "profile_path": "b.yaml"}]
    rows = _labels(4, "a")
    write_multi_profile_manifest(v2, "run", profiles=profiles, meta_rows=rows, label_rows=rows,
                                 splits={"train": []}, git_commit="abc")
    shutil.copy(v2 / DATASET_MANIFEST, tmp_path / "v2_new.json")
    paths.append(tmp_path / "v2_new.json")
    first = read_dataset_manifest(v2 / DATASET_MANIFEST)
    rows = _labels(6, "a") + _labels(2, "b", start=6)
    write_multi_profile_manifest(v2, "run", profiles=profiles, meta_rows=rows, label_rows=rows,
                                 splits={"train": [r.id for r in rows]}, existing=first, renderer_version="7")
    paths.append(v2 / DATASET_MANIFEST)
    return paths


# -- DatasetManifest ---------------------------------------------------------------------------

def test_manifest_round_trip_of_every_written_shape(tmp_path: Path) -> None:
    for path in _written_manifests(tmp_path):
        _assert_round_trip(path)


def test_manifest_round_trip_unknown_keys_and_order(tmp_path: Path) -> None:
    raw = {
        "zz_first": [1, {"b": 2, "a": 1}],
        "run_id": "r",
        "manifest_version": 1,
        "component_profile": {"profile_hash": "h", "note": "x", "profile_id": "p@1"},  # no profile_path
        "dataset_stats": {"total_samples": 5, "splits": {"val": 1, "train": 4}, "classes": {"OK": 5}},
        "extend_history": [{"samples_added": 5, "timestamp": "t"}],
        "generator": {"script": "s", "renderer_version": 3},
        "future": None,
    }
    typed = DatasetManifest.from_dict(raw)
    assert typed.to_dict() == raw and json.dumps(typed.to_dict()) == json.dumps(raw)
    assert typed.extra == {"zz_first": [1, {"b": 2, "a": 1}], "future": None}
    assert typed.profiles[0].extra == {"note": "x"} and typed.profiles[0].profile_path == ""
    assert typed.profile_ids == ("p@1",) and not typed.is_multi_profile
    assert typed.renderer_version == "3" and typed.total_samples == 5
    assert typed.split_counts == {"val": 1, "train": 4} and typed.class_counts == {"OK": 5}
    assert "created_at" not in typed.to_dict()  # missing known keys stay missing
    # to_dict() is a deep copy: mutating it leaves the record alone
    out = typed.to_dict()
    out["dataset_stats"]["total_samples"] = 0
    out["component_profile"]["profile_id"] = "other"
    assert typed.total_samples == 5 and typed.to_dict() == raw

    # odd shapes are kept verbatim in ``extra``
    odd = {"manifest_version": 2, "component_profiles": "not-a-list", "extend_history": {"x": 1}}
    t = DatasetManifest.from_dict(odd)
    assert t.profiles_key is None and t.profiles == () and t.to_dict() == odd


def test_manifest_profile_views_v1_v2(tmp_path: Path) -> None:
    paths = _written_manifests(tmp_path)
    v1 = load_dataset_manifest(paths[0])
    assert v1.version == 1 and v1.profile_ids == ("p@1",) and v1.renderer_version == "9"
    assert v1.git_commit == "abc"
    assert resolve_dataset_profile(v1) == resolve_dataset_profile(v1.to_dict()) == ("p@1", "sha256:x")
    assert v1.resolve_profile() == ("p@1", "sha256:x")
    v2 = load_dataset_manifest(paths[-1])
    assert v2.is_multi_profile and v2.profile_ids == ("a", "b")
    assert resolve_dataset_profile(v2) == ("multi", "multi")
    assert v2.profile("b") == ProfileRef("b", "h", "b.yaml", {}, ("profile_id", "profile_hash", "profile_path"))
    assert v2.profile("nope") is None
    assert v2.profile_counts == {"a": 6, "b": 2} and v2.renderer_version == "7"
    assert len(v2.extend_history) == 2


def test_load_dataset_manifest_errors_match_read(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset_manifest(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"manifest_version": 1}))
    with pytest.raises(ValueError, match="missing field 'created_at'"):
        load_dataset_manifest(bad)


def test_fresh_datasets_round_trip(tmp_path: Path, monkeypatch) -> None:
    """Manifests of a freshly generated v1 dataset (new and extended) and a mixed-profile dataset."""
    from simple_sim.generation.dataset import generate_dataset
    from simple_sim.generation.profile_dataset import generate_profile_dataset

    monkeypatch.delenv("IMAGE_FILTERS", raising=False)
    monkeypatch.setenv("SIMPLE_SIM_RENDER_EXECUTOR", "thread")
    cfg = yaml.safe_load((ROOT / "configs" / "run_0001.yaml").read_text(encoding="utf-8"))
    cfg["classes"] = {name: 1 for name in list(cfg["classes"])[:2]}
    cfg["roi"].update(width_px=48, height_px=48)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "ds"
    generate_dataset(cfg_path, out, project_root=ROOT)
    _assert_round_trip(out / DATASET_MANIFEST)
    generate_dataset(cfg_path, out, extend=True, project_root=ROOT)
    typed = _assert_round_trip(out / DATASET_MANIFEST)
    assert len(typed.extend_history) == 2 and typed.profile_ids == (cfg["run"]["component_profile"],)

    pcfg = yaml.safe_load((ROOT / "configs" / "run_profile_cls_quick.yaml").read_text(encoding="utf-8"))
    pcfg["classes"] = {"OK": 1, "MISSING": 1}
    pcfg["roi"].update(width_px=48, height_px=48)
    pcfg_path = tmp_path / "pcfg.yaml"
    pcfg_path.write_text(yaml.safe_dump(pcfg, sort_keys=False), encoding="utf-8")
    generate_profile_dataset(pcfg_path, tmp_path / "multi", project_root=ROOT)
    typed = _assert_round_trip(tmp_path / "multi" / DATASET_MANIFEST)
    assert typed.is_multi_profile and typed.profile_ids == tuple(pcfg["run"]["component_profiles"])


def test_checkout_datasets_round_trip() -> None:
    """Every manifest under ``outputs/`` of this checkout (none on a clean clone; hand-made ones
    may be formatted differently, so only the parsed content is compared there)."""
    found = sorted((ROOT / "outputs").rglob(DATASET_MANIFEST)) if (ROOT / "outputs").is_dir() else []
    for path in found:
        text = path.read_text(encoding="utf-8")
        _assert_round_trip(path, same_bytes=text == json.dumps(json.loads(text), indent=2) + "\n")


def test_dataset_index_manifest_cached_and_invalidated(tmp_path: Path) -> None:
    from simple_sim.dataset_index import DatasetIndex, clear_cache

    clear_cache()
    paths = _written_manifests(tmp_path)
    ds = paths[-1].parent
    idx = DatasetIndex(ds)
    m = idx.manifest()
    assert isinstance(m, DatasetManifest) and idx.manifest() is m
    assert DatasetIndex(ds).manifest() is m  # shared per file version
    assert idx.profile_ids() == ("a", "b")
    assert m.to_dict() == read_dataset_manifest(ds / DATASET_MANIFEST)

    data = m.to_dict()
    data["run_id"] = "changed-run"
    atomic_write_json(ds / DATASET_MANIFEST, data)
    st = (ds / DATASET_MANIFEST).stat()
    os.utime(ds / DATASET_MANIFEST, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert idx.manifest().run_id == "changed-run"
    with pytest.raises(FileNotFoundError):
        DatasetIndex(tmp_path / "nothing").manifest()
    clear_cache()


# -- writers: keyword-only (A6) ----------------------------------------------------------------

def test_writers_positional_form_still_works_silently(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(manifest_mod, "now_iso", lambda: "2026-11-07T00:00:00")
    for name in ("kw", "pos", "kw2", "pos2", "x"):
        (tmp_path / name).mkdir()
    rows = _labels(3)
    splits = {"train": [r.id for r in rows]}
    write_dataset_manifest(tmp_path / "kw", "run", profile_id="p@1", profile_hash="h", profile_path="p.yaml",
                           meta_rows=rows, label_rows=rows, splits=splits, git_commit="a")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # legacy positional calls (scripts, GUI) must stay silent
        write_dataset_manifest(tmp_path / "pos", "run", "p@1", "h", "p.yaml", rows, rows, splits, git_commit="a")
    assert (tmp_path / "kw" / DATASET_MANIFEST).read_bytes() == (tmp_path / "pos" / DATASET_MANIFEST).read_bytes()

    profiles = [{"profile_id": "a", "profile_hash": "h", "profile_path": "a.yaml"},
                {"profile_id": "b", "profile_hash": "h", "profile_path": "b.yaml"}]
    write_multi_profile_manifest(tmp_path / "kw2", "run", profiles=profiles, meta_rows=rows, label_rows=rows,
                                 splits=splits, script_name="s", git_commit="a")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_multi_profile_manifest(tmp_path / "pos2", "run", profiles, rows, rows, splits, "s", git_commit="a")
    assert (tmp_path / "kw2" / DATASET_MANIFEST).read_bytes() == (tmp_path / "pos2" / DATASET_MANIFEST).read_bytes()

    with pytest.raises(TypeError, match="multiple values"):
        write_dataset_manifest(tmp_path / "x", "run", "p@1", profile_id="p@1")
    with pytest.raises(TypeError, match="positional"):
        write_multi_profile_manifest(tmp_path / "x", "run", *([None] * 9))
    import inspect
    params = inspect.signature(write_dataset_manifest).parameters
    assert [n for n, p in params.items() if p.kind is p.POSITIONAL_OR_KEYWORD] == ["output_dir", "run_id"]


# -- Report / ModelRanking (A2) ----------------------------------------------------------------

def _write_report(path: Path, acc: float, **extra) -> None:
    path.write_text(json.dumps({"timestamp": "2026-03-01T00:00:00", "model_path": "m.pt",
                                "metrics": {"accuracy": acc, "macro_f1": acc - 0.1}, **extra}))


def test_report_is_a_read_only_mapping_with_legacy_path(tmp_path: Path) -> None:
    from simple_sim import reports

    reports.clear_report_cache()
    _write_report(tmp_path / "report_m1.json", 0.5, _path="stale")  # a stored "_path" is shadowed
    _write_report(tmp_path / "batch_report_m2_1.json", 0.7)
    rs = reports.load_reports([tmp_path])
    assert all(isinstance(r, reports.Report) for r in rs)
    r = next(x for x in rs if x.path.name == "report_m1.json")
    sp = str(tmp_path / "report_m1.json")
    assert r.path == Path(sp) and r["_path"] == sp and r.get("_path") == sp and "_path" in r
    assert reports.report_path(r) == sp and reports.report_path({"_path": "x"}) == "x"
    assert reports.report_path({}) is None
    assert r["metrics"]["accuracy"] == 0.5 and r.get("missing", 1) == 1 and "missing" not in r
    with pytest.raises(KeyError):
        r["missing"]
    # the former dict, key order included: file keys, then "_path" (appended where it was missing)
    other = next(x for x in rs if x is not r)
    legacy = json.loads((tmp_path / "batch_report_m2_1.json").read_text())
    legacy["_path"] = str(other.path)
    assert dict(other) == other.to_dict() == legacy and list(other) == list(legacy) and len(other) == len(legacy)
    assert other == legacy  # Mapping equality with the old dict
    legacy1 = json.loads((tmp_path / "report_m1.json").read_text())
    legacy1["_path"] = sp
    assert list(r) == list(legacy1) and len(r) == len(legacy1) and dict(r) == legacy1
    with pytest.raises(TypeError):
        r["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        r.data["x"] = 1  # type: ignore[index]
    assert reports.metric_value(r, "accuracy") == 0.5
    assert reports.report_timestamp(r) == parse_iso_ts("2026-03-01T00:00:00")
    assert reports.best_report(rs, "accuracy") is other
    reports.clear_report_cache()


def test_arena_same_rows_for_reports_and_legacy_dicts(tmp_path: Path) -> None:
    from simple_sim import arena, reports

    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    d = tmp_path / "outputs" / "models"
    d.mkdir(parents=True)
    _write_report(d / "report_m.json", 0.9, model_path=str(model), dataset_path="ds1", split="test")
    _write_report(d / "batch_report_m_1.json", 0.8, model_path=str(model), dataset_path="ds2", split="test")
    reports.clear_report_cache()
    rs = reports.load_reports([d])
    tracked = {str(model.resolve()): {"path": str(model)}}
    a = arena.best_rows_by_model(tmp_path, tracked, rs)
    b = arena.best_rows_by_model(tmp_path, tracked, [r.to_dict() for r in rs])
    assert a == b
    assert {row.report_path for row in a[str(model.resolve())]} == {str(r.path) for r in rs}

    overall = arena.overall_ranking(tmp_path, a)
    assert all(isinstance(it, arena.ModelRanking) for it in overall)
    assert list(overall[0].to_dict()) == ["model_abs", "model_rel", "model_name", "type", "avg_acc", "avg_f1",
                                          "datasets", "best_ds", "worst_ds", "last_ts"]
    assert overall[0].datasets == 2 and overall[0].type == "Single"
    assert overall[0].avg_acc == pytest.approx(0.85)
    reports.clear_report_cache()


# -- splits, timestamps, model_bundle (A6) -----------------------------------------------------

def test_stratified_splits_matches_config_form() -> None:
    from simple_sim.splits import generate_splits, stratified_splits

    from types import SimpleNamespace

    # only ``id`` and ``domain`` of a MetaRow are read
    meta = [SimpleNamespace(id=f"r/{dom}/{i:06d}", domain=dom) for dom in ("domain_A", "domain_B") for i in range(12)]
    labels = [LabelRow(schema_version=1, id=m.id, class_name="OK" if i % 3 else "MISSING")
              for i, m in enumerate(meta)]
    cfg = {"splits": {"train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2,
                      "train_domains": ["domain_A"], "val_domains": ["domain_A", "domain_B"],
                      "test_domains": ["domain_A", "domain_B"]}}
    expected = generate_splits(meta, labels, cfg, 7)
    got = stratified_splits(meta, labels, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=7,
                            split_domains={"train": ["domain_A"], "val": ["domain_A", "domain_B"],
                                           "test": ["domain_A", "domain_B"]})
    assert got == expected
    assert not any(i.startswith("r/domain_B") for i in got["train"])
    plain = stratified_splits(iter(meta), iter(labels), train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=7)
    assert plain == generate_splits(meta, labels, {"splits": {"train_frac": 0.6, "val_frac": 0.2,
                                                              "test_frac": 0.2}}, 7)
    with pytest.raises(KeyError):
        generate_splits(meta, labels, {"splits": {"train_frac": 0.6}}, 7)


def test_parse_iso_ts_failure_contract() -> None:
    assert parse_iso_ts("2026-03-01T00:00:00") > 0
    for bad in (None, "", "yesterday", 12, object()):
        assert parse_iso_ts(bad) == 0.0


def test_bundle_checkpoint_kind_literal(tmp_path: Path) -> None:
    import typing

    from simple_sim import model_bundle

    assert typing.get_args(model_bundle.CheckpointKind) == ("best", "last")
    assert model_bundle.bundle_checkpoint_path(tmp_path, "p@1", kind="last").name == "p@1_last.pt"
    assert model_bundle.bundle_checkpoint_path(tmp_path, "p@1").name == "p@1.pt"


# -- A4: config_schema deprecated, schema -> rows ------------------------------------------------

def test_config_schema_is_a_deprecated_shim() -> None:
    from simple_sim import config

    sys.modules.pop("simple_sim.config_schema", None)
    with pytest.warns(DeprecationWarning, match="config_schema is deprecated"):
        cs = importlib.import_module("simple_sim.config_schema")
    cfg = {"run": {"run_id": "r1", "seed": 4}, "roi": {"width_px": 8, "height_px": 8, "mm_per_px": 0.1},
           "train": {"model": "resnet18", "epochs": 2}, "eval": {"batch_size": 3}}
    typed = cs.parse_config_typed(cfg)
    assert typed.run.seed == 4 and typed.train.epochs == 2 and typed.eval.batch_size == 3
    assert config.core_section_values(cfg)["train"]["epochs"] == 2
    # validate_config keeps the former first-check error messages
    with pytest.raises(ValueError, match="Missing required top-level section"):
        config.validate_config({"run": {}, "roi": {}, "train": {}})
    bad = {**cfg, "train": {"epochs": "many"}}
    with pytest.raises(ValueError, match="Invalid value in config section 'train'"):
        config.validate_config(bad)
    import subprocess

    code = ("import sys, simple_sim.config, simple_sim.generation.dataset; "
            "assert 'simple_sim.config_schema' not in sys.modules")  # nothing imports the shim
    subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c", code], cwd=ROOT, check=True)


def test_schema_is_an_alias_of_rows() -> None:
    from simple_sim import rows, schema
    from simple_sim.schema import LabelRow as L, read_jsonl

    assert schema is rows and L is rows.LabelRow and read_jsonl is rows.read_jsonl
