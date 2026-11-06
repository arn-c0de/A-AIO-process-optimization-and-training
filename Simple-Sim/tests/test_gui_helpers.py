"""Tests for tk-free GUI helper modules (gui/utils/*)."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

import pytest
import yaml

from gui.utils.checkpoint_meta import (
    CheckpointMetaCache,
    checkpoint_class_names,
    checkpoint_profile,
    profile_label_suffix,
    short_profile_hash,
)
from gui.utils.dataset_catalog import DatasetCatalog
from gui.utils.filter_profile_store import (
    FILTER_SETTING_SPECS,
    float_or_default,
    formatted_filter_settings,
    image_filters_payload,
    normalized_filter_settings,
)
from gui.utils.formatting import fmt_bytes
from gui.utils.paths import newest_file
from gui.utils.pipeline_helpers import (
    DEFAULT_CLASS_NAMES,
    apply_precise_counts,
    chain_commands,
    config_class_names,
    continue_train_argv,
    decode_dataset_paths_json,
    encode_dataset_paths_json,
    eval_argv,
    mixed_profile_config,
    multi_train_argvs,
    parse_profile_ids,
    parse_run_count,
    parse_snap_every,
    parse_snap_keep,
    precise_counts,
    precise_counts_for_profile,
    profile_compat_status,
    profile_model_paths,
    profile_run_targets,
    resolve_under,
    sample_color_from_total,
    samples_text,
    summarize_labels,
    summarize_profile_ids,
    train_argv,
    unique_timestamped_dir,
    write_temp_yaml,
)
from gui.utils.settings_store import SettingsStore


# --------------------------------------------------------------------------- formatting / paths

def test_fmt_bytes_one_decimal() -> None:
    assert fmt_bytes(0) == "0 B"
    assert fmt_bytes(-5) == "0 B"
    assert fmt_bytes(1023) == "1023 B"
    assert fmt_bytes(1536) == "1.5 KB"
    assert fmt_bytes(5 * 1024**3) == "5.0 GB"


def test_newest_file(tmp_path: Path) -> None:
    assert newest_file(tmp_path / "missing", "*.json") is None
    assert newest_file(tmp_path, "*.json") is None
    a, b = tmp_path / "batch_report_a.json", tmp_path / "batch_report_b.json"
    a.write_text("{}")
    b.write_text("{}")
    os.utime(a, (1000, 1000))
    os.utime(b, (2000, 2000))
    (tmp_path / "other.txt").write_text("")
    assert newest_file(tmp_path, "batch_report_*.json") == b
    assert newest_file(tmp_path, "batch_report_*.json", min_mtime=1500) == b
    assert newest_file(tmp_path, "batch_report_*.json", min_mtime=2500) is None


# --------------------------------------------------------------------------- checkpoint metadata

def test_checkpoint_profile_and_labels() -> None:
    assert checkpoint_profile(None) == (None, None)
    assert checkpoint_profile({"component_profile": "x"}) == (None, None)
    assert checkpoint_profile({"component_profile": {"profile_id": "p", "profile_hash": "h"}}) == ("p", "h")
    assert short_profile_hash("sha256:0123456789abcdef") == "0123456789ab"
    assert short_profile_hash("0123456789abcdef") == "0123456789ab"
    assert profile_label_suffix({}) == " | Profile: ⚠ No profile (legacy)"
    ckpt = {"component_profile": {"profile_id": "chip@1", "profile_hash": "sha256:aaaabbbbccccdddd"}}
    assert profile_label_suffix(ckpt) == " | Profile: chip@1 (aaaabbbbcccc...)"


def test_checkpoint_meta_cache_and_class_names(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    p = tmp_path / "m.pt"
    torch.save({"component_profile": {"profile_id": "a"}, "class_names": ["OK", "MISSING"]}, p)

    cache = CheckpointMetaCache()
    assert cache.cached(tmp_path / "nope.pt") == {"error": "missing"}
    assert cache.cached(p) is None
    meta = cache.read(p)
    assert meta == {"component_profile": {"profile_id": "a"}, "class_names": ["OK", "MISSING"]}
    assert cache.cached(p) is meta

    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    assert "error" in cache.read(bad)

    bundle = tmp_path / "x.bundle"
    bundle.mkdir()
    assert checkpoint_class_names(bundle) == []
    torch.save({"class_names": ["A", "B"]}, bundle / "b.pt")
    assert checkpoint_class_names(bundle) == ["A", "B"]
    assert checkpoint_class_names(bad) == []


# --------------------------------------------------------------------------- filter settings

def test_float_or_default() -> None:
    assert float_or_default(" 1.5 ", default=0.0) == 1.5
    assert float_or_default("x", default=2) == 2.0
    assert float_or_default(None, default=3.0) == 3.0


def test_filter_settings_round_trip_and_defaults() -> None:
    defaults = normalized_filter_settings(lambda _k, d: d)
    assert list(defaults) == [k for k, _kind, _d in FILTER_SETTING_SPECS]
    assert defaults["enable_rotation"] is True and defaults["enable_reflection"] is False
    assert defaults["shadow_strength"] == "0.30"
    assert defaults["jpeg_quality"] == "85"

    raw = {"rotation_strength": "2", "jpeg_quality": "90.7", "enable_blur": 0, "hue_shift_deg": "bad"}
    formatted = formatted_filter_settings(lambda k: raw.get(k, ""))
    assert formatted["rotation_strength"] == "2.00"
    assert formatted["jpeg_quality"] == "90"
    assert formatted["enable_blur"] is False
    assert formatted["hue_shift_deg"] == "0.00"  # falls back to the default


def test_image_filters_payload() -> None:
    raw = {"enable_blur": True, "blur_strength": "0.5", "jpeg_quality": "70"}
    extras = {"blur_randomize": "yes", "blur_min": "0.1", "filter_mode": "custom", "unrelated": 1}
    payload = image_filters_payload(lambda k: raw.get(k, ""), extras)
    assert payload["enable"] is True
    assert payload["blur_strength"] == 0.5
    assert payload["shadow_size"] == 0.2  # default
    assert payload["jpeg_quality"] == 70
    assert payload["blur_randomize"] is True
    assert payload["blur_min"] == 0.1
    assert "unrelated" not in payload

    payload = image_filters_payload(lambda _k: "", {"filter_mode": "realism", "realism_enabled": "1", "realism_profile_id": "cam", "realism_noise": "0.2"})
    assert payload["enable"] is True  # realism mode enables the pipeline regardless of toggles
    assert payload["realism_profile_id"] == "cam"
    assert payload["realism_noise"] == 0.2


# --------------------------------------------------------------------------- precise mode

def test_precise_counts_variants() -> None:
    assert precise_counts({"total": 50}) == ({"_total": 50}, None)
    assert precise_counts({"total": 0})[1] == "Precise mode: invalid total count.\nOpen Precise Settings."
    assert precise_counts({"per_class": True, "classes": {"OK": "3", "MISSING": 0}}) == ({"OK": 3, "MISSING": 0}, None)
    assert precise_counts({"per_class": True, "classes": {"OK": -1}})[1] == "Precise mode: invalid count for class 'OK'."
    assert precise_counts({"per_class": True, "classes": {}})[0] is None

    multi = {"multi_profiles": ["a", "b"], "total": 7, "multi_totals": {"a": 3}}
    assert precise_counts(multi) == ({"_per_profile": {"a": {"_total": 3}, "b": {"_total": 7}}}, None)
    bad = {"multi_profiles": ["a"], "per_class": True, "multi_classes": {"a": {"OK": "x"}}}
    assert precise_counts(bad) == (None, "Precise mode: invalid count for 'a' / 'OK'.")


def test_precise_counts_application() -> None:
    per = {"_per_profile": {"a": {"_total": 3}, "b": {"OK": 1}}}
    assert precise_counts_for_profile(per, "b") == {"OK": 1}
    assert precise_counts_for_profile(per, "zzz") == {"_total": 3}
    assert precise_counts_for_profile({"_total": 9}, "a") == {"_total": 9}

    cfg = {"classes": {"OK": 10, "MISSING": 5}}
    assert apply_precise_counts(dict(cfg), {"_total": 2})["classes"] == {"OK": 2, "MISSING": 2}
    assert apply_precise_counts(dict(cfg), {"OK": "4"})["classes"] == {"OK": 4}


def test_write_temp_yaml_and_class_names(tmp_path: Path) -> None:
    out = write_temp_yaml({"classes": {"A": 1, "B": 2}}, tmp_path / "live", prefix="precise_")
    assert out.parent == tmp_path / "live" and out.name.startswith("precise_") and out.suffix == ".yaml"
    assert config_class_names(out) == ["A", "B"]
    assert config_class_names(None) == list(DEFAULT_CLASS_NAMES)
    assert config_class_names(tmp_path / "missing.yaml") == list(DEFAULT_CLASS_NAMES)


# --------------------------------------------------------------------------- run specs

def test_mixed_profile_config() -> None:
    cfg = mixed_profile_config({"run": {"component_profile": "x", "run_id": ""}, "render": {"backend": "blender_3d"}}, ["a", "b"])
    assert cfg["run"] == {"mode": "profile_classifier", "schema_version": 2, "component_profiles": ["a", "b"], "run_id": "run_profile_cls_selected"}
    assert cfg["render"]["backend"] == "opencv_2d"
    assert yaml.safe_load(yaml.safe_dump(cfg)) == cfg


def test_unique_timestamped_dir(tmp_path: Path) -> None:
    base = tmp_path / "ds"
    assert unique_timestamped_dir(base, "T") == tmp_path / "ds_T"
    (tmp_path / "ds_T").mkdir()
    (tmp_path / "ds_T_01").mkdir()
    assert unique_timestamped_dir(base, "T") == tmp_path / "ds_T_02"


def test_profile_run_targets() -> None:
    models = Path("/s/outputs/models")
    assert profile_run_targets(Path("runs/a"), Path("m.pt"), "r", multi=False, models_root=models) == (Path("runs/a"), Path("m.pt"))
    assert profile_run_targets(Path("x/runs"), Path("models/m.pt"), "r", multi=True, models_root=models) == (Path("x/runs/r"), Path("models/r.pt"))
    assert profile_run_targets(Path("x/runs/a"), Path("m.bundle"), "r", multi=True, models_root=models) == (Path("x/runs/r"), models / "r.pt")
    assert profile_run_targets(Path("x/y/a"), Path("m.pt"), "r", multi=True, models_root=models) == (Path("x/y/r"), Path("r.pt"))


def test_profile_model_paths(tmp_path: Path) -> None:
    cfg, ds, model = profile_model_paths(tmp_path, "quick", "TAG")
    assert cfg == tmp_path / "configs" / "run_profile_cls_quick.yaml"
    assert ds == tmp_path / "outputs" / "sim_data" / "runs" / "run_profile_cls_quick"
    assert model == tmp_path / "outputs" / "models" / "profile_classifier_quick.pt"
    ds.mkdir(parents=True)
    model.parent.mkdir(parents=True, exist_ok=True)
    (model.parent / "profile_classifier_v1.pt").write_bytes(b"")
    _cfg, ds2, model2 = profile_model_paths(tmp_path, "full", "TAG")
    assert ds2.name == "run_profile_cls"  # full dataset does not exist yet
    assert model2.name == "profile_classifier_v1_TAG.pt"
    assert profile_model_paths(tmp_path, "quick", "TAG")[1].name == "run_profile_cls_quick_TAG"


def test_parse_numeric_fields() -> None:
    assert parse_run_count("single", "x") == 1
    assert parse_run_count("precise", "") == 1
    assert parse_run_count("continuous", "") == -1
    assert parse_run_count("multiple", "4") == 4
    for bad in ("0", "x", ""):
        with pytest.raises(ValueError):
            parse_run_count("multiple", bad)
    assert [parse_snap_every(s) for s in ("", "x", "0", " 3 ")] == [1, 5, 1, 3]
    assert [parse_snap_keep(s) for s in ("", "x", "-2", "7")] == [0, 30, 0, 7]


# --------------------------------------------------------------------------- display text

def test_display_text_helpers() -> None:
    assert [sample_color_from_total(n) for n in (10, 1000, 5000, 10000)] == ["black", "orange", "blue", "green"]
    assert samples_text(12, 8, 2, 2) == "Samples: 12 (t8/v2/s2)"
    assert samples_text(12, 8, None, 2) == "Samples: 12"
    assert summarize_profile_ids([]) == ""
    assert summarize_profile_ids(["a", "b", "c"]) == "a, b, c"
    assert summarize_profile_ids(["a", "b", "c", "d", "e"]) == "a, b, c (+2)"
    assert summarize_labels([]) == "(none)"
    assert summarize_labels(["x"]) == "x"
    assert summarize_labels(["x", "y", "z"]) == "x (+2)"


def test_profile_compat_status() -> None:
    assert profile_compat_status("p", "h", None) == ("⚠ Model has no profile (legacy)", "orange")
    assert profile_compat_status("p", "h", {"profile_id": "multi"})[1] == "green"
    assert profile_compat_status("p", "h", {"profile_id": "q"}) == ("✗ INCOMPATIBLE: Model=q, Dataset=p", "red")
    assert profile_compat_status("p", "h", {"profile_id": "p", "profile_hash": "z"})[1] == "orange"
    assert profile_compat_status("p", "h", {"profile_id": "p", "profile_hash": "h"}) == ("✓ Compatible: p", "green")


# --------------------------------------------------------------------------- serialized selections

def test_parse_profile_ids() -> None:
    assert parse_profile_ids('["a", " b ", "a", 3]', set()) == ["a", "b"]
    assert parse_profile_ids("a, b,,c", {"a", "c"}) == ["a", "c"]
    assert parse_profile_ids("", {"a"}) == []


def test_dataset_paths_json_round_trip(tmp_path: Path) -> None:
    raw = json.dumps([str(tmp_path / "a"), "rel/b", str(tmp_path / "a"), 5])
    paths = decode_dataset_paths_json(raw, tmp_path)
    assert paths == [tmp_path / "a", (tmp_path / "rel" / "b").resolve()]
    assert decode_dataset_paths_json("not json", tmp_path) == []
    assert decode_dataset_paths_json('{"a": 1}', tmp_path) == []
    assert json.loads(encode_dataset_paths_json(paths, tmp_path)) == [str(p) for p in paths]
    assert resolve_under(tmp_path, "x/y") == tmp_path / "x" / "y"
    assert resolve_under(tmp_path, "/abs") == Path("/abs")


# --------------------------------------------------------------------------- command lines

def test_command_builders() -> None:
    ds_a, ds_b, model = Path("/d/a b"), Path("/d/c"), Path("/m/x.pt")
    assert chain_commands([train_argv(ds_a, model)]) == shlex.join(["./.venv/bin/python", "scripts/train.py", "--data", "/d/a b", "--out", "/m/x.pt"])
    assert " && " in chain_commands([train_argv(ds_a, model), train_argv(ds_b, model)])
    assert eval_argv(ds_b, model, allow_profile_mismatch=True)[-1] == "--allow-profile-mismatch"
    assert continue_train_argv(ds_b, model, extra_epochs=3, out_mode="last", allow_profile_mismatch=True)[6:] == [
        "--resume", "/m/x.pt", "--allow-profile-mismatch", "--extra-epochs", "3", "--out-mode", "last",
    ]


def test_multi_train_argvs_single_checkpoint_and_bundle() -> None:
    dss = [Path("/d/a"), Path("/d/b"), Path("/d/c")]
    model = Path("/m/x.pt")
    resume = ["--resume", "/m/x.pt", "--extra-epochs", "5", "--out-mode", "best"]

    argvs = multi_train_argvs(dss, model, ["p", "q", "p"], is_bundle=False, extra_epochs=5, out_mode="best", allow_profile_mismatch=True)
    assert argvs[0] == train_argv(dss[0], model)
    assert argvs[1] == train_argv(dss[1], model) + resume + ["--allow-profile-mismatch"]

    bundle = Path("/m/x.bundle")
    argvs = multi_train_argvs(dss, bundle, ["p", None, "p"], is_bundle=True, extra_epochs=5, out_mode="best", allow_profile_mismatch=True)
    assert argvs[0] == train_argv(dss[0], bundle)
    assert argvs[1] == train_argv(dss[1], bundle)  # first time for this (missing) profile
    assert argvs[2][-6:] == ["--resume", str(bundle), "--extra-epochs", "5", "--out-mode", "best"]


# --------------------------------------------------------------------------- dataset catalog / settings

def test_dataset_catalog_unique_labels(tmp_path: Path) -> None:
    store = SettingsStore(path=tmp_path / "settings.json")
    runs, versions = tmp_path / "runs", tmp_path / "versions"
    a, b, snap = runs / "a", runs / "b", versions / "g" / "a"
    for d in (a, b, snap):
        d.mkdir(parents=True)
    cat = DatasetCatalog(tmp_path, store)
    cat.set_category([b], "Mine")
    labels = cat.unique_labels([a, b, snap, a], runs_root=runs, versions_root=versions)
    assert list(labels) == ["a", "[Mine] b", "g:a", "a (2)"]
    assert cat.display_label(b, runs_root=runs, versions_root=versions) == "[Mine] b"


def test_settings_store_save_is_atomic_json(tmp_path: Path) -> None:
    store = SettingsStore(path=tmp_path / "sub" / "settings.json")
    store.set("k", "v")
    store.save()
    assert json.loads((tmp_path / "sub" / "settings.json").read_text()) == {"k": "v"}
    assert not (tmp_path / "sub" / "settings.json.tmp").exists()
