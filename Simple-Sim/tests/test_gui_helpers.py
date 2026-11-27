"""Tests for tk-free GUI helper modules (gui/utils/*)."""

from __future__ import annotations

import json
import os
import shlex
import sys
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
from gui.utils.dataset_catalog import DatasetCatalog, catalog_labels, sim_data_roots
from gui.utils.filter_profile_store import (
    FILTER_SETTING_SPECS,
    float_or_default,
    formatted_filter_settings,
    image_filters_payload,
    normalized_filter_settings,
)
from gui.utils.formatting import fmt_bytes, fmt_list
from gui.utils.paths import list_model_paths, newest_file
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
from simple_sim.arena import dataset_display_name


# --------------------------------------------------------------------------- formatting / paths

def test_fmt_bytes_one_decimal() -> None:
    assert fmt_bytes(0) == "0 B"
    assert fmt_bytes(-5) == "0 B"
    assert fmt_bytes(1023) == "1023 B"
    assert fmt_bytes(1536) == "1.5 KB"
    assert fmt_bytes(5 * 1024**3) == "5.0 GB"


def test_fmt_list_truncates_with_count() -> None:
    assert fmt_list([]) == ""
    assert fmt_list(["a", "b"], 2) == "a\nb"
    assert fmt_list(["a", "b", "c"], 2) == "a\nb\n... (+1 more)"


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


def test_list_model_paths(tmp_path: Path) -> None:
    root = tmp_path / "models"
    assert list_model_paths(root) == [] and root.is_dir()
    (root / "versions" / "x").mkdir(parents=True)
    (root / "imports").mkdir()
    (root / "a.bundle").mkdir()
    (root / "bundles" / "b").mkdir(parents=True)
    for f in (root / "m.pt", root / "versions" / "x" / "v.pt", root / "imports" / "i.pt", root / "notes.txt"):
        f.write_text("")
    got = {p.relative_to(root).as_posix() for p in list_model_paths(root)}
    assert got == {"m.pt", "versions/x/v.pt", "imports/i.pt", "a.bundle", "bundles/b"}
    assert "a.bundle" not in {p.name for p in list_model_paths(root, include_bundles=False)}


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
    # Stored relative to the sim root when below it (absolute otherwise); both decode to the same paths.
    assert json.loads(encode_dataset_paths_json(paths, tmp_path)) == ["a", str(Path("rel") / "b")]
    assert json.loads(encode_dataset_paths_json([Path("/elsewhere/c")], tmp_path)) == ["/elsewhere/c"]
    assert decode_dataset_paths_json(encode_dataset_paths_json(paths, tmp_path), tmp_path) == [
        (tmp_path / "a").resolve(), (tmp_path / "rel" / "b").resolve()]
    assert resolve_under(tmp_path, "x/y") == tmp_path / "x" / "y"
    assert resolve_under(tmp_path, "/abs") == Path("/abs")


# --------------------------------------------------------------------------- command lines

def test_command_builders() -> None:
    ds_a, ds_b, model = Path("/d/a b"), Path("/d/c"), Path("/m/x.pt")
    # P4/N9: the GUI's own interpreter and ``python -m simple_sim <command>``, no hard-coded venv or script path.
    assert chain_commands([train_argv(ds_a, model)]) == shlex.join(
        [sys.executable, "-m", "simple_sim", "train", "--data", "/d/a b", "--out", "/m/x.pt"])
    assert " && " in chain_commands([train_argv(ds_a, model), train_argv(ds_b, model)])
    assert eval_argv(ds_b, model, allow_profile_mismatch=True)[-1] == "--allow-profile-mismatch"
    assert continue_train_argv(ds_b, model, extra_epochs=3, out_mode="last", allow_profile_mismatch=True)[8:] == [
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

def test_catalog_labels(tmp_path: Path) -> None:
    store = SettingsStore(path=tmp_path / "settings.json")
    runs, versions = sim_data_roots(tmp_path)
    a, b, c, snap = runs / "a", runs / "b", runs / "c", versions / "g" / "a"
    for d in (a, b, c, snap):
        d.mkdir(parents=True)
    cat = DatasetCatalog(tmp_path, store)
    cat.set_category([b], "Mine")
    cat.set_archived([c], True)
    cat.save()  # catalog_labels() reloads from the store
    labels = catalog_labels(tmp_path, cat)
    assert labels == {"a": a, "[Mine] b": b, "g:a": snap}
    assert cat.display_label(b, runs_root=runs, versions_root=versions) == "[Mine] b"


def test_settings_store_save_is_atomic_json(tmp_path: Path) -> None:
    store = SettingsStore(path=tmp_path / "sub" / "settings.json")
    store.set("k", "v")
    store.save()
    assert json.loads((tmp_path / "sub" / "settings.json").read_text()) == {"k": "v"}
    assert not (tmp_path / "sub" / "settings.json.tmp").exists()


# --------------------------------------------------------------------------- render overrides (look / board)

def test_render_overrides_and_application() -> None:
    from gui.utils.pipeline_helpers import (
        BOARD_DRC_CHOICES,
        BOARD_ROUTING_CHOICES,
        BOARD_RULES_CHOICES,
        FROM_CONFIG,
        LOOK_CHOICES,
        PERSISTED_FIELD_MAP,
        apply_render_overrides,
        render_overrides,
        render_overrides_text,
    )
    from simple_sim.board import BOARD_ROUTERS, DRC_MODES, validate_board_options
    from simple_sim.board.rules import PRESETS
    from simple_sim.generators.opencv2d.photoreal.look import LOOKS, look_name

    assert (FROM_CONFIG, *LOOKS) == LOOK_CHOICES and (FROM_CONFIG, *BOARD_ROUTERS) == BOARD_ROUTING_CHOICES
    assert (FROM_CONFIG, *DRC_MODES) == BOARD_DRC_CHOICES and (FROM_CONFIG, *PRESETS) == BOARD_RULES_CHOICES
    assert render_overrides() == {} and render_overrides(look=FROM_CONFIG, routing="", drc=FROM_CONFIG) == {}
    ov = render_overrides(look="quins", routing="escape", drc="warn", design_rules="hdi")
    assert ov == {"look": "quins", "board": {"routing": "escape", "drc": "warn", "design_rules": "hdi"}}
    assert render_overrides_text(ov) == "look=quins, board.routing=escape, board.drc=warn, board.design_rules=hdi"
    with pytest.raises(ValueError):
        render_overrides(look="matte")
    with pytest.raises(ValueError):
        render_overrides(routing="grid")

    render = {"style": "photoreal", "board": {"pour_probability": 0.3, "routing": "legacy"}}
    out = apply_render_overrides(render, ov)
    assert out is render and out["look"] == "quins"
    assert out["board"] == {"pour_probability": 0.3, "routing": "escape", "drc": "warn", "design_rules": "hdi"}
    validate_board_options(out["board"]) and look_name(out)
    assert apply_render_overrides({"board": None}, {"board": {"drc": "strict"}}) == {"board": {"drc": "strict"}}
    assert apply_render_overrides({"look": "quins"}, {}) == {"look": "quins"}
    # Persisted like the other Pipeline-tab settings.
    keys = dict(PERSISTED_FIELD_MAP)
    assert {keys["pipeline.look"], keys["pipeline.board_routing"], keys["pipeline.board_drc"], keys["pipeline.board_rules"]} == {
        "var_look", "var_board_routing", "var_board_drc", "var_board_rules"}


# --------------------------------------------------------------------------- refactored shared helpers


def test_settings_store_json_values(tmp_path: Path) -> None:
    from gui.utils.settings_store import SettingsStore, parse_json

    store = SettingsStore(tmp_path / "s.json")
    assert store.get_json("k", want=list, fallback=["d"]) == ["d"]
    store.set_json("k", {"a": 1})
    assert store.get_json("k", want=dict, fallback={}) == {"a": 1}
    assert store.get_json("k", want=list, fallback=[]) == []  # wrong type -> fallback
    assert parse_json("{bad", want=dict, fallback=None) is None
    assert parse_json("  ", want=dict, fallback="x") == "x"


def test_dataset_label_helpers(tmp_path: Path) -> None:
    from gui.utils.dataset_catalog import label_for_dataset, same_path, sorted_dataset_labels

    runs, _versions = sim_data_roots(tmp_path)
    (runs / "b").mkdir()
    (runs / "a").mkdir()
    labels = sorted_dataset_labels(tmp_path)
    assert list(labels) == ["a", "b"]
    assert label_for_dataset(labels, runs / "b" / ".." / "b") == "b"
    assert label_for_dataset(labels, tmp_path / "nope", alt_label="a") == "a"
    assert label_for_dataset(labels, None) is None
    assert same_path(runs / "a", runs / "b" / ".." / "a")


def test_default_model_path(tmp_path: Path) -> None:
    from gui.utils.paths import default_model_path

    assert default_model_path(tmp_path, "ds") == tmp_path / "outputs" / "models" / "ds.pt"
    assert default_model_path(tmp_path, "m", ".bundle") == tmp_path / "outputs" / "models" / "m.bundle"


def test_manifest_profile(tmp_path: Path) -> None:
    from gui.utils.dataset_ops import manifest_profile

    assert manifest_profile(tmp_path) == (None, None)
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps({"manifest_version": 1, "component_profile": {"profile_id": "p@1", "profile_hash": "h"}}))
    assert manifest_profile(tmp_path) == ("p@1", "h")
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"manifest_version": 2, "profiles": []}))
    assert manifest_profile(tmp_path) == ("multi", None)


def test_analysis_sample_filtering() -> None:
    from gui.tabs.analysis.logic import dataset_stats_text, filter_options, filter_sample_tree, ordered_class_names

    ids = ["r/A/train/000001", "r/A/val/000002", "r/B/train/000003", "bad"]
    labels = {"r/A/train/000001": "OK", "r/A/val/000002": "MISSING", "r/B/train/000003": "OK"}
    profiles = {"r/A/train/000001": "p1"}
    assert filter_options(ids) == ({"All", "r"}, {"All", "A", "B"})
    all_f = dict.fromkeys(("class", "run", "domain", "split", "profile"), "All")
    tree = filter_sample_tree(ids, labels, profiles, filters={**all_f, "class": "OK"}, selected_profiles=set(),
                              query="", sort_mode="ID Z-A", mtime=lambda _s: 0.0)
    assert tree == {"r": {"A": {"train": ["r/A/train/000001"]}, "B": {"train": ["r/B/train/000003"]}}}
    tree = filter_sample_tree(ids, labels, profiles, filters=all_f, selected_profiles={"unknown_or_legacy"},
                              query="000002", sort_mode="Default", mtime=lambda _s: 0.0)
    assert tree == {"r": {"A": {"val": ["r/A/val/000002"]}}}
    assert dataset_stats_text([], {}) == "Dataset Stats: -"
    assert dataset_stats_text(ids, profiles) == (
        "Dataset Stats: total=4 | train=2, val=1, test=0 | profiles: unknown_or_legacy:3, p1:1")
    assert ordered_class_names({"Z", "MISSING", "OK", ""}) == ["OK", "MISSING", "Z"]


def test_pipeline_snapshots_honour_every_and_prune_their_meta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import queue

    from gui.tabs.pipeline.logic import PipelineLogic, replace_dir

    logic = PipelineLogic(tmp_path, queue.Queue(), queue.Queue(), tmp_path / "events.jsonl")
    model = tmp_path / "outputs" / "models" / "m.pt"
    model.parent.mkdir(parents=True)
    snaps = []
    for run_i in range(1, 7):
        model.write_bytes(b"x" * run_i)  # changed checkpoint every run
        os.utime(model, (run_i, run_i))
        snaps.append(logic._snapshot_model_checkpoint(model, run_i=run_i, snap_every=2, snap_keep=2))
    assert [s is not None for s in snaps] == [False, True, False, True, False, True]
    kept = sorted(p.name for p in snaps[-1].parent.iterdir())
    assert len([n for n in kept if n.endswith(".pt")]) == 2
    assert len([n for n in kept if n.endswith(".pt.meta.json")]) == 2  # pruned snapshots take their meta along

    # A failed snapshot is logged (not swallowed) and retried for the same checkpoint on the next run.
    from gui.tabs.pipeline import logic as pipeline_logic
    from gui.tabs.pipeline.logic import SnapshotOptions

    logic2 = PipelineLogic(tmp_path, queue.Queue(), queue.Queue(), tmp_path / "events.jsonl")
    lines: list[str] = []
    opts = SnapshotOptions(enabled=True, every=1, keep=0)
    def failing_snapshot(*_a, **_k):
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr(pipeline_logic, "snapshot_checkpoint", failing_snapshot)
        logic2.current_run_iteration = 7
        logic2._after_full_run(str(model), str(tmp_path), dataset_mode="new", snapshots=opts, log_callback=lines.append)
    assert any("[model snapshot] failed: disk full" in s for s in lines)
    assert Path(str(model) + ".meta.json").is_file()
    logic2._after_full_run(str(model), str(tmp_path), dataset_mode="new", snapshots=opts, log_callback=lines.append)
    assert lines[-1].startswith("[model snapshot] ") and "failed" not in lines[-1]

    src, dst = tmp_path / "new", tmp_path / "dst"
    src.mkdir()
    (src / "f").write_text("new")
    dst.mkdir()
    (dst / "f").write_text("old")
    replace_dir(src, dst)
    assert (dst / "f").read_text() == "new" and not src.exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("dst.")] == []


def test_arena_names_snapshots_by_dataset_and_version():
    assert dataset_display_name("outputs/sim_data/runs/run_0001") == "run_0001"
    assert dataset_display_name("outputs/sim_data/versions/run_A/v1") == "run_A/v1"
    assert dataset_display_name("outputs/sim_data/versions/run_B/v1") == "run_B/v1"
    assert dataset_display_name("") == "-"


def test_flag_manager_batches_and_survives_a_bad_line(tmp_path):
    from gui.utils.flag_manager import FlagManager

    fm = FlagManager(tmp_path)
    assert fm.add_flags([("a", "OUTLIER", "r1"), ("b", "DUPLICATE", "r2"), ("a", "OUTLIER", "again")]) == 2
    path = tmp_path / "flags.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    path.write_text(lines[0] + "\n{broken\n" + lines[1] + "\n")
    again = FlagManager(tmp_path)
    assert sorted(again.get_all_flagged_samples()) == ["a", "b"]  # the line after the bad one is kept
    again.clear()
    assert path.read_text() == "" and FlagManager(tmp_path).get_all_flagged_samples() == []
    assert [p.name for p in tmp_path.iterdir()] == ["flags.jsonl"]


def test_empty_model_field_falls_back_to_the_run_id_model():
    out, model = profile_run_targets(Path("/d/x"), Path(""), "run_7", multi=False, models_root=Path("/m"))
    assert out == Path("/d/x") and model == Path("/m/run_7.pt")


def test_precise_per_class_counts_only_cover_the_current_class_set():
    settings = {"per_class": True, "class_names": ["OK", "MISSING"],
                "classes": {"OK": 5, "MISSING": 6, "TOMBSTONE": 7}}
    assert precise_counts(settings) == ({"OK": 5, "MISSING": 6}, None)


def test_dataset_kind_tells_runs_from_snapshots(tmp_path: Path) -> None:
    from gui.utils.dataset_catalog import dataset_display_name as gui_display_name
    from gui.utils.dataset_catalog import dataset_kind, is_snapshot_dataset, sim_data_roots

    runs, versions = sim_data_roots(tmp_path)
    run, snap = runs / "run_a", versions / "run_a" / "v1"
    assert dataset_kind(tmp_path, run) == "run" and dataset_kind(tmp_path, snap) == "snapshot"
    assert dataset_kind(tmp_path, runs) is None and dataset_kind(tmp_path, tmp_path / "elsewhere") is None
    assert dataset_kind(tmp_path, tmp_path / "outputs" / "sim_data" / "runs_old" / "x") is None  # prefix, not a parent
    assert is_snapshot_dataset(tmp_path, snap) and not is_snapshot_dataset(tmp_path, run)
    assert gui_display_name(run, runs_root=runs, versions_root=versions) == "run_a"
    assert gui_display_name(snap, runs_root=runs, versions_root=versions) == "run_a:v1"
