"""Phase 2 shared helpers: devices (D1), argparse fragments (D2), manifest blocks (D5), dataset file
names (D7), bundle listing/resolution (D9/D10), profile hash of a loaded dict (D11), sha256 tag (D16)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from simple_sim import cli_utils, dataset_files
from simple_sim.io_utils import SHA256_PREFIX, sha256_tag
from simple_sim.manifest import hash_file, write_dataset_manifest, write_multi_profile_manifest
from simple_sim.model_bundle import list_bundle_checkpoints, resolve_bundle_checkpoint, scan_bundle_checkpoints
from simple_sim.profile_hash import hash_profile, hash_profile_dict, load_profile
from simple_sim.schema import LabelRow

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "configs" / "profiles"


# D1 ------------------------------------------------------------------------

def test_resolve_device() -> None:
    torch = pytest.importorskip("torch")
    from simple_sim.devices import default_device_name, resolve_device

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert default_device_name() == expected
    assert resolve_device(None) == torch.device(expected)
    assert resolve_device("cpu") == torch.device("cpu")
    dev = torch.device("cpu")
    assert resolve_device(dev) == dev
    with pytest.raises(RuntimeError):
        resolve_device("nope")


# D2 ------------------------------------------------------------------------

def test_argparse_fragments() -> None:
    ap = argparse.ArgumentParser()
    cli_utils.add_data_arg(ap, "d")
    cli_utils.add_model_arg(ap, "m", required=False)
    cli_utils.add_out_arg(ap, "o", required=False, default="x")
    cli_utils.add_amp_flags(ap)
    cli_utils.add_profile_mismatch_flag(ap, "p")
    args = ap.parse_args(["--data", "a"])
    assert (args.data, args.model, args.out, args.amp, args.allow_profile_mismatch) == ("a", None, "x", True, False)
    args = ap.parse_args(["--data", "a", "--no-amp", "--allow-profile-mismatch", "--model", "m.pt"])
    assert (args.amp, args.allow_profile_mismatch, args.model) == (False, True, "m.pt")
    with pytest.raises(SystemExit):
        ap.parse_args(["--data", "a", "--amp", "--no-amp"])
    with pytest.raises(SystemExit):
        ap.parse_args([])


# D5 ------------------------------------------------------------------------

def _rows(n: int, profile_id: str = "p@1") -> list[LabelRow]:
    return [LabelRow(schema_version=2, id=f"r/d/train/{i:06d}", class_name="OK", profile_id=profile_id) for i in range(n)]


def test_manifest_v1_blocks_and_key_order(tmp_path: Path) -> None:
    rows = _rows(3)
    write_dataset_manifest(tmp_path, "run", "p@1", "sha256:x", "p.yaml", rows, rows, {"train": [r.id for r in rows]},
                           renderer_version="9", git_commit="abc")
    m = json.loads((tmp_path / dataset_files.DATASET_MANIFEST).read_text())
    assert list(m) == ["manifest_version", "created_at", "run_id", "component_profile", "generator",
                       "dataset_stats", "extend_history"]
    assert list(m["generator"].items()) == [("version", m["generator"]["version"]), ("git_commit", "abc"),
                                            ("script", "scripts/generate.py"), ("renderer_version", "9")]
    entry = m["extend_history"][0]
    assert list(entry) == ["timestamp", "samples_added", "git_commit", "renderer_version"]
    assert (entry["samples_added"], entry["git_commit"], entry["renderer_version"]) == (3, "abc", "9")

    more = _rows(2)
    write_dataset_manifest(tmp_path, "run", "p@1", "sha256:x", "p.yaml", more, more, {"train": [r.id for r in rows + more]},
                           extend=True, git_commit="def")  # no renderer_version: entry without the key
    m = json.loads((tmp_path / dataset_files.DATASET_MANIFEST).read_text())
    assert list(m["extend_history"][1]) == ["timestamp", "samples_added", "git_commit"]
    assert m["extend_history"][1]["samples_added"] == 2 and m["generator"]["renderer_version"] == "9"


def test_manifest_v2_blocks_and_key_order(tmp_path: Path) -> None:
    profiles = [{"profile_id": "a", "profile_hash": "h", "profile_path": "a.yaml"},
                {"profile_id": "b", "profile_hash": "h", "profile_path": "b.yaml"}]
    rows = _rows(4, "a")
    write_multi_profile_manifest(tmp_path, "run", profiles, rows, rows, {"train": []}, git_commit="abc")
    first = json.loads((tmp_path / dataset_files.DATASET_MANIFEST).read_text())
    assert list(first["generator"]) == ["version", "git_commit", "script"]
    assert list(first["extend_history"][0]) == ["timestamp", "samples_added", "git_commit"]

    first["generator"]["renderer_version"] = "7"
    rows = _rows(6, "a")
    write_multi_profile_manifest(tmp_path, "run", profiles, rows, rows, {"train": []}, existing=first, git_commit="abc")
    m = json.loads((tmp_path / dataset_files.DATASET_MANIFEST).read_text())
    assert list(m["generator"]) == ["version", "git_commit", "script", "renderer_version"]
    assert m["generator"]["renderer_version"] == "7"
    assert list(m["extend_history"][1]) == ["timestamp", "samples_added", "git_commit", "renderer_version"]
    assert m["extend_history"][1]["samples_added"] == 2


# D7 ------------------------------------------------------------------------

def test_dataset_file_names() -> None:
    assert (dataset_files.DATASET_MANIFEST, dataset_files.META_JSONL, dataset_files.LABELS_JSONL,
            dataset_files.CONFIG_YAML, dataset_files.SPLITS_DIR) == (
        "dataset_manifest.json", "meta.jsonl", "labels.jsonl", "config.yaml", "splits")


# D9 / D10 ------------------------------------------------------------------

def test_list_bundle_checkpoints(tmp_path: Path) -> None:
    for name in ("b.pt", "a.pt", "a_last.pt", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in list_bundle_checkpoints(tmp_path)] == ["a.pt", "b.pt"]
    assert [p.name for p in list_bundle_checkpoints(tmp_path, include_last=True)] == ["a.pt", "a_last.pt", "b.pt"]
    assert scan_bundle_checkpoints(tmp_path) == {"a": "a.pt", "b": "b.pt"}  # unreadable files: stem as id


def test_resolve_bundle_checkpoint_error_text(tmp_path: Path) -> None:
    bundle = tmp_path / "m.bundle"
    bundle.mkdir()
    (bundle / "other.pt").write_bytes(b"x")
    with pytest.raises(FileNotFoundError) as exc:
        resolve_bundle_checkpoint(bundle, "p@1")
    assert str(exc.value) == (f"Multi-model bundle has no checkpoint for profile 'p@1':\n"
                              f"  bundle: {bundle}\n  expected: {bundle / 'p@1.pt'}")
    with pytest.raises(FileNotFoundError) as exc:
        resolve_bundle_checkpoint(bundle, "p@1", show_available=True, hint="\nTo add it: x")
    assert str(exc.value) == (f"Multi-model bundle has no checkpoint for profile 'p@1':\n"
                              f"  bundle: {bundle}\n  expected: {bundle / 'p@1.pt'}\n"
                              f"  available: other.pt\n\nTo add it: x")
    empty = tmp_path / "e.bundle"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=r"available: \(none\)"):
        resolve_bundle_checkpoint(empty, "p@1", show_available=True)
    (bundle / "p@1.pt").write_bytes(b"x")
    assert resolve_bundle_checkpoint(bundle, "p@1") == bundle / "p@1.pt"
    assert resolve_bundle_checkpoint(tmp_path / "plain.pt", "p@1") == tmp_path / "plain.pt"


def test_batch_predict_bundle_error_text_unchanged(tmp_path: Path) -> None:
    from scripts.batch_predict import resolve_model_path

    data = tmp_path / "ds"
    data.mkdir()
    rows = _rows(1)
    write_dataset_manifest(data, "run", "p@1", "sha256:x", "p.yaml", rows, rows, {"train": []}, git_commit="a")
    bundle = tmp_path / "m.bundle"
    bundle.mkdir()
    with pytest.raises(FileNotFoundError) as exc:
        resolve_model_path(bundle, data)
    assert str(exc.value) == (
        f"Multi-model bundle has no checkpoint for profile 'p@1':\n"
        f"  bundle: {bundle}\n"
        f"  expected: {bundle / 'p@1.pt'}\n"
        f"  available: (none)\n\n"
        f"To add it, train this dataset into the same bundle:\n"
        f"  ./.venv/bin/python scripts/train.py --data {data} --out {bundle}"
    )


def test_weights_tab_bundle_error_text_unchanged(tmp_path: Path) -> None:
    import queue

    from gui.tabs.weights.logic import WeightsLogic

    data = tmp_path / "ds"
    data.mkdir()
    rows = _rows(1)
    write_dataset_manifest(data, "run", "p@1", "sha256:x", "p.yaml", rows, rows, {"train": []}, git_commit="a")
    bundle = tmp_path / "m.bundle"
    bundle.mkdir()
    (bundle / "q@1.pt").write_bytes(b"x")
    logic = WeightsLogic(tmp_path, queue.Queue())
    with pytest.raises(FileNotFoundError) as exc:
        logic._predict_cmd(bundle, data, "test", "auto", "", False)
    assert str(exc.value) == (
        f"Multi-model bundle has no checkpoint for profile 'p@1':\n"
        f"  bundle: {bundle}\n"
        f"  expected: {bundle / 'p@1.pt'}\n"
        f"  available: q@1.pt\n\n"
        f"Train this dataset into the same bundle to add it:\n"
        f"  ./.venv/bin/python scripts/train.py --data {data} --out {bundle}\n"
    )


# D11 / D16 -----------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(PROFILES.glob("*.yaml"))[:6], ids=lambda p: p.stem)
def test_hash_profile_dict_matches_file_hash(path: Path) -> None:
    assert hash_profile_dict(load_profile(path.stem, PROFILES)) == hash_profile(path)


def test_sha256_tag(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc")
    assert sha256_tag(digest) == SHA256_PREFIX + digest.hexdigest() == hash_file(f)
    assert hash_file(f).startswith("sha256:")
