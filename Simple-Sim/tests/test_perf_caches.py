"""Per-process caches of plan S2/S5/S8: telemetry sink, git commit, reports, path sizes, bundle scan."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from simple_sim import io_utils, manifest, reports, telemetry


# --- S2: telemetry ---------------------------------------------------------------------------

@pytest.fixture
def clean_sink(monkeypatch):
    monkeypatch.delenv(telemetry.EVENT_LOG_ENV, raising=False)
    telemetry.reset_sink()
    yield
    telemetry.reset_sink()


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_emit_resolves_env_once_and_appends_lines(tmp_path, monkeypatch, clean_sink):
    log = tmp_path / "sub" / "events.jsonl"
    monkeypatch.setenv(telemetry.EVENT_LOG_ENV, str(log))
    telemetry.emit("a", n=1)
    telemetry.emit("b", n=2)
    assert [(e["event"], e["n"]) for e in _events(log)] == [("a", 1), ("b", 2)]  # flushed per event
    assert all(isinstance(e["ts"], float) for e in _events(log))
    monkeypatch.setenv(telemetry.EVENT_LOG_ENV, str(tmp_path / "other.jsonl"))
    telemetry.emit("c")
    assert len(_events(log)) == 3 and not (tmp_path / "other.jsonl").exists()
    telemetry.reset_sink()
    telemetry.emit("d")
    assert _events(tmp_path / "other.jsonl")[0]["event"] == "d"


def test_set_sink_overrides_and_disables(tmp_path, monkeypatch, clean_sink):
    monkeypatch.setenv(telemetry.EVENT_LOG_ENV, str(tmp_path / "env.jsonl"))
    telemetry.set_sink(tmp_path / "explicit.jsonl")
    telemetry.emit("x")
    assert telemetry.current_sink() == str(tmp_path / "explicit.jsonl")
    assert _events(tmp_path / "explicit.jsonl")[0]["event"] == "x"
    telemetry.set_sink(None)
    telemetry.emit("y")
    assert telemetry.current_sink() is None and not (tmp_path / "env.jsonl").exists()


def test_emit_never_raises(tmp_path, clean_sink):
    telemetry.emit("noop")  # unset: no-op
    telemetry.set_sink(tmp_path)  # a directory: open fails -> ignored
    telemetry.emit("bad")
    telemetry.set_sink(tmp_path / "ok.jsonl")
    telemetry.emit("obj", value=object())  # not serialisable -> ignored, nothing written
    telemetry.emit("fine")
    assert [e["event"] for e in _events(tmp_path / "ok.jsonl")] == ["fine"]


def test_git_commit_cached_and_overridable(tmp_path, monkeypatch):
    calls = []
    real_run = manifest.subprocess.run

    def run(*a, **k):
        calls.append(a)
        return real_run(*a, **k)

    manifest._get_git_commit.cache_clear()
    monkeypatch.setattr(manifest.subprocess, "run", run)
    try:
        first = manifest._get_git_commit()
        assert manifest._get_git_commit() == first and len(calls) == 1
    finally:
        manifest._get_git_commit.cache_clear()


def test_manifest_git_commit_parameter(tmp_path, monkeypatch):
    from simple_sim.schema import LabelRow

    monkeypatch.setattr(manifest, "_get_git_commit", lambda: pytest.fail("must not resolve"))
    manifest.write_multi_profile_manifest(
        tmp_path, "r", [{"profile_id": "p@1", "profile_hash": "sha256:x", "profile_path": "p.yaml"}],
        meta_rows=[], label_rows=[LabelRow(schema_version=2, id="a", class_name="OK", profile_id="p@1")],
        splits={"train": ["a"], "val": [], "test": []}, git_commit="abc1234",
    )
    m = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert m["generator"]["git_commit"] == "abc1234"


# --- S5: reports, arena, path sizes ----------------------------------------------------------

def _write_report(path: Path, acc: float, **extra) -> None:
    path.write_text(json.dumps({"metrics": {"accuracy": acc}, **extra}))


def test_load_reports_cache_and_invalidation(tmp_path, monkeypatch):
    reports.clear_report_cache()
    d = tmp_path / "models"
    d.mkdir()
    _write_report(d / "report_m1.json", 0.5)
    _write_report(d / "batch_report_m2_1.json", 0.7)
    (d / "report_bad.json").write_text("{")
    reads = []
    real = reports.read_json
    monkeypatch.setattr(reports, "read_json", lambda p, *a: (reads.append(Path(p).name), real(p, *a))[1])

    first = reports.load_reports([d])
    assert sorted(r["metrics"]["accuracy"] for r in first) == [0.5, 0.7]
    assert len(reads) == 3
    first[0]["_mutated"] = True  # top-level dicts are per call
    again = reports.load_reports([d])
    assert len(reads) == 3 and all("_mutated" not in r for r in again)
    assert [r["_path"] for r in again] == [r["_path"] for r in first]

    _write_report(d / "report_m1.json", 0.95)  # size changes
    st = (d / "report_m1.json").stat()
    os.utime(d / "report_m1.json", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    third = {Path(r["_path"]).name: r["metrics"]["accuracy"] for r in reports.load_reports([d])}
    assert third == {"report_m1.json": 0.95, "batch_report_m2_1.json": 0.7}
    assert reads.count("report_m1.json") == 2
    assert [Path(r["_path"]).name for r in reports.load_reports([d], model_stem="m1")] == ["report_m1.json"]
    reports.clear_report_cache()


def test_path_size_bytes_matches_os_walk(tmp_path):
    t = tmp_path / "tree"
    (t / "a" / "b").mkdir(parents=True)
    (t / "x.bin").write_bytes(b"1234")
    (t / "a" / "y.bin").write_bytes(b"12")
    (t / "a" / "b" / "z.bin").write_bytes(b"1")
    (t / "dirlink").symlink_to(t / "a")
    (t / "filelink").symlink_to(t / "x.bin")
    (t / "broken").symlink_to(t / "nope")

    expected = 0
    for root, _dirs, files in os.walk(str(t)):
        for fn in files:
            fp = Path(root) / fn
            if not fp.is_symlink():
                expected += fp.stat().st_size
    assert io_utils.path_size_bytes(t) == expected == 7
    assert io_utils.path_size_bytes(t / "x.bin") == 4
    assert io_utils.path_size_bytes(t / "filelink") == 4  # a file argument is followed, as before
    assert io_utils.path_size_bytes(t / "nope") == 0


def test_best_rows_pick_and_last_ts(tmp_path):
    from simple_sim import arena

    model = tmp_path / "outputs" / "models" / "m.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")
    tracked = {str(model.resolve()): {}}
    base = {"model_path": str(model), "dataset_path": str(tmp_path / "ds"), "split": "test"}
    rs = [
        {**base, "metrics": {"accuracy": 0.8, "macro_f1": 0.7}, "timestamp": "2026-01-01T00:00:00"},
        {**base, "metrics": {"accuracy": 0.8, "macro_f1": 0.7}, "timestamp": "2026-03-01T00:00:00", "_path": "b"},
        {**base, "metrics": {"accuracy": 0.6, "macro_f1": 0.9}, "timestamp": "2026-05-01T00:00:00"},
    ]
    rows = arena.best_rows_by_model(tmp_path, tracked, rs)[str(model.resolve())]
    assert len(rows) == 1
    assert rows[0].report_path == "b"  # tie on acc/f1 -> newest
    assert rows[0].last_ts == reports.report_timestamp(rs[2])


# --- S8: bundle scan without a full load -------------------------------------------------------

def test_scan_bundle_reads_profile_ids(tmp_path):
    torch = pytest.importorskip("torch")
    from simple_sim import model_bundle

    b = tmp_path / "x.bundle"
    b.mkdir()
    state = {"w": torch.zeros(4)}
    torch.save({"model_state_dict": state, "component_profile": {"profile_id": "a@1"}}, b / "a.pt")
    torch.save({"model_state_dict": state, "component_profile": {"profile_id": "b@2"}}, b / "legacy.pt",
               _use_new_zipfile_serialization=False)
    torch.save({"model_state_dict": state}, b / "noprof.pt")
    torch.save({"model_state_dict": state, "component_profile": {"profile_id": "a@1"}}, b / "a_last.pt")
    (b / "broken.pt").write_bytes(b"not a checkpoint")
    assert model_bundle.scan_bundle_checkpoints(b) == {
        "a@1": "a.pt", "broken": "broken.pt", "b@2": "legacy.pt", "noprof": "noprof.pt",
    }
