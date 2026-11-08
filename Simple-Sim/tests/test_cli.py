"""Tests for the agent entrypoint ``python -m simple_sim`` and tool --json result lines."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from simple_sim.cli import COMMANDS, collect_status, list_commands, main
from simple_sim.cli_utils import parse_result

SIM_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SIM_ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "simple_sim", *args],
        cwd=SIM_ROOT, env=env, capture_output=True, text=True, timeout=120, check=False,
    )


def test_commands_json_lists_all_scripts():
    proc = _run("commands", "--json")
    assert proc.returncode == 0, proc.stderr
    res = parse_result(proc.stdout)
    assert res["ok"] is True and res["command"] == "commands"
    by_name = {c["name"]: c for c in res["commands"]}
    assert {"commands", "status", *COMMANDS} == set(by_name)
    for name in COMMANDS:
        assert by_name[name]["exists"], name
        assert by_name[name]["description"], name
    assert by_name["validate"]["supports_json"] is True


def test_supports_json_comes_from_the_command_source():
    """Library commands declare ``SUPPORTS_JSON``; tool scripts are detected by ``add_json_flag``."""
    import importlib

    for c in list_commands():
        if c["module"]:
            assert c["supports_json"] == importlib.import_module(c["module"]).SUPPORTS_JSON, c["name"]
        elif c["script"]:
            src = (SIM_ROOT / c["script"]).read_text(encoding="utf-8")
            assert c["supports_json"] == ("add_json_flag" in src)


def test_library_commands_run_in_process(monkeypatch, capsys):
    """``main`` calls the command module (no subprocess) and passes a ``python -m simple_sim`` prog."""
    import subprocess as sp

    monkeypatch.setattr(sp, "call", lambda *a, **k: pytest.fail("library command spawned a subprocess"))
    assert main(["train", "--help"]) == 0
    assert capsys.readouterr().out.startswith("usage: python -m simple_sim train ")
    assert main(["train", "--no-such-flag"]) == 2


def test_every_script_is_a_shim_of_its_command():
    for c in list_commands():
        if c["module"]:
            src = (SIM_ROOT / c["script"]).read_text(encoding="utf-8")
            assert f"from {c['module']} import main" in src, c["name"]


def test_argv_for_builds_a_module_command_line():
    from simple_sim.cli import argv_for

    assert argv_for("train", data="/d", out="m.pt", amp=False, resume=None, json=True) == [
        sys.executable, "-m", "simple_sim", "train", "--data", "/d", "--out", "m.pt", "--json"]
    assert argv_for("eval", "--tta", python="py", model="m")[:5] == ["py", "-m", "simple_sim", "eval", "--tta"]
    with pytest.raises(KeyError):
        argv_for("no-such-command")


def test_status_json_on_empty_root(tmp_path):
    proc = _run("status", "--json", "--root", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    res = parse_result(proc.stdout)
    assert res["ok"] is True
    assert res["profiles"] == [] and res["run_configs"] == []
    assert res["datasets"] == [] and res["models"] == []
    assert res["reports"] == {"count": 0, "latest": None, "best_accuracy": None}


def test_status_json_on_repo():
    res = parse_result(_run("status", "--json").stdout)
    assert res["ok"] is True
    assert "chip_0603_resistor@1" in res["profiles"]
    assert any(c["name"] == "run_0001" for c in res["run_configs"])


def test_status_finds_datasets_models_and_reports(tmp_path):
    (tmp_path / "configs" / "profiles").mkdir(parents=True)
    (tmp_path / "configs" / "profiles" / "p@1.yaml").write_text("profile: {}\n")
    ds = tmp_path / "outputs" / "sim_data" / "runs" / "run_x"
    ds.mkdir(parents=True)
    (ds / "dataset_manifest.json").write_text(json.dumps({
        "manifest_version": 1, "created_at": "2026-01-01T00:00:00", "run_id": "run_x",
        "component_profile": {"profile_id": "p@1", "profile_hash": "h", "profile_path": "x"},
        "generator": {}, "dataset_stats": {"total_samples": 3, "splits": {"train": 3}, "classes": {"OK": 3}},
        "extend_history": [],
    }))
    models = tmp_path / "outputs" / "models"
    models.mkdir(parents=True)
    (models / "m.pt").write_bytes(b"x")
    (models / "m_last.pt").write_bytes(b"x")
    for name, acc, ts in (("report_a.json", 0.9, "2026-01-01T00:00:00"), ("report_b.json", 0.5, "2026-02-01T00:00:00")):
        (models / name).write_text(json.dumps({"timestamp": ts, "metrics": {"accuracy": acc, "macro_f1": acc}}))

    st = collect_status(tmp_path)
    assert st["profiles"] == ["p@1"]
    assert st["datasets"][0]["manifest"]["profiles"] == ["p@1"]
    assert st["datasets"][0]["manifest"]["total_samples"] == 3
    assert [m["path"] for m in st["models"]] == ["outputs/models/m.pt"]
    assert st["reports"]["count"] == 2
    assert st["reports"]["best_accuracy"]["accuracy"] == 0.9
    assert st["reports"]["latest"]["path"].endswith("report_b.json")


def test_unknown_command_and_usage(capsys):
    assert main(["no-such-command"]) == 2
    assert main([]) == 2
    assert main(["--help"]) == 0
    assert "validate" in capsys.readouterr().out


def test_validate_json_reports_failed_checks(tmp_path):
    proc = _run("validate", "--data", str(tmp_path), "--json")
    assert proc.returncode == 1
    res = parse_result(proc.stdout)
    assert res["ok"] is False and res["command"] == "validate"
    assert "file:meta.jsonl" in res["failed"]
    assert all({"name", "status", "detail"} <= set(c) for c in res["checks"])


SIM_TO_REAL_COMMANDS = ["sim-to-real", "calibrate-camera", "look-stats", "eval-real", "mix-real", "real-crops",
                        "look-montage"]


@pytest.mark.parametrize("name", SIM_TO_REAL_COMMANDS)
def test_sim_to_real_tool_commands_show_help(name):
    assert name in COMMANDS
    proc = _run(name, "--help")
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout


@pytest.mark.parametrize("script", ["tools/look_montage.py", "tools/real_crops.py"])
def test_tool_scripts_run_directly_without_pythonpath(script, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run([sys.executable, str(SIM_ROOT / script), "--help"], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
