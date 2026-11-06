"""Tests for the tk/torch-free shared helpers (io_utils, reports, cli_utils)."""

import json
import os

from simple_sim import io_utils, reports
from simple_sim.cli_utils import RESULT_JSON_PREFIX, emit_result, parse_result


def test_atomic_write_and_read_json(tmp_path):
    p = tmp_path / "a.json"
    io_utils.atomic_write_json(p, {"x": 1})
    assert io_utils.read_json(p) == {"x": 1}
    assert not (tmp_path / "a.json.tmp").exists()
    assert io_utils.read_json(tmp_path / "missing.json", default={}) == {}
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert io_utils.read_json(tmp_path / "bad.json") is None


def test_append_jsonl(tmp_path):
    p = tmp_path / "r.jsonl"
    io_utils.append_jsonl(p, [{"a": 1}])
    io_utils.append_jsonl(p, [{"a": 2}])
    assert [json.loads(line)["a"] for line in p.read_text().splitlines()] == [1, 2]


def test_unique_path(tmp_path):
    p = tmp_path / "f.txt"
    assert io_utils.unique_path(p) == p
    p.write_text("x")
    assert io_utils.unique_path(p) == tmp_path / "f_001.txt"


def test_format_bytes_and_size(tmp_path):
    assert io_utils.format_bytes(-5) == "0 B"
    assert io_utils.format_bytes(1536) == "1.50 KB"
    assert io_utils.format_bytes(1536, decimals=1) == "1.5 KB"
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a").write_bytes(b"12345")
    assert io_utils.path_size_bytes(tmp_path / "d") == 5
    assert io_utils.path_size_bytes(tmp_path / "nope") == 0


def test_reports_load_and_query(tmp_path):
    d = tmp_path / "outputs" / "models"
    d.mkdir(parents=True)
    io_utils.atomic_write_json(d / "report_m1.json", {"metrics": {"accuracy": 0.8, "critical_fn_rates": {"MISSING": 0.1}}})
    io_utils.atomic_write_json(d / "batch_report_m2_x.json", {"metrics": {"accuracy": 0.9}, "timestamp": "2026-01-01T00:00:00"})
    io_utils.atomic_write_json(d / "report_bad.json", {"no": "metrics"})
    dirs = reports.report_search_dirs(tmp_path)
    assert d in dirs
    all_r = reports.load_reports(dirs)
    assert len(all_r) == 2 and all("_path" in r for r in all_r)
    assert len(reports.load_reports(dirs, model_stem="m1")) == 1
    assert reports.best_report(all_r, "accuracy")["metrics"]["accuracy"] == 0.9
    m1 = reports.load_reports(dirs, model_stem="m1")[0]
    assert reports.metric_value(m1, "critical_fn_rate", "MISSING") == 0.1
    assert reports.metric_value(m1, "critical_fn_rate") is None
    assert reports.report_timestamp(m1) == os.path.getmtime(m1["_path"])


def test_result_line_roundtrip(capsys):
    print("noise")
    emit_result("demo", {"value": 3})
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith(RESULT_JSON_PREFIX)
    assert parse_result(out) == {"ok": True, "command": "demo", "value": 3}
    assert parse_result("nothing") is None
