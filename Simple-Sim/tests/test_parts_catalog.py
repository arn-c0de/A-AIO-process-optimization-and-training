"""Part catalog (simple_sim.parts) and its use in board3d - synthetic data only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_sim.board3d.parts import apply_part_catalog
from simple_sim.parts import PartCatalog, norm_mpn, upsert_entry, validate_entry


def _cat(tmp_path: Path, parts: list[dict]) -> Path:
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps({"schema": 1, "parts": parts}))
    return p


def test_builtin_catalog_is_valid_and_regex_match(tmp_path):
    cat = PartCatalog.load(local=[])                     # built-in layer only (generic public parts)
    assert cat.entries and all(not validate_entry(e.data) for e in cat.entries)
    m = cat.lookup(texts=["B13B-XH-A(LF)(SN)"])
    assert m is not None and m.entry.id == "jst-xh-top" and m.entry.origin == "builtin"
    cat = PartCatalog.load(local=[_cat(tmp_path, [{"id": "series", "match": ["\\bB\\d{1,2}B-XY-A"], "body": {"h": 6.0}}])])
    m = cat.lookup(texts=["Maker B4B-XY-A(LF)(SN)"])
    assert m is not None and m.entry.id == "series" and m.how == "regex" and m.entry.height_mm == 6.0
    assert cat.lookup(texts=["10k"], package="RESC1005") is None


def test_norm_mpn():
    assert norm_mpn("AB12-CD-3(X)") == "AB12CD3X"


def test_local_wins_and_mpn_beats_package(tmp_path):
    local = _cat(tmp_path, [
        {"id": "by-package", "packages": ["PKG-A"], "body": {"h": 1.0}},
        {"id": "by-mpn", "mpn": ["ABC-1234"], "body": {"h": 2.0}},
        {"id": "series", "match": ["B\\d+B-XY-A"], "body": {"h": 7.0}},
    ])
    cat = PartCatalog.load(local=[local])
    assert cat.lookup(texts=["Maker ABC1234"], package="PKG-A").entry.id == "by-mpn"
    assert cat.lookup(texts=["x"], package="PKG-A").entry.id == "by-package"
    assert cat.lookup(texts=["B4B-XY-A"]).entry.id == "series"


def test_upsert_and_validation(tmp_path):
    p = tmp_path / "sub" / "c.json"
    upsert_entry({"id": "a", "mpn": ["AAAA1"], "body": {"h": 1}}, p)
    upsert_entry({"id": "a", "mpn": ["AAAA1"], "body": {"h": 3}}, p)
    data = json.loads(p.read_text())
    assert [e["body"]["h"] for e in data["parts"]] == [3]
    with pytest.raises(ValueError):
        upsert_entry({"id": "b", "body": {"h": 1}}, p)            # nothing to match


def test_apply_to_board3d(tmp_path):
    cat = PartCatalog.load(local=[_cat(tmp_path, [{"id": "conn", "packages": ["CONN-X"], "body": {"h": 9.5},
                                                   "source": {"kind": "datasheet"}}])])
    b = {"components": [
        {"refdes": "J1", "package": "CONN-X", "value": "", "body": {"l": 5, "w": 4, "h": None, "heightSource": None}},
        {"refdes": "J2", "package": "OTHER", "value": "", "body": {"l": 5, "w": 4, "h": None, "heightSource": None}},
        {"refdes": "J3", "package": "CONN-X", "value": "", "body": {"l": 5, "w": 4, "h": 4.0, "heightSource": "measured"}},
    ]}
    res = apply_part_catalog(b, cat)
    j1, _j2, j3 = b["components"]
    assert j1["body"]["h"] == 9.5 and j1["body"]["heightSource"] == "catalog" and j1["body"]["catalog"]["id"] == "conn"
    assert j3["body"]["h"] == 4.0                                   # measured stays
    assert res["unmatched"] == ["OTHER"] and list(res["applied"]) == ["J1"]
