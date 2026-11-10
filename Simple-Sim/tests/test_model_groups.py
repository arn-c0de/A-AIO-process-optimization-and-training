"""Model grouping shared by the weights and merge tabs (gui.utils.model_groups, Tk-free)."""

from __future__ import annotations

from pathlib import Path

from gui.utils.model_groups import available_groups, group_for_model


def test_available_groups_order_and_dedup() -> None:
    assert available_groups([], {}) == ["Favorites", "Snapshots", "Imports", "Uncategorized"]
    groups = available_groups(["Mine", "", "Imports"], {"a.pt": "Zeta", "b.pt": "Alpha", "c.pt": "Mine"}, bundles=True)
    assert groups == ["Favorites", "Snapshots", "Imports", "Bundles", "Mine", "Alpha", "Zeta", "Uncategorized"]


def test_group_for_model_precedence(tmp_path: Path) -> None:
    root = tmp_path / "models"
    snap = root / "versions" / "m" / "v1.pt"
    imp = root / "imports" / "x.pt"
    plain = root / "x.pt"
    for p in (snap, imp, plain):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    groups = available_groups(["Mine"], {})

    def group(p: Path, **kw) -> str:
        args = {"model_root": root, "assignments": {}, "groups": groups, "favorite": False}
        args.update(kw)
        return group_for_model(p, p.name, **args)

    assert group(snap) == "Snapshots"
    assert group(imp) == "Imports"
    assert group(plain) == "Uncategorized"
    assert group(plain, bundled=True) == "Bundles"
    assert group(snap, favorite=True, bundled=True) == "Favorites"
    assert group(snap, favorite=True, assignments={"v1.pt": "Mine"}) == "Mine"
    # An assignment to a group that no longer exists is ignored.
    assert group(snap, assignments={"v1.pt": "Gone"}) == "Snapshots"
