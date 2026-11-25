"""Part catalog: list, look up and add real part geometry (datasheet heights), check a board3d against it.

Examples:
  python -m simple_sim parts list
  python -m simple_sim parts lookup --text "<mpn>" --package <package>
  python -m simple_sim parts check --board3d outputs/board3d/<product>/<side>
  python -m simple_sim parts add --entry new_part.json        # into the local catalog ($SIMPLE_SIM_PARTS_DB)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from simple_sim.cli_utils import add_json_flag, run_cli
from simple_sim.io_utils import load_json

COMMAND = "parts"
SUPPORTS_JSON = True


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Part catalog (datasheet geometry shared by all tools)")
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="all entries (local catalogs first, then the built-in one)")
    lk = sub.add_parser("lookup", help="best entry for a part")
    lk.add_argument("--text", action="append", default=[], help="value / part name / MPN (repeatable)")
    lk.add_argument("--package", default="", help="library package name")
    ck = sub.add_parser("check", help="which parts of a board3d the catalog covers")
    ck.add_argument("--board3d", type=Path, required=True)
    ad = sub.add_parser("add", help="add / replace an entry (JSON file) in the local catalog")
    ad.add_argument("--entry", type=Path, required=True)
    ad.add_argument("--catalog", type=Path, default=None, help="catalog file (default: first local one)")
    for p in sub.choices.values():
        add_json_flag(p)
    return ap


def _list() -> dict[str, Any]:
    from simple_sim.parts import PartCatalog, local_catalog_paths
    cat = PartCatalog.load()
    rows = [{"id": e.id, "origin": e.origin, "kind": e.data.get("kind"), "h": e.height_mm, "source": e.source_kind}
            for e in cat.entries]
    print(f"local catalogs: {', '.join(str(p) + ('' if p.is_file() else ' (missing)') for p in local_catalog_paths())}")
    for r in rows:
        print(f"  {r['id']:<34} h={r['h']!s:<6} {r['kind'] or '':<16} {r['source']:<12} {'builtin' if r['origin'] == 'builtin' else 'local'}")
    return {"entries": rows}


def _lookup(texts: list[str], package: str) -> dict[str, Any]:
    from simple_sim.parts import PartCatalog
    m = PartCatalog.load().lookup(texts=texts, package=package)
    print("no match" if m is None else f"{m.entry.id} ({m.how}, {m.entry.source_kind}): h={m.entry.height_mm} mm")
    return {"match": None if m is None else m.as_dict()}


def _check(folder: Path) -> dict[str, Any]:
    from simple_sim.board3d import load_board3d
    b = load_board3d(folder)
    pc = b["partCatalog"]
    est = sorted({c["body"]["catalog"]["id"] for c in b["components"]
                  if (c.get("body") or {}).get("catalog", {}).get("source") == "estimate"})
    print(f"{len(pc['applied'])} parts got a catalog height; without height: {', '.join(pc['unmatched']) or 'none'}")
    if est:
        print(f"estimates (check against the datasheet): {', '.join(est)}")
    return {"applied": len(pc["applied"]), "unmatched": pc["unmatched"], "estimates": est, "digest": pc["digest"]}


def _add(entry_path: Path, catalog: Path | None) -> dict[str, Any]:
    from simple_sim.parts import upsert_entry
    entry = load_json(entry_path)
    path = upsert_entry(entry, catalog)
    print(f"{entry['id']} -> {path}")
    return {"id": entry["id"], "catalog": str(path)}


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)

    def run() -> dict[str, Any]:
        if args.action == "list":
            return _list()
        if args.action == "lookup":
            return _lookup(args.text, args.package)
        if args.action == "check":
            return _check(args.board3d)
        return _add(args.entry, args.catalog)

    run_cli(COMMAND, run, getattr(args, "json", False))
    return 0
