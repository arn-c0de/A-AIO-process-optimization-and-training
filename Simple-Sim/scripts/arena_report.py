#!/usr/bin/env python3
"""Generate a markdown "Arena" ranking report for tracked models.

Reads:
  outputs/models/arena.json

Scans:
  report_*.json, batch_report_*.json in common outputs + per-dataset predictions dirs

Writes:
  ARENA_REPORT.md (in sim root, or --out)
  ARENA_REPORT_assets/*.svg (static charts referenced by the markdown, next to the report)

The logic lives in :mod:`simple_sim.arena`; this is a thin CLI wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.arena import detect_sim_root, write_arena_report
from simple_sim.cli_utils import add_json_flag, add_out_arg, run_cli

COMMAND = "arena_report"


def _run(args: argparse.Namespace) -> dict:
    sim_root = Path(args.sim_root).resolve() if args.sim_root else None
    if sim_root is None:
        sim_root = detect_sim_root(Path.cwd()) or detect_sim_root(Path(__file__).resolve())
    if sim_root is None:
        raise SystemExit("Could not auto-detect --sim-root (expected outputs/models/ under repo)")

    result = write_arena_report(sim_root, Path(args.out) if args.out else None)
    if result["n_tracked"]:
        print(f"[arena] wrote {result['out_path']}")
    else:
        print(f"[arena] no tracked models; wrote {result['out_path']}")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate ARENA_REPORT.md for Arena-tracked models")
    ap.add_argument("--sim-root", default=None, help="Simple-Sim root (default: auto-detect)")
    add_out_arg(ap, "Output markdown path (default: <sim-root>/ARENA_REPORT.md)", required=False)
    add_json_flag(ap)
    args = ap.parse_args(argv)

    run_cli(COMMAND, lambda: _run(args), args.json)


if __name__ == "__main__":
    main()
