"""Single agent-friendly entrypoint: ``python -m simple_sim <command> [args...]``.

Commands dispatch to the existing scripts/tools as a subprocess with the same Python
interpreter; all remaining arguments are passed through unchanged. Two built-ins:

* ``commands [--json]`` - list commands, their one-line description and ``--json`` support
* ``status [--json] [--root DIR]`` - profiles, run configs, datasets, models and latest reports

This module is torch-free and tk-free so it can be used for discovery anywhere.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from simple_sim.cli_utils import add_json_flag, emit_result
from simple_sim.io_utils import load_yaml, rel_to
from simple_sim.manifest import read_dataset_manifest
from simple_sim.model_bundle import is_bundle_dir
from simple_sim.reports import best_report, load_reports, metric_value, report_search_dirs, report_timestamp

SIM_ROOT = Path(__file__).resolve().parents[1]

# command name -> script path relative to SIM_ROOT (descriptions come from the script docstring)
COMMANDS: dict[str, str] = {
    "generate": "scripts/generate.py",
    "generate-profile-dataset": "scripts/generate_profile_dataset.py",
    "validate": "tools/validate_dataset.py",
    "train": "scripts/train.py",
    "train-profile": "scripts/train_profile.py",
    "eval": "scripts/eval.py",
    "eval-profile": "scripts/eval_profile.py",
    "predict": "scripts/predict.py",
    "predict-two-stage": "scripts/predict_two_stage.py",
    "sample-check": "scripts/sample_check.py",
    "batch-predict": "scripts/batch_predict.py",
    "arena-report": "scripts/arena_report.py",
    "render-debug-previews": "scripts/render_debug_previews.py",
    "create-multi-dataset": "tools/create_multi_dataset.py",
    "backfill-manifest": "tools/backfill_manifest.py",
    "backfill-labels": "tools/backfill_labels_v2.py",
    "update-sample-gallery": "tools/update_sample_gallery.py",
    "gui-screenshots": "tools/gui_screenshots.py",
}

BUILTINS: dict[str, str] = {
    "commands": "List available commands (name, script, description, --json support).",
    "status": "Report profiles, run configs, datasets, models and latest/best eval reports.",
}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def command_info(name: str, script: str, root: Path = SIM_ROOT) -> dict[str, Any]:
    """Describe one dispatched command by reading (not importing) its script source."""
    path = root / script
    try:
        src = path.read_text(encoding="utf-8")
        doc = ast.get_docstring(ast.parse(src)) or ""
    except (OSError, SyntaxError, ValueError):
        src, doc = "", ""
    return {
        "name": name,
        "script": script,
        "description": doc.strip().splitlines()[0] if doc.strip() else "",
        "supports_json": "add_json_flag" in src,
        "exists": path.is_file(),
    }


def list_commands(root: Path = SIM_ROOT) -> list[dict[str, Any]]:
    """All commands: built-ins first, then dispatched scripts."""
    out: list[dict[str, Any]] = [
        {"name": n, "script": None, "description": d, "supports_json": True, "exists": True}
        for n, d in BUILTINS.items()
    ]
    out.extend(command_info(n, s, root) for n, s in COMMANDS.items())
    return out


def _cmd_commands(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m simple_sim commands", description=BUILTINS["commands"])
    add_json_flag(parser)
    args = parser.parse_args(argv)
    cmds = list_commands()
    if args.json:
        emit_result("commands", {"commands": cmds})
    else:
        width = max(len(c["name"]) for c in cmds)
        for c in cmds:
            flag = "[json]" if c["supports_json"] else "      "
            print(f"  {c['name']:<{width}}  {flag}  {c['description']}")
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _run_config_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"name": path.stem, "path": rel_to(SIM_ROOT, path)}
    try:
        cfg = load_yaml(path) or {}
        run = cfg.get("run") or {}
        info.update(
            run_id=run.get("run_id"),
            component_profile=run.get("component_profile") or run.get("component_profiles"),
            mode=run.get("mode") or None,
            backend=(cfg.get("render") or {}).get("backend"),
        )
    except (OSError, yaml.YAMLError, AttributeError) as exc:
        info["error"] = str(exc)
    return info


def _dataset_dirs(sim_data: Path) -> list[Path]:
    runs = sorted(d for d in (sim_data / "runs").glob("*") if d.is_dir())
    versions = sorted(d for d in (sim_data / "versions").glob("*/*") if d.is_dir())
    return runs + versions


def _dataset_info(ds: Path, root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"name": ds.name, "path": rel_to(root, ds), "manifest": None}
    manifest_path = ds / "dataset_manifest.json"
    if not manifest_path.exists():
        info["has_labels"] = (ds / "labels.jsonl").exists()
        return info
    try:
        m = read_dataset_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        info["manifest_error"] = str(exc)
        return info
    if int(m.get("manifest_version", 1) or 1) == 1:
        profiles = [m["component_profile"]["profile_id"]]
    else:
        profiles = [p["profile_id"] for p in m.get("component_profiles", [])]
    stats = m.get("dataset_stats") or {}
    info["manifest"] = {
        "manifest_version": m.get("manifest_version"),
        "run_id": m.get("run_id"),
        "created_at": m.get("created_at"),
        "profiles": profiles,
        "total_samples": stats.get("total_samples"),
        "splits": stats.get("splits"),
        "classes": stats.get("classes"),
    }
    return info


def _models(models_dir: Path, root: Path) -> list[dict[str, Any]]:
    if not models_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(models_dir.iterdir()):
        if p.is_file() and p.suffix == ".pt" and not p.stem.endswith("_last"):
            out.append({"path": rel_to(root, p), "kind": "checkpoint", "size_bytes": p.stat().st_size})
        elif is_bundle_dir(p):
            out.append({"path": rel_to(root, p), "kind": "bundle",
                        "profiles": sorted(x.stem for x in p.glob("*.pt") if not x.stem.endswith("_last"))})
    return out


def _report_summary(r: dict[str, Any] | None, root: Path) -> dict[str, Any] | None:
    if r is None:
        return None
    return {
        "path": rel_to(root, Path(r["_path"])),
        "timestamp": r.get("timestamp"),
        "mode": r.get("mode", "eval"),
        "model_path": r.get("model_path"),
        "dataset_path": r.get("dataset_path"),
        "split": r.get("split"),
        "accuracy": metric_value(r, "accuracy"),
        "macro_f1": metric_value(r, "macro_f1"),
    }


def collect_status(root: Path = SIM_ROOT) -> dict[str, Any]:
    """Snapshot of what exists under *root* (a Simple-Sim checkout); pure file inspection."""
    root = Path(root)
    configs = root / "configs"
    outputs = root / "outputs"
    reports = load_reports(report_search_dirs(root))
    latest = max(reports, key=report_timestamp) if reports else None
    return {
        "root": str(root),
        "profiles": sorted(p.stem for p in (configs / "profiles").glob("*.yaml")),
        "run_configs": [_run_config_info(p) for p in sorted(configs.glob("run_*.yaml"))],
        "datasets": [_dataset_info(d, root) for d in _dataset_dirs(outputs / "sim_data")],
        "models": _models(outputs / "models", root),
        "reports": {
            "count": len(reports),
            "latest": _report_summary(latest, root),
            "best_accuracy": _report_summary(best_report(reports, "accuracy"), root),
        },
    }


def _cmd_status(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m simple_sim status", description=BUILTINS["status"])
    parser.add_argument("--root", default=str(SIM_ROOT), help="Simple-Sim root to inspect (default: this checkout)")
    add_json_flag(parser)
    args = parser.parse_args(argv)
    st = collect_status(Path(args.root))
    if args.json:
        emit_result("status", st)
        return 0
    print(f"Root: {st['root']}")
    print(f"Profiles ({len(st['profiles'])}): {', '.join(st['profiles']) or '-'}")
    print(f"Run configs ({len(st['run_configs'])}): {', '.join(c['name'] for c in st['run_configs']) or '-'}")
    print(f"Datasets ({len(st['datasets'])}):")
    for d in st["datasets"]:
        m = d["manifest"] or {}
        print(f"  {d['path']}  samples={m.get('total_samples', '?')}  profiles={','.join(m.get('profiles', [])) or '?'}")
    print(f"Models ({len(st['models'])}):")
    for m in st["models"]:
        print(f"  {m['path']}  [{m['kind']}]")
    rep = st["reports"]
    print(f"Reports: {rep['count']}")
    for label in ("latest", "best_accuracy"):
        r = rep[label]
        if r:
            print(f"  {label}: {r['path']}  accuracy={r['accuracy']}  macro_f1={r['macro_f1']}")
    return 0


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def _usage() -> str:
    lines = ["usage: python -m simple_sim <command> [args...]", "", "commands:"]
    width = max(len(n) for n in [*BUILTINS, *COMMANDS])
    for c in list_commands():
        lines.append(f"  {c['name']:<{width}}  {c['description']}")
    lines += ["", "Run 'python -m simple_sim <command> --help' for command options.",
              "Most commands accept --json (see 'commands --json')."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name == "commands":
        return _cmd_commands(rest)
    if name == "status":
        return _cmd_status(rest)
    script = COMMANDS.get(name)
    if script is None:
        print(f"Unknown command: {name}\n\n{_usage()}", file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, str(SIM_ROOT / script), *rest])


if __name__ == "__main__":
    sys.exit(main())
