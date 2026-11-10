"""Single agent-friendly entrypoint: ``python -m simple_sim <command> [args...]``.

Commands are registered in :data:`COMMANDS`. Library commands (:mod:`simple_sim.commands`) run
in-process via their ``main(argv)``; the helper tools under ``tools/`` still run as a subprocess
with the same Python interpreter. All remaining arguments are passed through unchanged. Two
built-ins:

* ``commands [--json]`` - list commands, their one-line description and ``--json`` support
* ``status [--json] [--root DIR]`` - profiles, run configs, datasets, models and latest reports

:func:`argv_for` builds the command line other programs (the GUI, shell launchers) should spawn.
This module is torch-free and tk-free so it can be used for discovery anywhere: listing reads the
command sources, it does not import them.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from simple_sim.cli_utils import add_json_flag, emit_result
from simple_sim.dataset_files import DATASET_MANIFEST, LABELS_JSONL
from simple_sim.io_utils import load_yaml, rel_to
from simple_sim.manifest import read_dataset_manifest
from simple_sim.model_bundle import is_bundle_dir, list_bundle_checkpoints
from simple_sim.reports import best_report, is_partial_report, is_profile_classifier_report, load_reports, metric_value, report_search_dirs, report_timestamp

SIM_ROOT = Path(__file__).resolve().parents[1]

PROG = "python -m simple_sim"


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One ``python -m simple_sim`` command.

    ``module`` (``simple_sim.commands.<name>``) runs in-process: its ``main(argv, prog=...)`` is
    called, its docstring is the description and its ``SUPPORTS_JSON`` constant the ``--json``
    flag. ``script`` (relative to :data:`SIM_ROOT`) is the file users can also run directly: the
    ``scripts/*.py`` shim of a module command, or a ``tools/*.py`` script that is dispatched as a
    subprocess (description from its docstring, ``--json`` support detected from its source).
    """

    name: str
    script: str
    module: str | None = None


def _library(name: str) -> CommandSpec:
    stem = name.replace("-", "_")
    return CommandSpec(name, f"scripts/{stem}.py", f"simple_sim.commands.{stem}")


def _tool(name: str, script: str) -> CommandSpec:
    return CommandSpec(name, script)


_SPECS = [
    _library("generate"),
    _library("generate-profile-dataset"),
    _tool("validate", "tools/validate_dataset.py"),
    _library("train"),
    _library("train-profile"),
    _library("eval"),
    _library("eval-profile"),
    _library("predict"),
    _library("predict-two-stage"),
    _library("sample-check"),
    _library("batch-predict"),
    _library("arena-report"),
    _library("render-debug-previews"),
    _tool("create-multi-dataset", "tools/create_multi_dataset.py"),
    _tool("backfill-manifest", "tools/backfill_manifest.py"),
    _tool("backfill-labels", "tools/backfill_labels_v2.py"),
    _tool("update-sample-gallery", "tools/update_sample_gallery.py"),
    _tool("gui-screenshots", "tools/gui_screenshots.py"),
    # sim-to-real helpers (real camera crops vs. the synthetic look)
    _tool("sim-to-real", "tools/sim_to_real.py"),
    _tool("calibrate-camera", "tools/calibrate_camera.py"),
    _tool("look-stats", "tools/look_stats.py"),
    _tool("look-montage", "tools/look_montage.py"),
    _tool("real-crops", "tools/real_crops.py"),
    _tool("eval-real", "tools/eval_real.py"),
    _tool("mix-real", "tools/mix_real.py"),
    # DeepVision app: reviewed crops, board backgrounds, judge export
    _tool("dv-crops", "tools/dv_crops.py"),
    _tool("board-backgrounds", "tools/board_backgrounds.py"),
    _tool("export-deepvision", "tools/export_deepvision.py"),
]

#: command name -> :class:`CommandSpec`
COMMANDS: dict[str, CommandSpec] = {c.name: c for c in _SPECS}

BUILTINS: dict[str, str] = {
    "commands": "List available commands (name, script, description, --json support).",
    "status": "Report profiles, run configs, datasets, models and latest/best eval reports.",
}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def _module_source(module: str) -> Path | None:
    """File of *module* without importing it (only its parent packages are imported)."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    return Path(spec.origin) if spec is not None and spec.origin else None


def _constant(tree: ast.Module, name: str) -> Any:
    """Value of a top-level ``NAME = <literal>`` assignment (None when absent)."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return None
    return None


def command_info(name: str, spec: CommandSpec | str, root: Path = SIM_ROOT) -> dict[str, Any]:
    """Describe one command by reading (not importing) its source."""
    if isinstance(spec, str):  # former signature: (name, script path)
        spec = CommandSpec(name, spec)
    source = _module_source(spec.module) if spec.module else root / spec.script
    try:
        src = source.read_text(encoding="utf-8") if source is not None else ""
        tree = ast.parse(src)
        doc = ast.get_docstring(tree) or ""
    except (OSError, SyntaxError, ValueError):
        src, tree, doc = "", None, ""
    if spec.module:
        supports_json = bool(tree is not None and _constant(tree, "SUPPORTS_JSON"))
    else:
        supports_json = "add_json_flag" in src
    return {
        "name": name,
        "script": spec.script,
        "module": spec.module,
        "description": doc.strip().splitlines()[0] if doc.strip() else "",
        "supports_json": supports_json,
        "exists": source is not None and source.is_file() and (root / spec.script).is_file(),
    }


def list_commands(root: Path = SIM_ROOT) -> list[dict[str, Any]]:
    """All commands: built-ins first, then dispatched scripts."""
    out: list[dict[str, Any]] = [
        {"name": n, "script": None, "module": None, "description": d, "supports_json": True, "exists": True}
        for n, d in BUILTINS.items()
    ]
    out.extend(command_info(n, s, root) for n, s in COMMANDS.items())
    return out


def _cmd_commands(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{PROG} commands", description=BUILTINS["commands"])
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

def _run_config_info(path: Path, root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"name": path.stem, "path": rel_to(root, path)}
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
    manifest_path = ds / DATASET_MANIFEST
    if not manifest_path.exists():
        info["has_labels"] = (ds / LABELS_JSONL).exists()
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
                        "profiles": sorted(x.stem for x in list_bundle_checkpoints(p))})
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


def _is_ranked_report(r: Any) -> bool:
    """Reports ``best_accuracy`` may pick: full test-split runs of defect models (as the Arena ranks)."""
    return (
        not is_profile_classifier_report(r)
        and str(r.get("split") or "test") == "test"
        and not is_partial_report(r)
    )


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
        "run_configs": [_run_config_info(p, root) for p in sorted(configs.glob("run_*.yaml"))],
        "datasets": [_dataset_info(d, root) for d in _dataset_dirs(outputs / "sim_data")],
        "models": _models(outputs / "models", root),
        "reports": {
            "count": len(reports),
            "latest": _report_summary(latest, root),
            "best_accuracy": _report_summary(
                best_report([r for r in reports if _is_ranked_report(r)], "accuracy"), root),
        },
    }


def _cmd_status(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{PROG} status", description=BUILTINS["status"])
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
    lines = [f"usage: {PROG} <command> [args...]", "", "commands:"]
    width = max(len(n) for n in [*BUILTINS, *COMMANDS])
    for c in list_commands():
        lines.append(f"  {c['name']:<{width}}  {c['description']}")
    lines += ["", "Run 'python -m simple_sim <command> --help' for command options.",
              "Most commands accept --json (see 'commands --json')."]
    return "\n".join(lines)


def argv_for(command: str, *args: Any, python: str | None = None, **options: Any) -> list[str]:
    """Command line that runs ``python -m simple_sim <command>`` with the current interpreter.

    *args* are appended as they are; each keyword becomes ``--<name-with-dashes>``: ``True`` adds
    the bare flag, ``False``/``None`` omit it, anything else adds ``--name value``. Callers that
    spawn commands (GUI, launchers) use this instead of hard-coding ``.venv/bin/python
    scripts/<name>.py``, so the venv and the installed package are always the ones in use.
    """
    if command not in COMMANDS and command not in BUILTINS:
        raise KeyError(f"Unknown command: {command}")
    out = [python or sys.executable, "-m", "simple_sim", command, *(str(a) for a in args)]
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        out.append(flag) if value is True else out.extend([flag, str(value)])
    return out


def run_command(name: str, argv: list[str]) -> int:
    """Run registered command *name* with *argv*; return its exit status."""
    spec = COMMANDS[name]
    if spec.module is None:
        return subprocess.call([sys.executable, str(SIM_ROOT / spec.script), *argv])
    mod = importlib.import_module(spec.module)
    try:
        return int(mod.main(argv, prog=f"{PROG} {name}") or 0)
    except SystemExit as exc:  # argparse --help / usage errors, UsageError, run_cli failures
        code = exc.code
        if code is None or isinstance(code, int):
            return int(code or 0)
        print(code, file=sys.stderr)
        return 1


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
    if name not in COMMANDS:
        print(f"Unknown command: {name}\n\n{_usage()}", file=sys.stderr)
        return 2
    return run_command(name, rest)


if __name__ == "__main__":
    sys.exit(main())
