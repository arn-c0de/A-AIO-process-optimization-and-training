#!/usr/bin/env python3
"""Render small per-profile preview images for debugging 2D/3D profiles.

Goal: quickly verify that components/pads/defect states render correctly before AI training.

This script:
- Detects available profiles under `configs/profiles/` (2D: "opencv_2d", 3D: "blender_3d")
- Lets you choose backend mode (2D, 3D, or both), then select one or more profiles
- Renders exactly 1 image per defect state (OK/MISALIGNED/MISSING/TOMBSTONE, etc.)
- Writes previews under `<out>/previews/<profile_id>/<backend>/...` and an `<out>/index.html`

Rendering logic: :mod:`simple_sim.preview.render`; Tk viewer: :mod:`simple_sim.preview.viewer_tk`.

Examples:
  .venv/bin/python scripts/render_debug_previews.py
  .venv/bin/python scripts/render_debug_previews.py --backend opencv_2d --all
  .venv/bin/python scripts/render_debug_previews.py --backend both --profiles chip_0603_resistor@1,chip_0603_resistor_3d@1
  .venv/bin/python scripts/render_debug_previews.py --all
  .venv/bin/python scripts/render_debug_previews.py --profiles chip_0603_resistor_3d@1,sot23_transistor_3d@1
  .venv/bin/python scripts/render_debug_previews.py --config configs/run_qfn32_3d.yaml --profiles qfn32_ic_3d@1
  .venv/bin/python scripts/render_debug_previews.py --backend opencv_2d --all --json   # headless, machine-readable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.cli_utils import add_json_flag, run_cli
from simple_sim.config import load_config, validate_config
from simple_sim.filter_profiles import load_shared_filter_profiles, shared_settings_path
from simple_sim.preview.render import (
    BACKEND_2D,
    BACKEND_3D,
    BACKEND_CHOICES,
    PreviewOptions,
    PreviewRec,
    ProfileEntry,
    auto_select_profiles_with_missing_previews,
    collect_existing_previews,
    discover_profiles,
    parse_csv_list,
    parse_roi,
    parse_selection,
    render_previews,
    write_index_html,
)

COMMAND = "render_debug_previews"
SIM_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render per-profile debug previews (2D OpenCV and/or 3D Blender)")
    parser.add_argument("--profiles-dir", default="configs/profiles", help="Directory with component profile YAMLs")
    parser.add_argument("--configs-dir", default="configs", help="Directory with run_*.yaml configs (for auto matching)")
    parser.add_argument("--out", default="outputs/debug_previews", help="Output directory root")
    parser.add_argument("--settings-file", default="", help="Shared GUI settings JSON (default: outputs/gui/settings.json)")

    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="auto", help="Render backend: opencv_2d|blender_3d|both|auto")
    parser.add_argument("--profiles", default="", help="Comma-separated profile IDs to render")
    parser.add_argument("--all", action="store_true", help="Render all discovered profiles (within backend selection)")
    parser.add_argument("--non-interactive", action="store_true", help="Do not prompt; auto-renders new/incomplete profiles when --profiles/--all is omitted")

    parser.add_argument("--config", default="", help="Optional run config YAML to use for ALL selected profiles (must match backend unless --backend=both)")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed for deterministic previews")
    parser.add_argument("--variable-seeds", action="store_true", help="Generate different images on each run (like real pipeline)")
    parser.add_argument(
        "--disable-cardinal-rotation-90",
        action="store_true",
        help="Disable random 0/90/180/270 base orientation in previews (keep only rotation_deg_range jitter)",
    )
    parser.add_argument("--states", default="", help="Comma-separated defect states to render (default: from profile.defect_set)")

    parser.add_argument("--blender", default="", help="(3D) Override blender executable (else from config)")
    parser.add_argument("--samples", type=int, default=0, help="(3D) Override Cycles samples (else from config)")
    parser.add_argument("--device", default="", help="(3D) Override device (CPU/GPU; else from config)")

    parser.add_argument("--roi", default="", help="Override ROI as WIDTHxHEIGHT@MM_PER_PX, e.g. 256x256@0.01")
    parser.add_argument("--dry-run", action="store_true", help="Only write index/jobs; do not render (2D or 3D)")
    parser.add_argument("--view", choices=["tk", "browser", "none"], default="tk", help="How to open previews after run (default: tk)")
    # Back-compat flags
    open_group = parser.add_mutually_exclusive_group()
    open_group.add_argument("--open", dest="open_browser", action="store_true", help="(legacy) open in browser after run")
    open_group.add_argument("--no-open", dest="open_browser", action="store_false", help="(legacy) do not auto-open after run")
    parser.set_defaults(open_browser=None)
    add_json_flag(parser)
    parser.epilog = "--json implies --non-interactive and --view none."
    return parser


def _prompt_backend() -> str:
    print("\nSelect backend mode:\n")
    print(f"  1. 3D ({BACKEND_3D})")
    print(f"  2. 2D ({BACKEND_2D})")
    print("  3. Both (default)")
    print("\nPress Enter for default, or enter your choice:")
    sel = input("> ").strip().lower()
    if not sel or sel in {"3", "both", "b"}:
        return "both"
    if sel in {"1", "3d", BACKEND_3D}:
        return BACKEND_3D
    if sel in {"2", "2d", BACKEND_2D}:
        return BACKEND_2D
    raise SystemExit("Invalid backend selection.")


def _prompt_profiles(discovered: list[ProfileEntry]) -> list[str]:
    print("\nAvailable profiles:\n")
    for i, (pid, prof, b) in enumerate(discovered, 1):
        desc = str((prof.get("profile") or {}).get("description", "") or "")
        print(f"{i:2d}. {pid}  [{'/'.join(sorted(b))}]" + (f"  ({desc})" if desc else ""))
    print("\nSelect profiles by number (e.g. 1,3-4) or 'all' (default):")
    print("Press Enter for all profiles, or enter your selection:")
    sel = input("> ").strip() or "all"
    return [discovered[i][0] for i in parse_selection(sel, len(discovered))]


def _show(
    view: str,
    out_root: Path,
    previews: list[PreviewRec],
    *,
    opts: PreviewOptions,
    settings_file: Path,
    profiles_dir: Path,
    configs_dir: Path,
    all_profiles: list[ProfileEntry],
    selected_ids: list[str] | None,
) -> None:
    if view == "browser":
        import webbrowser

        webbrowser.open((out_root / "index.html").resolve().as_uri())
    elif view == "tk":
        from simple_sim.preview.viewer_tk import open_tk_viewer

        open_tk_viewer(
            out_root,
            previews,
            sim_root=SIM_ROOT,
            profiles_dir=profiles_dir,
            configs_dir=configs_dir,
            all_profiles=all_profiles,
            options=opts,
            settings_file=settings_file,
            selected_profile_ids=selected_ids,
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.json:
        args.non_interactive = True
        args.view = "none"
    profiles_dir = (SIM_ROOT / args.profiles_dir).resolve()
    configs_dir = (SIM_ROOT / args.configs_dir).resolve()
    out_root = (SIM_ROOT / args.out).resolve()
    settings_file = shared_settings_path(SIM_ROOT, args.settings_file)
    _profiles, _active, shared_filter_settings = load_shared_filter_profiles(settings_file)

    discovered_all = discover_profiles(profiles_dir, backend_mode="both")
    if not discovered_all:
        raise SystemExit(f"No profiles found under: {profiles_dir}")

    # Either one config for all profiles, or auto matching per profile/backend.
    forced_cfg: dict[str, Any] | None = None
    forced_backend: str | None = None
    if args.config:
        forced_cfg = load_config(SIM_ROOT / args.config)
        validate_config(forced_cfg)
        forced_backend = str((forced_cfg.get("render") or {}).get("backend", ""))

    try:
        roi_override = parse_roi(args.roi) if args.roi else None
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    backend_mode = str(args.backend or "auto")
    if backend_mode == "auto" and forced_backend:
        backend_mode = forced_backend
    if backend_mode != "auto" and forced_backend and backend_mode not in {"both", forced_backend}:
        raise SystemExit(f"--backend={backend_mode!r} conflicts with --config backend={forced_backend!r}")
    if backend_mode == "auto":
        backend_mode = "both" if args.non_interactive else _prompt_backend()

    discovered = [e for e in discovered_all if backend_mode == "both" or backend_mode in e[2]]
    if not discovered:
        raise SystemExit(f"No profiles match backend={backend_mode!r} under: {profiles_dir}")

    selected_ids = [pid for pid, _p, _b in discovered] if args.all else parse_csv_list(args.profiles)
    # Auto mode: if nothing explicitly selected, render profiles with missing previews.
    if not selected_ids:
        selected_ids = auto_select_profiles_with_missing_previews(
            discovered,
            profiles_dir=profiles_dir,
            out_root=out_root,
            backend_mode=backend_mode,
            states_override=str(args.states or ""),
        )
        if selected_ids:
            print(f"[auto] Rendering {len(selected_ids)} new/incomplete profile(s): {', '.join(selected_ids)}")
    if not selected_ids and not args.non_interactive:
        selected_ids = _prompt_profiles(discovered)

    # Translate legacy flags into --view behavior if explicitly set.
    if args.open_browser is True:
        args.view = "browser"
    elif args.open_browser is False:
        args.view = "none"

    opts = PreviewOptions(
        backend_mode=backend_mode,
        forced_cfg=forced_cfg,
        forced_label=str(args.config),
        roi_override=roi_override,
        seed_base=int(args.seed),
        variable_seeds=bool(args.variable_seeds),
        states=str(args.states or ""),
        enable_cardinal_rotation_90=(
            not bool(args.disable_cardinal_rotation_90) and bool(shared_filter_settings.get("cardinal_rotation_90", True))
        ),
        blender=str(args.blender or ""),
        samples=int(args.samples or 0),
        device=str(args.device or ""),
    )
    show_kwargs = dict(
        opts=opts, settings_file=settings_file, profiles_dir=profiles_dir, configs_dir=configs_dir, all_profiles=discovered_all
    )
    result: dict[str, Any] = {"out_dir": str(out_root), "backend_mode": backend_mode, "dry_run": bool(args.dry_run)}

    if not selected_ids:
        if not args.non_interactive:
            raise SystemExit("No profiles selected and no new/incomplete profiles found.")
        # Keep index/view in sync even when nothing new needs rendering.
        existing = collect_existing_previews(out_root)
        if not existing:
            print("[auto] No new/incomplete profiles found and no existing previews available.")
            return {**result, "profiles": [], "rendered": False, "index_path": None, "jobs_path": None, "images": []}
        index_path = write_index_html(out_root, existing)
        print(f"[auto] No new/incomplete profiles found. Reusing {len(existing)} existing preview(s).")
        _show(args.view, out_root, existing, selected_ids=None, **show_kwargs)
        return {
            **result,
            "profiles": sorted({r[0] for r in existing}),
            "rendered": False,
            "index_path": str(index_path),
            "jobs_path": None,
            "images": [str(out_root / r[3]) for r in existing],
        }

    pid_to_entry = {e[0]: e for e in discovered_all}
    unknown = [pid for pid in selected_ids if pid not in pid_to_entry]
    if unknown:
        raise SystemExit(f"Unknown profiles: {unknown}")

    try:
        run_res = render_previews(
            [pid_to_entry[pid] for pid in selected_ids],
            opts=opts,
            sim_root=SIM_ROOT,
            configs_dir=configs_dir,
            out_root=out_root,
            image_filters=shared_filter_settings,
            dry_run=bool(args.dry_run),
        )
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    _show(args.view, out_root, run_res.previews, selected_ids=selected_ids, **show_kwargs)
    return {
        **result,
        "profiles": list(selected_ids),
        "rendered": not args.dry_run,
        "index_path": str(run_res.index_path),
        "jobs_path": str(run_res.jobs_path) if run_res.jobs_path else None,
        "images": [str(out_root / r[3]) for r in run_res.previews],
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_cli(COMMAND, lambda: run(args), args.json)


if __name__ == "__main__":
    main()
