#!/usr/bin/env python3
"""Capture screenshots of every monitor tab in light and dark mode.

Runs the GUI on a throw-away sandbox root (your real outputs/ and settings are not
touched), optionally with a tiny generated demo dataset, and writes PNGs:

    python tools/gui_screenshots.py --out /tmp/gui_shots            # empty state
    python tools/gui_screenshots.py --out /tmp/gui_shots --demo     # with a demo dataset
    xvfb-run -s "-screen 0 1600x1000x24" python tools/gui_screenshots.py --out shots

Needs an X display (real or Xvfb) and Pillow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path
from simple_sim.io_utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_demo_dataset(sim_root: Path) -> Path | None:
    import yaml

    cfg = load_yaml(ROOT / "configs" / "run_0001.yaml")
    cfg["classes"] = {k: 4 for k in cfg.get("classes", {"OK": 4})}
    cfg.setdefault("run", {})["run_id"] = "demo"
    cfg_path = sim_root / "outputs" / "demo_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = sim_root / "outputs" / "sim_data" / "runs" / "demo"
    proc = subprocess.run([sys.executable, "-m", "simple_sim", "generate", "--config", str(cfg_path), "--out", str(out)],
                          cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print("demo dataset generation failed:\n" + proc.stdout[-2000:] + proc.stderr[-2000:], file=sys.stderr)
        return None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Directory for the PNG files")
    ap.add_argument("--demo", action="store_true", help="Generate a tiny demo dataset first")
    ap.add_argument("--size", default="1440x900", help="Window size (default 1440x900)")
    ap.add_argument("--themes", default="light,dark", help="Comma list of themes (light,dark)")
    args = ap.parse_args(argv)
    from simple_sim.cli_utils import configure_cli_logging

    configure_cli_logging()

    import tkinter as tk

    from PIL import ImageGrab

    from gui.devtools.sandbox import make_sandbox_root
    from gui.monitor import MonitorAppTabbed

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="simsim_gui_"))
    sim_root = make_sandbox_root(tmp / "sim")
    # Skip the first-start quick-start dialog.
    (sim_root / "outputs" / "gui" / "settings.json").write_text('{"app.welcome_shown": true}', encoding="utf-8")
    if args.demo:
        ds = _make_demo_dataset(sim_root)
        print(f"demo dataset: {ds}")

    try:
        return _shoot(args, tk, ImageGrab, MonitorAppTabbed, sim_root, out_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _shoot(args, tk, ImageGrab, MonitorAppTabbed, sim_root: Path, out_dir: Path) -> int:
    root = tk.Tk(className="SimpleSim")
    root.geometry(args.size + "+0+0")
    app = MonitorAppTabbed(root, sim_root=sim_root)
    shots: list[Path] = []

    def settle(ms: int = 700) -> None:
        end = root.tk.call("clock", "milliseconds") + ms
        while root.tk.call("clock", "milliseconds") < end:
            root.update()

    for theme in [t.strip() for t in args.themes.split(",") if t.strip()]:
        app.var_dark.set(theme == "dark")
        app._apply_theme_from_var()
        for idx, key in enumerate(app._tab_order):
            app.notebook.select(idx)
            settle()
            x, y = root.winfo_rootx(), root.winfo_rooty()
            w, h = root.winfo_width(), root.winfo_height()
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            path = out_dir / f"{idx + 1:02d}_{key}_{theme}.png"
            img.save(path)
            shots.append(path)
            print(path)

    app._on_close()
    print(f"{len(shots)} screenshots in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
