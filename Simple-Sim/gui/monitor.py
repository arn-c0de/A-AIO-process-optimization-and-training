#!/usr/bin/env python3
"""Tiny Tkinter GUI to monitor the Simple-Sim pipeline live.

Shows:
- Live stdout/stderr log stream
- Parsed JSONL events (SIMPLE_SIM_EVENT_LOG)
- Latest images (last seen) as thumbnails

Notes:
- Requires a system Tk install (on Ubuntu: apt install python3-tk).
- Uses Pillow for image thumbnails.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import messagebox
except Exception as e:
    raise SystemExit(
        "Tkinter is not available. On Ubuntu/Debian install it with:\n"
        "  sudo apt-get install python3-tk\n"
        f"Original error: {e}"
    )


try:
    from PIL import Image, ImageTk
except Exception as e:
    raise SystemExit(
        "Pillow is required for thumbnails. Install dependencies:\n"
        "  .venv/bin/python -m pip install -r requirements.txt\n"
        f"Original error: {e}"
    )


@dataclass
class UiState:
    phase: str = "idle"
    epoch: str = "-"
    img_per_s: str = "-"
    last_id: str = "-"
    last_img: str = "-"
    dataset_dir: Optional[Path] = None


class MonitorApp:
    def __init__(self, root: tk.Tk, sim_root: Path) -> None:
        self.root = root
        self.sim_root = sim_root
        self.state = UiState()

        self.proc: Optional[subprocess.Popen[str]] = None
        self.stop_evt = threading.Event()

        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.event_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()

        self.event_log_path = self.sim_root / "outputs" / "live" / "events.jsonl"
        self._event_fp = None
        self._event_pos = 0

        self._thumb_refs = []  # keep PhotoImage references
        self._recent_imgs: list[str] = []
        self._dataset_dirs: list[Path] = []

        self._build_ui()
        self._refresh_datasets()
        self._tick_ui()

    def _build_ui(self) -> None:
        self.root.title("Simple-Sim Live Monitor")
        self.root.geometry("1200x750")

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # Top controls
        top = ttk.Frame(outer)
        top.pack(fill="x")

        self.btn_start = ttk.Button(top, text="Start Pipeline", command=self.start_pipeline)
        self.btn_start.pack(side="left")

        self.btn_stop = ttk.Button(top, text="Stop", command=self.stop_pipeline, state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(top, text="Config:").pack(side="left", padx=(0, 6))
        self.var_config = tk.StringVar(value="configs/run_0001.yaml")
        ttk.Entry(top, textvariable=self.var_config, width=28).pack(side="left")

        ttk.Label(top, text="Out:").pack(side="left", padx=(10, 6))
        self.var_out = tk.StringVar(value="outputs/sim_data/runs/run_0001")
        ttk.Entry(top, textvariable=self.var_out, width=34).pack(side="left")

        ttk.Label(top, text="Model:").pack(side="left", padx=(10, 6))
        self.var_model = tk.StringVar(value="outputs/models/run_0001.pt")
        ttk.Entry(top, textvariable=self.var_model, width=28).pack(side="left")

        ttk.Label(top, text="Wheelhouse:").pack(side="left", padx=(20, 6))
        self.var_wheelhouse = tk.StringVar(value="wheelhouse")
        ttk.Entry(top, textvariable=self.var_wheelhouse, width=40).pack(side="left")

        ttk.Label(top, text="Event log:").pack(side="left", padx=(20, 6))
        self.var_eventlog = tk.StringVar(value=str(self.event_log_path))
        ttk.Entry(top, textvariable=self.var_eventlog, width=55).pack(side="left")

        # Status line
        status = ttk.Frame(outer)
        status.pack(fill="x", pady=(10, 0))

        self.var_phase = tk.StringVar(value="phase: idle")
        self.var_epoch = tk.StringVar(value="epoch: -")
        self.var_ips = tk.StringVar(value="img/s: -")
        self.var_last = tk.StringVar(value="last: -")

        ttk.Label(status, textvariable=self.var_phase).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_epoch).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_ips).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_last).pack(side="left", padx=(0, 14))

        # Main split
        main = ttk.Panedwindow(outer, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(10, 0))

        # Left: datasets + image grid
        left = ttk.Frame(main, padding=8)
        main.add(left, weight=1)
        ttk.Label(left, text="Datasets").pack(anchor="w")

        dsbar = ttk.Frame(left)
        dsbar.pack(fill="x", pady=(6, 0))

        self.var_dataset = tk.StringVar(value="")
        self.dataset_combo = ttk.Combobox(dsbar, textvariable=self.var_dataset, state="readonly", width=38)
        self.dataset_combo.pack(side="left", fill="x", expand=True)
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_selected)

        ttk.Button(dsbar, text="Refresh", command=self._refresh_datasets).pack(side="left", padx=(8, 0))
        ttk.Button(dsbar, text="Delete", command=self._delete_dataset).pack(side="left", padx=(8, 0))

        dsbtns = ttk.Frame(left)
        dsbtns.pack(fill="x", pady=(6, 0))
        ttk.Button(dsbtns, text="Validate", command=self._validate_selected).pack(side="left")
        ttk.Button(dsbtns, text="Train", command=self._train_selected).pack(side="left", padx=(8, 0))
        ttk.Button(dsbtns, text="Eval", command=self._eval_selected).pack(side="left", padx=(8, 0))

        ttk.Label(left, text="Latest Images").pack(anchor="w", pady=(10, 0))

        self.img_grid = ttk.Frame(left)
        self.img_grid.pack(fill="both", expand=True, pady=(6, 0))

        self.img_labels = []
        rows, cols = 3, 3
        for r in range(rows):
            for c in range(cols):
                frm = ttk.Frame(self.img_grid, relief="groove", borderwidth=1)
                frm.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
                self.img_grid.grid_rowconfigure(r, weight=1)
                self.img_grid.grid_columnconfigure(c, weight=1)

                lbl = ttk.Label(frm, text="(no image)", anchor="center")
                lbl.pack(fill="both", expand=True)
                self.img_labels.append(lbl)

        # Right: logs
        right = ttk.Frame(main, padding=8)
        main.add(right, weight=2)
        ttk.Label(right, text="Live Logs").pack(anchor="w")

        self.txt = tk.Text(right, wrap="none", height=10)
        self.txt.pack(fill="both", expand=True, pady=(6, 0))
        self.txt.configure(state="disabled")

        yscroll = ttk.Scrollbar(right, command=self.txt.yview)
        yscroll.place(in_=self.txt, relx=1.0, rely=0, relheight=1.0, anchor="ne")
        self.txt["yscrollcommand"] = yscroll.set

    def _append_log(self, s: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", s)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def start_pipeline(self) -> None:
        if self.proc is not None:
            return

        wheelhouse = self.var_wheelhouse.get().strip()
        eventlog = Path(self.var_eventlog.get().strip())
        if not eventlog.is_absolute():
            eventlog = (self.sim_root / eventlog).resolve()
        self.event_log_path = eventlog

        cfg = self.var_config.get().strip()
        out_dir = self.var_out.get().strip()
        model_path = self.var_model.get().strip()

        env = os.environ.copy()
        env["WHEELHOUSE"] = wheelhouse
        env["EVENT_LOG"] = str(self.event_log_path)
        env["CONFIG"] = cfg
        env["DATA_DIR"] = out_dir
        env["MODEL_PATH"] = model_path

        cmd = ["bash", "-lc", "./run_pipeline.sh"]
        self.stop_evt.clear()

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.sim_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        t = threading.Thread(target=self._read_process_output, daemon=True)
        t.start()

        te = threading.Thread(target=self._tail_events, daemon=True)
        te.start()

        self.state.phase = "running"

    def stop_pipeline(self) -> None:
        self.stop_evt.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _run_simple_cmd(self, args: list[str], extra_env: Optional[Dict[str, str]] = None) -> None:
        """Run a Simple-Sim script and stream its output into the log pane."""
        if self.proc is not None:
            messagebox.showwarning("Busy", "A process is already running. Stop it first.")
            return

        env = os.environ.copy()
        env["EVENT_LOG"] = str(self.event_log_path)
        env["SIMPLE_SIM_EVENT_LOG"] = str(self.event_log_path)
        if extra_env:
            env.update(extra_env)

        cmd = ["bash", "-lc", " ".join(args)]
        self.stop_evt.clear()

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.sim_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        t = threading.Thread(target=self._read_process_output, daemon=True)
        t.start()

        te = threading.Thread(target=self._tail_events, daemon=True)
        te.start()

    def _read_process_output(self) -> None:
        assert self.proc is not None
        fp = self.proc.stdout
        assert fp is not None
        for line in fp:
            self.log_q.put(line)
            if self.stop_evt.is_set():
                break

        try:
            rc = self.proc.wait(timeout=1.0)
        except Exception:
            rc = None

        self.log_q.put(f"\n[process exited rc={rc}]\n")
        self.proc = None

    def _tail_events(self) -> None:
        # Poll for file existence; run_pipeline creates/truncates it early.
        path = self.event_log_path
        for _ in range(200):
            if self.stop_evt.is_set():
                return
            if path.exists():
                break
            time.sleep(0.05)

        pos = 0
        while not self.stop_evt.is_set():
            try:
                if not path.exists():
                    time.sleep(0.1)
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except Exception:
                            continue
                        self.event_q.put(evt)
                    pos = f.tell()
            except Exception:
                time.sleep(0.1)
            time.sleep(0.1)

    def _handle_event(self, evt: Dict[str, Any]) -> None:
        et = evt.get("event")
        if et == "gen_start":
            self.state.phase = "generating"
            out_dir = evt.get("output_dir")
            if out_dir:
                self.state.dataset_dir = Path(out_dir)
        elif et == "gen_progress":
            self.state.phase = "generating"
            self.state.img_per_s = f"{evt.get('samp_per_s', '-')}"
            last_img = evt.get("last_image_path")
            if last_img:
                self._push_image(last_img)
        elif et == "gen_done":
            self.state.phase = "generated"
            out_dir = evt.get("output_dir")
            if out_dir:
                self.state.dataset_dir = Path(out_dir)
            last_img = evt.get("last_image_path")
            if last_img:
                self._push_image(last_img)
        elif et == "train_start":
            self.state.phase = "training"
            ds = evt.get("dataset_dir")
            if ds:
                self.state.dataset_dir = Path(ds)
        elif et == "epoch_start":
            self.state.phase = "training"
            self.state.epoch = f"{evt.get('epoch', '-')}/{evt.get('epochs', '-')}"
        elif et == "train_batch":
            self.state.phase = "training"
            ips = evt.get("img_per_s")
            if ips is not None:
                self.state.img_per_s = f"{ips:.1f}"
            lid = evt.get("last_id")
            if lid:
                self.state.last_id = str(lid)
            limg = evt.get("last_image_path")
            if limg:
                self.state.last_img = str(limg)
                self._push_image(str(limg))
        elif et == "eval_start":
            self.state.phase = "evaluating"
            ds = evt.get("dataset_dir")
            if ds:
                self.state.dataset_dir = Path(ds)
        elif et == "eval_batch":
            self.state.phase = "evaluating"
            ips = evt.get("img_per_s")
            if ips is not None:
                self.state.img_per_s = f"{ips:.1f}"
            limg = evt.get("last_image_path")
            if limg:
                self.state.last_img = str(limg)
                self._push_image(str(limg))
        elif et == "eval_done":
            self.state.phase = "done"
        elif et == "validate_start":
            self.state.phase = "validating"
        elif et == "validate_done":
            ok = evt.get("ok")
            self.state.phase = "validated_ok" if ok else "validated_fail"

        self.var_phase.set(f"phase: {self.state.phase}")
        self.var_epoch.set(f"epoch: {self.state.epoch}")
        self.var_ips.set(f"img/s: {self.state.img_per_s}")
        self.var_last.set(f"last: {Path(self.state.last_img).name if self.state.last_img != '-' else '-'}")

    def _push_image(self, rel_path: str) -> None:
        # rel_path is like "images/000123.png".
        if not self.state.dataset_dir:
            return
        img_path = self.state.dataset_dir / rel_path
        if not img_path.exists():
            return
        name = str(img_path)
        if self._recent_imgs and self._recent_imgs[-1] == name:
            return
        self._recent_imgs.append(name)
        self._recent_imgs = self._recent_imgs[-9:]
        self._refresh_thumbnails()

    def _refresh_thumbnails(self) -> None:
        self._thumb_refs.clear()
        for i, lbl in enumerate(self.img_labels):
            if i >= len(self._recent_imgs):
                lbl.configure(image="", text="(no image)")
                continue

            path = Path(self._recent_imgs[-1 - i])
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((340, 220))
                tkimg = ImageTk.PhotoImage(img)
                self._thumb_refs.append(tkimg)
                lbl.configure(image=tkimg, text=path.name)
                lbl.image = tkimg
            except Exception:
                lbl.configure(image="", text=f"(failed) {path.name}")

    def _tick_ui(self) -> None:
        # Drain queues.
        try:
            while True:
                line = self.log_q.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass

        try:
            while True:
                evt = self.event_q.get_nowait()
                self._handle_event(evt)
        except queue.Empty:
            pass

        # Reset buttons when process finishes.
        if self.proc is None and self.btn_start["state"] == "disabled":
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")

        self.root.after(120, self._tick_ui)

    def _refresh_datasets(self) -> None:
        base = self.sim_root / "outputs" / "sim_data" / "runs"
        base.mkdir(parents=True, exist_ok=True)
        ds = [p for p in base.iterdir() if p.is_dir()]
        ds.sort(key=lambda p: p.name)
        self._dataset_dirs = ds
        names = [p.name for p in ds]
        self.dataset_combo["values"] = names

        # Auto-select current, if any.
        if self.var_dataset.get() in names:
            return
        if names:
            self.var_dataset.set(names[-1])
            self._on_dataset_selected()

    def _selected_dataset_dir(self) -> Optional[Path]:
        name = self.var_dataset.get().strip()
        if not name:
            return None
        for p in self._dataset_dirs:
            if p.name == name:
                return p
        return None

    def _on_dataset_selected(self, _evt: Optional[object] = None) -> None:
        ds = self._selected_dataset_dir()
        if not ds:
            return
        self.state.dataset_dir = ds
        self._recent_imgs.clear()

        img_dir = ds / "images"
        if img_dir.exists():
            imgs = sorted(img_dir.glob("*.png"))
            for p in imgs[-9:]:
                self._recent_imgs.append(str(p))
            self._refresh_thumbnails()

    def _delete_dataset(self) -> None:
        ds = self._selected_dataset_dir()
        if not ds:
            return
        if self.proc is not None:
            messagebox.showwarning("Busy", "Stop the running process before deleting datasets.")
            return
        if not messagebox.askyesno("Delete dataset", f"Delete dataset folder?\n\n{ds}"):
            return
        import shutil
        try:
            shutil.rmtree(ds)
            self._append_log(f"\n[deleted dataset {ds}]\n")
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))
        self._refresh_datasets()

    def _validate_selected(self) -> None:
        ds = self._selected_dataset_dir()
        if not ds:
            return
        self._run_simple_cmd(["./.venv/bin/python", "tools/validate_dataset.py", "--data", str(ds)])

    def _train_selected(self) -> None:
        ds = self._selected_dataset_dir()
        if not ds:
            return
        model_out = self.sim_root / "outputs" / "models" / f"{ds.name}.pt"
        self._run_simple_cmd(["./.venv/bin/python", "scripts/train.py", "--data", str(ds), "--out", str(model_out)])

    def _eval_selected(self) -> None:
        ds = self._selected_dataset_dir()
        if not ds:
            return
        model_in = self.sim_root / "outputs" / "models" / f"{ds.name}.pt"
        if not model_in.exists():
            messagebox.showerror("Missing model", f"Model not found:\n{model_in}\n\nRun Train first.")
            return
        self._run_simple_cmd(["./.venv/bin/python", "scripts/eval.py", "--data", str(ds), "--model", str(model_in)])


def main() -> None:
    sim_root = Path(__file__).resolve().parent.parent
    root = tk.Tk()
    app = MonitorApp(root, sim_root=sim_root)
    root.mainloop()


if __name__ == "__main__":
    main()
