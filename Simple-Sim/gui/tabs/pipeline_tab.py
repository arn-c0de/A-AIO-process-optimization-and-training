"""Pipeline Control Tab - Enhanced version of the original monitor."""

from __future__ import annotations
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageTk

from .base_tab import BaseTab
from gui.state import UiState

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from simple_sim.schema import read_jsonl, LabelRow, MetaRow


class PipelineControlTab(BaseTab):
    """Tab 1: Pipeline control with enhanced monitoring."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)

        # Process management
        self.proc: Optional[subprocess.Popen[str]] = None
        self.stop_evt = threading.Event()

        # Queues for thread communication
        self.log_q: queue.Queue[str] = queue.Queue()
        self.event_q: queue.Queue[Dict[str, Any]] = queue.Queue()

        # Event log handling
        self.event_log_path = self.sim_root / "outputs" / "live" / "events.jsonl"
        self._event_fp = None
        self._event_pos = 0

        # Image display
        self._thumb_refs = []  # keep PhotoImage references
        self._recent_imgs: list[str] = []
        self._dataset_dirs: list[Path] = []
        self._label_dict: dict[str, str] = {}  # Map sample_id to class label
        self._image_to_sample: dict[str, str] = {}  # Map image_path to sample_id

        # UI components (will be created in build_ui)
        self.btn_start: ttk.Button
        self.btn_stop: ttk.Button
        self.var_config: tk.StringVar
        self.var_out: tk.StringVar
        self.var_model: tk.StringVar
        self.var_wheelhouse: tk.StringVar
        self.var_eventlog: tk.StringVar
        self.var_phase: tk.StringVar
        self.var_epoch: tk.StringVar
        self.var_ips: tk.StringVar
        self.var_last: tk.StringVar
        self.var_dataset: tk.StringVar
        self.dataset_combo: ttk.Combobox
        self.img_labels: list[ttk.Label]
        self.txt: tk.Text

    def build_ui(self) -> None:
        """Build the pipeline control UI."""
        # Note: Don't pack self.frame - it's managed by the notebook

        # Top controls - Row 1: Start/Stop and Pipeline Configuration
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 5))

        self.btn_start = ttk.Button(top, text="▶ Start Pipeline", command=self.start_pipeline, width=15)
        self.btn_start.pack(side="left")

        self.btn_stop = ttk.Button(top, text="⏹ Stop", command=self.stop_pipeline, state="disabled", width=10)
        self.btn_stop.pack(side="left", padx=(8, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        # Pipeline run mode
        ttk.Label(top, text="Run Mode:").pack(side="left", padx=(0, 6))
        self.var_run_mode = tk.StringVar(value="single")
        run_mode_frame = ttk.Frame(top)
        run_mode_frame.pack(side="left")

        ttk.Radiobutton(run_mode_frame, text="Single", variable=self.var_run_mode,
                       value="single").pack(side="left")
        ttk.Radiobutton(run_mode_frame, text="Multiple", variable=self.var_run_mode,
                       value="multiple").pack(side="left", padx=(5, 0))
        ttk.Radiobutton(run_mode_frame, text="Continuous", variable=self.var_run_mode,
                       value="continuous").pack(side="left", padx=(5, 0))

        ttk.Label(top, text="Count:").pack(side="left", padx=(10, 6))
        self.var_run_count = tk.StringVar(value="10")
        self.run_count_entry = ttk.Entry(top, textvariable=self.var_run_count, width=5)
        self.run_count_entry.pack(side="left")

        # Enable/disable count based on mode
        def on_run_mode_change(*args):
            if self.var_run_mode.get() == "multiple":
                self.run_count_entry.configure(state="normal")
            else:
                self.run_count_entry.configure(state="disabled")
        self.var_run_mode.trace("w", on_run_mode_change)
        on_run_mode_change()

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        # Dataset mode
        ttk.Label(top, text="Dataset:").pack(side="left", padx=(0, 6))
        self.var_dataset_mode = tk.StringVar(value="new")
        ttk.Radiobutton(top, text="Create New", variable=self.var_dataset_mode,
                       value="new").pack(side="left")
        ttk.Radiobutton(top, text="Extend Existing", variable=self.var_dataset_mode,
                       value="extend").pack(side="left", padx=(5, 0))

        # Top controls - Row 2: Config paths
        top2 = ttk.Frame(self.frame)
        top2.pack(fill="x", pady=(0, 5))

        ttk.Label(top2, text="Config:").pack(side="left", padx=(0, 6))
        self.var_config = tk.StringVar(value="configs/run_0001.yaml")
        ttk.Entry(top2, textvariable=self.var_config, width=30).pack(side="left")

        ttk.Label(top2, text="Out:").pack(side="left", padx=(10, 6))
        self.var_out = tk.StringVar(value="outputs/sim_data/runs/run_0001")
        ttk.Entry(top2, textvariable=self.var_out, width=35).pack(side="left")

        ttk.Label(top2, text="Model:").pack(side="left", padx=(10, 6))
        self.var_model = tk.StringVar(value="outputs/models/run_0001.pt")
        ttk.Entry(top2, textvariable=self.var_model, width=30).pack(side="left")

        # Status line
        status = ttk.Frame(self.frame)
        status.pack(fill="x", pady=(10, 0))

        self.var_phase = tk.StringVar(value="phase: idle")
        self.var_run_progress = tk.StringVar(value="run: -")
        self.var_epoch = tk.StringVar(value="epoch: -")
        self.var_ips = tk.StringVar(value="img/s: -")
        self.var_last = tk.StringVar(value="last: -")

        ttk.Label(status, textvariable=self.var_phase).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_run_progress).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_epoch).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_ips).pack(side="left", padx=(0, 14))
        ttk.Label(status, textvariable=self.var_last).pack(side="left", padx=(0, 14))

        # Track current run iteration
        self.current_run_iteration = 0
        self.total_run_count = 0

        # Main split
        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(10, 0))

        # Left: datasets + image grid (4x3 = 12 thumbnails)
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

        ttk.Label(left, text="Latest Images (4x3 grid)").pack(anchor="w", pady=(10, 0))

        img_grid = ttk.Frame(left)
        img_grid.pack(fill="both", expand=True, pady=(6, 0))

        self.img_labels = []
        rows, cols = 4, 3  # Enhanced: 12 thumbnails instead of 9
        for r in range(rows):
            for c in range(cols):
                frm = ttk.Frame(img_grid, relief="groove", borderwidth=1)
                frm.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
                img_grid.grid_rowconfigure(r, weight=1)
                img_grid.grid_columnconfigure(c, weight=1)

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

        # Initialize datasets and start UI ticker
        self._refresh_datasets()
        self._tick_ui()

    def start_pipeline(self) -> None:
        """Start the pipeline process with configured run mode."""
        if self.proc is not None:
            return

        # Get run configuration
        run_mode = self.var_run_mode.get()
        dataset_mode = self.var_dataset_mode.get()

        # Determine output directory
        if dataset_mode == "extend":
            # Use selected dataset
            selected_ds = self._selected_dataset_dir()
            if not selected_ds:
                messagebox.showerror("Error", "No dataset selected to extend.\n\nSelect a dataset or use 'Create New' mode.")
                return
            out_dir = str(selected_ds)
            self._append_log(f"\n=== Extending existing dataset: {out_dir} ===\n")
        else:
            # Create new dataset
            out_dir = self.var_out.get().strip()
            self._append_log(f"\n=== Creating new dataset: {out_dir} ===\n")

        # Get run count
        if run_mode == "single":
            run_count = 1
        elif run_mode == "multiple":
            try:
                run_count = int(self.var_run_count.get())
                if run_count < 1:
                    raise ValueError()
            except:
                messagebox.showerror("Error", "Invalid run count. Please enter a positive integer.")
                return
        else:  # continuous
            run_count = -1  # Infinite

        self._append_log(f"Run mode: {run_mode}" + (f" ({run_count}x)" if run_count > 0 else " (continuous)") + "\n\n")

        # Start pipeline runner in background thread
        threading.Thread(target=self._run_pipeline_loop, args=(out_dir, run_count), daemon=True).start()

    def _run_pipeline_loop(self, out_dir: str, run_count: int) -> None:
        """Run pipeline in a loop (runs in background thread).

        Args:
            out_dir: Output directory
            run_count: Number of runs (-1 for infinite)
        """
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.stop_evt.clear()

        self.current_run_iteration = 0
        self.total_run_count = run_count

        while run_count == -1 or self.current_run_iteration < run_count:
            if self.stop_evt.is_set():
                self._append_log(f"\n=== Pipeline stopped by user after {self.current_run_iteration} runs ===\n")
                break

            self.current_run_iteration += 1

            # Update status display
            if run_count > 0:
                self.var_run_progress.set(f"run: {self.current_run_iteration}/{run_count}")
            else:
                self.var_run_progress.set(f"run: {self.current_run_iteration} (∞)")

            self._append_log(f"\n{'='*60}\n")
            self._append_log(f"Pipeline Run {self.current_run_iteration}" + (f"/{run_count}" if run_count > 0 else " (continuous)") + "\n")
            self._append_log(f"{'='*60}\n\n")

            # Run single pipeline iteration
            success = self._run_single_pipeline(out_dir)

            if not success and run_count > 1:
                self._append_log(f"\n⚠ Run {self.current_run_iteration} failed, but continuing...\n")

            # Small delay between runs
            if run_count != 1 and not self.stop_evt.is_set():
                time.sleep(2)

        self._append_log(f"\n{'='*60}\n")
        self._append_log(f"All pipeline runs complete! Total: {self.current_run_iteration}\n")
        self._append_log(f"{'='*60}\n")

        # Reset buttons and status
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.state.phase = "idle"
        self.var_run_progress.set("run: -")
        self.current_run_iteration = 0
        self.total_run_count = 0

    def _run_single_pipeline(self, out_dir: str) -> bool:
        """Run a single pipeline iteration.

        Args:
            out_dir: Output directory

        Returns:
            True if successful, False otherwise
        """
        eventlog = self.event_log_path
        cfg = self.var_config.get().strip()
        model_path = self.var_model.get().strip()

        env = os.environ.copy()
        env["WHEELHOUSE"] = "wheelhouse"
        env["EVENT_LOG"] = str(eventlog)
        env["CONFIG"] = cfg
        env["DATA_DIR"] = out_dir
        env["MODEL_PATH"] = model_path

        cmd = ["bash", "-lc", "./run_pipeline.sh"]

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.sim_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            # Start monitoring threads
            t = threading.Thread(target=self._read_process_output, daemon=True)
            t.start()

            te = threading.Thread(target=self._tail_events, daemon=True)
            te.start()

            self.state.phase = "running"

            # Wait for process to complete
            self.proc.wait()
            rc = self.proc.returncode
            self.proc = None

            return rc == 0

        except Exception as e:
            self._append_log(f"\nError running pipeline: {e}\n")
            self.proc = None
            return False

    def stop_pipeline(self) -> None:
        """Stop the pipeline process."""
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
        """Read process output in background thread."""
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
        """Tail the event log file in background thread."""
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
        """Handle telemetry events."""
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
        """Add image to recent images list."""
        if not self.state.dataset_dir:
            return
        img_path = self.state.dataset_dir / rel_path
        if not img_path.exists():
            return
        name = str(img_path)
        if self._recent_imgs and self._recent_imgs[-1] == name:
            return
        self._recent_imgs.append(name)
        self._recent_imgs = self._recent_imgs[-12:]  # Keep last 12
        self._refresh_thumbnails()

    def _refresh_thumbnails(self) -> None:
        """Refresh thumbnail display."""
        self._thumb_refs.clear()
        for i, lbl in enumerate(self.img_labels):
            if i >= len(self._recent_imgs):
                lbl.configure(image="", text="(no image)")
                continue

            path = Path(self._recent_imgs[-1 - i])
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((280, 180))
                tkimg = ImageTk.PhotoImage(img)
                self._thumb_refs.append(tkimg)

                # Get class label for this image
                class_label = ""
                if self.state.dataset_dir:
                    # Get relative path from dataset root
                    try:
                        rel_path = path.relative_to(self.state.dataset_dir)
                        rel_path_str = str(rel_path).replace('\\', '/')  # Normalize path separators

                        # Look up sample_id from image path
                        sample_id = self._image_to_sample.get(rel_path_str)
                        if sample_id:
                            # Get class label from sample_id
                            class_name = self._label_dict.get(sample_id, "")
                            if class_name:
                                class_label = f" [{class_name}]"
                    except ValueError:
                        pass  # Path is not relative to dataset_dir

                display_text = f"{path.name}{class_label}"
                lbl.configure(image=tkimg, text=display_text)
                lbl.image = tkimg
            except Exception:
                lbl.configure(image="", text=f"(failed) {path.name}")

    def _tick_ui(self) -> None:
        """UI update ticker."""
        # Drain queues
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

        # Reset buttons when process finishes
        if self.proc is None and self.btn_start["state"] == "disabled":
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")

        self.frame.after(120, self._tick_ui)

    def _append_log(self, s: str) -> None:
        """Append text to log widget."""
        self.txt.configure(state="normal")
        self.txt.insert("end", s)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _refresh_datasets(self) -> None:
        """Refresh dataset dropdown list."""
        base = self.sim_root / "outputs" / "sim_data" / "runs"
        base.mkdir(parents=True, exist_ok=True)
        ds = [p for p in base.iterdir() if p.is_dir()]
        ds.sort(key=lambda p: p.name)
        self._dataset_dirs = ds
        names = [p.name for p in ds]
        self.dataset_combo["values"] = names

        # Auto-select current, if any
        if self.var_dataset.get() in names:
            return
        if names:
            self.var_dataset.set(names[-1])
            self._on_dataset_selected()

    def _selected_dataset_dir(self) -> Optional[Path]:
        """Get currently selected dataset directory."""
        name = self.var_dataset.get().strip()
        if not name:
            return None
        for p in self._dataset_dirs:
            if p.name == name:
                return p
        return None

    def _on_dataset_selected(self, _evt: Optional[object] = None) -> None:
        """Handle dataset selection change."""
        ds = self._selected_dataset_dir()
        if not ds:
            return
        self.state.dataset_dir = ds
        self._recent_imgs.clear()
        self._label_dict.clear()
        self._image_to_sample.clear()

        # Load metadata and labels
        try:
            meta_path = ds / "meta.jsonl"
            if meta_path.exists():
                meta_rows = read_jsonl(meta_path, MetaRow)
                self._image_to_sample = {row.image_path: row.id for row in meta_rows}

            labels_path = ds / "labels.jsonl"
            if labels_path.exists():
                label_rows = read_jsonl(labels_path, LabelRow)
                self._label_dict = {row.id: row.class_name for row in label_rows}
        except Exception as e:
            print(f"Failed to load metadata/labels: {e}")

        img_dir = ds / "images"
        if img_dir.exists():
            imgs = sorted(img_dir.glob("*.png"))
            for p in imgs[-12:]:
                self._recent_imgs.append(str(p))
            self._refresh_thumbnails()

        # Notify other tabs of dataset change
        self.on_dataset_changed()

    def _delete_dataset(self) -> None:
        """Delete selected dataset."""
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
        """Validate selected dataset."""
        ds = self._selected_dataset_dir()
        if not ds:
            return
        self._run_simple_cmd(["./.venv/bin/python", "tools/validate_dataset.py", "--data", str(ds)])

    def _train_selected(self) -> None:
        """Train model on selected dataset."""
        ds = self._selected_dataset_dir()
        if not ds:
            return
        model_out = self.sim_root / "outputs" / "models" / f"{ds.name}.pt"
        self._run_simple_cmd(["./.venv/bin/python", "scripts/train.py", "--data", str(ds), "--out", str(model_out)])

    def _eval_selected(self) -> None:
        """Evaluate model on selected dataset."""
        ds = self._selected_dataset_dir()
        if not ds:
            return
        model_in = self.sim_root / "outputs" / "models" / f"{ds.name}.pt"
        if not model_in.exists():
            messagebox.showerror("Missing model", f"Model not found:\n{model_in}\n\nRun Train first.")
            return
        self._run_simple_cmd(["./.venv/bin/python", "scripts/eval.py", "--data", str(ds), "--model", str(model_in)])
