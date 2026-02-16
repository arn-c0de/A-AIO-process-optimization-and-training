"""Logic for the Pipeline Control Tab."""

from __future__ import annotations
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Callable
import shutil
from datetime import datetime
import re
import shlex
import tempfile

import torch
import cv2
import yaml

from simple_sim.profile_hash import hash_profile
from simple_sim.schema import read_jsonl, LabelRow, MetaRow, write_jsonl
from simple_sim.model_bundle import bundle_checkpoint_path
from simple_sim.manifest import read_dataset_manifest, write_dataset_manifest, write_multi_profile_manifest


class PipelineLogic:
    def __init__(self, sim_root: Path, log_q: queue.Queue[str], event_q: queue.Queue[Dict[str, Any]], event_log_path: Path):
        self.sim_root = sim_root
        self.log_q = log_q
        self.event_q = event_q
        self.event_log_path = event_log_path

        self.proc: Optional[subprocess.Popen[str]] = None
        self.stop_evt = threading.Event()
        self._temp_merge_cleanup_dir: Optional[Path] = None

        self._cpu_prev_total: Optional[int] = None
        self._cpu_prev_idle: Optional[int] = None
        self._last_snap_sig: Optional[tuple[int, int]] = None  # (mtime_ns, size)
        
        self._profile_config_cache: Dict[str, Dict[str, Any]] = {}

    def _model_root(self) -> Path:
        return self.sim_root / "outputs" / "models"

    def _resolve_out_dir(self, out_dir: str) -> Path:
        p = Path(out_dir)
        if p.is_absolute(): return p
        return (self.sim_root / p).resolve()

    def _resolve_model_path(self, model_path: str) -> Path:
        p = Path(model_path)
        if p.is_absolute(): return p
        return (self.sim_root / p).resolve()

    def start_pipeline(self, run_specs: list[dict[str, str]], run_count: int, dataset_mode: str, task: str,
                       event_callback: Callable[[Dict[str, Any]], None], log_callback: Callable[[str], None],
                       ui_update_callback: Callable[[], None], ui_reset_callback: Callable[[], None],
                       status_vars: dict) -> None:
        self.stop_evt.clear()
        self.current_run_iteration = 0
        self.total_run_count = run_count

        def _run_pipeline_loop_thread():
            while run_count == -1 or self.current_run_iteration < run_count:
                if self.stop_evt.is_set():
                    log_callback(f"\n=== Pipeline stopped by user after {self.current_run_iteration} runs ===\n")
                    break

                self.current_run_iteration += 1
                
                # Update status display
                if run_count > 0:
                    self._safe_set_var(status_vars.get("run_progress"), f"run: {self.current_run_iteration}/{run_count}")
                else:
                    self._safe_set_var(status_vars.get("run_progress"), f"run: {self.current_run_iteration} (∞)")

                log_callback(f"\n{'='*60}\n")
                log_callback(f"Pipeline Run {self.current_run_iteration}" + (f"/{run_count}" if run_count > 0 else " (continuous)") + "\n")
                log_callback(f"{'='*60}\n\n")

                overall_ok = True
                for spec in run_specs:
                    if self.stop_evt.is_set():
                        overall_ok = False
                        break
                    
                    pid = spec.get("profile_id", "")
                    cfg = spec.get("config", "")
                    out_dir = spec.get("out_dir", "")
                    model_path = spec.get("model_path", "")

                    log_callback(f"[profile] {pid}\n")
                    log_callback(f"[config]  {cfg}\n")
                    log_callback(f"[out]     {out_dir}\n")
                    if task == "full": log_callback(f"[model]   {model_path}\n")
                    log_callback("\n")

                    status_vars["dataset_dir"] = Path(out_dir)

                    ok = self._run_single_pipeline(
                        out_dir, dataset_mode=dataset_mode, task=task, config_path=cfg, model_path=model_path,
                        status_vars=status_vars, log_callback=log_callback, event_callback=event_callback
                    )
                    overall_ok = overall_ok and bool(ok)

                    if task == "full":
                        try:
                            mp = self._resolve_model_path(model_path)
                            if mp.exists():
                                self._write_model_meta(mp, run_i=self.current_run_iteration, dataset_dir=status_vars["dataset_dir"], dataset_mode=dataset_mode, out_dir=self._resolve_out_dir(out_dir))
                            snap_every = status_vars["snap_every"]
                            if status_vars["autosnap"] and mp.exists() and mp.is_file() and self.current_run_iteration % snap_every == 0:
                                snap = self._snapshot_model_checkpoint(
                                    mp,
                                    run_i=self.current_run_iteration,
                                    snap_every=snap_every,
                                    snap_keep=status_vars.get("snap_keep", 30),
                                )
                                if snap: log_callback(f"[model snapshot] {snap}\n")
                        except Exception: pass

                success = overall_ok

                if not success and run_count > 1: log_callback(f"\n⚠ Run {self.current_run_iteration} failed, but continuing...\n")

                ui_update_callback()

                if run_count != 1 and not self.stop_evt.is_set(): time.sleep(2)

            log_callback(f"\n{'='*60}\n")
            log_callback(f"All pipeline runs complete! Total: {self.current_run_iteration}\n")
            log_callback(f"{'='*60}\n")

            ui_reset_callback()

        threading.Thread(target=_run_pipeline_loop_thread, daemon=True).start()

    def _run_single_pipeline(self, out_dir: str, *, dataset_mode: str, task: str, config_path: str, model_path: str,
                             status_vars: dict, log_callback: Callable[[str], None], event_callback: Callable[[Dict[str, Any]], None]) -> bool:
        eventlog = self.event_log_path
        cfg = str(config_path).strip()
        model_path = str(model_path).strip()

        env = os.environ.copy()
        env["WHEELHOUSE"] = "wheelhouse"
        env["EVENT_LOG"] = str(eventlog)
        env["CONFIG"] = cfg
        env["DATA_DIR"] = out_dir
        env["MODEL_PATH"] = model_path
        env["DATASET_MODE"] = dataset_mode

        if task == "full": cmd = ["bash", "-lc", "./run_pipeline.sh"]
        elif task == "generate_only":
            cmd_str = (
                "./.venv/bin/python scripts/generate.py "
                "--config \"$CONFIG\" "
                "--out \"$DATA_DIR\" "
                "$([[ \"${DATASET_MODE}\" == \"extend\" ]] && echo --extend) "
                "&& "
                "./.venv/bin/python tools/validate_dataset.py --data \"$DATA_DIR\""
            )
            cmd = ["bash", "-lc", cmd_str]
        elif task == "generate_mixed":
            cmd_str = (
                "./.venv/bin/python scripts/generate_profile_dataset.py "
                "--config \"$CONFIG\" "
                "--out \"$DATA_DIR\" "
                "&& "
                "./.venv/bin/python tools/validate_dataset.py --data \"$DATA_DIR\""
            )
            cmd = ["bash", "-lc", cmd_str]
        else: raise ValueError(f"Unknown task: {task}")

        try:
            self.proc = subprocess.Popen(cmd, cwd=str(self.sim_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, start_new_session=True)
            t_stdout = threading.Thread(target=self._read_process_output_thread, args=(log_callback,), daemon=True)
            t_events = threading.Thread(target=self._tail_events_thread, args=(event_callback,), daemon=True)
            t_stdout.start()
            t_events.start()
            
            self._safe_set_var(status_vars.get("phase"), "running")
            self.proc.wait()
            rc = self.proc.returncode
            self.proc = None

            cleanup = self._temp_merge_cleanup_dir
            self._temp_merge_cleanup_dir = None
            if cleanup is not None:
                try:
                    if rc == 0: shutil.rmtree(cleanup); log_callback(f"[merge] cleaned up temp dataset: {cleanup}\n")
                    else: log_callback(f"[merge] keeping temp dataset (rc={rc}): {cleanup}\n")
                except Exception as e: log_callback(f"[merge] cleanup failed: {cleanup} ({e})\n")

            return rc == 0

        except Exception as e:
            log_callback(f"\nError running pipeline: {e}\n")
            self.proc = None
            return False

    def stop_pipeline(self) -> None:
        self.stop_evt.set()
        if self.proc is not None:
            try: os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception:
                try: self.proc.terminate()
                except Exception: pass
            
            # Escalate to SIGKILL after a grace period.
            def _kill_later() -> None:
                p = self.proc
                if p is None: return
                try:
                    if p.poll() is None: os.killpg(p.pid, signal.SIGKILL)
                except Exception: pass
            
            threading.Timer(2.0, _kill_later).start()

    def _read_process_output_thread(self, log_callback: Callable[[str], None]) -> None:
        assert self.proc is not None
        fp = self.proc.stdout
        assert fp is not None
        for line in fp:
            log_callback(line)
            if self.stop_evt.is_set(): break
        self.log_q.put(f"\n[process exited rc={self.proc.returncode}]\n")

    def _safe_set_var(self, tk_var: Any, value: str) -> None:
        try:
            if tk_var is not None:
                tk_var.set(value)
        except Exception:
            pass

    def _tail_events_thread(self, event_callback: Callable[[Dict[str, Any]], None]) -> None:
        path = self.event_log_path
        for _ in range(200):
            if self.stop_evt.is_set(): return
            if path.exists(): break
            time.sleep(0.05)

        pos = 0
        while not self.stop_evt.is_set():
            try:
                if not path.exists(): time.sleep(0.1); continue
                with open(path, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        event_callback(json.loads(line))
                    pos = f.tell()
            except Exception: time.sleep(0.1)
            time.sleep(0.1)

    def _write_model_meta(self, model_path: Path, *, run_i: int, dataset_dir: Optional[Path], dataset_mode: str, out_dir: Path) -> None:
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"), "run_iteration": int(run_i),
            "dataset_mode": dataset_mode, "dataset_dir": str(dataset_dir) if dataset_dir else None,
            "out_dir": str(out_dir), "model_path": str(model_path),
        }
        Path(str(model_path) + ".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _snapshot_model_checkpoint(self, model_path: Path, *, run_i: int, snap_every: int, snap_keep: int) -> Optional[Path]:
        try:
            model_path = Path(model_path)
            if not model_path.exists(): return None
            st = model_path.stat()
            sig = (int(st.st_mtime_ns), int(st.st_size))
            if self._last_snap_sig == sig: return None
            self._last_snap_sig = sig
            
            dst_root = self.sim_root / "outputs" / "models" / "versions" / model_path.stem
            dst_root.mkdir(parents=True, exist_ok=True)
            tag = time.strftime("%Y%m%d_%H%M%S")
            base = dst_root / f"{model_path.stem}_{tag}_run{int(run_i):03d}.pt"
            dst = base
            if dst.exists():
                for i in range(1, 1000):
                    cand = dst_root / f"{model_path.stem}_{tag}_run{int(run_i):03d}_{i:03d}.pt"
                    if not cand.exists():
                        dst = cand
                        break
            shutil.copy2(model_path, dst)
            self._write_snapshot_meta(dst, src=model_path, run_i=run_i)
            self._prune_model_snapshots(dst_root, snap_keep)
            return dst
        except Exception: return None

    def _write_snapshot_meta(self, snapshot_path: Path, *, src: Path, run_i: int) -> None:
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"), "run_iteration": int(run_i),
            "snapshot_path": str(snapshot_path), "source_model_path": str(src),
        }
        Path(str(snapshot_path) + ".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _prune_model_snapshots(self, dst_root: Path, snap_keep: int) -> None:
        if snap_keep == 0: return
        try:
            snaps = sorted([p for p in dst_root.glob("*.pt") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
            for p in snaps[snap_keep:]: p.unlink()
        except Exception: pass

    def _read_cpu_percent(self) -> Optional[float]:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f: line = f.readline().strip()
            parts = line.split()
            if not parts or parts[0] != "cpu" or len(parts) < 5: return None
            nums = [int(x) for x in parts[1:]]
            total, idle = sum(nums), nums[3] + (nums[4] if len(nums) > 4 else 0)
            if self._cpu_prev_total is None or self._cpu_prev_idle is None:
                self._cpu_prev_total, self._cpu_prev_idle = total, idle
                return None
            dt, di = total - self._cpu_prev_total, idle - self._cpu_prev_idle
            self._cpu_prev_total, self._cpu_prev_idle = total, idle
            return 100.0 * (1.0 - (di / dt)) if dt > 0 else None
        except Exception: return None

    def _read_ram_percent(self) -> Optional[tuple[float, int, int]]:
        try:
            total_kb, avail_kb = None, None
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"): total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"): avail_kb = int(line.split()[1])
                    if total_kb is not None and avail_kb is not None: break
            if total_kb is None or avail_kb is None or total_kb <= 0: return None
            used_kb = total_kb - avail_kb
            return 100.0 * (used_kb / total_kb), used_kb * 1024, total_kb * 1024
        except Exception: return None

    def _read_gpu_percent(self) -> Optional[float]:
        try:
            proc = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=0.5, check=False)
            out = (proc.stdout or "").strip()
            if not out: return None
            return float(out.splitlines()[0].strip())
        except Exception: return None

    def get_system_stats(self) -> Tuple[Optional[float], Optional[Tuple[float, int, int]], Optional[float]]:
        return self._read_cpu_percent(), self._read_ram_percent(), self._read_gpu_percent()

    def _compute_dir_size_bytes(self, root: Path) -> int:
        total = 0
        try:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    try: total += int(os.stat(Path(dirpath) / fn, follow_symlinks=False).st_size)
                    except Exception: pass
        except Exception: pass
        return total

    def start_dataset_size_calc(self, ds: Path, update_callback: Callable[[int], None]) -> None:
        def worker() -> None:
            size_b = self._compute_dir_size_bytes(ds)
            update_callback(size_b)
        threading.Thread(target=worker, daemon=True).start()

    def get_dataset_profile_id(self, ds: Path) -> Optional[str]:
        if not ds: return None
        manifest_path = ds / "dataset_manifest.json"
        if not manifest_path.exists(): return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f: manifest = json.load(f)
            mver = int(manifest.get("manifest_version", 1) or 1)
            if mver == 1: return manifest.get("component_profile", {}).get("profile_id")
            if mver == 2: return "multi"
            return None
        except Exception: return None

    def get_dataset_profile_hash(self, ds: Path) -> Optional[str]:
        if not ds: return None
        manifest_path = ds / "dataset_manifest.json"
        if not manifest_path.exists(): return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f: manifest = json.load(f)
            mver = int(manifest.get("manifest_version", 1) or 1)
            if mver == 1: return manifest.get("component_profile", {}).get("profile_hash")
            return None
        except Exception: return None

    def load_model_profile(self, model_path: Path) -> tuple[Optional[str], Optional[str]]:
        try:
            if model_path.is_dir(): return "multi", None
        except Exception: pass
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(comp := checkpoint.get("component_profile"), dict):
                return comp.get("profile_id"), comp.get("profile_hash")
        except Exception: pass
        return None, None

    def get_all_model_paths(self) -> List[Path]:
        root = self.sim_root / "outputs" / "models"
        root.mkdir(parents=True, exist_ok=True)
        cand: list[Path] = []
        cand.extend(sorted(root.glob("*.pt")))
        cand.extend([p for p in sorted(root.glob("*.bundle")) if p.is_dir()])
        cand.extend([p for p in sorted((root / "bundles").glob("*")) if p.is_dir()])
        cand.extend(sorted((root / "imports").glob("*.pt")))
        cand.extend(sorted((root / "versions").glob("**/*.pt")))
        return list({str(p.resolve()): p for p in cand if p.is_file() or p.is_dir()}.values())

    def get_profile_classifier_models(self) -> List[Path]:
        root = self.sim_root / "outputs" / "models"
        cand: list[Path] = []
        if root.exists():
            cand.extend(sorted(root.glob("profile_classifier_*.pt")))
            cand.extend(sorted(root.glob("**/profile_classifier_*.pt")))
        return list({str(p.resolve()): p for p in cand if p.is_file()}.values())

    def build_profile_model(self, preset: str, config_path_str: str, out_ds_str: str, out_model_str: str, log_callback: Callable[[str], None], run_simple_cmd_callback: Callable[[List[str], Optional[Dict[str, str]]], None]) -> None:
        if preset == "full":
            cfg = self.sim_root / "configs" / "run_profile_cls.yaml"
            out_ds = self.sim_root / "outputs" / "sim_data" / "runs" / "run_profile_cls"
            out_model = self.sim_root / "outputs" / "models" / "profile_classifier_v1.pt"
        else:
            cfg = self.sim_root / "configs" / "run_profile_cls_quick.yaml"
            out_ds = self.sim_root / "outputs" / "sim_data" / "runs" / "run_profile_cls_quick"
            out_model = self.sim_root / "outputs" / "models" / "profile_classifier_quick.pt"

        if not cfg.exists(): raise FileNotFoundError(f"Config not found:\\n{cfg}")
        
        tag = time.strftime("%Y%m%d_%H%M%S")
        if out_ds.exists(): out_ds = out_ds.parent / f"{out_ds.name}_{tag}"
        if out_model.exists(): out_model = out_model.with_name(f"{out_model.stem}_{tag}{out_model.suffix}")

        rel_cfg = str(cfg.resolve().relative_to(self.sim_root.resolve()) if str(cfg).startswith(str(self.sim_root)) else str(cfg))
        rel_out_ds = str(out_ds.resolve().relative_to(self.sim_root.resolve()) if str(out_ds).startswith(str(self.sim_root)) else str(out_ds))
        rel_out_model = str(out_model.resolve().relative_to(self.sim_root.resolve()) if str(out_model).startswith(str(self.sim_root)) else str(out_model))
        
        cmd = [
            "./.venv/bin/python", "scripts/generate_profile_dataset.py", "--config", rel_cfg, "--out", rel_out_ds,
            "&&", "./.venv/bin/python", "scripts/train_profile.py", "--data", rel_out_ds, "--out", rel_out_model,
        ]
        cmd_q = " ".join(shlex.quote(x) for x in cmd)
        log_callback(f"\n[build profile model] preset={preset}\nconfig: {cfg}\ndata:   {out_ds}\nout:    {out_model}\n[cmd] {cmd_q}\n\n")
        run_simple_cmd_callback([cmd_q])

    def get_profile_paths(self, render_backend: str) -> List[Path]:
        profiles_dir = self.sim_root / "configs" / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        profile_files = sorted(profiles_dir.glob("*.yaml"))

        values: List[str] = []
        for p in profile_files:
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8"))
                supported = (((data or {}).get("profile") or {}).get("supported_render_backends") or [])
                if not supported: supported = ["opencv_2d"]
                if render_backend not in supported: continue
            except Exception:
                if render_backend != "opencv_2d": continue
            values.append(p.stem)
        return values

    def slugify_name(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "unnamed"

    def ensure_dataset_skeleton(self, profile_id: str, config_path: Path, out_dir: Path, log_callback: Callable[[str], None]) -> bool:
        if not config_path.exists(): return False
        out_dir = out_dir.resolve()
        manifest_path = out_dir / "dataset_manifest.json"
        if manifest_path.exists(): return False
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "images").mkdir(exist_ok=True)
        (out_dir / "splits").mkdir(exist_ok=True)
        for split in ["train", "val", "test"]: (out_dir / "splits" / f"{split}.txt").write_text("", encoding="utf-8")
        Path(out_dir / "meta.jsonl").write_text("", encoding="utf-8")
        Path(out_dir / "labels.jsonl").write_text("", encoding="utf-8")
        try: shutil.copy2(config_path, out_dir / "config.yaml")
        except Exception: pass

        cfg_data, run_block = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}), {}
        if isinstance(cfg_data, dict): run_block = cfg_data.get("run") or {}
        run_id, classes = str(run_block.get("run_id") or out_dir.name), run_block.get("classes", {})
        
        profiles_dir = self.sim_root / "configs" / "profiles"
        profile_path = profiles_dir / f"{profile_id}.yaml"
        profile_hash = hash_profile(profile_path) if profile_path.exists() else ""

        manifest = {
            "manifest_version": 1, "created_at": datetime.utcnow().isoformat() + "Z", "run_id": run_id,
            "component_profile": {"profile_id": profile_id, "profile_hash": profile_hash, "profile_path": str(profile_path) if profile_path.exists() else ""},
            "generator": {"version": "1.0.1", "git_commit": None, "script": "gui.profile_placeholder"},
            "dataset_stats": {"total_samples": 0, "splits": {"train": 0, "val": 0, "test": 0}, "classes": {str(k): 0 for k in classes.keys()}},
            "extend_history": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log_callback(f"[profile] created placeholder dataset manifest for {out_dir.name}\n")
        return True

    def config_for_profile(self, profile_id: str, want_backend: Optional[str] = None) -> Optional[Dict[str, Any]]:
        cache_key = f"{profile_id}|{want_backend or ''}"
        if cache_key in self._profile_config_cache: return self._profile_config_cache[cache_key]

        configs_root = self.sim_root / "configs"
        best_entry = None
        for cfg in sorted(configs_root.glob("*.yaml")):
            try:
                data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
            except Exception: continue
            
            run_block = data.get("run") or {}
            if run_block.get("component_profile") != profile_id: continue
            if want_backend:
                backend = ((data.get("render") or {}).get("backend") or "").strip()
                if backend and backend != want_backend: continue
            
            best_entry = {"path": cfg, "run_id": str(run_block.get("run_id") or cfg.stem)}
            break

        if best_entry is None:
            needle = f"component_profile: {profile_id}"
            needle_quoted = f'component_profile: "{profile_id}"'
            for cfg in sorted(configs_root.glob("*.yaml")):
                try: text = cfg.read_text(encoding="utf-8")
                except Exception: continue
                if needle in text or needle_quoted in text:
                    if want_backend:
                        try:
                            data = yaml.safe_load(text)
                            backend = ((data.get("render") or {}).get("backend") or "").strip()
                            if backend and backend != want_backend: continue
                        except Exception: continue
                    best_entry = {"path": cfg, "run_id": cfg.stem}
                    break

        self._profile_config_cache[cache_key] = best_entry
        return best_entry

    def _datasets_require_bundle(self, dss: List[Path], get_dataset_profile_id_func: Callable[[Path], Optional[str]]) -> bool:
        pids: set[str] = set()
        for ds in dss:
            if pid := get_dataset_profile_id_func(ds): pids.add(pid)
        return len(pids) > 1

    def _is_model_bundle_target(self, model_path: Path) -> bool:
        try: return model_path.exists() and model_path.is_dir() or str(model_path).endswith(".bundle")
        except Exception: return False
        
    def _slugify_name(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "unnamed"

    def merge_datasets_for_training(self, sources: List[Path], out_dir: Path, log_callback: Callable[[str], None], log_error_callback: Callable[[str], None]) -> None:
        out_dir = Path(out_dir)
        if out_dir.exists(): raise FileExistsError(f"Output directory already exists: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "images").mkdir(parents=True, exist_ok=False)
        (out_dir / "splits").mkdir(parents=True, exist_ok=False)

        if not sources: raise ValueError("No sources provided")

        cfg0_path = sources[0] / "config.yaml"
        if not cfg0_path.exists(): raise FileNotFoundError(f"Missing config.yaml in {sources[0]}")
        cfg0_text = cfg0_path.read_text(encoding="utf-8")
        try: cfg0 = yaml.safe_load(cfg0_text) or {}
        except Exception: cfg0 = {}
        classes0 = cfg0.get("classes") or {}
        class_keys0 = tuple(sorted((classes0.keys() if isinstance(classes0, dict) else [])))
        roi0 = cfg0.get("roi") or {}
        roi0_w, roi0_h = int(roi0.get("width_px") or 0), int(roi0.get("height_px") or 0)

        src_profiles: Dict[str, Dict[str, str]] = {}
        run_id = sources[0].name
        for ds in sources:
            manifest_path = ds / "dataset_manifest.json"
            if not manifest_path.exists(): raise FileNotFoundError(f"Missing dataset_manifest.json in {ds}")
            manifest = read_dataset_manifest(manifest_path)
            if (mver := int(manifest.get("manifest_version", 1) or 1)) != 1: raise ValueError(f"Only manifest_version=1 source datasets are supported for merge (got {mver} in {ds})")
            run_id = str(manifest.get("run_id") or run_id)
            comp = manifest.get("component_profile") or {}
            pid, phash, ppath = str(comp.get("profile_id") or ""), str(comp.get("profile_hash") or ""), str(comp.get("profile_path") or "")
            if not pid or not phash: raise ValueError(f"Invalid manifest in {ds}: missing profile_id/profile_hash")
            src_profiles[str(ds.resolve())] = {"profile_id": pid, "profile_hash": phash, "profile_path": ppath}

        target_w, target_h = (roi0_w if roi0_w > 0 else None), (roi0_h if roi0_h > 0 else None)
        for ds in sources[1:]:
            cfg_path = ds / "config.yaml"
            if not cfg_path.exists(): raise FileNotFoundError(f"Missing config.yaml in {ds}")
            try: cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception: cfg = {}
            classes = cfg.get("classes") or {}
            if (class_keys := tuple(sorted((classes.keys() if isinstance(classes, dict) else [])))) != class_keys0:
                raise ValueError(f"Cannot merge datasets with different class sets.\\n  first: {class_keys0}\\n  {ds.name}: {class_keys}")
            roi = cfg.get("roi") or {}
            w, h = int(roi.get("width_px") or 0), int(roi.get("height_px") or 0)
            if w > 0: target_w = w if target_w is None else min(target_w, w)
            if h > 0: target_h = h if target_h is None else min(target_h, h)

        if target_w is None or target_h is None or target_w <= 0 or target_h <= 0: raise ValueError("Cannot determine target ROI size for merged dataset (missing roi.width_px/height_px).")
        target_w, target_h = int(target_w), int(target_h)

        meta_out, label_out = [], []
        split_counters: Dict[str, int] = {}
        splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

        def link_or_copy(src: Path, dst: Path) -> None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            try: os.link(src, dst)
            except Exception: shutil.copy2(src, dst)

        for ds in sources:
            meta_rows = read_jsonl(ds / "meta.jsonl", MetaRow)
            label_map = {row.id: row.class_name for row in read_jsonl(ds / "labels.jsonl", LabelRow)}
            prof = src_profiles.get(str(ds.resolve())) or {}
            pid = prof.get("profile_id") or ""

            for row in meta_rows:
                if not (class_name := label_map.get(row.id)): continue
                split = row.split
                idx = split_counters.setdefault(split, 0)
                new_id, split_counters[split] = f"{run_id}/{row.domain}/{split}/{idx:06d}", idx + 1
                new_image_name = f"{split}_{idx:06d}.png"
                dst_image, src_image = Path(images_dir) / new_image_name, ds / row.image_path
                try:
                    if (img := cv2.imread(str(src_image))) is None: raise ValueError("imread returned None")
                    h_src, w_src = img.shape[:2]
                    if w_src != target_w or h_src != target_h:
                        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA if (w_src > target_w or h_src > target_h) else cv2.INTER_LINEAR)
                        dst_image.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(dst_image), img): raise IOError("imwrite failed")
                    else: link_or_copy(src_image, dst_image)
                except Exception: link_or_copy(src_image, dst_image)

                meta_out.append(MetaRow(schema_version=int(row.schema_version), id=new_id, run_id=str(run_id), domain=row.domain, split=split, seed=int(row.seed), image_path=str(Path("images") / new_image_name), render_backend=row.render_backend, footprint=row.footprint, nominal=row.nominal, defect=row.defect, augment=row.augment, render_meta=getattr(row, "render_meta", {}) or {}))
                label_out.append(LabelRow(schema_version=2, id=new_id, class_name=class_name, profile_id=pid))
                splits[split].append(new_id)

        write_jsonl(out_dir / "meta.jsonl", meta_out)
        write_jsonl(out_dir / "labels.jsonl", label_out)
        for split in ["train", "val", "test"]: Path(out_dir / "splits" / f"{split}.txt").write_text("\n".join(splits[split]) + "\n", encoding="utf-8")
        
        try:
            cfg_out = dict(cfg0) if isinstance(cfg0, dict) else {}
            roi_out = dict(cfg_out.get("roi") or {})
            roi_out["width_px"], roi_out["height_px"] = target_w, target_h
            cfg_out["roi"] = roi_out
            Path(out_dir / "config.yaml").write_text(yaml.safe_dump(cfg_out, sort_keys=False), encoding="utf-8")
        except Exception: Path(out_dir / "config.yaml").write_text(cfg0_text, encoding="utf-8")

        uniq_profiles: Dict[str, Dict[str, str]] = {prof["profile_id"]: prof for prof in src_profiles.values()}
        if len(uniq_profiles) == 1:
            only = next(iter(uniq_profiles.values()))
            write_dataset_manifest(out_dir, run_id=str(run_id), profile_id=str(only["profile_id"]), profile_hash=str(only["profile_hash"]), profile_path=str(only["profile_path"]), meta_rows=meta_out, label_rows=label_out, splits=splits, extend=False)
        else:
            write_multi_profile_manifest(out_dir, run_id=str(run_id), profiles=[{"profile_id": p["profile_id"], "profile_hash": p["profile_hash"], "profile_path": p["profile_path"]} for p in sorted(uniq_profiles.values(), key=lambda x: x["profile_id"])], meta_rows=meta_out, label_rows=label_out, splits=splits, script_name="gui.merge_datasets")
