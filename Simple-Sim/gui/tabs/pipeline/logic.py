"""Logic for the Pipeline Control Tab."""

from __future__ import annotations
import json
from dataclasses import dataclass
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from collections.abc import Callable
import shutil
import re
import shlex

import yaml

from gui.utils.checkpoint_meta import CheckpointMetaCache
from gui.utils.dataset_ops import manifest_profile
from gui.utils.process import StreamingProcessLogic
from gui.utils.paths import list_model_paths, model_root
from simple_sim.cli import argv_for
from simple_sim.dataset_merge import MergeResult, merge_datasets
from simple_sim.dataset_files import CONFIG_YAML, DATASET_MANIFEST, LABELS_JSONL, META_JSONL, SPLITS_DIR
from simple_sim.io_utils import atomic_write_json, load_yaml, now_iso, rel_to
from simple_sim.profile_hash import hash_profile
from simple_sim.manifest import write_dataset_manifest
from simple_sim.model_store import snapshot_checkpoint
from gui.utils.tk_safe import quiet


@dataclass(frozen=True)
class SnapshotOptions:
    """Model snapshots after full runs: every ``every``-th run when ``enabled``, keeping the newest ``keep``
    (``keep == 0``: keep all)."""

    enabled: bool = False
    every: int = 1
    keep: int = 30


def replace_dir(src: Path, dst: Path) -> None:
    """Move directory *src* to *dst*, replacing an existing *dst* without a window where neither exists.

    The old *dst* is renamed aside first and deleted last; if the final rename fails it is put back.
    """
    if not dst.exists():
        src.rename(dst)
        return
    old = dst.with_name(f"{dst.name}.__old_{os.getpid()}")
    shutil.rmtree(old, ignore_errors=True)
    dst.rename(old)
    try:
        src.rename(dst)
    except BaseException:
        old.rename(dst)
        raise
    shutil.rmtree(old, ignore_errors=True)


class PipelineLogic(StreamingProcessLogic):
    """Pipeline runs: ``self.proc`` / ``self.stop_evt`` / ``stop()`` come from :class:`StreamingProcessLogic`.

    The pipeline reads its child's output itself (``launch_process``): chunk-wise with CR handling for
    the tqdm progress lines and an events.jsonl tailer per run, which ``run_streaming`` does not do.
    """

    def __init__(self, sim_root: Path, log_q: queue.Queue[str], event_q: queue.Queue[dict[str, Any]], event_log_path: Path):
        super().__init__(sim_root, log_q)
        self.event_q = event_q
        self.event_log_path = event_log_path

        self._temp_merge_cleanup_dir: Path | None = None

        self._last_snap_sig: tuple[int, int] | None = None  # (mtime_ns, size)
        
        self._profile_config_cache: dict[str, dict[str, Any]] = {}

        # True while the start_pipeline loop thread is alive (also between runs, when no process is running).
        self.loop_running: bool = False
        # Done-event of the currently active events.jsonl tailer (only one tailer at a time).
        self._tail_done: threading.Event | None = None
        self._tail_lock = threading.Lock()
        self._ckpt_meta = CheckpointMetaCache()

    def is_busy(self) -> bool:
        return self.loop_running or self.is_process_running()

    def is_process_running(self) -> bool:
        p = self.proc
        if p is None:
            return False
        try:
            if p.poll() is None:
                return True
        except Exception:
            return True
        self.proc = None
        return False

    def _model_root(self) -> Path:
        return model_root(self.sim_root)

    def start_pipeline(self, run_specs: list[dict[str, str]], run_count: int, dataset_mode: str, task: str,
                       image_filters: dict[str, Any],
                       event_callback: Callable[[dict[str, Any]], None], log_callback: Callable[[str], None],
                       ui_update_callback: Callable[[], None], ui_reset_callback: Callable[[], None],
                       snapshots: SnapshotOptions | None = None,
                       on_status: Callable[[str, str], None] | None = None,
                       run_complete_callback: Callable[[int, list[dict[str, str]], Callable[[str], None]], bool] | None = None) -> None:
        """Run *run_specs* *run_count* times (-1: until stopped) on a worker thread.

        ``on_status(field, text)`` (``field`` in ``"phase"``, ``"run_progress"``) is called from that thread.
        """
        snapshots = snapshots or SnapshotOptions()
        self.stop_evt.clear()
        self.current_run_iteration = 0
        self.total_run_count = run_count
        self.loop_running = True

        def status(field: str, text: str) -> None:
            if on_status is not None:
                try: on_status(field, text)
                except Exception as e: self.log_q.put(f"[ui] failed to update status: {e}\n")

        def _run_pipeline_loop_thread() -> None:
            try:
                self._pipeline_loop(run_specs, run_count, dataset_mode=dataset_mode, task=task, image_filters=image_filters,
                                    snapshots=snapshots, status=status, log_callback=log_callback,
                                    event_callback=event_callback, ui_update_callback=ui_update_callback,
                                    run_complete_callback=run_complete_callback)
            except Exception as e: log_callback(f"\n[pipeline] loop crashed: {e}\n")
            finally:
                self.loop_running = False
                ui_reset_callback()

        try: threading.Thread(target=_run_pipeline_loop_thread, daemon=True).start()
        except Exception: self.loop_running = False; raise

    def _pipeline_loop(self, run_specs: list[dict[str, str]], run_count: int, *, dataset_mode: str, task: str,
                       image_filters: dict[str, Any], snapshots: SnapshotOptions, status: Callable[[str, str], None],
                       log_callback: Callable[[str], None], event_callback: Callable[[dict[str, Any]], None],
                       ui_update_callback: Callable[[], None],
                       run_complete_callback: Callable[[int, list[dict[str, str]], Callable[[str], None]], bool] | None) -> None:
        """Body of the ``start_pipeline`` worker thread: *run_count* runs (-1: until stopped) of all *run_specs*."""
        rule = "=" * 60
        while run_count == -1 or self.current_run_iteration < run_count:
            if self.stop_evt.is_set():
                log_callback(f"\n=== Pipeline stopped by user after {self.current_run_iteration} runs ===\n")
                break

            self.current_run_iteration += 1
            run_i = self.current_run_iteration
            status("run_progress", f"run: {run_i}/{run_count}" if run_count > 0 else f"run: {run_i} (∞)")
            log_callback(f"\n{rule}\n")
            log_callback(f"Pipeline Run {run_i}" + (f"/{run_count}" if run_count > 0 else " (continuous)") + "\n")
            log_callback(f"{rule}\n\n")

            overall_ok = True
            for spec in run_specs:
                if self.stop_evt.is_set():
                    overall_ok = False
                    break
                ok = self._run_spec(spec, dataset_mode=dataset_mode, task=task, image_filters=image_filters,
                                    snapshots=snapshots, status=status, log_callback=log_callback,
                                    event_callback=event_callback)
                overall_ok = overall_ok and ok

            success = overall_ok
            if success and run_complete_callback is not None:
                try:
                    success = bool(run_complete_callback(run_i, run_specs, log_callback))
                except Exception as e:
                    log_callback(f"[post-run] failed: {e}\n")
                    success = False

            if not success and run_count != 1: log_callback(f"\n⚠ Run {run_i} failed, but continuing...\n")

            ui_update_callback()

            if run_count != 1 and not self.stop_evt.is_set(): self.stop_evt.wait(2)

        log_callback(f"\n{rule}\n")
        log_callback(f"All pipeline runs complete! Total: {self.current_run_iteration}\n")
        log_callback(f"{rule}\n")

    def _run_spec(self, spec: dict[str, str], *, dataset_mode: str, task: str, image_filters: dict[str, Any],
                  snapshots: SnapshotOptions, status: Callable[[str, str], None], log_callback: Callable[[str], None],
                  event_callback: Callable[[dict[str, Any]], None]) -> bool:
        """One profile of a pipeline run (+ model meta / snapshot after a full run); True when it succeeded."""
        out_dir, model_path = spec.get("out_dir", ""), spec.get("model_path", "")
        cfg = spec.get("config", "")
        log_callback(f"[profile] {spec.get('profile_id', '')}\n")
        log_callback(f"[config]  {cfg}\n")
        log_callback(f"[out]     {out_dir}\n")
        if task == "full": log_callback(f"[model]   {model_path}\n")
        log_callback("\n")

        ok = self._run_single_pipeline(
            out_dir, dataset_mode=dataset_mode, task=task, config_path=cfg, model_path=model_path,
            image_filters=image_filters, on_status=status, log_callback=log_callback, event_callback=event_callback)
        if task == "full":
            self._after_full_run(model_path, out_dir, dataset_mode=dataset_mode, snapshots=snapshots,
                                 log_callback=log_callback)
        return bool(ok)

    def _after_full_run(self, model_path: str, out_dir: str, *, dataset_mode: str, snapshots: SnapshotOptions,
                        log_callback: Callable[[str], None]) -> None:
        """Write the model's ``.meta.json`` and the optional snapshot; failures are logged, not raised."""
        run_i = self.current_run_iteration
        try:
            mp = self.resolve_path(model_path)
            if mp.exists():
                self._write_model_meta(mp, run_i=run_i, dataset_dir=Path(out_dir), dataset_mode=dataset_mode,
                                       out_dir=self.resolve_path(out_dir))
        except Exception as e:
            log_callback(f"[model meta] failed: {e}\n")
            return
        if not (snapshots.enabled and mp.is_file()): return
        try:
            if snap := self._snapshot_model_checkpoint(mp, run_i=run_i, snap_every=snapshots.every, snap_keep=snapshots.keep):
                log_callback(f"[model snapshot] {snap}\n")
        except Exception as e:
            log_callback(f"[model snapshot] failed: {e}\n")

    def _run_single_pipeline(self, out_dir: str, *, dataset_mode: str, task: str, config_path: str, model_path: str,
                             image_filters: dict[str, Any],
                             on_status: Callable[[str, str], None], log_callback: Callable[[str], None], event_callback: Callable[[dict[str, Any]], None]) -> bool:
        eventlog = self.event_log_path
        cfg = str(config_path).strip()
        model_path = str(model_path).strip()

        env = os.environ.copy()
        env["WHEELHOUSE"] = "wheelhouse"
        env["EVENT_LOG"] = str(eventlog)
        # generate.py/validate_dataset.py read SIMPLE_SIM_EVENT_LOG directly (generate-only modes skip run_pipeline.sh's bridge).
        env["SIMPLE_SIM_EVENT_LOG"] = str(eventlog)
        env["CONFIG"] = cfg
        env["DATA_DIR"] = out_dir
        env["MODEL_PATH"] = model_path
        env["DATASET_MODE"] = dataset_mode
        env["IMAGE_FILTERS"] = json.dumps(image_filters, ensure_ascii=True)
        env["PYTHONUNBUFFERED"] = "1"

        # Generate-only modes run ``python -m simple_sim <command>`` with the GUI's interpreter (plan P4/N9);
        # run_pipeline.sh bootstraps and uses the project venv itself.
        if task == "full": cmd = ["bash", "-lc", "./run_pipeline.sh"]
        elif task == "generate_only":
            cmd_str = (
                f"{shlex.join(argv_for('generate'))} "
                "--config \"$CONFIG\" "
                "--out \"$DATA_DIR\" "
                "$([[ \"${DATASET_MODE}\" == \"extend\" ]] && echo --extend) "
                "&& "
                f"{shlex.join(argv_for('validate'))} --data \"$DATA_DIR\""
            )
            cmd = ["bash", "-lc", cmd_str]
        elif task == "generate_mixed":
            cmd_str = (
                f"{shlex.join(argv_for('generate-profile-dataset'))} "
                "--config \"$CONFIG\" "
                "--out \"$DATA_DIR\" "
                "&& "
                f"{shlex.join(argv_for('validate'))} --data \"$DATA_DIR\""
            )
            cmd = ["bash", "-lc", cmd_str]
        else: raise ValueError(f"Unknown task: {task}")

        try:
            p = self.launch_process(cmd, env, log_callback, event_callback)
            on_status("phase", "running")
            p.wait()
            rc = p.returncode
            self.proc = None
            return rc == 0

        except Exception as e:
            log_callback(f"\nError running pipeline: {e}\n")
            self.proc = None
            return False

    def launch_process(self, cmd: list[str], env: dict[str, str], log_callback: Callable[[str], None], event_callback: Callable[[dict[str, Any]], None],
                       on_exit: Callable[[int | None], None] | None = None) -> subprocess.Popen[str]:
        """Start cmd as self.proc with one stdout reader and one events.jsonl tailer bound to this run.

        The tailer starts at the current end of the event log (no replay of older runs) and stops after a final
        drain once the reader saw EOF/process exit. on_exit(rc) runs on the reader thread after that."""
        try: start_pos = self.event_log_path.stat().st_size
        except Exception: start_pos = 0
        done = threading.Event()
        with self._tail_lock:
            if self._tail_done is not None: self._tail_done.set()  # retire any previous tailer
            self._tail_done = done
        try: self.proc = p = subprocess.Popen(cmd, cwd=str(self.sim_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, start_new_session=True)
        except Exception: done.set(); raise
        threading.Thread(target=self._read_process_output_thread, args=(p, log_callback), kwargs={"done": done, "on_exit": on_exit}, daemon=True).start()
        threading.Thread(target=self._tail_events_thread, args=(event_callback,), kwargs={"done": done, "start_pos": start_pos}, daemon=True).start()
        return p

    def cleanup_temp_merge_dir(self, cleanup: Path | None, rc: int | None, log_callback: Callable[[str], None]) -> None:
        if cleanup is None: return
        try:
            if rc == 0: shutil.rmtree(cleanup); log_callback(f"[merge] cleaned up temp dataset: {cleanup}\n")
            else: log_callback(f"[merge] keeping temp dataset (rc={rc}): {cleanup}\n")
        except Exception as e: log_callback(f"[merge] cleanup failed: {cleanup} ({e})\n")

    def stop_pipeline(self) -> None:
        self.stop()

    def _read_process_output_thread(self, proc: subprocess.Popen[str], log_callback: Callable[[str], None], *,
                                    done: threading.Event | None = None, on_exit: Callable[[int | None], None] | None = None) -> None:
        try: self._read_process_output(proc, log_callback)
        finally:
            rc = proc.poll()
            if done is not None: done.set()
            if on_exit is not None:
                try: on_exit(rc)
                except Exception as e: self.log_q.put(f"[process exit hook] failed: {e}\n")

    def _read_process_output(self, proc: subprocess.Popen[str], log_callback: Callable[[str], None]) -> None:
        fp = proc.stdout
        assert fp is not None

        # Read in chunks and flush on both LF and CR so tqdm/progress output
        # becomes visible live instead of showing only at the end.
        buf: list[str] = []
        last_transient_emit_ts = 0.0

        def _is_transient_progress_line(s: str) -> bool:
            t = s.strip()
            if not t:
                return False
            return t.startswith("Training:") or t.startswith("Validation:") or t.startswith("Evaluating:")

        def _emit_buffer(delim: str) -> None:
            nonlocal last_transient_emit_ts
            if not buf:
                if delim == "\n":
                    log_callback("\n")
                return

            line = "".join(buf)
            buf.clear()

            # Carriage-return progress updates can arrive very frequently.
            # Throttle transient progress lines to keep Tk responsive.
            if delim == "\r" and _is_transient_progress_line(line):
                now = time.time()
                if now - last_transient_emit_ts < 0.35:
                    return
                last_transient_emit_ts = now
            log_callback(line + "\n")

        while True:
            try:
                chunk = fp.read(256)
            except Exception:
                break
            if chunk == "":
                break

            for ch in chunk:
                if ch in ("\n", "\r"):
                    _emit_buffer(ch)
                else:
                    buf.append(ch)

            if self.stop_evt.is_set():
                break

        if buf:
            log_callback("".join(buf) + "\n")

        rc = proc.poll()
        if rc is None:
            with quiet():
                rc = proc.wait(timeout=0.2 if self.stop_evt.is_set() else None)
        self.log_q.put(f"\n[process exited rc={rc}]\n")

    def _tail_events_thread(self, event_callback: Callable[[dict[str, Any]], None], *, done: threading.Event | None = None, start_pos: int = 0) -> None:
        """Tail events.jsonl from start_pos until done is set (then drain once more); resets on truncation."""
        path = self.event_log_path
        done = done or self.stop_evt
        pos, partial = max(0, int(start_pos)), ""
        while True:
            finished = done.is_set()
            try:
                if path.exists():
                    if path.stat().st_size < pos: pos, partial = 0, ""  # truncated (run_pipeline.sh resets the log)
                    with open(path, encoding="utf-8") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    lines = (partial + chunk).split("\n")
                    partial = lines.pop()  # keep an incomplete trailing line for the next read
                    for line in lines:
                        if not (line := line.strip()): continue
                        with quiet(): event_callback(json.loads(line))
            except Exception: pass
            if finished: return
            done.wait(0.1)

    def _write_model_meta(self, model_path: Path, *, run_i: int, dataset_dir: Path | None, dataset_mode: str, out_dir: Path) -> None:
        meta = {
            "timestamp": now_iso(timespec="seconds"), "run_iteration": int(run_i),
            "dataset_mode": dataset_mode, "dataset_dir": str(dataset_dir) if dataset_dir else None,
            "out_dir": str(out_dir), "model_path": str(model_path),
        }
        atomic_write_json(Path(str(model_path) + ".meta.json"), meta)

    def _snapshot_model_checkpoint(self, model_path: Path, *, run_i: int, snap_every: int, snap_keep: int) -> Path | None:
        """Snapshot *model_path* after run *run_i* when it is a multiple of *snap_every* (and the file changed).

        ``None`` when no snapshot is due; I/O errors propagate (the caller logs them).
        """
        if snap_every < 1 or run_i % snap_every != 0: return None
        model_path = Path(model_path)
        if not model_path.exists(): return None
        st = model_path.stat()
        sig = (int(st.st_mtime_ns), int(st.st_size))
        if self._last_snap_sig == sig: return None
        dst = snapshot_checkpoint(self._model_root(), model_path, run=int(run_i))
        self._last_snap_sig = sig  # only after the copy exists: a failed snapshot is retried next run
        self._write_snapshot_meta(dst, src=model_path, run_i=run_i)
        self._prune_model_snapshots(dst.parent, snap_keep)
        return dst

    def _write_snapshot_meta(self, snapshot_path: Path, *, src: Path, run_i: int) -> None:
        meta = {
            "timestamp": now_iso(timespec="seconds"), "run_iteration": int(run_i),
            "snapshot_path": str(snapshot_path), "source_model_path": str(src),
        }
        atomic_write_json(Path(str(snapshot_path) + ".meta.json"), meta)

    def _prune_model_snapshots(self, dst_root: Path, snap_keep: int) -> None:
        """Keep the newest *snap_keep* snapshots in *dst_root* (0: all); a removed one takes its ``.meta.json``."""
        if snap_keep == 0: return
        try:
            snaps = sorted([p for p in dst_root.glob("*.pt") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
            for p in snaps[snap_keep:]:
                p.unlink()
                Path(str(p) + ".meta.json").unlink(missing_ok=True)
        except Exception: pass

    def get_dataset_profile_id(self, ds: Path) -> str | None:
        return manifest_profile(ds)[0]

    def get_dataset_profile_hash(self, ds: Path) -> str | None:
        return manifest_profile(ds)[1]

    def cached_checkpoint_meta(self, p: Path) -> dict[str, Any] | None:
        """Cached metadata for a checkpoint file, or None if not cached (or the file changed)."""
        return self._ckpt_meta.cached(p)

    def read_checkpoint_meta(self, p: Path) -> dict[str, Any]:
        """{component_profile, class_names[, error]} of a .pt checkpoint; cached by (path, mtime, size). Slow when uncached."""
        return self._ckpt_meta.read(p)

    def get_all_model_paths(self) -> list[Path]:
        return list_model_paths(self._model_root())

    def get_profile_classifier_models(self) -> list[Path]:
        root = self._model_root()
        cand: list[Path] = []
        if root.exists():
            cand.extend(sorted(root.glob("profile_classifier_*.pt")))
            cand.extend(sorted(root.glob("**/profile_classifier_*.pt")))
        return list({str(p.resolve()): p for p in cand if p.is_file()}.values())

    def build_profile_model(self, preset: str, config_path_str: str, out_ds_str: str, out_model_str: str, log_callback: Callable[[str], None], run_simple_cmd_callback: Callable[[list[str], dict[str, str] | None], None]) -> None:
        """Generate a profile dataset and train a profile classifier on it (paths from ``profile_model_paths``)."""
        cfg, out_ds, out_model = Path(config_path_str), Path(out_ds_str), Path(out_model_str)
        if not cfg.exists(): raise FileNotFoundError(f"Config not found:\n{cfg}")
        rel_cfg, rel_out_ds, rel_out_model = (rel_to(self.sim_root, p) for p in (cfg, out_ds, out_model))

        # "&&" must stay a shell operator, so quote each command separately.
        gen = argv_for("generate-profile-dataset", config=rel_cfg, out=rel_out_ds)
        train = argv_for("train-profile", data=rel_out_ds, out=rel_out_model)
        cmd_q = shlex.join(gen) + " && " + shlex.join(train)
        log_callback(f"\n[build profile model] preset={preset}\nconfig: {cfg}\ndata:   {out_ds}\nout:    {out_model}\n[cmd] {cmd_q}\n\n")
        run_simple_cmd_callback([cmd_q])

    def get_profile_paths(self, render_backend: str) -> list[Path]:
        profiles_dir = self.sim_root / "configs" / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        profile_files = sorted(profiles_dir.glob("*.yaml"))

        values: list[str] = []
        for p in profile_files:
            try:
                data = load_yaml(p)
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
        manifest_path = out_dir / DATASET_MANIFEST
        if manifest_path.exists(): return False
        # A legacy dataset without manifest: never truncate its rows.
        if (out_dir / META_JSONL).exists() or (out_dir / LABELS_JSONL).exists(): return False
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "images").mkdir(exist_ok=True)
        (out_dir / SPLITS_DIR).mkdir(exist_ok=True)
        for split in ["train", "val", "test"]: (out_dir / SPLITS_DIR / f"{split}.txt").write_text("", encoding="utf-8")
        Path(out_dir / META_JSONL).write_text("", encoding="utf-8")
        Path(out_dir / LABELS_JSONL).write_text("", encoding="utf-8")
        with quiet(): shutil.copy2(config_path, out_dir / CONFIG_YAML)

        cfg_data, run_block = (load_yaml(config_path) or {}), {}
        if isinstance(cfg_data, dict): run_block = cfg_data.get("run") or {}
        run_id = str(run_block.get("run_id") or out_dir.name)

        profile_path = self.sim_root / "configs" / "profiles" / f"{profile_id}.yaml"
        has_profile = profile_path.exists()
        # The regular v1 manifest writer (an empty dataset: no rows, empty splits).
        write_dataset_manifest(out_dir, run_id, profile_id=profile_id,
                               profile_hash=hash_profile(profile_path) if has_profile else "",
                               profile_path=str(profile_path) if has_profile else "", meta_rows=[], label_rows=[],
                               splits={"train": [], "val": [], "test": []}, script_name="gui.profile_placeholder")
        log_callback(f"[profile] created placeholder dataset manifest for {out_dir.name}\n")
        return True

    def chosen_config_for_profile(self, config_str: str, profile_id: str, want_backend: str | None = None) -> dict[str, Any] | None:
        """The config of the Config field (*config_str*) as ``{"path", "run_id"}`` when it is a config
        of *profile_id* (and of *want_backend*, if given and the config names one), else ``None``.

        Several configs share a profile (looks, cross-domain, phone, ...): the one the user picked
        wins over :meth:`config_for_profile`, which returns the alphabetically first.
        """
        if not config_str.strip():
            return None
        path = Path(config_str.strip())
        path = path if path.is_absolute() else self.sim_root / path
        try:
            data = load_yaml(path)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        run_block = data.get("run") or {}
        if str(run_block.get("component_profile") or "").strip() != str(profile_id).strip():
            return None
        backend = str((data.get("render") or {}).get("backend") or "").strip()
        if want_backend and backend and backend != want_backend:
            return None
        return {"path": path, "run_id": str(run_block.get("run_id") or path.stem)}

    def config_for_profile(self, profile_id: str, want_backend: str | None = None) -> dict[str, Any] | None:
        cache_key = f"{profile_id}|{want_backend or ''}"
        if cache_key in self._profile_config_cache: return self._profile_config_cache[cache_key]

        configs_root = self.sim_root / "configs"
        best_entry = None
        for cfg in sorted(configs_root.glob("*.yaml")):
            try:
                data = load_yaml(cfg)
            except Exception: continue
            if not isinstance(data, dict): continue

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

    def _datasets_require_bundle(self, dss: list[Path], get_dataset_profile_id_func: Callable[[Path], str | None]) -> bool:
        pids: set[str] = set()
        for ds in dss:
            if pid := get_dataset_profile_id_func(ds): pids.add(pid)
        return len(pids) > 1

    def _is_model_bundle_target(self, model_path: Path) -> bool:
        try: return (model_path.exists() and model_path.is_dir()) or str(model_path).endswith(".bundle")
        except Exception: return False

    def merge_mixed_outputs(self, src_dirs: list[Path], log_callback: Callable[[str], None], *, final_out: Path,
                            tmp_root: Path, extend: bool, replace: bool) -> bool:
        """Merge per-profile temp datasets into *final_out* (worker thread); temp data is kept on failure.

        With ``replace`` an existing *final_out* is only swapped out once the merge succeeded (with
        ``extend`` it is a merge source as well), so a failed merge leaves it untouched.
        """
        ok = True
        src_dirs = [Path(p).resolve() for p in src_dirs]
        try:
            sources = list(src_dirs)
            if replace and final_out.exists():
                if extend:
                    sources = [final_out, *sources]
                    log_callback(f"[mixed] extend mode: including existing dataset as merge source: {final_out}\n")
                tmp_target = final_out.parent / f"{final_out.name}.__merge_tmp_{int(time.time())}_{os.getpid()}"
                shutil.rmtree(tmp_target, ignore_errors=True)
                log_callback(f"[mixed] merging {len(sources)} source datasets into: {tmp_target}\n")
                self.merge_datasets_for_training(sources, tmp_target, log_callback, log_callback)
                replace_dir(tmp_target, final_out)
                log_callback(f"[mixed] replaced target dataset with merged result: {final_out}\n")
            else:
                log_callback(f"[mixed] merging {len(sources)} source datasets into: {final_out}\n")
                self.merge_datasets_for_training(sources, final_out, log_callback, log_callback)
            log_callback(f"[mixed] merged dataset ready: {final_out}\n")
        except Exception as e:
            log_callback(f"[mixed] merge failed: {e}\n")
            ok = False
        if not ok:
            # Keep freshly generated samples so a failed merge can be retried manually.
            log_callback(f"[mixed] temp generation data kept after failed merge: {tmp_root}\n")
            return False
        try:
            for p in src_dirs:
                shutil.rmtree(p, ignore_errors=True)
            shutil.rmtree(tmp_root, ignore_errors=True)
            log_callback(f"[mixed] cleaned temp generation data: {tmp_root}\n")
        except Exception as e:
            log_callback(f"[mixed] temp cleanup warning: {e}\n")
        return True

    def merge_datasets_for_training(self, sources: list[Path], out_dir: Path, log_callback: Callable[[str], None],
                                    log_error_callback: Callable[[str], None]) -> MergeResult:
        """Merge *sources* into the new dataset *out_dir* (:func:`simple_sim.dataset_merge.merge_datasets`).

        Equal class sets required, images resized to the smallest ROI and hard-linked when unchanged,
        tolerant (``strict=False``) about the rows. Raises ``MergeError`` (a ``ValueError``),
        ``FileNotFoundError`` or ``FileExistsError``; a failed merge leaves no *out_dir* behind.
        """
        result = merge_datasets(
            [Path(p) for p in sources], Path(out_dir), require_same_classes=True, resize_to_common_roi=True,
            link_images=True, include_unused_profiles=True, script_name="gui.merge_datasets",
            on_warning=lambda msg: log_error_callback(f"[merge] warning: {msg}\n"), strict=False)
        log_callback(f"[merge] {len(result.meta_rows)} samples from {len(sources)} dataset(s) -> {result.out_dir}"
                     + (f" (ROI {result.roi[0]}x{result.roi[1]})" if result.roi else "") + "\n")
        return result
