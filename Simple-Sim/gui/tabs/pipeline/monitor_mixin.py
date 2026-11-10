"""Live monitoring of a run: events, thumbnails, status timer and dataset statistics.

Mixin of :class:`gui.tabs.pipeline.tab.PipelineControlTab`; it relies on the attributes that class sets up
(``ui``, ``logic``, ``state``, ``sim_root`` ...).
"""

from __future__ import annotations
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any


from gui.components.overlay_renderer import draw_defect_overlay
from gui.utils.pipeline_helpers import (
    parse_snap_every,
    parse_snap_keep,
    sample_color_from_total,
    samples_text,
)

from simple_sim.dataset_files import DATASET_MANIFEST
from simple_sim.io_utils import append_jsonl


_LOG = logging.getLogger(__name__)

#: Period of the remaining status timer (process liveness, run buttons, dataset size); log lines and
#: events are pushed through the dispatcher and do not wait for it.
_STATUS_TICK_MS = 250


class PipelineMonitorMixin:
    def _handle_event(self, evt: dict[str, Any]) -> None:
        # Event tailing runs in a worker thread; marshal to the Tk thread.
        if threading.current_thread() is not threading.main_thread():
            self.event_q.put(evt)
            return
        et = evt.get("event")
        # The dataset of the running process is kept in _monitor_dir (thumbnails, size display).
        # It used to be written to the shared state.dataset_dir without a <<DatasetChanged>>: the
        # other tabs kept showing dataset A while their actions (analyze, delete images, predict,
        # validate) went to the dataset being generated.
        if et == "gen_start":
            self.ui.var_phase.set("phase: generating")
            if out_dir := evt.get("output_dir"): self._monitor_dir = Path(out_dir)
        elif et == "gen_progress":
            self.ui.var_phase.set("phase: generating")
            rate = evt.get("samp_per_s")
            self.ui.var_ips.set(f"img/s: {rate:.1f}" if isinstance(rate, (int, float)) else "img/s: -")
            # A new dataset is rendered into <out>.tmp and moved in at the end: image_root says where.
            if root := evt.get("image_root"): self._monitor_image_root = Path(root)
            if last_img := evt.get("last_image_path"): self._push_image(last_img, root=self._monitor_image_root)
        elif et == "gen_done":
            self.ui.var_phase.set("phase: generated")
            if out_dir := evt.get("output_dir"):
                self._monitor_dir = Path(out_dir)
                if (tmp_root := self._monitor_image_root) is not None and tmp_root != self._monitor_dir:
                    self._recent_imgs = [n.replace(str(tmp_root), str(self._monitor_dir), 1) for n in self._recent_imgs]
            self._monitor_image_root = None
            if last_img := evt.get("last_image_path"): self._push_image(last_img)
        elif et == "train_start":
            self.ui.var_phase.set("phase: training")
            if ds := evt.get("dataset_dir"): self._monitor_dir = Path(ds)
        elif et == "epoch_start":
            self.ui.var_phase.set("phase: training")
            self.ui.var_epoch.set(f"epoch: {evt.get('epoch', '-')}/{evt.get('epochs', '-')}")
        elif et == "train_batch":
            self.ui.var_phase.set("phase: training")
            if ips := evt.get("img_per_s"): self.ui.var_ips.set(f"img/s: {ips:.1f}")
            if lid := evt.get("last_id"): self.ui.var_last.set(f"last: {lid}")
            if limg := evt.get("last_image_path"): self._push_image(limg)
        elif et == "eval_start":
            self.ui.var_phase.set("phase: evaluating")
            if ds := evt.get("dataset_dir"): self._monitor_dir = Path(ds)
        elif et == "eval_batch":
            self.ui.var_phase.set("phase: evaluating")
            if ips := evt.get("img_per_s"): self.ui.var_ips.set(f"img/s: {ips:.1f}")
            if limg := evt.get("last_image_path"): self._push_image(limg)
        elif et == "eval_done":
            self.ui.var_phase.set("phase: done")
        elif et == "validate_start":
            self.ui.var_phase.set("phase: validating")
        elif et == "validate_done":
            ok = evt.get("ok")
            self.ui.var_phase.set("phase: validated_ok" if ok else "phase: validated_fail")

    _monitor_dir: Path | None = None         # dataset of the running process (telemetry events)
    _monitor_image_root: Path | None = None  # where its new images are written right now

    def _run_dataset_dir(self) -> Path | None:
        """Dataset the monitor shows: the running process's, else the selected one."""
        return self._monitor_dir or self.state.dataset_dir

    def _push_image(self, rel_path: str, root: Path | None = None) -> None:
        base = root or self._run_dataset_dir()
        if not base: return
        img_path = base / rel_path
        if not img_path.exists(): return
        name = str(img_path)
        if self._recent_imgs and self._recent_imgs[-1] == name: return
        self._recent_imgs.append(name)
        self._recent_imgs = self._recent_imgs[-12:]
        # Rendering 12 thumbnails can be expensive: throttled (min interval) and decoded off the Tk thread.
        self._thumb_refresh_pending = True
        self._schedule_thumb_refresh()

    def _schedule_thumb_refresh(self) -> None:
        """One timer per pending refresh, at most every ``_thumb_refresh_min_interval_s``."""
        if self._thumb_after_id is not None:
            return
        wait_s = self._thumb_refresh_min_interval_s - (time.time() - self._last_thumb_refresh_ts)
        try: self._thumb_after_id = self.frame.after(max(0, int(wait_s * 1000)), self._on_thumb_timer)
        except Exception: self._thumb_after_id = None

    def _on_thumb_timer(self) -> None:
        self._thumb_after_id = None
        self._maybe_refresh_thumbnails()

    def _maybe_refresh_thumbnails(self) -> None:
        if not self._thumb_refresh_pending:
            return
        now = time.time()
        if now - self._last_thumb_refresh_ts < self._thumb_refresh_min_interval_s:
            self._schedule_thumb_refresh()
            return
        self._last_thumb_refresh_ts = now
        self._thumb_refresh_pending = False
        self._refresh_thumbnails()

    def _refresh_thumbnails(self) -> None:
        """Decode + overlay the recent images in a worker; only the PhotoImages are made on the Tk thread."""
        if self._thumb_busy:
            self._thumb_rerun = True  # coalesce: one more pass with the newest list when this one ends
            return
        self._thumb_busy, self._thumb_rerun = True, False
        self._thumb_gen += 1
        gen = self._thumb_gen
        args = (list(self._recent_imgs), self._run_dataset_dir(), self._label_dict,
                self._image_to_sample, self._meta_by_sample, self._meta_by_image_rel, draw_defect_overlay)
        prepare = self.ui.prepare_thumbnails

        def on_done(prepared: Any) -> None:
            if gen == self._thumb_gen:
                self.ui.show_thumbnails(prepared)

        def on_finally() -> None:
            self._thumb_busy = False
            if self._thumb_rerun:
                self._refresh_thumbnails()

        self.quiet_tasks.run(lambda _t: prepare(*args), name="Thumbnails", on_done=on_done, on_finally=on_finally)

    def _on_log_batch(self, lines: list[str]) -> None:
        # Up to 200 lines per drain, appended as one block (keeps the UI responsive under heavy output).
        self._append_log("".join(lines))

    def _on_event_batch(self, events: list[dict[str, Any]]) -> None:
        for evt in events: self._handle_event(evt)

    def _tick_status(self) -> None:
        """Cheap timer for state that is polled, not pushed (process alive, run buttons, dataset size)."""
        try:
            if not self.logic.is_busy():
                if self.ui.btn_start["state"] == "disabled" or self.ui.btn_start_generate["state"] == "disabled": self.ui.set_run_buttons_state(False)
            self._maybe_update_stats()
        finally:
            self._status_after_id = self.frame.after(_STATUS_TICK_MS, self._tick_status)

    def _maybe_update_stats(self) -> None:
        now = time.time()
        if now - self._last_stats_ts < 1.0: return
        self._last_stats_ts = now

        proc_running = self.logic.is_process_running()
        if self._prev_proc_running and not proc_running:
            if ds := self._run_dataset_dir(): self._start_dataset_size_calc(ds)
            self._last_ds_size_ts = now
            self._monitor_dir = self._monitor_image_root = None  # run over: back to the selected dataset
        self._prev_proc_running = proc_running

        if proc_running and (now - self._last_ds_size_ts) >= 5.0:
            if ds := self._run_dataset_dir(): self._start_dataset_size_calc(ds)
            self._last_ds_size_ts = now

    def _snap_every_n(self) -> int:
        return parse_snap_every(str(self.ui.var_snap_every.get()))

    def _snap_keep_n(self) -> int:
        return parse_snap_keep(str(self.ui.var_snap_keep.get()))

    def _start_dataset_size_calc(self, ds: Path) -> None:
        self._dataset_size_job_id += 1
        job_id = self._dataset_size_job_id
        self.ui.var_ds_size.set("DS: calculating...")
        
        def apply(size_b: int) -> None:
            if job_id == self._dataset_size_job_id:
                self.ui.var_ds_size.set(f"DS: {self.ui.fmt_bytes(size_b)} ({ds.name})")

        def update_ui(size_b: int) -> None:  # worker thread
            self.post(lambda: apply(size_b))

        self.logic.start_dataset_size_calc(ds, update_ui)

    def _refresh_dataset_stats(self) -> None:
        ds = self.state.dataset_dir
        if not ds:
            self.ui.set_dataset_samples_info("Samples: -", "black")
            self.ui.var_ds_size.set("DS: -")
            return

        manifest_path = ds / DATASET_MANIFEST
        if manifest_path.exists():
            try:
                with open(manifest_path, encoding="utf-8") as f: manifest = json.load(f)
                stats = manifest.get("dataset_stats", {})
                splits = stats.get("splits", {})
                self._set_dataset_samples_from_manifest(ds, stats.get("total_samples"), splits.get("train"), splits.get("val"), splits.get("test"))
            except Exception as exc:
                _LOG.warning("Failed to refresh dataset stats: %s", exc)
                self._set_dataset_samples_info("Samples: manifest error", "")
        else: self._set_dataset_samples_info("Samples: manifest missing", "")

        self._start_dataset_size_calc(ds)

    def _set_dataset_samples_from_manifest(self, ds: Path, total: int | None, train: int | None, val: int | None, test: int | None) -> None:
        if total is None: return self._set_dataset_samples_info("Samples: manifest missing", "")
        self._set_dataset_samples_info(samples_text(total, train, val, test), sample_color_from_total(total))
        self._check_milestone(ds, total)

    def _set_dataset_samples_info(self, text: str, color: str) -> None:
        self.ui.set_dataset_samples_info(text, color)

    def _check_milestone(self, ds: Path, total: int) -> None:
        name = ds.name
        current = self._dataset_milestones.get(name, 0)
        for threshold in self._milestone_thresholds:
            if total >= threshold and threshold > current:
                self._dataset_milestones[name] = threshold
                self._log_dataset_milestone(ds, threshold, total)
                break

    def _log_dataset_milestone(self, ds: Path, threshold: int, total: int) -> None:
        event = {
            "ts": time.time(), "event": "dataset_milestone", "dataset": ds.name,
            "threshold": threshold, "samples": total,
        }
        try:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.event_log_path, [event])
        except Exception as e: _LOG.warning("Failed to log milestone: %s", e)
