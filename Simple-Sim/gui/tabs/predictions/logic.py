"""Logic for the Predictions Tab."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any

from gui.utils.dataset_catalog import list_dataset_dirs
from gui.utils.paths import list_model_paths, model_root
from gui.utils.process import run_streaming, terminate_process
from simple_sim.cli import argv_for
from simple_sim.cli_utils import parse_result_line
from simple_sim.dataset_files import LABELS_JSONL, META_JSONL
from simple_sim.dataset_index import DatasetIndex
from simple_sim.eval_utils import dataset_profile
from simple_sim.model_bundle import bundle_checkpoint_path


class PredictionsLogic:
    def __init__(self, sim_root: Path, log_q: queue.Queue[str]) -> None:
        self.sim_root = sim_root
        self.log_q = log_q
        self.proc: subprocess.Popen[str] | None = None
        self.stop_evt = threading.Event()

    def get_datasets(self) -> list[Path]:
        return list_dataset_dirs(self.sim_root)

    def get_models(self) -> list[Path]:
        paths = list_model_paths(model_root(self.sim_root))
        return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)

    def get_profile_models(self) -> list[Path]:
        root = model_root(self.sim_root)
        cand = sorted(root.glob("**/profile_classifier_*.pt")) if root.exists() else []
        seen: dict[str, Path] = {str(p.resolve()): p for p in cand}
        return sorted(list(seen.values()), key=lambda p: p.stat().st_mtime, reverse=True)

    def stop_predictions(self) -> None:
        self.stop_evt.set()
        terminate_process(self.proc)

    def _set_proc(self, proc: subprocess.Popen[str]) -> None:
        self.proc = proc

    def resolve_dataset_path(self, s: str) -> Path:
        p = Path(s)
        if p.is_absolute():
            return p
        return (self.sim_root / p).resolve()

    def run_predictions(
        self,
        model_path: Path,
        dataset_dirs: list[Path],
        split: str,
        device: str,
        max_samples: str,
        save_preds: bool,
        auto_train_bundle: bool,
        profile_model_path: Path | None,
        on_complete: callable,
    ) -> None:
        self.stop_evt.clear()

        def worker() -> None:
            last_report_path: Path | None = None
            last_preds_path: Path | None = None
            last_dataset_dir: Path | None = None
            multi_results: list[dict[str, Any]] = []
            multi_result_by_key: dict[str, dict[str, Any]] = {}

            try:
                for idx, data_dir in enumerate(dataset_dirs, 1):
                    if self.stop_evt.is_set():
                        break

                    if not (data_dir / META_JSONL).exists() or not (data_dir / LABELS_JSONL).exists():
                        self.log_q.put(f"[error] dataset missing meta.jsonl/labels.jsonl: {data_dir}\n")
                        continue

                    model_arg_path = model_path
                    if model_path.exists() and model_path.is_dir():
                        # v1 and v2 manifests (v2 -> the bundle's "multi" checkpoint, as train writes it; plan X7).
                        try:
                            profile_id = str(dataset_profile(data_dir)[0] or "").strip()
                        except FileNotFoundError:
                            self.log_q.put(f"[error] dataset manifest missing: {data_dir}\n")
                            continue
                        except Exception as e:
                            self.log_q.put(f"[error] failed to read dataset manifest: {data_dir} ({e})\n")
                            continue
                        if not profile_id:
                            self.log_q.put(f"[error] dataset manifest has no profile id: {data_dir}\n")
                            continue
                        expected = bundle_checkpoint_path(model_path, profile_id, kind="best")
                        if not expected.exists():
                            if auto_train_bundle:
                                cmd_train = argv_for("train", data=data_dir, out=model_path,
                                                     device=None if device == "auto" else device)
                                self.log_q.put(f"[bundle] missing '{profile_id}.pt' -> training into bundle...\n")
                                self.log_q.put("[cmd] " + " ".join(cmd_train) + "\n")
                                try:
                                    # Track as self.proc (on_start) so Stop can terminate the training run.
                                    rc = run_streaming(
                                        cmd_train,
                                        cwd=self.sim_root,
                                        log_q=self.log_q,
                                        stop_evt=self.stop_evt,
                                        on_start=self._set_proc,
                                    )
                                    if self.stop_evt.is_set():
                                        self.log_q.put("\n[stopped]\n")
                                        break
                                    if rc != 0:
                                        self.log_q.put(f"\n[error] train exited with code {rc}\n")
                                        continue
                                except Exception as e:
                                    self.log_q.put(f"\n[error] failed to run train: {e}\n")
                                    continue
                                if not expected.exists():
                                    self.log_q.put(f"[error] bundle still missing checkpoint for profile '{profile_id}'\n")
                                    continue
                            else:
                                self.log_q.put(f"[error] multi-model bundle missing checkpoint for profile '{profile_id}'\n")
                                continue
                        model_arg_path = expected

                    out_dir = data_dir / "predictions"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    cmd = argv_for(
                        "batch-predict", model=model_arg_path, data=data_dir, split=split, out_dir=out_dir,
                        device=None if device == "auto" else device, max_samples=max_samples or None,
                        save_preds=bool(save_preds), profile_model=profile_model_path, json=True,
                    )

                    self.log_q.put(f"\n[dataset {idx}/{len(dataset_dirs)}] {data_dir}\n")
                    self.log_q.put("[cmd] " + " ".join(cmd) + "\n")

                    report_path: Path | None = None
                    preds_path: Path | None = None

                    def on_line(line: str) -> None:
                        # The --json result line carries the paths (plan D22: no scraping of human output).
                        nonlocal report_path, preds_path
                        if (res := parse_result_line(line)) is not None:
                            report_path = Path(res["report_path"]) if res.get("report_path") else None
                            preds_path = Path(res["preds_path"]) if res.get("preds_path") else None

                    rc = run_streaming(
                        cmd,
                        cwd=self.sim_root,
                        log_q=self.log_q,
                        stop_evt=self.stop_evt,
                        on_line=on_line,
                        on_start=self._set_proc,
                    )
                    if self.stop_evt.is_set():
                        self.log_q.put("\n[stopped]\n")
                        break
                    if rc != 0:
                        self.log_q.put(f"\n[error] batch-predict exited with code {rc}\n")
                        continue

                    try:
                        key = str(data_dir.resolve())
                        label = data_dir.name
                        result = {"key": key, "label": label, "dataset_dir": data_dir, "report_path": report_path, "preds_path": preds_path}
                        multi_results.append(result)
                        multi_result_by_key[key] = result
                    except Exception:
                        pass
                    
                    last_report_path = report_path
                    last_preds_path = preds_path
                    last_dataset_dir = data_dir
            except Exception as e:
                self.log_q.put(f"\n[error] Failed to run predictions: {e}\n")
            finally:
                on_complete(last_report_path, last_preds_path, last_dataset_dir, multi_results, multi_result_by_key, self.stop_evt.is_set())

        threading.Thread(target=worker, daemon=True).start()

    def load_report(self, report_path: Path) -> tuple[dict[str, Any] | None, str | None]:
        try:
            obj = json.loads(report_path.read_text(encoding="utf-8"))
            return obj, None
        except Exception as e:
            return None, f"Failed to load report:\n{e}"

    def load_preds(self, preds_path: Path) -> tuple[list[dict[str, Any]] | None, str | None]:
        rows: list[dict[str, Any]] = []
        try:
            with open(preds_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows, None
        except Exception as e:
            return None, f"Failed to load predictions:\n{e}"

    def enrich_rows_with_gt_profile(self, rows: list[dict[str, Any]], active_dataset_dir: Path | None) -> None:
        gt_profile_map = self._load_gt_profile_map_from_labels(active_dataset_dir)
        if not gt_profile_map:
            return

        for r in rows:
            sid = str(r.get("id") or "")
            if not sid:
                continue

            if not r.get("gt_profile"):
                pid = gt_profile_map.get(sid, "")
                if pid:
                    r["gt_profile"] = pid

            if r.get("pred_profile") and r.get("gt_profile") and r.get("profile_correct") is None:
                r["profile_correct"] = bool(str(r.get("pred_profile")) == str(r.get("gt_profile")))

    def _load_gt_profile_map_from_labels(self, data_dir: Path | None) -> dict[str, str]:
        """Sample id -> GT ``profile_id`` from ``labels.jsonl`` (shared :class:`DatasetIndex`, plan L11)."""
        if not data_dir or not (data_dir / LABELS_JSONL).exists():
            return {}
        try:
            return DatasetIndex(data_dir).profile_map()
        except (OSError, ValueError, TypeError, KeyError):
            return {}

    def resolve_model_path(self, s: str) -> Path:
        p = Path(s)
        if p.is_absolute():
            return p
        return (self.sim_root / p).resolve()
