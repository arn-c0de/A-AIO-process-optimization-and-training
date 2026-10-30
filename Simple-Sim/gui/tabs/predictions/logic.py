"""Logic for the Predictions Tab."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from gui.utils.dataset_catalog import list_dataset_dirs
from gui.utils.process import run_streaming, terminate_process
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
        root = self.sim_root / "outputs" / "models"
        cand: list[Path] = []
        for search_path in [root, root / "versions", root / "imports"]:
            if not search_path.exists(): continue
            cand.extend(sorted(search_path.glob("**/*.pt" if search_path.name == "versions" else "*.pt")))
            cand.extend([p for p in sorted(search_path.glob("*.bundle")) if p.is_dir()])
            if (bundles_subdir := search_path / "bundles").exists():
                cand.extend([p for p in sorted(bundles_subdir.iterdir()) if p.is_dir()])
        
        uniq: dict[str, Path] = {str(p.resolve()): p for p in cand if p.exists()}
        return sorted(list(uniq.values()), key=lambda p: p.stat().st_mtime, reverse=True)

    def get_profile_models(self) -> list[Path]:
        root = self.sim_root / "outputs" / "models"
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
            started_s = time.time()

            try:
                for idx, data_dir in enumerate(dataset_dirs, 1):
                    if self.stop_evt.is_set():
                        break

                    if not (data_dir / "meta.jsonl").exists() or not (data_dir / "labels.jsonl").exists():
                        self.log_q.put(f"[error] dataset missing meta.jsonl/labels.jsonl: {data_dir}\n")
                        continue

                    model_arg_path = model_path
                    if model_path.exists() and model_path.is_dir():
                        manifest_path = data_dir / "dataset_manifest.json"
                        if not manifest_path.exists():
                            self.log_q.put(f"[error] dataset manifest missing: {manifest_path}\n")
                            continue
                        try:
                            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                            profile_id = str((manifest.get("component_profile") or {}).get("profile_id") or "").strip()
                        except Exception as e:
                            self.log_q.put(f"[error] failed to read dataset manifest: {manifest_path} ({e})\n")
                            continue
                        if not profile_id:
                            self.log_q.put(f"[error] dataset manifest missing component_profile.profile_id: {manifest_path}\n")
                            continue
                        expected = bundle_checkpoint_path(model_path, profile_id, kind="best")
                        if not expected.exists():
                            if auto_train_bundle:
                                py = self.sim_root / ".venv" / "bin" / "python"
                                cmd_train: list[str] = [str(py), "scripts/train.py", "--data", str(data_dir), "--out", str(model_path)]
                                if device != "auto":
                                    cmd_train.extend(["--device", device])
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
                                        self.log_q.put(f"\n[error] train.py exited with code {rc}\n")
                                        continue
                                except Exception as e:
                                    self.log_q.put(f"\n[error] failed to run train.py: {e}\n")
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

                    cmd: list[str] = ["bash", str(self.sim_root / "predict.sh"), "--model", str(model_arg_path), "--data", str(data_dir), "--split", split, "--out-dir", str(out_dir)]
                    if device != "auto":
                        cmd.extend(["--device", device])
                    if max_samples:
                        cmd.extend(["--max-samples", max_samples])
                    if save_preds:
                        cmd.append("--save-preds")
                    if profile_model_path is not None:
                        cmd.extend(["--profile-model", str(profile_model_path)])

                    self.log_q.put(f"\n[dataset {idx}/{len(dataset_dirs)}] {data_dir}\n")
                    self.log_q.put("[cmd] " + " ".join(cmd) + "\n")

                    report_path: Path | None = None
                    preds_path: Path | None = None

                    def on_line(line: str) -> None:
                        nonlocal report_path, preds_path
                        if "Report saved to:" in line:
                            try:
                                report_path = Path(line.split("Report saved to:", 1)[-1].strip())
                            except Exception:
                                report_path = None
                        if "Predictions saved to:" in line:
                            try:
                                preds_path = Path(line.split("Predictions saved to:", 1)[-1].strip())
                            except Exception:
                                preds_path = None

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
                        self.log_q.put(f"\n[error] predict.sh exited with code {rc}\n")
                        continue
                    
                    try:
                        if report_path is None:
                            cand = [p for p in out_dir.glob("batch_report_*.json") if p.stat().st_mtime >= started_s - 0.5]
                            report_path = sorted(cand, key=lambda p: p.stat().st_mtime)[-1] if cand else None
                        if save_preds and preds_path is None:
                            cand = [p for p in out_dir.glob("batch_preds_*.jsonl") if p.stat().st_mtime >= started_s - 0.5]
                            preds_path = sorted(cand, key=lambda p: p.stat().st_mtime)[-1] if cand else None
                    except Exception:
                        pass
                    
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
        if not data_dir:
            return {}
        labels_path = data_dir / "labels.jsonl"
        if not labels_path.exists():
            return {}

        gt_profile_map: dict[str, str] = {}
        with open(labels_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sid = str(obj.get("id") or "")
                    pid = str(obj.get("profile_id") or "")
                    if sid and pid:
                        gt_profile_map[sid] = pid
                except Exception:
                    continue
        return gt_profile_map

    def resolve_model_path(self, s: str) -> Path:
        p = Path(s)
        if p.is_absolute():
            return p
        return (self.sim_root / p).resolve()
