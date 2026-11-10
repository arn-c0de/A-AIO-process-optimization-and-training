"""Logic for the Predictions Tab."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gui.utils.dataset_catalog import list_dataset_dirs
from gui.utils.paths import list_model_paths, model_root
from gui.utils.process import StreamingProcessLogic
from simple_sim.cli import argv_for
from simple_sim.cli_utils import parse_result_line
from simple_sim.dataset_files import LABELS_JSONL, META_JSONL
from simple_sim.dataset_index import DatasetIndex
from simple_sim.eval_utils import dataset_profile
from simple_sim.io_utils import iter_json_lines
from simple_sim.model_bundle import bundle_checkpoint_path


class _Stopped(Exception):
    """Stop was requested while a child process of the prediction run was active."""


class PredictionsLogic(StreamingProcessLogic):
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
        on_complete: Callable[..., None],
    ) -> None:
        self.stop_evt.clear()

        def worker() -> None:
            last: dict[str, Any] = {}
            multi_results: list[dict[str, Any]] = []
            multi_result_by_key: dict[str, dict[str, Any]] = {}
            try:
                for idx, data_dir in enumerate(dataset_dirs, 1):
                    if self.stop_evt.is_set():
                        break
                    try:
                        result = self._predict_dataset(
                            model_path, data_dir, progress=f"[dataset {idx}/{len(dataset_dirs)}]", split=split, device=device, max_samples=max_samples,
                            save_preds=save_preds, auto_train_bundle=auto_train_bundle,
                            profile_model_path=profile_model_path,
                        )
                    except _Stopped:
                        self.log_q.put("\n[stopped]\n")
                        break
                    if result is None:
                        continue
                    if result["key"] is not None:
                        multi_results.append(result)
                        multi_result_by_key[result["key"]] = result
                    last = result
            except Exception as e:
                self.log_q.put(f"\n[error] Failed to run predictions: {e}\n")
            finally:
                on_complete(last.get("report_path"), last.get("preds_path"), last.get("dataset_dir"),
                            multi_results, multi_result_by_key, self.stop_evt.is_set())

        threading.Thread(target=worker, daemon=True).start()

    def _bundle_checkpoint(self, bundle_dir: Path, data_dir: Path, *, device: str, auto_train: bool) -> Path | None:
        """Checkpoint of ``bundle_dir`` for the profile of ``data_dir``, trained on demand with ``auto_train``.

        ``None`` (reason logged) when the dataset has to be skipped; raises :class:`_Stopped` on Stop.
        """
        # v1 and v2 manifests (v2 -> the bundle's "multi" checkpoint, as train writes it; plan X7).
        try:
            profile_id = str(dataset_profile(data_dir)[0] or "").strip()
        except FileNotFoundError:
            self.log_q.put(f"[error] dataset manifest missing: {data_dir}\n")
            return None
        except Exception as e:
            self.log_q.put(f"[error] failed to read dataset manifest: {data_dir} ({e})\n")
            return None
        if not profile_id:
            self.log_q.put(f"[error] dataset manifest has no profile id: {data_dir}\n")
            return None
        expected = bundle_checkpoint_path(bundle_dir, profile_id, kind="best")
        if expected.exists():
            return expected
        if not auto_train:
            self.log_q.put(f"[error] multi-model bundle missing checkpoint for profile '{profile_id}'\n")
            return None

        cmd_train = argv_for("train", data=data_dir, out=bundle_dir, device=None if device == "auto" else device)
        self.log_q.put(f"[bundle] missing '{profile_id}.pt' -> training into bundle...\n")
        self.log_q.put("[cmd] " + " ".join(cmd_train) + "\n")
        try:
            rc = self._stream(cmd_train)
        except Exception as e:
            self.log_q.put(f"\n[error] failed to run train: {e}\n")
            return None
        if self.stop_evt.is_set():
            raise _Stopped
        if rc != 0:
            self.log_q.put(f"\n[error] train exited with code {rc}\n")
            return None
        if not expected.exists():
            self.log_q.put(f"[error] bundle still missing checkpoint for profile '{profile_id}'\n")
            return None
        return expected

    def _predict_dataset(self, model_path: Path, data_dir: Path, *, progress: str, split: str, device: str, max_samples: str,
                         save_preds: bool, auto_train_bundle: bool, profile_model_path: Path | None) -> dict[str, Any] | None:
        """Run ``batch-predict`` on one dataset; the result entry, or ``None`` (logged) when it was skipped.

        Raises :class:`_Stopped` when Stop was requested while a child process ran.
        """
        if not (data_dir / META_JSONL).exists() or not (data_dir / LABELS_JSONL).exists():
            self.log_q.put(f"[error] dataset missing meta.jsonl/labels.jsonl: {data_dir}\n")
            return None

        model_arg_path: Path | None = model_path
        if model_path.exists() and model_path.is_dir():
            model_arg_path = self._bundle_checkpoint(model_path, data_dir, device=device, auto_train=auto_train_bundle)
            if model_arg_path is None:
                return None

        out_dir = data_dir / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = argv_for(
            "batch-predict", model=model_arg_path, data=data_dir, split=split, out_dir=out_dir,
            device=None if device == "auto" else device, max_samples=max_samples or None,
            save_preds=bool(save_preds), profile_model=profile_model_path, json=True,
        )
        self.log_q.put(f"\n{progress} {data_dir}\n")
        self.log_q.put("[cmd] " + " ".join(cmd) + "\n")

        paths: dict[str, Path | None] = {"report_path": None, "preds_path": None}

        def on_line(line: str) -> None:
            # The --json result line carries the paths (plan D22: no scraping of human output).
            if (res := parse_result_line(line)) is not None:
                for name in paths:
                    paths[name] = Path(res[name]) if res.get(name) else None

        rc = self._stream(cmd, on_line=on_line)
        if self.stop_evt.is_set():
            raise _Stopped
        if rc != 0:
            self.log_q.put(f"\n[error] batch-predict exited with code {rc}\n")
            return None

        try:
            key: str | None = str(data_dir.resolve())
        except OSError:
            key = None
        return {"key": key, "label": data_dir.name, "dataset_dir": data_dir, **paths}

    def load_report(self, report_path: Path) -> tuple[dict[str, Any] | None, str | None]:
        try:
            obj = json.loads(report_path.read_text(encoding="utf-8"))
            return obj, None
        except Exception as e:
            return None, f"Failed to load report:\n{e}"

    def load_preds(self, preds_path: Path) -> tuple[list[dict[str, Any]] | None, str | None]:
        try:
            return list(iter_json_lines(preds_path)), None
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
