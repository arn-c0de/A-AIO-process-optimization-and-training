"""Logic for the Weights Tab."""

from __future__ import annotations
import json
import os
import queue
import shutil
import subprocess
import threading
import re
import zipfile
from pathlib import Path
from typing import Any

from gui.utils.checkpoint_meta import checkpoint_class_names
from gui.utils.dataset_catalog import dataset_display_name, sorted_dataset_dirs
from gui.utils.paths import list_model_paths, model_root
from gui.utils.process import run_streaming, terminate_process
from simple_sim.cli import argv_for
from simple_sim.cli_utils import parse_result_line
from simple_sim.eval_utils import dataset_profile
from simple_sim.io_utils import atomic_write_json, load_yaml, now_iso, now_tag, read_json, rel_to
from simple_sim.dataset_files import DATASET_MANIFEST, LABELS_JSONL
from simple_sim.model_bundle import resolve_bundle_checkpoint
from simple_sim.model_store import SNAPSHOT_RUN_RE, snapshot_checkpoint, snapshot_path
from simple_sim.reports import load_reports, report_search_dirs
import yaml


class WeightsLogic:
    def __init__(self, sim_root: Path, log_q: queue.Queue[str]) -> None:
        self.sim_root = sim_root
        self.log_q = log_q
        self.proc: subprocess.Popen[str] | None = None
        self.stop_evt = threading.Event()

    def _model_root(self) -> Path:
        return model_root(self.sim_root)

    def get_models(self) -> list[Path]:
        return list_model_paths(self._model_root())

    def rel_path(self, p: Path) -> str:
        return rel_to(self.sim_root, p)

    def _favorites_path(self) -> Path:
        return self._model_root() / "favorites.json"

    def _arena_path(self) -> Path:
        return self._model_root() / "arena.json"

    @staticmethod
    def _load_path_items(p: Path, list_key: str) -> dict[str, dict[str, Any]]:
        """``{path: item}`` from a ``{list_key: [{"path": ...}, ...]}`` JSON file (``{}`` if missing/invalid)."""
        obj = read_json(p)
        items = obj.get(list_key) if isinstance(obj, dict) else None
        out: dict[str, dict[str, Any]] = {}
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and (rp := str(it.get("path") or "").strip()):
                out[rp] = it
        return out

    @staticmethod
    def _save_path_items(p: Path, list_key: str, items: dict[str, dict[str, Any]]) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(p, {list_key: sorted(items.values(), key=lambda it: str(it.get("path") or ""))})

    def load_favorites(self) -> dict[str, dict[str, Any]]:
        return self._load_path_items(self._favorites_path(), "favorites")

    def save_favorites(self, favorites: dict[str, dict[str, Any]]) -> None:
        self._save_path_items(self._favorites_path(), "favorites", favorites)

    def load_arena(self) -> dict[str, dict[str, Any]]:
        return self._load_path_items(self._arena_path(), "arena_models")

    def save_arena(self, arena: dict[str, dict[str, Any]]) -> None:
        self._save_path_items(self._arena_path(), "arena_models", arena)

    def add_to_arena(self, paths: list[Path], arena: dict[str, dict[str, Any]]) -> None:
        for p in paths:
            rel = self.rel_path(p)
            if rel in arena:
                continue
            arena[rel] = {
                "path": rel,
                "added_at": now_iso(timespec="seconds"),
                "note": "",
            }

    def remove_from_arena(self, paths: list[Path], arena: dict[str, dict[str, Any]]) -> None:
        for p in paths:
            rel = self.rel_path(p)
            if rel in arena:
                arena.pop(rel, None)

    def generate_arena_report(self) -> None:
        # The GUI's own interpreter (its venv), not the system python3 (plan X11/P4).
        cmd = argv_for("arena-report", sim_root=self.sim_root)
        self.log_q.put(f"\n$ {' '.join(cmd)}\n")

        def worker() -> None:
            try:
                rc = run_streaming(cmd, cwd=self.sim_root, log_q=self.log_q)
                if rc != 0:
                    self.log_q.put(f"\n[error] arena-report exited with code {rc}\n")
            except Exception as e:
                self.log_q.put(f"\n[error] Failed to run arena-report: {e}\n")

        threading.Thread(target=worker, daemon=True).start()

    def snapshot_active_model(self, active_model_path: Path) -> Path:
        return snapshot_checkpoint(self._model_root(), active_model_path)

    def _snapshot_path_for(self, model_path: Path) -> Path:
        return snapshot_path(self._model_root(), model_path)

    def import_model(self, src_p: Path) -> Path:
        tag = now_tag(micro=False)
        if src_p.suffix.lower() == ".zip":
            bundles_dir = self._model_root() / "bundles"
            bundles_dir.mkdir(parents=True, exist_ok=True)

            stem = src_p.stem
            if not stem.endswith(".bundle"):
                stem = f"{stem}.bundle"
            dst = bundles_dir / f"{tag}_{stem}"

            if dst.exists():
                for i in range(1, 1000):
                    cand = bundles_dir / f"{tag}_{src_p.stem}_{i:03d}.bundle"
                    if not cand.exists():
                        dst = cand
                        break

            tmp = bundles_dir / f".tmp_import_{tag}"
            try:
                if tmp.exists():
                    shutil.rmtree(tmp)
                tmp.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(src_p, "r") as zf:
                    members = zf.namelist()
                    if not members:
                        raise ValueError("Empty zip archive")

                    for m in members:
                        mp = Path(m)
                        if mp.is_absolute() or ".." in mp.parts:
                            raise ValueError(f"Unsafe path in zip: {m}")
                    zf.extractall(tmp)

                kids = list(tmp.iterdir())
                extracted_root = tmp
                if len(kids) == 1 and kids[0].is_dir():
                    extracted_root = kids[0]

                if not extracted_root.name.endswith(".bundle"):
                    bundle_dirs = [p for p in extracted_root.iterdir() if p.is_dir() and p.name.endswith(".bundle")]
                    if len(bundle_dirs) == 1:
                        extracted_root = bundle_dirs[0]

                pts = list(extracted_root.glob("*.pt"))
                if not pts:
                    raise ValueError("Bundle archive contains no .pt checkpoints at the expected level")

                shutil.move(str(extracted_root), str(dst))
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp)
            return dst
        else:
            dst_dir = self._model_root() / "imports"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{tag}_{src_p.name}"
            shutil.copy2(src_p, dst)
            return dst

    def export_model(self, src_p: Path, dst: Path) -> None:
        if src_p.is_dir():
            base = dst
            if base.suffix.lower() == ".zip":
                base = base.with_suffix("")
            shutil.make_archive(str(base), "zip", root_dir=str(src_p.parent), base_dir=src_p.name)
        else:
            shutil.copy2(src_p, dst)

    def activate_model(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def is_deletable_checkpoint(self, p: Path) -> bool:
        try:
            rp = p.resolve()
            root = (self.sim_root / "outputs" / "models").resolve()
            return str(rp).startswith(str(root) + os.sep)
        except Exception:
            return False

    def delete_model(self, p: Path) -> None:
        if p.is_file() or p.is_symlink():
            meta = Path(str(p) + ".meta.json")
            if meta.exists():
                meta.unlink()
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()

    def show_model_profile_info(self, p: Path) -> tuple[str, str | None]:
        from simple_sim.models import load_checkpoint  # lazy: pulls in torch

        try:
            checkpoint = load_checkpoint(p)
            profile_data = checkpoint.get('component_profile')

            if not profile_data:
                return "This model checkpoint has no component profile metadata.\n\nIt was likely trained before the profile system was implemented.", None

            profile_id = profile_data.get('profile_id', 'unknown')
            profile_hash = profile_data.get('profile_hash', '')
            dataset_path = checkpoint.get('trained_on_dataset', 'unknown')
            manifest_hash = checkpoint.get('dataset_manifest_hash', '')

            profiles_dir = self.sim_root / "configs" / "profiles"
            profile_path = profiles_dir / f"{profile_id}.yaml"

            info_text = f"Model: {p.name}\nPath: {p}\n\n"
            info_text += "="*60 + "\n" + "COMPONENT PROFILE\n" + "="*60 + "\n"
            info_text += f"Profile ID: {profile_id}\nProfile Hash: {profile_hash}\nTrained on dataset: {dataset_path}\nDataset manifest hash: {manifest_hash}\n"

            if profile_path.exists():
                info_text += "\n" + "="*60 + "\n" + "PROFILE DETAILS\n" + "="*60 + "\n\n"
                profile_yaml = load_yaml(profile_path)
                info_text += yaml.dump(profile_yaml, default_flow_style=False, sort_keys=False)
            else:
                info_text += f"\n\n⚠ Profile file not found at: {profile_path}"

            info_text += "\n" + "="*60 + "\n" + "CHECKPOINT METADATA\n" + "="*60 + "\n"
            info_text += f"Epoch: {checkpoint.get('epoch', 'unknown')}\n"
            info_text += f"Val F1: {checkpoint.get('val_f1', 'unknown')}\n"
            info_text += f"Val Accuracy: {checkpoint.get('val_accuracy', 'unknown')}\n"
            info_text += f"Classes: {checkpoint.get('class_names', 'unknown')}\n"
            return info_text, f"Model Profile: {p.name}"

        except Exception as e:
            return "", str(e)

    def rename_model(self, src: Path, dst: Path) -> None:
        src.rename(dst)
        src_meta = Path(str(src) + ".meta.json")
        dst_meta = Path(str(dst) + ".meta.json")
        if src_meta.exists() and not dst_meta.exists():
            src_meta.rename(dst_meta)

    def duplicate_model(self, src: Path, dst: Path) -> None:
        shutil.copy2(src, dst)
        meta_src = Path(str(src) + ".meta.json")
        meta_dst = Path(str(dst) + ".meta.json")
        if meta_src.exists():
            shutil.copy2(meta_src, meta_dst)

    def stop_prediction(self) -> None:
        self.stop_evt.set()
        terminate_process(self.proc)

    def _set_proc(self, proc: subprocess.Popen[str]) -> None:
        self.proc = proc

    def _predict_cmd(self, model_path: Path, data_dir: Path, split: str, device: str, max_samples_s: str, save_preds: bool) -> tuple[list[str], Path]:
        """``batch-predict --json`` command line and its report dir; a bundle dir resolves to the dataset's profile.

        The profile comes from the manifest via :func:`simple_sim.manifest.resolve_dataset_profile`, so
        multi-profile (v2) datasets resolve to the bundle's ``multi`` checkpoint (plan X7).
        """
        if model_path.exists() and model_path.is_dir():
            profile_id = dataset_profile(data_dir)[0]
            model_path = resolve_bundle_checkpoint(
                model_path, profile_id, show_available=True,
                hint=("\nTrain this dataset into the same bundle to add it:\n"
                      f"  python -m simple_sim train --data {data_dir} --out {model_path}\n"),
            )

        out_dir = data_dir / "predictions" / "weights_tab"
        out_dir.mkdir(parents=True, exist_ok=True)

        if max_samples_s:
            int(max_samples_s)
        cmd = argv_for(
            "batch-predict", model=model_path, data=data_dir, split=split, out_dir=out_dir,
            device=None if device == "auto" else device, max_samples=max_samples_s or None,
            save_preds=bool(save_preds), json=True,
        )
        return cmd, out_dir

    def run_predict(self, model_path: Path, data_dir: Path, label: str, split: str, device: str, max_samples_s: str, save_preds: bool, on_complete: callable) -> None:
        self.stop_evt.clear()
        self.log_q.put(f"\n[run] {label}\n")

        def worker() -> None:
            report_path: Path | None = None

            def on_line(line: str) -> None:
                nonlocal report_path
                if (res := parse_result_line(line)) is not None and res.get("report_path"):
                    report_path = Path(res["report_path"])

            try:
                cmd, _out_dir = self._predict_cmd(model_path, data_dir, split, device, max_samples_s, save_preds)
                self.log_q.put(f"\n$ {' '.join(cmd)}\n")
                rc = run_streaming(cmd, cwd=self.sim_root, log_q=self.log_q, stop_evt=self.stop_evt, on_line=on_line, on_start=self._set_proc)
                if self.stop_evt.is_set(): self.log_q.put("\n[stopped]\n")
                elif rc != 0: self.log_q.put(f"\n[error] batch-predict exited with code {rc}\n")
            except Exception as e: self.log_q.put(f"\n[error] Failed to run batch-predict: {e}\n")
            finally:
                on_complete(report_path, self.stop_evt.is_set())

        threading.Thread(target=worker, daemon=True).start()

    def run_predict_blocking(self, model_path: Path, data_dir: Path, label: str, split: str, device: str, max_samples_s: str, save_preds: bool) -> dict[str, Any] | None:
        try:
            cmd, _out_dir = self._predict_cmd(model_path, data_dir, split, device, max_samples_s, save_preds)
        except Exception as e:
            self.log_q.put(f"[error] {e}\n")
            return None

        # stop_evt is cleared once by the caller (compare run); clearing it here would swallow a Stop between runs.
        if self.stop_evt.is_set(): return None
        self.log_q.put(f"\n$ {' '.join(cmd)}\n")
        self.log_q.put(f"[run] {label}\n")
        result: dict[str, Any] | None = None

        def on_line(line: str) -> None:
            nonlocal result
            if (res := parse_result_line(line)) is not None:
                result = res

        try:
            # Track as self.proc (via on_start) so stop_prediction() can terminate it and the "already running" guard sees it.
            # run_streaming terminates the child before wait() on stop, so it cannot block on a full stdout pipe.
            rc = run_streaming(cmd, cwd=self.sim_root, log_q=self.log_q, stop_evt=self.stop_evt, on_line=on_line, on_start=self._set_proc)
            if self.stop_evt.is_set():
                self.log_q.put("\n[stopped]\n")
                return None
            if rc != 0:
                error = str((result or {}).get("error") or "")
                if "do not match dataset classes" in error: self.log_q.put("[skip] incompatible model/dataset (class mismatch)\n")
                else: self.log_q.put(f"\n[error] batch-predict exited with code {rc}\n")
                return None
        except Exception as e:
            self.log_q.put(f"\n[error] Failed to run batch-predict: {e}\n")
            return None

        report_path = Path(result["report_path"]) if result and result.get("report_path") else None
        if not report_path or not report_path.exists(): return None
        return read_json(report_path)

    _RE_SNAP_RUN = SNAPSHOT_RUN_RE

    def runs_for_model(self, p: Path) -> str:
        """Pipeline run count of a checkpoint: ``<ckpt>.meta.json`` run_iteration, else ``_runNNN`` in the name."""
        meta = read_json(Path(str(p) + ".meta.json"))
        if isinstance(meta, dict):
            try: return str(int(meta.get("run_iteration")))
            except (TypeError, ValueError): pass
        if m := self._RE_SNAP_RUN.search(p.name):
            return str(int(m.group("run")))
        return "-"

    def load_reports(self) -> list[dict[str, Any]]:
        return load_reports(report_search_dirs(self.sim_root))

    def short_report(self, report_path: Path) -> str:
        obj = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = obj.get("metrics", {})
        acc, f1 = metrics.get("accuracy"), metrics.get("macro_f1")
        seen, split = obj.get("seen_samples", "-"), obj.get("split", "-")
        model, ds = Path(obj.get("model_path", "-")).name, Path(obj.get("dataset_path", "-")).name

        lines = ["\n" + "-" * 60, f"[report] {report_path.name}", f"  model: {model}", f"  data:  {ds}",
                 f"  split: {split}  seen: {seen}"]
        if acc is not None: lines.append(f"  accuracy: {float(acc):.4f}")
        if f1 is not None: lines.append(f"  macro_f1:  {float(f1):.4f}")
        lines.append("-" * 60 + "\n")
        return "\n".join(lines)

    def compare_summary_multi(self, paired: list[tuple[Path, dict[str, Any], dict[str, Any]]], model_a: Path, model_b: Path, split: str) -> str:
        def metric(r: dict[str, Any], k: str) -> float | None:
            try: return float((r.get("metrics") or {}).get(k))
            except Exception: return None
        def seen(r: dict[str, Any]) -> int:
            try: return int(r.get("seen_samples") or 0)
            except Exception: return 0
        def wavg(which: str, k: str) -> float | None:
            num, den = 0.0, 0.0
            for _ds, ra, rb in paired:
                r, s = (ra if which == "A" else rb), seen(ra if which == "A" else rb)
                if (v := metric(r, k)) is None or s <= 0: continue
                num, den = num + v * float(s), den + float(s)
            return (num / den) if den > 0 else None

        acc_a, acc_b, f1_a, f1_b = wavg("A", "accuracy"), wavg("B", "accuracy"), wavg("A", "macro_f1"), wavg("B", "macro_f1")
        winner = "TIE"
        eps = 1e-12
        if f1_a is not None and f1_b is not None and abs(f1_a - f1_b) > eps: winner = "A" if f1_a > f1_b else "B"
        elif acc_a is not None and acc_b is not None and abs(acc_a - acc_b) > eps: winner = "A" if acc_a > acc_b else "B"

        lines: list[str] = ["\n[compare multi]", f"  split: {split}", f"  datasets ({len(paired)}): {', '.join([ds.name for ds, _, _ in paired])}"]
        if acc_a is not None and acc_b is not None: lines.append(f"  accuracy (wavg): A={acc_a:.4f}  B={acc_b:.4f}  delta(B-A)={acc_b-acc_a:+.4f}")
        if f1_a is not None and f1_b is not None: lines.append(f"  macro_f1 (wavg):  A={f1_a:.4f}  B={f1_b:.4f}  delta(B-A)={f1_b-f1_a:+.4f}")
        lines.append(f"  winner: {winner}")
        lines.append(f"  A: {model_a.name}\n  B: {model_b.name}\n")
        return "\n".join(lines)

    def compare_summary(self, ra: dict[str, Any], rb: dict[str, Any], model_a: Path, model_b: Path, split: str) -> str:
        def fmetric(r: dict[str, Any], k: str) -> float | None:
            try: return float((r.get("metrics") or {}).get(k))
            except Exception: return None
        def fn_rate(r: dict[str, Any], cls: str) -> float | None:
            try: return float((r.get("metrics") or {}).get("critical_fn_rates", {}).get(cls))
            except Exception: return None

        dsname = Path(str(ra.get("dataset_path") or rb.get("dataset_path") or "-")).name
        runs_a, runs_b = self.runs_for_model(model_a), self.runs_for_model(model_b)
        acc_a, acc_b, f1_a, f1_b, cfn_a, cfn_b = fmetric(ra, "accuracy"), fmetric(rb, "accuracy"), fmetric(ra, "macro_f1"), fmetric(rb, "macro_f1"), fn_rate(ra, "MISALIGNED"), fn_rate(rb, "MISALIGNED")
        
        winner = "TIE"
        eps = 1e-12
        if f1_a is not None and f1_b is not None and abs(f1_a - f1_b) > eps: winner = "A" if f1_a > f1_b else "B"
        elif acc_a is not None and acc_b is not None and abs(acc_a - acc_b) > eps: winner = "A" if acc_a > acc_b else "B"

        lines: list[str] = ["\n[compare]", f"  dataset: {dsname}   split: {split}", f"  A: {model_a.name}  (runs={runs_a})", f"  B: {model_b.name}  (runs={runs_b})"]
        if acc_a is not None and acc_b is not None: lines.append(f"  accuracy: A={acc_a:.4f}  B={acc_b:.4f}  delta(B-A)={acc_b-acc_a:+.4f}")
        if f1_a is not None and f1_b is not None: lines.append(f"  macro_f1:  A={f1_a:.4f}  B={f1_b:.4f}  delta(B-A)={f1_b-f1_a:+.4f}")
        if cfn_a is not None and cfn_b is not None: lines.append(f"  MISALIGNED FN rate: A={cfn_a:.4f}  B={cfn_b:.4f}  delta(B-A)={cfn_b-cfn_a:+.4f}")
        lines.append(f"  winner: {winner}\n")
        return "\n".join(lines)

    def get_cmp_datasets(self) -> list[Path]:
        return sorted_dataset_dirs(self.sim_root)

    def _display_for_dataset(self, p: Path, *, runs: Path, versions: Path) -> str:
        return dataset_display_name(p, runs_root=runs, versions_root=versions)

    def describe_dataset_samples(self, ds: Path) -> str:
        try:
            manifest_path = ds / DATASET_MANIFEST
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                stats = manifest.get("dataset_stats") or {}
                total = stats.get("total_samples")
                splits = stats.get("splits") or {}
                tr = splits.get("train")
                va = splits.get("val")
                te = splits.get("test")
                if isinstance(total, int):
                    if all(isinstance(x, int) for x in (tr, va, te)):
                        return f"Samples: {total} (t{tr}/v{va}/s{te})"
                    return f"Samples: {total}"
            labels_path = ds / LABELS_JSONL
            if labels_path.exists():
                n = 0
                with labels_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            n += 1
                return f"Samples: {n}"
        except Exception:
            pass
        return "Samples: -"

    def get_dataset_classes(self, ds: Path) -> list[str]:
        classes: set[str] = set()
        labels_path = ds / LABELS_JSONL
        if not labels_path.exists():
            return []
        try:
            with labels_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    cls = str(obj.get("class_name") or "").strip()
                    if cls:
                        classes.add(cls)
        except Exception:
            return []
        return sorted(classes)

    def get_model_classes(self, model_path: Path) -> list[str]:
        return checkpoint_class_names(model_path)
