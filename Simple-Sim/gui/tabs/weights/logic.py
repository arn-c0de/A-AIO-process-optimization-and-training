"""Logic for the Weights Tab."""

from __future__ import annotations
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from simple_sim.model_bundle import bundle_checkpoint_path
import torch
import yaml


class WeightsLogic:
    def __init__(self, sim_root: Path, log_q: queue.Queue[str]) -> None:
        self.sim_root = sim_root
        self.log_q = log_q
        self.proc: Optional[subprocess.Popen[str]] = None
        self.stop_evt = threading.Event()

    def _model_root(self) -> Path:
        return self.sim_root / "outputs" / "models"

    def get_models(self) -> List[Path]:
        root = self._model_root()
        root.mkdir(parents=True, exist_ok=True)

        cand: List[Path] = []
        cand.extend(sorted(root.glob("*.pt")))
        cand.extend(sorted((root / "versions").glob("**/*.pt")))
        cand.extend(sorted((root / "imports").glob("*.pt")))
        cand.extend([p for p in sorted(root.glob("*.bundle")) if p.is_dir()])
        bundles_subdir = root / "bundles"
        if bundles_subdir.exists():
            cand.extend([p for p in sorted(bundles_subdir.iterdir()) if p.is_dir()])

        uniq: Dict[str, Path] = {}
        for p in cand:
            try:
                if p.is_file() or p.is_dir():
                    uniq[str(p.resolve())] = p
            except Exception:
                continue
        return list(uniq.values())

    def rel_path(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.sim_root.resolve()))
        except Exception:
            return str(p)

    def _favorites_path(self) -> Path:
        return self.sim_root / "outputs" / "models" / "favorites.json"

    def _arena_path(self) -> Path:
        return self.sim_root / "outputs" / "models" / "arena.json"

    def load_favorites(self) -> Dict[str, Dict[str, Any]]:
        p = self._favorites_path()
        favorites: Dict[str, Dict[str, Any]] = {}
        if not p.exists():
            return favorites
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return favorites
        if not isinstance(obj, dict):
            return favorites
        favs = obj.get("favorites")
        if isinstance(favs, list):
            for it in favs:
                if not isinstance(it, dict):
                    continue
                rp = str(it.get("path") or "").strip()
                if not rp:
                    continue
                favorites[rp] = it
        return favorites

    def save_favorites(self, favorites: Dict[str, Dict[str, Any]]) -> None:
        p = self._favorites_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        favs = list(favorites.values())
        favs.sort(key=lambda it: str(it.get("path") or ""))
        obj = {"favorites": favs}
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def load_arena(self) -> Dict[str, Dict[str, Any]]:
        p = self._arena_path()
        arena: Dict[str, Dict[str, Any]] = {}
        if not p.exists():
            return arena
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return arena
        if not isinstance(obj, dict):
            return arena
        items = obj.get("arena_models")
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                rp = str(it.get("path") or "").strip()
                if not rp:
                    continue
                arena[rp] = it
        return arena

    def save_arena(self, arena: Dict[str, Dict[str, Any]]) -> None:
        p = self._arena_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        items = list(arena.values())
        items.sort(key=lambda it: str(it.get("path") or ""))
        obj = {"arena_models": items}
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def add_to_arena(self, paths: List[Path], arena: Dict[str, Dict[str, Any]]) -> None:
        for p in paths:
            rel = self.rel_path(p)
            if rel in arena:
                continue
            arena[rel] = {
                "path": rel,
                "added_at": datetime.now().isoformat(timespec="seconds"),
                "note": "",
            }

    def remove_from_arena(self, paths: List[Path], arena: Dict[str, Dict[str, Any]]) -> None:
        for p in paths:
            rel = self.rel_path(p)
            if rel in arena:
                arena.pop(rel, None)

    def generate_arena_report(self) -> None:
        script = self.sim_root / "scripts" / "arena_report.py"
        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")

        cmd = ["python3", str(script), "--sim-root", str(self.sim_root)]
        self.log_q.put(f"\n$ {' '.join(cmd)}\n")

        def worker() -> None:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.sim_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log_q.put(line)
                rc = proc.wait()
                if rc != 0:
                    self.log_q.put(f"\n[error] arena_report.py exited with code {rc}\n")
            except Exception as e:
                self.log_q.put(f"\n[error] Failed to run arena_report.py: {e}\n")

        threading.Thread(target=worker, daemon=True).start()

    def snapshot_active_model(self, active_model_path: Path) -> Path:
        dst = self._snapshot_path_for(active_model_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(active_model_path, dst)
        return dst

    def _snapshot_path_for(self, model_path: Path) -> Path:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = self._model_root() / "versions" / model_path.stem
        base = root / f"{model_path.stem}_{tag}.pt"
        if not base.exists():
            return base
        for i in range(1, 1000):
            cand = root / f"{model_path.stem}_{tag}_{i:03d}.pt"
            if not cand.exists():
                return cand
        return base

    def import_model(self, src_p: Path) -> Path:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    def show_model_profile_info(self, p: Path) -> Tuple[str, Optional[str]]:
        try:
            checkpoint = torch.load(p, map_location='cpu')
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
                with open(profile_path, 'r') as f:
                    profile_yaml = yaml.safe_load(f)
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
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def _predict_cmd(self, model_path: Path, data_dir: Path, split: str, device: str, max_samples_s: str, save_preds: bool) -> Tuple[List[str], Path]:
        if model_path.exists() and model_path.is_dir():
            manifest_path = data_dir / "dataset_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            profile_id = str((manifest.get("component_profile") or {}).get("profile_id") or "").strip()
            if not profile_id:
                raise ValueError(f"Dataset manifest missing component_profile.profile_id: {manifest_path}")
            expected = bundle_checkpoint_path(model_path, profile_id, kind="best")
            if not expected.exists():
                avail = sorted([p.name for p in model_path.glob("*.pt")])[:12]
                raise FileNotFoundError(
                    f"Multi-model bundle has no checkpoint for profile '{profile_id}':\\n"
                    f"  bundle: {model_path}\\n"
                    f"  expected: {expected}\\n"
                    f"  available: {', '.join(avail) if avail else '(none)'}\\n\\n"
                    f"Train this dataset into the same bundle to add it:\\n"
                    f"  ./.venv/bin/python scripts/train.py --data {data_dir} --out {model_path}\\n"
                )
            model_path = expected

        out_dir = data_dir / "predictions" / "weights_tab"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [
            "bash", str(self.sim_root / "predict.sh"), "--model", str(model_path), "--data", str(data_dir), "--split", split, "--out-dir", str(out_dir),
        ]
        if device != "auto": cmd.extend(["--device", device])
        if max_samples_s:
            int(max_samples_s)
            cmd.extend(["--max-samples", max_samples_s])
        if save_preds: cmd.append("--save-preds")
        return cmd, out_dir

    def run_predict(self, model_path: Path, data_dir: Path, label: str, split: str, device: str, max_samples_s: str, save_preds: bool, on_complete: callable) -> None:
        self.stop_evt.clear()
        self.log_q.put(f"\n[run] {label}\n")

        def worker() -> None:
            report_path: Optional[Path] = None
            try:
                cmd, out_dir = self._predict_cmd(model_path, data_dir, split, device, max_samples_s, save_preds)
                self.log_q.put(f"\n$ {' '.join(cmd)}\n")
                self.proc = subprocess.Popen(cmd, cwd=str(self.sim_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    if self.stop_evt.is_set(): break
                    self.log_q.put(line)
                    if "Report saved to:" in line: report_path = Path(line.split("Report saved to:", 1)[-1].strip())
                rc = self.proc.wait()
                if self.stop_evt.is_set(): self.log_q.put("\n[stopped]\n")
                elif rc != 0: self.log_q.put(f"\n[error] predict.sh exited with code {rc}\n")
            except Exception as e: self.log_q.put(f"\n[error] Failed to run predict.sh: {e}\n")
            finally:
                if report_path is None:
                    try:
                        cand = sorted(out_dir.glob("batch_report_*.json"), key=lambda p: p.stat().st_mtime)
                        report_path = cand[-1] if cand else None
                    except Exception: pass
                on_complete(report_path, self.stop_evt.is_set())

        threading.Thread(target=worker, daemon=True).start()

    def run_predict_blocking(self, model_path: Path, data_dir: Path, label: str, split: str, device: str, max_samples_s: str, save_preds: bool) -> Optional[Dict[str, Any]]:
        try:
            cmd, out_dir = self._predict_cmd(model_path, data_dir, split, device, max_samples_s, save_preds)
        except Exception as e:
            self.log_q.put(f"[error] {e}\\n")
            return None

        self.stop_evt.clear()
        self.log_q.put(f"\\n$ {' '.join(cmd)}\\n")
        self.log_q.put(f"[run] {label}\\n")
        report_path: Optional[Path] = None
        try:
            proc = subprocess.Popen(cmd, cwd=str(self.sim_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            class_mismatch = False
            for line in proc.stdout:
                if self.stop_evt.is_set(): break
                if "do not match dataset classes" in line: class_mismatch = True
                self.log_q.put(line)
                if "Report saved to:" in line: report_path = Path(line.split("Report saved to:", 1)[-1].strip())
            rc = proc.wait()
            if self.stop_evt.is_set():
                self.log_q.put("\\n[stopped]\\n")
                return None
            if rc != 0:
                if class_mismatch: self.log_q.put(f"[skip] incompatible model/dataset (class mismatch)\\n")
                else: self.log_q.put(f"\\n[error] predict.sh exited with code {rc}\\n")
                return None
        except Exception as e:
            self.log_q.put(f"\\n[error] Failed to run predict.sh: {e}\\n")
            return None

        if report_path is None:
            try:
                cand = sorted(out_dir.glob("batch_report_*.json"), key=lambda p: p.stat().st_mtime)
                report_path = cand[-1] if cand else None
            except Exception: pass
        if not report_path or not report_path.exists(): return None

        try: return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception: return None

    _RE_SNAP_RUN = re.compile(r"_run(?P<run>\d+)(?:_|\\.|$)")

    def runs_for_model(self, p: Path) -> str:
        meta = self._read_model_meta(p)
        if meta is not None:
            try: return str(int(meta.get("run_iteration")))
            except Exception: pass
        m = self._RE_SNAP_RUN.search(p.name)
        if m:
            try: return str(int(m.group("run")))
            except Exception: pass
        sib = Path(str(p) + ".meta.json")
        if sib.exists():
            try: return str(int(json.loads(sib.read_text(encoding="utf-8")).get("run_iteration")))
            except Exception: pass
        return "-"

    def _read_model_meta(self, p: Path) -> Optional[Dict[str, Any]]:
        mp = Path(str(p) + ".meta.json")
        if not mp.exists(): return None
        try: obj = json.loads(mp.read_text(encoding="utf-8"))
        except Exception: return None
        return obj if isinstance(obj, dict) else None

    def _report_search_dirs(self) -> List[Path]:
        dirs: List[Path] = [self.sim_root / "outputs" / "models", self.sim_root / "outputs" / "models" / "history"]
        sim_data = self.sim_root / "outputs" / "sim_data"
        runs = sim_data / "runs"
        versions = sim_data / "versions"
        
        def add_dataset_prediction_dirs(ds: Path) -> None:
            dirs.append(ds / "predictions")
            dirs.append(ds / "predictions" / "weights_tab")
        
        for root_dir in [runs, versions]:
            if root_dir.exists():
                for ds in (root_dir.iterdir() if root_dir == runs else root_dir.glob("*/*")):
                    if ds.is_dir(): add_dataset_prediction_dirs(ds)
        
        seen = set()
        return [d for d in dirs if (rp := str(d.resolve())) not in seen and not seen.add(rp)]

    def load_reports(self) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        seen_paths = set()
        for d in self._report_search_dirs():
            if not d.exists(): continue
            for p in sorted(list(d.glob("report_*.json")) + list(d.glob("batch_report_*.json")), key=lambda x: x.stat().st_mtime, reverse=True):
                if (sp := str(p)) in seen_paths: continue
                seen_paths.add(sp)
                try: obj = json.loads(p.read_text(encoding="utf-8"))
                except Exception: continue
                if not isinstance(obj, dict) or "metrics" not in obj: continue
                obj["_path"] = sp
                reports.append(obj)
        return reports

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

    def compare_summary_multi(self, paired: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]], model_a: Path, model_b: Path, split: str) -> str:
        def metric(r: Dict[str, Any], k: str) -> Optional[float]:
            try: return float((r.get("metrics") or {}).get(k))
            except Exception: return None
        def seen(r: Dict[str, Any]) -> int:
            try: return int(r.get("seen_samples") or 0)
            except Exception: return 0
        def wavg(which: str, k: str) -> Optional[float]:
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

        lines: List[str] = ["\n[compare multi]", f"  split: {split}", f"  datasets ({len(paired)}): {', '.join([ds.name for ds, _, _ in paired])}"]
        if acc_a is not None and acc_b is not None: lines.append(f"  accuracy (wavg): A={acc_a:.4f}  B={acc_b:.4f}  delta(B-A)={acc_b-acc_a:+.4f}")
        if f1_a is not None and f1_b is not None: lines.append(f"  macro_f1 (wavg):  A={f1_a:.4f}  B={f1_b:.4f}  delta(B-A)={f1_b-f1_a:+.4f}")
        lines.append(f"  winner: {winner}")
        lines.append(f"  A: {model_a.name}\\n  B: {model_b.name}\\n")
        return "\\n".join(lines)

    def compare_summary(self, ra: Dict[str, Any], rb: Dict[str, Any], model_a: Path, model_b: Path, split: str) -> str:
        def fmetric(r: Dict[str, Any], k: str) -> Optional[float]:
            try: return float((r.get("metrics") or {}).get(k))
            except Exception: return None
        def fn_rate(r: Dict[str, Any], cls: str) -> Optional[float]:
            try: return float((r.get("metrics") or {}).get("critical_fn_rates", {}).get(cls))
            except Exception: return None

        dsname = Path(str(ra.get("dataset_path") or rb.get("dataset_path") or "-")).name
        runs_a, runs_b = self.runs_for_model(model_a), self.runs_for_model(model_b)
        acc_a, acc_b, f1_a, f1_b, cfn_a, cfn_b = fmetric(ra, "accuracy"), fmetric(rb, "accuracy"), fmetric(ra, "macro_f1"), fmetric(rb, "macro_f1"), fn_rate(ra, "MISALIGNED"), fn_rate(rb, "MISALIGNED")
        
        winner = "TIE"
        eps = 1e-12
        if f1_a is not None and f1_b is not None and abs(f1_a - f1_b) > eps: winner = "A" if f1_a > f1_b else "B"
        elif acc_a is not None and acc_b is not None and abs(acc_a - acc_b) > eps: winner = "A" if acc_a > acc_b else "B"

        lines: List[str] = ["\n[compare]", f"  dataset: {dsname}   split: {split}", f"  A: {model_a.name}  (runs={runs_a})", f"  B: {model_b.name}  (runs={runs_b})"]
        if acc_a is not None and acc_b is not None: lines.append(f"  accuracy: A={acc_a:.4f}  B={acc_b:.4f}  delta(B-A)={acc_b-acc_a:+.4f}")
        if f1_a is not None and f1_b is not None: lines.append(f"  macro_f1:  A={f1_a:.4f}  B={f1_b:.4f}  delta(B-A)={f1_b-f1_a:+.4f}")
        if cfn_a is not None and cfn_b is not None: lines.append(f"  MISALIGNED FN rate: A={cfn_a:.4f}  B={cfn_b:.4f}  delta(B-A)={cfn_b-cfn_a:+.4f}")
        lines.append(f"  winner: {winner}\\n")
        return "\\n".join(lines)

    def get_cmp_datasets(self) -> List[Path]:
        runs, versions = self.sim_root / "outputs" / "sim_data" / "runs", self.sim_root / "outputs" / "sim_data" / "versions"
        runs.mkdir(parents=True, exist_ok=True)
        versions.mkdir(parents=True, exist_ok=True)
        cand: List[Path] = [p for p in runs.iterdir() if p.is_dir()]
        cand.extend([p for p in versions.glob("*/*") if p.is_dir()])
        cand.sort(key=lambda p: (1, -p.stat().st_mtime, p.name) if str(p.resolve()).startswith(str(versions.resolve()) + os.sep) else (0, p.name))
        return cand

    def _display_for_dataset(self, p: Path, *, runs: Path, versions: Path) -> str:
        try:
            rp = p.resolve()
            if str(rp).startswith(str(runs.resolve()) + os.sep):
                return rp.name
            if str(rp).startswith(str(versions.resolve()) + os.sep):
                return f"{rp.parent.name}:{rp.name}"
        except Exception:
            pass
        return p.name

    def describe_dataset_samples(self, ds: Path) -> str:
        try:
            manifest_path = ds / "dataset_manifest.json"
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
            labels_path = ds / "labels.jsonl"
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

    def get_dataset_classes(self, ds: Path) -> List[str]:
        classes: set[str] = set()
        labels_path = ds / "labels.jsonl"
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

    def get_model_classes(self, model_path: Path) -> List[str]:
        try:
            p = model_path
            if p.is_dir():
                candidates = sorted(p.glob("*.pt"), key=lambda x: x.name)
                if not candidates:
                    return []
                p = candidates[0]
            ckpt = torch.load(p, map_location="cpu")
            classes = ckpt.get("class_names")
            if isinstance(classes, list):
                return [str(c) for c in classes]
        except Exception:
            pass
        return []
