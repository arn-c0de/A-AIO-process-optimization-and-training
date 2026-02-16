"""Logic for the Merge Tab."""

from __future__ import annotations
import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from simple_sim.model_bundle import (
    bundle_checkpoint_path,
    upsert_bundle_meta,
    is_bundle_dir,
    read_bundle_meta,
)

class MergeLogic:
    def __init__(self, sim_root: Path):
        self.sim_root = sim_root

    def _model_root(self) -> Path:
        return self.sim_root / "outputs" / "models"

    def _rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.sim_root.resolve()))
        except Exception:
            return str(p)
            
    def scan_models(self) -> Dict[str, List[Tuple[str, Path]]]:
        root = self._model_root()
        root.mkdir(parents=True, exist_ok=True)

        cand: List[Path] = []
        cand.extend(sorted(root.glob("*.pt")))
        cand.extend(sorted((root / "versions").glob("**/*.pt")))
        cand.extend(sorted((root / "imports").glob("*.pt")))
        
        bundle_dirs: List[Path] = []
        for p in sorted(root.glob("*.bundle")):
            if is_bundle_dir(p):
                bundle_dirs.append(p)
        bundles_subdir = root / "bundles"
        if bundles_subdir.exists():
            for p in sorted(bundles_subdir.iterdir()):
                if is_bundle_dir(p):
                    bundle_dirs.append(p)
        for bd in bundle_dirs:
            cand.extend(sorted(bd.glob("*.pt")))
            
        uniq: Dict[str, Path] = {str(p.resolve()): p for p in cand if p.is_file()}

        profile_models: Dict[str, List[Tuple[str, Path]]] = {}
        
        for p in uniq.values():
            profile_id, _, _, _ = self.load_model_meta(p, {})
            if profile_id is None or profile_id == "multi":
                continue
            
            disp = p.name
            bd = self._bundle_root_for_path(p)
            if bd is not None:
                disp = f"[Bundle:{bd.name}] {p.name}"

            profile_models.setdefault(profile_id, []).append((disp, p))

        for pid in profile_models:
            profile_models[pid].sort(key=lambda t: t[1].stat().st_mtime if t[1].exists() else 0.0, reverse=True)
            
        return profile_models
        
    def load_model_meta(self, model_path: Path, cache: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[float]]:
        key = str(model_path.resolve())
        if key in cache:
            return cache[key]

        result: Tuple[Optional[str], Optional[str], Optional[float], Optional[float]] = (None, None, None, None)
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            comp = checkpoint.get("component_profile")
            pid, phash = (comp.get("profile_id"), comp.get("profile_hash")) if isinstance(comp, dict) else (None, None)
            val_acc = float(checkpoint.get("val_accuracy")) if checkpoint.get("val_accuracy") is not None else None
            val_f1 = float(checkpoint.get("val_f1")) if checkpoint.get("val_f1") is not None else None
            result = (pid, phash, val_acc, val_f1)
        except Exception:
            pass
        
        cache[key] = result
        return result

    def _bundle_root_for_path(self, p: Path) -> Optional[Path]:
        try:
            rp = p.resolve()
            root = self._model_root().resolve()
        except Exception:
            rp = p
            root = self._model_root()
            
        cur = rp if rp.is_dir() else rp.parent
        while cur != root and cur.parent != cur:
            if is_bundle_dir(cur):
                return cur
            cur = cur.parent
        return None
        
    def do_merge(self, bundle_dir: Path, selected_weights: Dict[str, Path], model_meta_cache: Dict) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        profiles_merged = []
        model_details: Dict[str, Any] = {}
        sources: Dict[str, Any] = {}

        for profile_id, src_path in sorted(selected_weights.items()):
            dst = bundle_checkpoint_path(bundle_dir, profile_id, kind="best")
            shutil.copy2(src_path, dst)
            upsert_bundle_meta(bundle_dir, profile_id, dst)
            profiles_merged.append(profile_id)

            _, phash, val_acc, val_f1 = self.load_model_meta(src_path, model_meta_cache)
            src_size = src_path.stat().st_size
            src_mtime = datetime.fromtimestamp(src_path.stat().st_mtime).isoformat(timespec="seconds")
            
            entry: Dict[str, Any] = {
                "source_path": str(src_path), "source_name": src_path.name, "profile_hash": phash,
                "val_accuracy": val_acc, "val_f1": val_f1, "source_size_bytes": src_size, "source_modified": src_mtime,
            }

            bd = self._bundle_root_for_path(src_path)
            if bd is not None:
                bd_key = str(bd)
                if bd_key not in sources:
                    src_obj: Dict[str, Any] = {"bundle_dir": bd_key}
                    try:
                        meta = read_bundle_meta(bd)
                        if meta: src_obj["bundle_meta"] = {"bundle_version": meta.bundle_version, "created_at": meta.created_at, "updated_at": meta.updated_at, "checkpoints": dict(meta.checkpoints)}
                        details = self._read_bundle_details(bd)
                        if details: src_obj["bundle_details"] = details
                    except Exception: pass
                    sources[bd_key] = src_obj
                entry["provenance"] = {"type": "bundle", "bundle_dir": bd_key}
                try:
                    bd_details = (sources.get(bd_key) or {}).get("bundle_details") or {}
                    if isinstance(bd_details, dict):
                        if profile_id in (models := bd_details.get("models") or {}):
                            entry["upstream"] = models.get(profile_id)
                except Exception: pass
            else:
                entry["provenance"] = {"type": "checkpoint"}

            model_details[profile_id] = entry
            
        return profiles_merged, model_details, sources
        
    def write_bundle_details(self, bundle_dir: Path, model_details: Dict[str, Any], sources: Optional[Dict[str, Any]] = None):
        details_path = bundle_dir / "bundle_details.json"
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        obj = {"created_at": now, "models": model_details, "sources": sources or {}}
        details_path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _read_bundle_details(self, bundle_dir: Path) -> Optional[Dict[str, Any]]:
        details_path = bundle_dir / "bundle_details.json"
        if not details_path.exists(): return None
        try: return json.loads(details_path.read_text(encoding="utf-8"))
        except Exception: return None

    def get_existing_bundles(self) -> List[Path]:
        root = self._model_root()
        bundle_dirs: List[Path] = [p for p in sorted(root.glob("*.bundle")) if is_bundle_dir(p)]
        bundles_subdir = root / "bundles"
        if bundles_subdir.exists():
            bundle_dirs.extend([p for p in sorted(bundles_subdir.iterdir()) if is_bundle_dir(p)])
        return bundle_dirs
        
    def delete_bundle(self, bundle_path: Path):
        shutil.rmtree(bundle_path)

    def do_merge_ensemble(self, out_path: Path, selected_weights: Dict[str, Path], log_q: queue.Queue):
        import torch

        items: List[Dict[str, Any]] = []
        class_names: Optional[List[str]] = None
        profiles: List[str] = []

        for profile_id, src_path in sorted(selected_weights.items()):
            log_q.put(f"[ensemble] loading {profile_id}: {src_path}\n")
            ckpt = torch.load(src_path, map_location="cpu")
            cn = list(ckpt.get("class_names") or [])
            if not cn: raise ValueError(f"Checkpoint missing class_names: {src_path}")
            if class_names is None: class_names = cn
            elif cn != class_names: raise ValueError("Class names mismatch")

            phash = (ckpt.get("component_profile") or {}).get("profile_hash")
            
            slim = {
                "model_state_dict": ckpt.get("model_state_dict"), "class_names": cn, "config": ckpt.get("config"),
                "val_f1": ckpt.get("val_f1"), "val_accuracy": ckpt.get("val_accuracy"),
                "component_profile": ckpt.get("component_profile"), "trained_on_dataset": ckpt.get("trained_on_dataset"),
            }
            if slim["model_state_dict"] is None: raise ValueError(f"Checkpoint missing model_state_dict: {src_path}")
            
            items.append({"profile_id": profile_id, "profile_hash": phash, "source_path": str(src_path), "checkpoint": slim})
            profiles.append(profile_id)
        
        if class_names is None or not items: raise ValueError("No valid checkpoints")
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "format": "simple_sim_ensemble_v1", "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "class_names": class_names, "models": items,
            "component_profile": {"profile_id": "multi", "profiles": profiles},
        }
        log_q.put(f"[ensemble] writing: {out_path}\n")
        torch.save(obj, out_path)
