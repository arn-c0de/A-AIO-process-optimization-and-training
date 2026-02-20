"""Logic for the Analysis Tab."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import json
import numpy as np

from gui.utils.model_inference import load_model, ModelWrapper
from simple_sim.schema import read_jsonl, MetaRow, LabelRow
from simple_sim.manifest import hash_file
from simple_sim.metrics import compute_metrics


class AnalysisLogic:
    def __init__(self, sim_root: Path):
        self.sim_root = sim_root
        self.model: Optional[ModelWrapper] = None

    def sim_data_roots(self) -> Tuple[Path, Path]:
        sim_data = self.sim_root / "outputs" / "sim_data"
        runs = sim_data / "runs"
        versions = sim_data / "versions"
        runs.mkdir(parents=True, exist_ok=True)
        versions.mkdir(parents=True, exist_ok=True)
        return runs, versions

    def get_datasets(self) -> List[Path]:
        runs, versions = self.sim_data_roots()
        
        cand: list[Path] = []
        cand.extend([p for p in runs.iterdir() if p.is_dir()])
        cand.extend([p for p in versions.glob("*/*") if p.is_dir()])
        return cand

    def load_dataset(self, dataset_dir: Path) -> Tuple[Dict[str, MetaRow], Dict[str, str], Dict[str, str], List[str]]:
        meta_rows = read_jsonl(dataset_dir / "meta.jsonl", MetaRow)
        label_rows = read_jsonl(dataset_dir / "labels.jsonl", LabelRow)
        meta_dict = {row.id: row for row in meta_rows}
        label_dict = {row.id: row.class_name for row in label_rows}
        profile_dict = {row.id: row.profile_id for row in label_rows}
        sample_ids = [row.id for row in meta_rows]
        return meta_dict, label_dict, profile_dict, sample_ids

    def try_load_model(self, dataset_dir: Path, current_model_path: Optional[Path]) -> Optional[ModelWrapper]:
        self.model = None
        manifest_hash = None
        manifest_path = dataset_dir / "dataset_manifest.json"
        if manifest_path.exists():
            try:
                manifest_hash = hash_file(manifest_path)
            except Exception as e:
                print(f"Failed to hash dataset manifest: {e}")

        if current_model_path and current_model_path.exists():
            if manifest_hash is None or self._checkpoint_matches_manifest(current_model_path, manifest_hash):
                if self._load_model_from_path(current_model_path):
                    return self.model

        dataset_model = self.sim_root / "outputs" / "models" / f"{dataset_dir.name}.pt"
        if dataset_model.exists() and self._load_model_from_path(dataset_model):
            return self.model

        if manifest_hash:
            match = self._find_model_by_manifest_hash(manifest_hash)
            if match and self._load_model_from_path(match):
                return self.model
        return None

    def _load_model_from_path(self, path: Path) -> bool:
        try:
            self.model = load_model(path)
            print(f"Loaded model: {path}")
            return True
        except Exception as e:
            print(f"Failed to load model {path}: {e}")
            self.model = None
            return False

    def _find_model_by_manifest_hash(self, manifest_hash: str) -> Optional[Path]:
        models_root = self.sim_root / "outputs" / "models"
        candidates = [p for p in models_root.glob("*.pt")]
        candidates.extend([p for p in (models_root / "imports").glob("*.pt")])
        candidates.extend([p for p in (models_root / "versions").glob("**/*.pt")])
        
        seen = {str(p.resolve()): p for p in candidates if p.is_file()}
        ordered = sorted(seen.values(), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)

        for path in ordered:
            if self._checkpoint_manifest_hash(path) == manifest_hash:
                return path
        return None

    def _checkpoint_matches_manifest(self, path: Path, manifest_hash: str) -> bool:
        ckpt_hash = self._checkpoint_manifest_hash(path)
        return ckpt_hash == manifest_hash if ckpt_hash else False

    def _checkpoint_manifest_hash(self, path: Path) -> Optional[str]:
        try:
            checkpoint = torch.load(path, map_location="cpu")
            return checkpoint.get("dataset_manifest_hash")
        except Exception as exc:
            print(f"Failed to read checkpoint metadata {path}: {exc}")
            return None

    def analyze_dataset(
        self,
        dataset_dir: Path,
        sample_ids: List[str],
        meta_dict: Dict[str, MetaRow],
        label_dict: Dict[str, str],
        effective_label_dict: Optional[Dict[str, str]] = None,
        feedback_summary: Optional[Dict] = None,
    ) -> Dict:
        if not self.model:
            raise ValueError("No model loaded.")

        results = {
            'dataset_path': str(dataset_dir),
            'model_path': str(self.model.model_path),
            'samples': [],
            'summary': {},
        }
        correct_count = 0
        failed_count = 0
        y_true_labels: list[str] = []
        y_pred_labels: list[str] = []

        for sample_id in sample_ids:
            meta_row = meta_dict[sample_id]
            original_ground_truth = label_dict[sample_id]
            ground_truth = (
                effective_label_dict.get(sample_id, original_ground_truth)
                if effective_label_dict
                else original_ground_truth
            )
            image_path = dataset_dir / meta_row.image_path
            try:
                predicted, confidence, _ = self.model.predict(image_path)
                is_correct = (predicted == ground_truth)
                if is_correct:
                    correct_count += 1
                y_true_labels.append(ground_truth)
                y_pred_labels.append(predicted)
                results['samples'].append({
                    'id': sample_id,
                    'ground_truth': ground_truth,
                    'ground_truth_original': original_ground_truth,
                    'predicted': predicted,
                    'confidence': confidence,
                    'correct': is_correct,
                })
            except Exception as e:
                print(f"Failed to analyze {sample_id}: {e}")
                failed_count += 1

        analyzed_count = len(y_true_labels)
        class_names = self._ordered_class_names(set(y_true_labels) | set(y_pred_labels))
        metrics = self._compute_metrics(y_true_labels, y_pred_labels, class_names)

        results['summary'] = {
            'total_requested': len(sample_ids),
            'analyzed': analyzed_count,
            'failed': failed_count,
            'correct': correct_count,
            'accuracy': (correct_count / analyzed_count) if analyzed_count > 0 else 0.0,
        }
        results['metrics'] = metrics
        results['class_names'] = class_names
        results['used_feedback'] = bool(effective_label_dict)
        if feedback_summary:
            results['feedback_summary'] = feedback_summary

        output_path = dataset_dir / "analysis_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        return results

    def _ordered_class_names(self, classes: set[str]) -> list[str]:
        priority = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]
        ordered = [p for p in priority if p in classes]
        ordered.extend(sorted([c for c in classes if c and c not in priority]))
        return ordered

    def _compute_metrics(self, y_true_labels: list[str], y_pred_labels: list[str], class_names: list[str]) -> Dict:
        if not y_true_labels or not y_pred_labels or not class_names:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "per_class": {},
                "confusion_matrix": [],
                "critical_fn_rates": {},
            }
        idx = {name: i for i, name in enumerate(class_names)}
        y_true = np.array([idx[s] for s in y_true_labels], dtype=np.int64)
        y_pred = np.array([idx[s] for s in y_pred_labels], dtype=np.int64)
        return compute_metrics(y_true, y_pred, class_names)
