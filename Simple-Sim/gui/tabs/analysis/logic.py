"""Logic for the Analysis Tab."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
import numpy as np

from gui.utils.checkpoint_meta import read_checkpoint_dict
from gui.utils.dataset_catalog import sim_data_roots
from gui.utils.model_inference import load_model, ModelWrapper
from gui.utils.paths import default_model_path, model_root
from simple_sim.classes import STANDARD_CLASSES
from simple_sim.dataset_files import DATASET_MANIFEST, LABELS_JSONL, META_JSONL
from simple_sim.schema import read_jsonl, MetaRow, LabelRow
from simple_sim.manifest import hash_file
from simple_sim.io_utils import atomic_write_json
from simple_sim.metrics import compute_metrics


def ordered_class_names(classes: set[str]) -> list[str]:
    """The standard classes in their canonical order first, then the others alphabetically."""
    ordered = [c for c in STANDARD_CLASSES if c in classes]
    ordered.extend(sorted(c for c in classes if c and c not in STANDARD_CLASSES))
    return ordered


UNKNOWN_PROFILE = "unknown_or_legacy"
SORT_MODES = ("Default", "Newest first", "Oldest first", "ID A-Z", "ID Z-A")


def sample_id_parts(sample_id: str) -> list[str]:
    """``run/domain/split/leaf`` parts of a sample id (moved samples may carry a suffixed leaf)."""
    return sample_id.split("/")


def sample_profile(profiles: dict[str, str], sample_id: str) -> str:
    return str(profiles.get(sample_id, "")).strip() or UNKNOWN_PROFILE


def filter_options(sample_ids: list[str]) -> tuple[set[str], set[str]]:
    """``(run ids, domains)`` occurring in *sample_ids*, each with ``"All"``."""
    runs, domains = {"All"}, {"All"}
    for sid in sample_ids:
        parts = sample_id_parts(sid)
        if len(parts) > 1:
            runs.add(parts[0])
            domains.add(parts[1])
    return runs, domains


def sort_sample_ids(sample_ids: list[str], sort_mode: str, mtime: Callable[[str], float]) -> list[str]:
    """*sample_ids* in *sort_mode* order (one of :data:`SORT_MODES`; ``mtime(sid)`` for the date modes)."""
    if sort_mode == "ID A-Z":
        return sorted(sample_ids)
    if sort_mode == "ID Z-A":
        return sorted(sample_ids, reverse=True)
    if sort_mode in {"Newest first", "Oldest first"}:
        return sorted(sample_ids, key=mtime, reverse=sort_mode == "Newest first")
    return list(sample_ids)


def filter_sample_tree(sample_ids: list[str], labels: dict[str, str], profiles: dict[str, str], *,
                       filters: dict[str, str], selected_profiles: set[str], query: str, sort_mode: str,
                       mtime: Callable[[str], float]) -> dict[str, dict[str, dict[str, list[str]]]]:
    """``{run: {domain: {split: [sample ids]}}}`` of the samples passing the filters.

    *filters* holds ``class``/``run``/``domain``/``split``/``profile`` values (``"All"`` = no filter);
    a non-empty *selected_profiles* replaces the single profile filter. *query* is lower-case.
    """
    tree: dict[str, dict[str, dict[str, list[str]]]] = {}
    for sid in sample_ids:
        label = labels.get(sid, "?")
        if filters["class"] != "All" and label != filters["class"]:
            continue
        pid = sample_profile(profiles, sid)
        if selected_profiles:
            if pid not in selected_profiles:
                continue
        elif filters["profile"] != "All" and pid != filters["profile"]:
            continue
        parts = sample_id_parts(sid)
        if len(parts) < 3:
            continue
        run, domain, split = parts[0], parts[1], parts[2]
        if any(filters[k] != "All" and v != filters[k] for k, v in (("run", run), ("domain", domain), ("split", split))):
            continue
        if query and query not in f"{sid} {parts[-1]} {label}".lower():
            continue
        tree.setdefault(run, {}).setdefault(domain, {}).setdefault(split, []).append(sid)
    for run_node in tree.values():
        for domain_node in run_node.values():
            for split, ids in domain_node.items():
                domain_node[split] = sort_sample_ids(ids, sort_mode, mtime)
    return tree


def dataset_stats_text(sample_ids: list[str], profiles: dict[str, str]) -> str:
    """One-line summary: totals per split and per profile (most frequent first)."""
    if not sample_ids:
        return "Dataset Stats: -"
    split_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    for sid in sample_ids:
        parts = sample_id_parts(sid)
        split_counts[parts[2] if len(parts) > 2 else "unknown"] += 1
        profile_counts[sample_profile(profiles, sid)] += 1
    split_text = ", ".join(f"{s}={split_counts.get(s, 0)}" for s in ("train", "val", "test"))
    profiles_text = ", ".join(
        f"{pid}:{cnt}" for pid, cnt in sorted(profile_counts.items(), key=lambda x: (-x[1], x[0])))
    return f"Dataset Stats: total={len(sample_ids)} | {split_text} | profiles: {profiles_text}"


def sample_mtime(dataset_dir: Path | None, meta: MetaRow | None) -> float:
    """Modification time of a sample's image (0.0 when unknown)."""
    if dataset_dir is None or meta is None:
        return 0.0
    try:
        path = dataset_dir / meta.image_path
        return path.stat().st_mtime if path.exists() else 0.0
    except Exception:
        return 0.0


class AnalysisLogic:
    def __init__(self, sim_root: Path):
        self.sim_root = sim_root
        self.model: ModelWrapper | None = None

    def sim_data_roots(self) -> tuple[Path, Path]:
        return sim_data_roots(self.sim_root)

    def load_dataset(self, dataset_dir: Path) -> tuple[dict[str, MetaRow], dict[str, str], dict[str, str], list[str]]:
        meta_rows = read_jsonl(dataset_dir / META_JSONL, MetaRow)
        label_rows = read_jsonl(dataset_dir / LABELS_JSONL, LabelRow)
        meta_dict = {row.id: row for row in meta_rows}
        label_dict = {row.id: row.class_name for row in label_rows}
        profile_dict = {row.id: row.profile_id for row in label_rows}
        sample_ids = [row.id for row in meta_rows]
        return meta_dict, label_dict, profile_dict, sample_ids

    def try_load_model(self, dataset_dir: Path, current_model_path: Path | None) -> ModelWrapper | None:
        self.model = None
        manifest_hash = None
        manifest_path = dataset_dir / DATASET_MANIFEST
        if manifest_path.exists():
            try:
                manifest_hash = hash_file(manifest_path)
            except Exception as e:
                print(f"Failed to hash dataset manifest: {e}")

        if current_model_path and current_model_path.exists():
            if manifest_hash is None or self._checkpoint_matches_manifest(current_model_path, manifest_hash):
                if self._load_model_from_path(current_model_path):
                    return self.model

        dataset_model = default_model_path(self.sim_root, dataset_dir.name)
        if dataset_model.exists() and self._load_model_from_path(dataset_model):
            return self.model

        if manifest_hash:
            match = self._find_model_by_manifest_hash(manifest_hash)
            if match and self._load_model_from_path(match):
                return self.model
        return None

    def _load_model_from_path(self, path: Path) -> bool:
        try:
            self.model = load_model(path)  # None when loading failed (logged there)
            if self.model is None:
                return False
            print(f"Loaded model: {path}")
            return True
        except Exception as e:
            print(f"Failed to load model {path}: {e}")
            self.model = None
            return False

    def _find_model_by_manifest_hash(self, manifest_hash: str) -> Path | None:
        models_root = model_root(self.sim_root)
        candidates = list(models_root.glob("*.pt"))
        candidates.extend(list((models_root / "imports").glob("*.pt")))
        candidates.extend(list((models_root / "versions").glob("**/*.pt")))
        
        seen = {str(p.resolve()): p for p in candidates if p.is_file()}
        ordered = sorted(seen.values(), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)

        for path in ordered:
            if self._checkpoint_manifest_hash(path) == manifest_hash:
                return path
        return None

    def _checkpoint_matches_manifest(self, path: Path, manifest_hash: str) -> bool:
        ckpt_hash = self._checkpoint_manifest_hash(path)
        return ckpt_hash == manifest_hash if ckpt_hash else False

    def _checkpoint_manifest_hash(self, path: Path) -> str | None:
        try:
            return read_checkpoint_dict(path).get("dataset_manifest_hash")
        except Exception as exc:
            print(f"Failed to read checkpoint metadata {path}: {exc}")
            return None

    def analyze_dataset(
        self,
        dataset_dir: Path,
        sample_ids: list[str],
        meta_dict: dict[str, MetaRow],
        label_dict: dict[str, str],
        effective_label_dict: dict[str, str] | None = None,
        feedback_summary: dict | None = None,
        progress: Callable[[float | None, str | None], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict:
        """Predict every sample and write ``analysis_results.json``.

        ``cancelled()`` is polled per sample; when it returns True the run stops
        with ``TaskCancelled`` *before* anything is written.
        """
        # One model for the whole run: a dataset change swaps self.model under the worker thread.
        model = self.model
        if not model:
            raise ValueError("No model loaded.")

        results = {
            'dataset_path': str(dataset_dir),
            'model_path': str(model.model_path),
            'samples': [],
            'summary': {},
        }
        correct_count = 0
        failed_count = 0
        y_true_labels: list[str] = []
        y_pred_labels: list[str] = []

        total = len(sample_ids)
        for i, sample_id in enumerate(sample_ids):
            if cancelled is not None and cancelled():
                from gui.core.tasks import TaskCancelled

                raise TaskCancelled("analysis")
            if progress is not None and (i % 10 == 0 or i == total - 1):
                progress(i / max(1, total), f"{i}/{total} samples")
            meta_row = meta_dict[sample_id]
            original_ground_truth = label_dict[sample_id]
            ground_truth = (
                effective_label_dict.get(sample_id, original_ground_truth)
                if effective_label_dict
                else original_ground_truth
            )
            image_path = dataset_dir / meta_row.image_path
            try:
                predicted, confidence, _ = model.predict(image_path)
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
        class_names = ordered_class_names(set(y_true_labels) | set(y_pred_labels))
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

        atomic_write_json(dataset_dir / "analysis_results.json", results)
        return results

    def _compute_metrics(self, y_true_labels: list[str], y_pred_labels: list[str], class_names: list[str]) -> dict:
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
