"""Extended dataset validation with outlier and duplicate detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import cv2

from simple_sim.schema import read_jsonl, MetaRow, LabelRow
from tools.validate_dataset import validate_dataset

_LOG = logging.getLogger(__name__)

# Tolerance thresholds used in edge-case detection.
_OK_SHIFT_PX = 3.0
_OK_ROTATION_DEG = 5.0
_TOMBSTONE_TILT_DEG = 75.0


def _detect_metric_outliers(
    sample_ids: List[str],
    scores: List[float],
    metric: str,
    threshold_sigma: float,
    reason_template: str,
) -> List[Dict[str, Any]]:
    """Detect outliers in a list of per-sample metric scores using z-scores.

    Args:
        sample_ids: Sample identifier for each score entry.
        scores: Metric score for each sample.
        metric: Metric name used in the returned flag dictionary.
        threshold_sigma: Minimum absolute z-score to flag as an outlier.
        reason_template: Format string receiving ``z`` (float), e.g.
            ``"Blur score {z:.1f}σ from mean"``.

    Returns:
        List of outlier dictionaries with keys ``sample_id``, ``flag``,
        ``metric``, ``value``, ``z_score``, and ``reason``.
    """
    if not scores:
        return []

    mean = np.mean(scores)
    std = np.std(scores)

    if std == 0:
        return []

    outliers: List[Dict[str, Any]] = []
    for sample_id, value in zip(sample_ids, scores):
        z_score = abs(value - mean) / std
        if z_score > threshold_sigma:
            outliers.append({
                'sample_id': sample_id,
                'flag': 'OUTLIER',
                'metric': metric,
                'value': value,
                'z_score': z_score,
                'reason': reason_template.format(z=z_score),
            })

    return outliers


def detect_image_outliers(dataset_dir: Path,
                          threshold_sigma: float = 3.0) -> List[Dict[str, Any]]:
    """Detect statistical outliers in image quality metrics.

    Args:
        dataset_dir: Dataset directory
        threshold_sigma: Number of standard deviations for outlier threshold

    Returns:
        List of outlier dictionaries with sample_id, metric, value, reason
    """
    dataset_dir = Path(dataset_dir)
    meta_rows = read_jsonl(dataset_dir / 'meta.jsonl', MetaRow)

    # Collect metrics for all images
    blur_scores = []
    brightness_scores = []
    sample_ids = []

    for meta_row in meta_rows:
        image_path = dataset_dir / meta_row.image_path

        if not image_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        # Blur detection using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brightness
        brightness = np.mean(gray)

        blur_scores.append(blur_score)
        brightness_scores.append(brightness)
        sample_ids.append(meta_row.id)

    outliers = _detect_metric_outliers(
        sample_ids, blur_scores, 'blur', threshold_sigma,
        'Blur score {z:.1f}\u03c3 from mean',
    )
    outliers += _detect_metric_outliers(
        sample_ids, brightness_scores, 'brightness', threshold_sigma,
        'Brightness {z:.1f}\u03c3 from mean',
    )

    return outliers


def detect_duplicate_images(dataset_dir: Path,
                            hamming_threshold: int = 5) -> List[Tuple[str, str, int]]:
    """Detect perceptually similar images using perceptual hashing.

    Args:
        dataset_dir: Dataset directory
        hamming_threshold: Maximum Hamming distance for duplicates

    Returns:
        List of tuples (id1, id2, hamming_distance)
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        _LOG.warning("imagehash not installed, skipping duplicate detection")
        return []

    dataset_dir = Path(dataset_dir)
    meta_rows = read_jsonl(dataset_dir / 'meta.jsonl', MetaRow)

    # Compute hashes
    hashes: Dict[str, Any] = {}

    for meta_row in meta_rows:
        image_path = dataset_dir / meta_row.image_path

        if not image_path.exists():
            continue

        try:
            img = Image.open(image_path)
            img_hash = imagehash.average_hash(img)
            hashes[meta_row.id] = img_hash
        except Exception:
            continue

    # Find duplicates
    duplicates = []
    sample_ids = list(hashes.keys())

    for i in range(len(sample_ids)):
        for j in range(i + 1, len(sample_ids)):
            id1 = sample_ids[i]
            id2 = sample_ids[j]

            hamming_dist = hashes[id1] - hashes[id2]

            if hamming_dist <= hamming_threshold:
                duplicates.append((id1, id2, hamming_dist))

    return duplicates


def detect_edge_cases(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Detect samples with defect parameters near tolerance boundaries.

    Args:
        dataset_dir: Dataset directory

    Returns:
        List of edge case dictionaries
    """
    dataset_dir = Path(dataset_dir)
    meta_rows = read_jsonl(dataset_dir / 'meta.jsonl', MetaRow)
    label_rows = read_jsonl(dataset_dir / 'labels.jsonl', LabelRow)

    id_to_label = {row.id: row.class_name for row in label_rows}

    edge_cases = []

    for meta_row in meta_rows:
        defect_params = meta_row.defect
        label = id_to_label.get(meta_row.id)

        shift_x = defect_params['shift_x']
        shift_y = defect_params['shift_y']
        rotation_deg = abs(defect_params['rotation_deg'])
        tilt_deg = abs(defect_params['tilt_deg'])

        shift_magnitude = np.sqrt(shift_x**2 + shift_y**2)

        # Check if near OK/MISALIGNED boundary
        if label in ['OK', 'MISALIGNED']:
            # Within 10% of threshold
            shift_margin = 0.1 * _OK_SHIFT_PX
            rot_margin = 0.1 * _OK_ROTATION_DEG

            if abs(shift_magnitude - _OK_SHIFT_PX) < shift_margin:
                edge_cases.append({
                    'sample_id': meta_row.id,
                    'flag': 'EDGE_CASE',
                    'reason': f'Shift magnitude {shift_magnitude:.2f}px near threshold {_OK_SHIFT_PX}px',
                    'label': label
                })

            if abs(rotation_deg - _OK_ROTATION_DEG) < rot_margin:
                edge_cases.append({
                    'sample_id': meta_row.id,
                    'flag': 'EDGE_CASE',
                    'reason': f'Rotation {rotation_deg:.2f}\u00b0 near threshold {_OK_ROTATION_DEG}\u00b0',
                    'label': label
                })

        # Check if near TOMBSTONE boundary
        if label == 'TOMBSTONE':
            tilt_margin = 0.1 * (90 - _TOMBSTONE_TILT_DEG)

            if abs(tilt_deg - _TOMBSTONE_TILT_DEG) < tilt_margin:
                edge_cases.append({
                    'sample_id': meta_row.id,
                    'flag': 'EDGE_CASE',
                    'reason': f'Tilt {tilt_deg:.2f}\u00b0 near threshold {_TOMBSTONE_TILT_DEG}\u00b0',
                    'label': label
                })

    return edge_cases


def run_validation_suite(dataset_dir: Path,
                         run_schema: bool = True,
                         run_outliers: bool = True,
                         run_duplicates: bool = True,
                         run_edge_cases: bool = True) -> Dict[str, Any]:
    """Run complete validation suite.

    Args:
        dataset_dir: Dataset directory
        run_schema: Run schema validation
        run_outliers: Run outlier detection
        run_duplicates: Run duplicate detection
        run_edge_cases: Run edge case detection

    Returns:
        Dictionary with validation results
    """
    results = {
        'dataset_path': str(dataset_dir),
        'schema_validation': None,
        'outliers': [],
        'duplicates': [],
        'edge_cases': []
    }

    # Schema validation (reuse existing tool)
    if run_schema:
        try:
            schema_ok = validate_dataset(dataset_dir)
            results['schema_validation'] = {
                'passed': schema_ok,
                'message': 'All schema checks passed' if schema_ok else 'Some schema checks failed'
            }
        except Exception as e:
            results['schema_validation'] = {
                'passed': False,
                'message': f'Schema validation error: {e}'
            }

    # Outlier detection
    if run_outliers:
        try:
            outliers = detect_image_outliers(dataset_dir)
            results['outliers'] = outliers
        except Exception as e:
            _LOG.warning("Outlier detection failed: %s", e)

    # Duplicate detection
    if run_duplicates:
        try:
            duplicates = detect_duplicate_images(dataset_dir)
            results['duplicates'] = duplicates
        except Exception as e:
            _LOG.warning("Duplicate detection failed: %s", e)

    # Edge case detection
    if run_edge_cases:
        try:
            edge_cases = detect_edge_cases(dataset_dir)
            results['edge_cases'] = edge_cases
        except Exception as e:
            _LOG.warning("Edge case detection failed: %s", e)

    return results
