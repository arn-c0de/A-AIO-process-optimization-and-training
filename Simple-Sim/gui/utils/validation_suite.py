"""Extended dataset validation with outlier and duplicate detection."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simple_sim.schema import read_jsonl, MetaRow, LabelRow
from tools.validate_dataset import validate_dataset


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

    # Detect outliers
    outliers = []

    if len(blur_scores) > 0:
        blur_mean = np.mean(blur_scores)
        blur_std = np.std(blur_scores)

        for i, (sample_id, blur_score) in enumerate(zip(sample_ids, blur_scores)):
            if blur_std > 0:
                z_score = abs(blur_score - blur_mean) / blur_std
                if z_score > threshold_sigma:
                    outliers.append({
                        'sample_id': sample_id,
                        'flag': 'OUTLIER',
                        'metric': 'blur',
                        'value': blur_score,
                        'z_score': z_score,
                        'reason': f'Blur score {z_score:.1f}σ from mean'
                    })

    if len(brightness_scores) > 0:
        bright_mean = np.mean(brightness_scores)
        bright_std = np.std(brightness_scores)

        for i, (sample_id, brightness) in enumerate(zip(sample_ids, brightness_scores)):
            if bright_std > 0:
                z_score = abs(brightness - bright_mean) / bright_std
                if z_score > threshold_sigma:
                    outliers.append({
                        'sample_id': sample_id,
                        'flag': 'OUTLIER',
                        'metric': 'brightness',
                        'value': brightness,
                        'z_score': z_score,
                        'reason': f'Brightness {z_score:.1f}σ from mean'
                    })

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
        print("Warning: imagehash not installed, skipping duplicate detection")
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

    # Typical tolerances (should be read from config, but using defaults here)
    ok_shift_px = 3.0
    ok_rotation_deg = 5.0
    tombstone_tilt_deg = 75.0

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
            shift_margin = 0.1 * ok_shift_px
            rot_margin = 0.1 * ok_rotation_deg

            if abs(shift_magnitude - ok_shift_px) < shift_margin:
                edge_cases.append({
                    'sample_id': meta_row.id,
                    'flag': 'EDGE_CASE',
                    'reason': f'Shift magnitude {shift_magnitude:.2f}px near threshold {ok_shift_px}px',
                    'label': label
                })

            if abs(rotation_deg - ok_rotation_deg) < rot_margin:
                edge_cases.append({
                    'sample_id': meta_row.id,
                    'flag': 'EDGE_CASE',
                    'reason': f'Rotation {rotation_deg:.2f}° near threshold {ok_rotation_deg}°',
                    'label': label
                })

        # Check if near TOMBSTONE boundary
        if label == 'TOMBSTONE':
            tilt_margin = 0.1 * (90 - tombstone_tilt_deg)

            if abs(tilt_deg - tombstone_tilt_deg) < tilt_margin:
                edge_cases.append({
                    'sample_id': meta_row.id,
                    'flag': 'EDGE_CASE',
                    'reason': f'Tilt {tilt_deg:.2f}° near threshold {tombstone_tilt_deg}°',
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
            print(f"Outlier detection failed: {e}")

    # Duplicate detection
    if run_duplicates:
        try:
            duplicates = detect_duplicate_images(dataset_dir)
            results['duplicates'] = duplicates
        except Exception as e:
            print(f"Duplicate detection failed: {e}")

    # Edge case detection
    if run_edge_cases:
        try:
            edge_cases = detect_edge_cases(dataset_dir)
            results['edge_cases'] = edge_cases
        except Exception as e:
            print(f"Edge case detection failed: {e}")

    return results
