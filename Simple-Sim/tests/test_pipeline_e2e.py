"""End-to-end pipeline integration test."""

import pytest
import tempfile
import yaml
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate import generate_dataset
from tools.validate_dataset import validate_dataset


def test_pipeline_e2e():
    """Test complete pipeline: generate -> validate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create minimal config
        config = {
            'run': {
                'run_id': 'test_run',
                'seed': 42,
                'schema_version': 1
            },
            'roi': {
                'width_px': 128,  # Smaller for speed
                'height_px': 128,
                'mm_per_px': 0.01
            },
            'classes': {
                'OK': 5,
                'MISSING': 5,
                'MISALIGNED': 5,
                'TOMBSTONE': 5
            },
            'tolerances': {
                '0603': {
                    'ok_shift_px': 3.0,
                    'ok_rotation_deg': 5.0,
                    'tombstone_tilt_deg': 75.0
                }
            },
            'domains': {
                'domain_A': {
                    'lighting_brightness': [0.9, 1.1],
                    'blur_sigma': [0.0, 1.0],
                    'noise_stddev': [0.0, 10.0]
                }
            },
            'splits': {
                'train_domains': ['domain_A'],
                'val_domains': ['domain_A'],
                'test_domains': ['domain_A'],
                'train_frac': 0.7,
                'val_frac': 0.15,
                'test_frac': 0.15
            },
            'render': {
                'backend': 'opencv_2d',
                'substrate_color': [40, 90, 40],
                'copper_color': [60, 120, 180],
                'component_color': [20, 20, 20],
                'solder_mask_alpha': 0.3
            },
            'augment': {
                'rotation_deg_range': [-5, 5],
                'brightness_factor_range': [0.85, 1.15],
                'contrast_factor_range': [0.9, 1.1]
            },
            'train': {
                'model': 'resnet18',
                'pretrained': False,
                'epochs': 1,
                'batch_size': 4,
                'num_workers': 0,
                'lr': 0.001,
                'optimizer': 'adam',
                'weight_decay': 0.0001
            },
            'eval': {
                'batch_size': 4,
                'num_workers': 0,
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
            }
        }

        # Write config
        config_path = tmpdir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Generate dataset
        output_dir = tmpdir / 'dataset'
        generate_dataset(config_path, output_dir)

        # Validate dataset
        success = validate_dataset(output_dir)
        assert success, "Dataset validation failed"

        # Check expected files exist
        assert (output_dir / 'meta.jsonl').exists()
        assert (output_dir / 'labels.jsonl').exists()
        assert (output_dir / 'config.yaml').exists()
        assert (output_dir / 'images').exists()
        assert (output_dir / 'splits' / 'train.txt').exists()
        assert (output_dir / 'splits' / 'val.txt').exists()
        assert (output_dir / 'splits' / 'test.txt').exists()

        # Check image count
        images = list((output_dir / 'images').glob('*.png'))
        assert len(images) == 20, f"Expected 20 images, got {len(images)}"


if __name__ == '__main__':
    test_pipeline_e2e()
    print("✓ E2E pipeline test passed")
