"""Configuration loading and validation."""

import yaml
from pathlib import Path
from typing import Any, Dict, List

VALID_CLASSES = {'OK', 'MISSING', 'MISALIGNED', 'TOMBSTONE', 'SOLDER_BRIDGE', 'CORNER_LIFT'}

_SUPPORTED_SCHEMA_VERSIONS = (1, 2)
_SUPPORTED_MODELS = {'resnet18'}
_SUPPORTED_OPTIMIZERS = {'adam'}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Empty config file: {path}")

    return config


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate configuration structure and values.

    Args:
        cfg: Configuration dictionary

    Raises:
        ValueError: If validation fails
    """
    schema_version, mode = _validate_run(cfg)
    required_sections = _required_sections(schema_version, mode)

    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required section: {section}")

    _validate_roi(cfg['roi'])
    _validate_classes(cfg['classes'])

    if 'tolerances' in cfg:
        validate_tolerances(cfg)
    elif schema_version == 1:
        raise ValueError("schema_version=1 requires tolerances section")

    if 'geometry_ranges' in cfg:
        _validate_geometry_ranges(cfg['geometry_ranges'])

    _validate_domains(cfg['domains'])
    validate_splits(cfg)
    _validate_render(cfg['render'], schema_version, mode)
    _validate_augment(cfg['augment'])
    _validate_train(cfg['train'])
    _validate_eval(cfg['eval'])


def validate_splits(cfg: Dict[str, Any]) -> None:
    """Validate split configuration.

    Args:
        cfg: Configuration dictionary

    Raises:
        ValueError: If validation fails
    """
    splits = cfg['splits']
    required_split_fields = ['train_domains', 'val_domains', 'test_domains',
                             'train_frac', 'val_frac', 'test_frac']
    for field in required_split_fields:
        if field not in splits:
            raise ValueError(f"Missing splits.{field}")

    available_domains = set(cfg['domains'].keys())
    for split_type in ['train', 'val', 'test']:
        domain_field = f"{split_type}_domains"
        domains = splits[domain_field]
        if not isinstance(domains, list) or not domains:
            raise ValueError(f"splits.{domain_field} must be a non-empty list")
        for domain in domains:
            if domain not in available_domains:
                raise ValueError(f"Unknown domain in splits.{domain_field}: {domain}")

    train_frac = splits['train_frac']
    val_frac = splits['val_frac']
    test_frac = splits['test_frac']

    for name, val in [('train_frac', train_frac), ('val_frac', val_frac), ('test_frac', test_frac)]:
        if not isinstance(val, (int, float)) or not 0 < val < 1:
            raise ValueError(f"splits.{name} must be in (0, 1)")

    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")


def validate_tolerances(cfg: Dict[str, Any]) -> None:
    """Validate tolerance configuration.

    Args:
        cfg: Configuration dictionary

    Raises:
        ValueError: If validation fails
    """
    tolerances = cfg['tolerances']
    if not tolerances:
        raise ValueError("tolerances cannot be empty")

    required_fields = ['ok_shift_px', 'ok_rotation_deg', 'tombstone_tilt_deg']

    for footprint, tol_cfg in tolerances.items():
        for field in required_fields:
            if field not in tol_cfg:
                raise ValueError(f"Missing tolerances.{footprint}.{field}")
            value = tol_cfg[field]
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"tolerances.{footprint}.{field} must be a non-negative number")

        if tol_cfg['ok_shift_px'] <= 0:
            raise ValueError(f"tolerances.{footprint}.ok_shift_px must be positive")
        if tol_cfg['ok_rotation_deg'] <= 0:
            raise ValueError(f"tolerances.{footprint}.ok_rotation_deg must be positive")
        if tol_cfg['tombstone_tilt_deg'] <= 0 or tol_cfg['tombstone_tilt_deg'] >= 90:
            raise ValueError(f"tolerances.{footprint}.tombstone_tilt_deg must be in (0, 90)")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_run(cfg: Dict[str, Any]):
    """Validate run section; return (schema_version, mode)."""
    if 'run' not in cfg:
        raise ValueError("Missing required section: run")

    run = cfg['run']
    for field in ('run_id', 'seed', 'schema_version'):
        if field not in run:
            raise ValueError(f"Missing run.{field}")

    if not isinstance(run['seed'], int) or run['seed'] < 0:
        raise ValueError("run.seed must be a non-negative integer")

    schema_version = run['schema_version']
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported schema version: {schema_version}")

    mode = run.get('mode', '')

    if mode == 'profile_classifier':
        if 'component_profiles' not in run:
            raise ValueError("mode=profile_classifier requires run.component_profiles (list of 2+ profile IDs)")
        profiles = run['component_profiles']
        if not isinstance(profiles, list) or len(profiles) < 2:
            raise ValueError("run.component_profiles must be a list of 2+ profile ID strings")
        for p in profiles:
            if not isinstance(p, str) or not p:
                raise ValueError("run.component_profiles entries must be non-empty strings")
    elif schema_version == 2:
        if 'component_profile' not in run:
            raise ValueError("schema_version=2 requires run.component_profile")

    return schema_version, mode


def _required_sections(schema_version: int, mode: str) -> List[str]:
    """Return the list of required top-level config sections."""
    base = ['run', 'roi', 'classes', 'domains', 'splits', 'render', 'augment', 'train', 'eval']
    if schema_version == 1 and mode != 'profile_classifier':
        return base + ['tolerances']
    return base


def _validate_roi(roi: Dict[str, Any]) -> None:
    for field in ('width_px', 'height_px', 'mm_per_px'):
        if field not in roi:
            raise ValueError(f"Missing roi.{field}")
        if not isinstance(roi[field], (int, float)) or roi[field] <= 0:
            raise ValueError(f"roi.{field} must be a positive number")


def _validate_classes(classes: Dict[str, Any]) -> None:
    if not classes:
        raise ValueError("classes cannot be empty")
    for class_name, count in classes.items():
        if class_name not in VALID_CLASSES:
            raise ValueError(f"Invalid class name: {class_name}")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"classes.{class_name} must be a positive integer")


def _validate_geometry_ranges(gr: Any) -> None:
    if not isinstance(gr, dict) or not gr:
        raise ValueError("geometry_ranges must be a non-empty mapping if provided")
    for k, v in gr.items():
        if not isinstance(k, str):
            raise ValueError("geometry_ranges keys must be strings")
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError(f"geometry_ranges.{k} must be a list of 2 numbers")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError(f"geometry_ranges.{k} entries must be numbers")
        if v[0] > v[1]:
            raise ValueError(f"geometry_ranges.{k} min > max")


def _validate_domains(domains: Dict[str, Any]) -> None:
    if not domains:
        raise ValueError("domains cannot be empty")
    required = ['lighting_brightness', 'blur_sigma', 'noise_stddev']
    for domain_name, domain_cfg in domains.items():
        for field in required:
            if field not in domain_cfg:
                raise ValueError(f"Missing domains.{domain_name}.{field}")
            if not isinstance(domain_cfg[field], list) or len(domain_cfg[field]) != 2:
                raise ValueError(f"domains.{domain_name}.{field} must be a list of 2 numbers")
            if domain_cfg[field][0] > domain_cfg[field][1]:
                raise ValueError(f"domains.{domain_name}.{field} min > max")


def _validate_render(render: Dict[str, Any], schema_version: int, mode: str) -> None:
    if 'backend' not in render:
        raise ValueError("Missing render.backend")
    backend = render['backend']

    if mode == 'profile_classifier' and backend != 'opencv_2d':
        raise ValueError("mode=profile_classifier currently supports only render.backend='opencv_2d'")

    if backend == 'opencv_2d':
        _validate_render_2d(render, schema_version)
    elif backend == 'blender_3d':
        _validate_render_3d(render)
    else:
        raise ValueError(f"Unsupported render backend: {backend}")


def _validate_render_2d(render: Dict[str, Any], schema_version: int) -> None:
    if schema_version == 1:
        required = ['backend', 'substrate_color', 'copper_color', 'component_color', 'solder_mask_alpha']
    else:
        required = ['backend', 'substrate_color', 'copper_color', 'solder_mask_alpha']

    for field in required:
        if field not in render:
            raise ValueError(f"Missing render.{field}")

    color_fields = ['substrate_color', 'copper_color']
    if schema_version == 1 or 'component_color' in render:
        color_fields.append('component_color')

    for color_field in color_fields:
        if color_field in render:
            color = render[color_field]
            if not isinstance(color, list) or len(color) != 3:
                raise ValueError(f"render.{color_field} must be a list of 3 integers (BGR)")
            if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
                raise ValueError(f"render.{color_field} values must be in [0, 255]")

    if not isinstance(render['solder_mask_alpha'], (int, float)) or not 0 <= render['solder_mask_alpha'] <= 1:
        raise ValueError("render.solder_mask_alpha must be in [0, 1]")


def _validate_render_3d(render: Dict[str, Any]) -> None:
    blender = render.get('blender')
    if not isinstance(blender, dict):
        raise ValueError("render.blender must be a mapping for backend=blender_3d")

    samples = blender.get('samples')
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError("render.blender.samples must be a positive integer")

    exe = blender.get('executable', 'blender')
    if not isinstance(exe, str) or not exe.strip():
        raise ValueError("render.blender.executable must be a non-empty string if provided")

    engine = blender.get('engine', 'CYCLES')
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError("render.blender.engine must be a non-empty string if provided")


def _validate_augment(augment: Dict[str, Any]) -> None:
    required = ['rotation_deg_range', 'brightness_factor_range', 'contrast_factor_range']
    for field in required:
        if field not in augment:
            raise ValueError(f"Missing augment.{field}")
        if not isinstance(augment[field], list) or len(augment[field]) != 2:
            raise ValueError(f"augment.{field} must be a list of 2 numbers")
        if augment[field][0] > augment[field][1]:
            raise ValueError(f"augment.{field} min > max")


def _validate_train(train: Dict[str, Any]) -> None:
    required = ['model', 'epochs', 'batch_size', 'lr', 'optimizer', 'weight_decay']
    for field in required:
        if field not in train:
            raise ValueError(f"Missing train.{field}")

    if train['model'] not in _SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {train['model']}")
    if train['optimizer'] not in _SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {train['optimizer']}")
    if 'pretrained' in train and not isinstance(train['pretrained'], bool):
        raise ValueError("train.pretrained must be a boolean if provided")
    if 'num_workers' in train:
        if not isinstance(train['num_workers'], int) or train['num_workers'] < 0:
            raise ValueError("train.num_workers must be an integer >= 0 if provided")
    if not isinstance(train['epochs'], int) or train['epochs'] <= 0:
        raise ValueError("train.epochs must be a positive integer")
    if not isinstance(train['batch_size'], int) or train['batch_size'] <= 0:
        raise ValueError("train.batch_size must be a positive integer")
    if not isinstance(train['lr'], (int, float)) or train['lr'] <= 0:
        raise ValueError("train.lr must be a positive number")


def _validate_eval(eval_cfg: Dict[str, Any]) -> None:
    if 'batch_size' not in eval_cfg:
        raise ValueError("Missing eval.batch_size")
    if not isinstance(eval_cfg['batch_size'], int) or eval_cfg['batch_size'] <= 0:
        raise ValueError("eval.batch_size must be a positive integer")
    if 'num_workers' in eval_cfg:
        if not isinstance(eval_cfg['num_workers'], int) or eval_cfg['num_workers'] < 0:
            raise ValueError("eval.num_workers must be an integer >= 0 if provided")
    if 'critical_classes' in eval_cfg:
        cc = eval_cfg['critical_classes']
        if not isinstance(cc, list) or not cc:
            raise ValueError("eval.critical_classes must be a non-empty list if provided")
        for c in cc:
            if not isinstance(c, str):
                raise ValueError("eval.critical_classes entries must be strings")
            if c not in VALID_CLASSES:
                raise ValueError(f"Invalid class name in eval.critical_classes: {c}")
