"""Configuration loading and validation."""

import yaml
from pathlib import Path
from typing import Any, Dict

VALID_CLASSES = {'OK', 'MISSING', 'MISALIGNED', 'TOMBSTONE', 'SOLDER_BRIDGE', 'CORNER_LIFT'}


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
    # Validate run section first to determine schema version
    if 'run' not in cfg:
        raise ValueError("Missing required section: run")

    run = cfg['run']
    if 'run_id' not in run:
        raise ValueError("Missing run.run_id")
    if 'seed' not in run:
        raise ValueError("Missing run.seed")
    if not isinstance(run['seed'], int) or run['seed'] < 0:
        raise ValueError("run.seed must be a non-negative integer")
    if 'schema_version' not in run:
        raise ValueError("Missing run.schema_version")

    schema_version = run['schema_version']
    if schema_version not in [1, 2]:
        raise ValueError(f"Unsupported schema version: {schema_version}")

    # Schema version-specific validation
    mode = run.get('mode', '')

    if mode == 'profile_classifier':
        # Profile classifier mode: requires list of 2+ profiles, no singular component_profile needed
        if 'component_profiles' not in run:
            raise ValueError("mode=profile_classifier requires run.component_profiles (list of 2+ profile IDs)")
        profiles = run['component_profiles']
        if not isinstance(profiles, list) or len(profiles) < 2:
            raise ValueError("run.component_profiles must be a list of 2+ profile ID strings")
        for p in profiles:
            if not isinstance(p, str) or not p:
                raise ValueError("run.component_profiles entries must be non-empty strings")
        required_sections = ['run', 'roi', 'classes', 'domains', 'splits', 'render', 'augment', 'train', 'eval']
    elif schema_version == 1:
        # v1: tolerances and geometry_ranges required, component_profile optional
        required_sections = ['run', 'roi', 'classes', 'tolerances', 'domains', 'splits', 'render', 'augment', 'train', 'eval']
    elif schema_version == 2:
        # v2: component_profile required, tolerances and geometry_ranges optional (come from profile)
        required_sections = ['run', 'roi', 'classes', 'domains', 'splits', 'render', 'augment', 'train', 'eval']
        if 'component_profile' not in run:
            raise ValueError("schema_version=2 requires run.component_profile")

    # Check required top-level sections
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required section: {section}")

    # Validate ROI section
    roi = cfg['roi']
    required_roi = ['width_px', 'height_px', 'mm_per_px']
    for field in required_roi:
        if field not in roi:
            raise ValueError(f"Missing roi.{field}")
        if not isinstance(roi[field], (int, float)) or roi[field] <= 0:
            raise ValueError(f"roi.{field} must be a positive number")

    # Validate classes
    classes = cfg['classes']
    if not classes:
        raise ValueError("classes cannot be empty")
    valid_classes = VALID_CLASSES
    for class_name, count in classes.items():
        if class_name not in valid_classes:
            raise ValueError(f"Invalid class name: {class_name}")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"classes.{class_name} must be a positive integer")

    # Validate tolerances (required for v1, optional for v2)
    if 'tolerances' in cfg:
        validate_tolerances(cfg)
    elif schema_version == 1:
        raise ValueError("schema_version=1 requires tolerances section")

    # Validate optional geometry ranges for nominal sampling
    if 'geometry_ranges' in cfg:
        gr = cfg['geometry_ranges']
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

    # Validate domains
    domains = cfg['domains']
    if not domains:
        raise ValueError("domains cannot be empty")
    for domain_name, domain_cfg in domains.items():
        required_domain_fields = ['lighting_brightness', 'blur_sigma', 'noise_stddev']
        for field in required_domain_fields:
            if field not in domain_cfg:
                raise ValueError(f"Missing domains.{domain_name}.{field}")
            if not isinstance(domain_cfg[field], list) or len(domain_cfg[field]) != 2:
                raise ValueError(f"domains.{domain_name}.{field} must be a list of 2 numbers")
            if domain_cfg[field][0] > domain_cfg[field][1]:
                raise ValueError(f"domains.{domain_name}.{field} min > max")

    # Validate splits
    validate_splits(cfg)

    # Validate render section
    render = cfg['render']
    if 'backend' not in render:
        raise ValueError("Missing render.backend")
    backend = render.get('backend')

    if mode == 'profile_classifier':
        # Keep this mode simple for now; mixed-profile dataset generation is currently 2D-only.
        if backend != 'opencv_2d':
            raise ValueError("mode=profile_classifier currently supports only render.backend='opencv_2d'")

    if backend == 'opencv_2d':
        # component_color is optional for schema v2 (comes from profile)
        if schema_version == 1:
            required_render = ['backend', 'substrate_color', 'copper_color', 'component_color', 'solder_mask_alpha']
        else:
            required_render = ['backend', 'substrate_color', 'copper_color', 'solder_mask_alpha']

        for field in required_render:
            if field not in render:
                raise ValueError(f"Missing render.{field}")

        # Validate color fields (component_color optional for v2)
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

    elif backend == 'blender_3d':
        # 3D renderer config: keep required fields minimal; the generator will fail fast if Blender is missing.
        blender = render.get('blender')
        if not isinstance(blender, dict):
            raise ValueError("render.blender must be a mapping for backend=blender_3d")

        # Optional (defaults allowed): executable, engine. Required: samples.
        samples = blender.get('samples')
        if not isinstance(samples, int) or samples <= 0:
            raise ValueError("render.blender.samples must be a positive integer")

        exe = blender.get('executable', 'blender')
        if not isinstance(exe, str) or not exe.strip():
            raise ValueError("render.blender.executable must be a non-empty string if provided")

        engine = blender.get('engine', 'CYCLES')
        if not isinstance(engine, str) or not engine.strip():
            raise ValueError("render.blender.engine must be a non-empty string if provided")

    else:
        raise ValueError(f"Unsupported render backend: {backend}")

    # Validate augment section
    augment = cfg['augment']
    required_augment = ['rotation_deg_range', 'brightness_factor_range', 'contrast_factor_range']
    for field in required_augment:
        if field not in augment:
            raise ValueError(f"Missing augment.{field}")
        if not isinstance(augment[field], list) or len(augment[field]) != 2:
            raise ValueError(f"augment.{field} must be a list of 2 numbers")
        if augment[field][0] > augment[field][1]:
            raise ValueError(f"augment.{field} min > max")

    # Validate train section
    train = cfg['train']
    required_train = ['model', 'epochs', 'batch_size', 'lr', 'optimizer', 'weight_decay']
    for field in required_train:
        if field not in train:
            raise ValueError(f"Missing train.{field}")
    if train['model'] != 'resnet18':
        raise ValueError(f"Unsupported model: {train['model']}")
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
    if train['optimizer'] != 'adam':
        raise ValueError(f"Unsupported optimizer: {train['optimizer']}")

    # Validate eval section
    eval_cfg = cfg['eval']
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
        # Ensure they're valid class names.
        valid_classes = VALID_CLASSES
        for c in cc:
            if not isinstance(c, str):
                raise ValueError("eval.critical_classes entries must be strings")
            if c not in valid_classes:
                raise ValueError(f"Invalid class name in eval.critical_classes: {c}")


def validate_splits(cfg: Dict[str, Any]) -> None:
    """Validate split configuration.

    Args:
        cfg: Configuration dictionary

    Raises:
        ValueError: If validation fails
    """
    splits = cfg['splits']
    required_split_fields = ['train_domains', 'val_domains', 'test_domains', 'train_frac', 'val_frac', 'test_frac']
    for field in required_split_fields:
        if field not in splits:
            raise ValueError(f"Missing splits.{field}")

    # Check domain lists
    available_domains = set(cfg['domains'].keys())
    for split_type in ['train', 'val', 'test']:
        domain_field = f"{split_type}_domains"
        domains = splits[domain_field]
        if not isinstance(domains, list) or not domains:
            raise ValueError(f"splits.{domain_field} must be a non-empty list")
        for domain in domains:
            if domain not in available_domains:
                raise ValueError(f"Unknown domain in splits.{domain_field}: {domain}")

    # Check fractions sum to 1.0
    train_frac = splits['train_frac']
    val_frac = splits['val_frac']
    test_frac = splits['test_frac']

    for frac_name, frac_val in [('train_frac', train_frac), ('val_frac', val_frac), ('test_frac', test_frac)]:
        if not isinstance(frac_val, (int, float)) or not 0 < frac_val < 1:
            raise ValueError(f"splits.{frac_name} must be in (0, 1)")

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

    for footprint, tol_cfg in tolerances.items():
        required_tol_fields = ['ok_shift_px', 'ok_rotation_deg', 'tombstone_tilt_deg']
        for field in required_tol_fields:
            if field not in tol_cfg:
                raise ValueError(f"Missing tolerances.{footprint}.{field}")
            value = tol_cfg[field]
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"tolerances.{footprint}.{field} must be a non-negative number")

        # Check logical constraints
        if tol_cfg['ok_shift_px'] <= 0:
            raise ValueError(f"tolerances.{footprint}.ok_shift_px must be positive")
        if tol_cfg['ok_rotation_deg'] <= 0:
            raise ValueError(f"tolerances.{footprint}.ok_rotation_deg must be positive")
        if tol_cfg['tombstone_tilt_deg'] <= 0 or tol_cfg['tombstone_tilt_deg'] >= 90:
            raise ValueError(f"tolerances.{footprint}.tombstone_tilt_deg must be in (0, 90)")
