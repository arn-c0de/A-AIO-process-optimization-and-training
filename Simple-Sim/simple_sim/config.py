"""Configuration loading and validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


from simple_sim.classes import VALID_CLASSES  # noqa: F401  (re-exported for existing importers)
from simple_sim.generators.backends import Backend, backend_of, parse_backend, supporting
from simple_sim.generators.options import LOOKS, POST_FILTER_MODES, SOLDER_STYLES, SUPERSAMPLING_CHOICES, validate_camera_config
from simple_sim.io_utils import load_yaml

log = logging.getLogger(__name__)

RENDER_STYLES = {"photoreal", "flat"}
DEFAULT_COMPONENT_PROFILE = "chip_0603_resistor@2"


def merge_profile_render(render_cfg: dict[str, Any], profile: dict[str, Any],
                         roi: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render config with the profile's defaults filled in (schema v2: colour/type come from the profile).

    With *roi* (the run config's ``roi`` section) the ROI scale ``mm_per_px`` is passed along too:
    the board model (``render.board``) works in mm, profiles are in pixels at that scale.
    """
    out = dict(render_cfg)
    prof_render = profile.get("render") or {}
    if "component_color" not in out and "component_color_bgr" in prof_render:
        out["component_color"] = list(prof_render["component_color_bgr"])
    out.setdefault("component_type", str((profile.get("component") or {}).get("type", "")))
    if "marking_probability" in prof_render:
        out.setdefault("marking_probability", float(prof_render["marking_probability"]))
    if roi and roi.get("mm_per_px"):
        out["mm_per_px"] = float(roi["mm_per_px"])
    return out

def core_section_values(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Primitive casts of ``run`` / ``roi`` / ``train`` / ``eval`` (plan A4).

    The first check of :func:`validate_config`, formerly ``config_schema.parse_config_typed``
    (whose typed result nothing used). Raises ``ValueError("Missing required top-level section:
    ...")`` or ``ValueError("Invalid value in config section '<name>': ...")``; returns the cast
    values per section (the fields of the deprecated ``config_schema`` dataclasses).
    """
    try:
        run_raw = cfg["run"]
        roi_raw = cfg["roi"]
        train_raw = cfg["train"]
        eval_raw = cfg["eval"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing required top-level section: {exc}") from exc

    section = "run"
    try:
        run = dict(
            run_id=str(run_raw.get("run_id", "")),
            seed=int(run_raw.get("seed", -1)),
            schema_version=int(run_raw.get("schema_version", -1)),
            mode=str(run_raw.get("mode", "")),
        )
        section = "roi"
        roi = dict(
            width_px=float(roi_raw.get("width_px", 0)),
            height_px=float(roi_raw.get("height_px", 0)),
            mm_per_px=float(roi_raw.get("mm_per_px", 0)),
        )
        section = "train"
        train = dict(
            model=str(train_raw.get("model", "")),
            epochs=int(train_raw.get("epochs", 0)),
            batch_size=int(train_raw.get("batch_size", 0)),
            lr=float(train_raw.get("lr", 0.0)),
            optimizer=str(train_raw.get("optimizer", "")),
            weight_decay=float(train_raw.get("weight_decay", 0.0)),
            pretrained=train_raw.get("pretrained", None),
            num_workers=train_raw.get("num_workers", None),
        )
        section = "eval"
        eval_cfg = dict(
            batch_size=int(eval_raw.get("batch_size", 0)),
            num_workers=eval_raw.get("num_workers", None),
        )
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid value in config section '{section}': {exc}") from exc
    return {"run": run, "roi": roi, "train": train, "eval": eval_cfg}


_SUPPORTED_SCHEMA_VERSIONS = (1, 2)
_SUPPORTED_MODELS = {'resnet18'}
_SUPPORTED_OPTIMIZERS = {'adam'}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = load_yaml(path)

    if config is None:
        raise ValueError(f"Empty config file: {path}")

    return config


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate configuration structure and values.

    Args:
        cfg: Configuration dictionary

    Raises:
        ValueError: If validation fails
    """
    # Primitive casts of the core sections first (their error messages predate the validators below).
    core_section_values(cfg)
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

    _validate_domains(cfg['domains'], cfg['render'])
    validate_splits(cfg)
    _validate_render(cfg['render'], schema_version, mode)
    _validate_augment(cfg['augment'])
    _validate_defects(cfg.get('defects'))
    _validate_train(cfg['train'])
    _validate_eval(cfg['eval'])


def validate_splits(cfg: dict[str, Any]) -> None:
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


def validate_tolerances(cfg: dict[str, Any]) -> None:
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

def _validate_run(cfg: dict[str, Any]) -> tuple[int, str]:
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


def _required_sections(schema_version: int, mode: str) -> list[str]:
    """Return the list of required top-level config sections."""
    base = ['run', 'roi', 'classes', 'domains', 'splits', 'render', 'augment', 'train', 'eval']
    if schema_version == 1 and mode != 'profile_classifier':
        return base + ['tolerances']
    return base


def _validate_roi(roi: dict[str, Any]) -> None:
    for field in ('width_px', 'height_px', 'mm_per_px'):
        if field not in roi:
            raise ValueError(f"Missing roi.{field}")
        if not isinstance(roi[field], (int, float)) or roi[field] <= 0:
            raise ValueError(f"roi.{field} must be a positive number")


def _validate_classes(classes: dict[str, Any]) -> None:
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


def _validate_domains(domains: dict[str, Any], render: dict[str, Any] | None = None) -> None:
    """``domains``: samples are spread evenly over all domains (``index % n``); each domain has its
    own augmentation ranges and, for the 2D backend, an optional ``camera`` override of ``render.camera``."""
    if not domains:
        raise ValueError("domains cannot be empty")
    required = ['lighting_brightness', 'blur_sigma', 'noise_stddev']
    for domain_name, domain_cfg in domains.items():
        if not isinstance(domain_cfg, dict):
            raise ValueError(f"domains.{domain_name} must be a mapping")
        for field in required:
            if field not in domain_cfg:
                raise ValueError(f"Missing domains.{domain_name}.{field}")
            if not isinstance(domain_cfg[field], list) or len(domain_cfg[field]) != 2:
                raise ValueError(f"domains.{domain_name}.{field} must be a list of 2 numbers")
            if domain_cfg[field][0] > domain_cfg[field][1]:
                raise ValueError(f"domains.{domain_name}.{field} min > max")
        if 'camera' in domain_cfg:
            if render is not None and not backend_of(render).domain_camera:
                raise ValueError(f"domains.{domain_name}.camera is only supported by render.backend: "
                                 f"{' | '.join(supporting('domain_camera'))}")
            try:
                _validate_camera(domain_cfg['camera'])
            except ValueError as exc:
                raise ValueError(str(exc).replace("render.camera", f"domains.{domain_name}.camera")) from exc


def domain_camera(render: dict[str, Any], domain_cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    """Effective ``render.camera`` for a sample of a domain: the domain's ``camera`` keys override the
    render section's (a ``preset`` in the domain replaces the render preset)."""
    base = render.get("camera")
    override = (domain_cfg or {}).get("camera")
    if not override:
        return base
    return {**(base or {}), **override}


def _validate_defects(defects: Any) -> None:
    """``defects`` (optional): how defect parameters are drawn (``placement_sampling``, ``units``)."""
    if defects is None:
        return
    from simple_sim.defects import DEFECT_UNITS, PLACEMENT_SAMPLING

    if not isinstance(defects, dict) or set(defects) - {'placement_sampling', 'units'}:
        raise ValueError("defects must be a mapping with keys from ['placement_sampling', 'units']")
    placement = defects.get('placement_sampling')
    if placement is not None and placement not in PLACEMENT_SAMPLING:
        raise ValueError(f"defects.placement_sampling must be one of {list(PLACEMENT_SAMPLING)}")
    units = defects.get('units')
    if units is not None and units not in DEFECT_UNITS:
        raise ValueError(f"defects.units must be one of {list(DEFECT_UNITS)}")


def placement_sampling(cfg: dict[str, Any]) -> str:
    """``defects.placement_sampling`` of a run config (default ``legacy``)."""
    from simple_sim.defects import DEFAULT_PLACEMENT_SAMPLING

    return str((cfg.get('defects') or {}).get('placement_sampling') or DEFAULT_PLACEMENT_SAMPLING)


def defect_units(cfg: dict[str, Any]) -> str:
    """``defects.units`` of a run config (default ``px``)."""
    from simple_sim.defects import DEFAULT_DEFECT_UNITS

    return str((cfg.get('defects') or {}).get('units') or DEFAULT_DEFECT_UNITS)


def _validate_render(render: dict[str, Any], schema_version: int, mode: str) -> None:
    if 'backend' not in render:
        raise ValueError("Missing render.backend")
    backend = parse_backend(render['backend'])

    if mode == 'profile_classifier' and not backend_of(render).profile_classifier:
        raise ValueError("mode=profile_classifier currently supports only render.backend="
                         + " | ".join(repr(b) for b in supporting('profile_classifier')))

    if backend is Backend.OPENCV_2D:
        _validate_render_2d(render, schema_version)
    else:
        _validate_render_3d(render)
    _validate_camera(render.get('camera'))
    _validate_supersampling(render.get('supersampling'))
    _validate_solder(render.get('solder'))
    _validate_look(render.get('look'))
    _validate_post_filters(render)
    _validate_board(render.get('board'))
    _validate_board_look(render)


def _is_probability(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0 <= value <= 1


def _validate_board_look(render: dict[str, Any]) -> None:
    """Photoreal board settings shared by both backends (3D bakes the same board, see ``bake_fields``)."""
    for key in ('rework_artifacts', 'mask_gloss'):
        value = render.get(key)
        if value is not None and not _is_probability(value):
            raise ValueError(f"render.{key} must be a probability in [0, 1]")

    texture_dir = render.get('texture_dir')
    if texture_dir is not None:
        from simple_sim.images import is_image

        if not isinstance(texture_dir, str) or not Path(texture_dir).is_dir():
            raise ValueError(f"render.texture_dir must be an existing directory: {texture_dir!r}")
        if not any(is_image(p) for p in Path(texture_dir).iterdir()):  # = photoreal texture_files() non-empty
            raise ValueError(f"render.texture_dir contains no images: {texture_dir!r}")


def _validate_board(board: Any) -> None:
    """``render.board`` (both backends): router, design-rule preset, pour probability, DRC mode."""
    from simple_sim.board import validate_board_options

    validate_board_options(board)


def _validate_supersampling(factor: Any) -> None:
    """``render.supersampling`` (both backends; 3D: the baked board): photoreal render scale, default 2."""
    if factor is None:
        return
    if isinstance(factor, bool) or factor not in SUPERSAMPLING_CHOICES:
        raise ValueError(f"render.supersampling must be one of {list(SUPERSAMPLING_CHOICES)}")


def _validate_solder(style: Any) -> None:
    """``render.solder`` (both backends; 3D: the baked neighbour joints): solder joint model, default legacy."""
    if style is None:
        return
    if style not in SOLDER_STYLES:
        raise ValueError(f"render.solder must be one of {list(SOLDER_STYLES)}")


def _validate_look(look: Any) -> None:
    """``render.look`` (both backends; 3D: the baked board and the camera): target look, default legacy."""
    if look is None:
        return
    if look not in LOOKS:
        raise ValueError(f"render.look must be one of {list(LOOKS)}")


def _validate_post_filters(render: dict[str, Any]) -> None:
    """``render.post_filters`` (2D photoreal): where the domain blur/noise and the rotation are applied."""
    mode = render.get('post_filters')
    if mode is None:
        return
    if mode not in POST_FILTER_MODES:
        raise ValueError(f"render.post_filters must be one of {list(POST_FILTER_MODES)}")
    if mode == 'camera' and (not backend_of(render).post_filters_camera or render.get('style', 'photoreal') != 'photoreal'):
        raise ValueError("render.post_filters: camera requires render.backend: opencv_2d with render.style: photoreal")


def _validate_camera(camera: Any) -> None:
    """``render.camera`` (both backends): preset and overrides of the camera model (photoreal.CAMERA_DEFAULTS)."""
    if camera is None:
        return
    validate_camera_config(camera)


def _validate_render_2d(render: dict[str, Any], schema_version: int) -> None:
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

    if render.get('style', 'photoreal') not in RENDER_STYLES:
        raise ValueError(f"render.style must be one of {sorted(RENDER_STYLES)}")

    if render.get('style', 'photoreal') == 'flat':
        ignored = [k for k in ('board', 'supersampling', 'camera', 'mask_gloss', 'rework_artifacts', 'texture_dir', 'solder',
                                'look')
                   if render.get(k) is not None]
        if ignored:
            log.warning("render.style=flat ignores render.%s (photoreal-only settings)", ", render.".join(ignored))


def _validate_render_3d(render: dict[str, Any]) -> None:
    blender = render.get('blender')
    if not isinstance(blender, dict):
        raise ValueError("render.blender must be a mapping for backend=blender_3d")

    samples = blender.get('samples')
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError("render.blender.samples must be a positive integer")

    exe = blender.get('executable', 'blender')
    if not isinstance(exe, str) or not exe.strip():
        raise ValueError("render.blender.executable must be a non-empty string if provided")

    engine = blender.get('engine', 'BLENDER_EEVEE')
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError("render.blender.engine must be a non-empty string if provided")


def _validate_augment(augment: dict[str, Any]) -> None:
    required = ['rotation_deg_range', 'contrast_factor_range']
    # Brightness is sampled from domains.*.lighting_brightness; brightness_factor_range
    # is legacy/unused by the samplers and only validated when present.
    optional = ['brightness_factor_range']
    for field in required + optional:
        if field not in augment:
            if field in optional:
                continue
            raise ValueError(f"Missing augment.{field}")
        if not isinstance(augment[field], list) or len(augment[field]) != 2:
            raise ValueError(f"augment.{field} must be a list of 2 numbers")
        if augment[field][0] > augment[field][1]:
            raise ValueError(f"augment.{field} min > max")


def _validate_train(train: dict[str, Any]) -> None:
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
    _validate_train_extras(train)


_TRAIN_SCHEDULERS = ('none', 'cosine')
_TRAIN_FREEZE_STAGES = ('stem', 'layer1', 'layer2', 'layer3', 'layer4')
_TRAIN_AUGMENT_FLAGS = ('hflip', 'vflip', 'rot90')
_TRAIN_AUGMENT_JITTER = ('brightness', 'contrast', 'saturation', 'hue')


def _validate_train_extras(train: dict[str, Any]) -> None:
    """Optional training knobs: ``scheduler``, ``warmup_epochs``, ``label_smoothing``, ``augment``,
    ``freeze`` (ResNet stages without gradients), ``cache_images`` (decoded images kept in RAM)."""
    if 'scheduler' in train and train['scheduler'] not in _TRAIN_SCHEDULERS:
        raise ValueError(f"train.scheduler must be one of {list(_TRAIN_SCHEDULERS)}")
    if 'cache_images' in train and not isinstance(train['cache_images'], bool):
        raise ValueError("train.cache_images must be a boolean")
    if 'freeze' in train and train['freeze'] is not None:
        freeze = train['freeze']
        if not isinstance(freeze, list) or any(f not in _TRAIN_FREEZE_STAGES for f in freeze):
            raise ValueError(f"train.freeze must be a list of stages from {list(_TRAIN_FREEZE_STAGES)}")
    if 'warmup_epochs' in train:
        if isinstance(train['warmup_epochs'], bool) or not isinstance(train['warmup_epochs'], int) or train['warmup_epochs'] < 0:
            raise ValueError("train.warmup_epochs must be an integer >= 0")
        if train['warmup_epochs'] >= train['epochs']:
            raise ValueError("train.warmup_epochs must be smaller than train.epochs")
    if 'label_smoothing' in train:
        ls = train['label_smoothing']
        if isinstance(ls, bool) or not isinstance(ls, (int, float)) or not 0 <= ls < 1:
            raise ValueError("train.label_smoothing must be a number in [0, 1)")
    if 'augment' in train:
        aug = train['augment']
        if aug is None:
            return
        if not isinstance(aug, dict) or set(aug) - set(_TRAIN_AUGMENT_FLAGS) - set(_TRAIN_AUGMENT_JITTER):
            raise ValueError(f"train.augment must be a mapping with keys from {list(_TRAIN_AUGMENT_FLAGS + _TRAIN_AUGMENT_JITTER)}")
        for key in _TRAIN_AUGMENT_FLAGS:
            if key in aug and not isinstance(aug[key], bool):
                raise ValueError(f"train.augment.{key} must be a boolean")
        for key in _TRAIN_AUGMENT_JITTER:
            if key in aug:
                v = aug[key]
                limit = 0.5 if key == 'hue' else 1.0
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= limit:
                    raise ValueError(f"train.augment.{key} must be a number in [0, {limit}]")


def _validate_eval(eval_cfg: dict[str, Any]) -> None:
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
