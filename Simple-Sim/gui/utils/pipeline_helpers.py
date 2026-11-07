"""Pure (tk-free) helpers behind the Pipeline tab: run targets, precise counts, command lines.

Kept separate from ``gui/tabs/pipeline`` so they can be reused and tested without tkinter.
"""

from __future__ import annotations

import json
import shlex
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from gui.utils.filter_profile_store import FILTER_SETTING_SPECS
from gui.utils.paths import model_root
from simple_sim.board import BOARD_ROUTERS, DRC_MODES
from simple_sim.board.rules import PRESETS as DESIGN_RULE_PRESETS
from simple_sim.classes import STANDARD_CLASSES
from simple_sim.generators.options import LOOKS
from simple_sim.io_utils import load_yaml

PYTHON = "./.venv/bin/python"

# (SettingsStore key, PipelineUI variable attribute) pairs persisted by the Pipeline tab.
PERSISTED_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("pipeline.run_mode", "var_run_mode"),
    ("pipeline.run_count", "var_run_count"),
    ("pipeline.dataset_mode", "var_dataset_mode"),
    ("pipeline.config", "var_config"),
    ("pipeline.out", "var_out"),
    ("pipeline.model", "var_model"),
    ("pipeline.autosnap", "var_autosnap"),
    ("pipeline.snap_every", "var_snap_every"),
    ("pipeline.snap_keep", "var_snap_keep"),
    ("pipeline.continue_epochs", "var_continue_epochs"),
    ("pipeline.continue_out_mode", "var_continue_out_mode"),
    ("pipeline.name", "var_name"),
    ("pipeline.name_ts", "var_name_ts"),
    ("pipeline.dataset_selection", "var_dataset"),
    ("pipeline.dataset_multi_enabled", "var_dataset_multi"),
    ("pipeline.dataset_multi_paths", "var_dataset_multi_paths"),
    ("pipeline.profile_model", "var_profile_model"),
    ("pipeline.profile_model_locked", "var_profile_model_lock"),
    ("pipeline.profile_build_preset", "var_profile_build_preset"),
    ("pipeline.render_backend", "var_render_backend"),
    ("pipeline.look", "var_look"),
    ("pipeline.board_routing", "var_board_routing"),
    ("pipeline.board_drc", "var_board_drc"),
    ("pipeline.board_rules", "var_board_rules"),
) + tuple((f"pipeline.filter.{key}", f"var_filter_{key}") for key, _kind, _default in FILTER_SETTING_SPECS)

# Render options the GUI can override on top of a run config (``render.look``, ``render.board.*``).
# Every selector starts at FROM_CONFIG: the config's value (or the renderer default) is kept.
FROM_CONFIG = "from config"
LOOK_CHOICES: tuple[str, ...] = (FROM_CONFIG, *LOOKS)
BOARD_ROUTING_CHOICES: tuple[str, ...] = (FROM_CONFIG, *BOARD_ROUTERS)
BOARD_DRC_CHOICES: tuple[str, ...] = (FROM_CONFIG, *DRC_MODES)
BOARD_RULES_CHOICES: tuple[str, ...] = (FROM_CONFIG, *DESIGN_RULE_PRESETS)

# Persisted keys that selecting a dataset overwrites (``_apply_dataset_settings``); restored again afterwards.
DATASET_DERIVED_KEYS: frozenset[str] = frozenset(
    {"pipeline.config", "pipeline.out", "pipeline.model", "pipeline.name", "pipeline.dataset_mode"}
)
DEFAULT_CLASS_NAMES: tuple[str, ...] = STANDARD_CLASSES


def resolve_under(sim_root: Path, s: str) -> Path:
    """*s* as a path, relative paths taken relative to *sim_root* (not resolved)."""
    return Path(s) if Path(s).is_absolute() else Path(sim_root) / s


def config_class_names(cfg_path: Path | None) -> list[str]:
    """Class names (``classes`` keys) of a run config, or the default four classes."""
    if cfg_path is not None:
        try:
            data = load_yaml(Path(cfg_path)) or {}
            names = list((data.get("classes") or {}).keys())
            if names:
                return names
        except Exception:
            pass
    return list(DEFAULT_CLASS_NAMES)


def write_temp_yaml(data: Any, out_dir: Path, *, prefix: str) -> Path:
    """Dump *data* to a new ``<prefix>*.yaml`` file in *out_dir* (created) and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", prefix=prefix, dir=str(out_dir), delete=False) as tf:
        yaml.safe_dump(data, tf, sort_keys=False)
        return Path(tf.name)


# ---------------------------------------------------------------------------
# Precise mode
# ---------------------------------------------------------------------------

def _non_negative_counts(cls_dict: dict[str, Any]) -> dict[str, int] | str:
    """``{cls: int}`` or the first class name whose count is not a non-negative int."""
    counts: dict[str, int] = {}
    for cls, val in cls_dict.items():
        try:
            n = int(val)
        except (TypeError, ValueError, OverflowError):
            return cls
        if n < 0:
            return cls
        counts[cls] = n
    return counts


def _positive_int(v: Any) -> int | None:
    try:
        n = int(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if n >= 1 else None


def precise_counts(settings: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Sample-count overrides from precise-mode settings as ``(counts, error_message)``.

    Shapes: ``{"_total": n}``, ``{cls: n, ...}`` or ``{"_per_profile": {pid: <either>}}``.
    """
    s = settings
    multi_profiles: list[str] = s.get("multi_profiles") or []
    per_class = bool(s.get("per_class", False))

    if multi_profiles:
        per_profile: dict[str, Any] = {}
        for pid in multi_profiles:
            if per_class:
                cls_dict: dict[str, Any] = (s.get("multi_classes") or {}).get(pid) or {}
                if not cls_dict:
                    return None, f"Precise mode: no class entries for '{pid}'.\nOpen Precise Settings and click ↻."
                counts = _non_negative_counts(cls_dict)
                if isinstance(counts, str):
                    return None, f"Precise mode: invalid count for '{pid}' / '{counts}'."
                per_profile[pid] = counts
            else:
                if (n := _positive_int((s.get("multi_totals") or {}).get(pid, s.get("total", 100)))) is None:
                    return None, f"Precise mode: invalid count for profile '{pid}'."
                per_profile[pid] = {"_total": n}
        return {"_per_profile": per_profile}, None

    if per_class:
        classes: dict[str, Any] = s.get("classes") or {}
        if not classes:
            return None, "Precise mode: no class entries.\nOpen Precise Settings and click ↻."
        counts = _non_negative_counts(classes)
        if isinstance(counts, str):
            return None, f"Precise mode: invalid count for class '{counts}'."
        return counts, None
    if (total := _positive_int(s.get("total", 100))) is None:
        return None, "Precise mode: invalid total count.\nOpen Precise Settings."
    return {"_total": total}, None


def precise_counts_for_profile(counts: dict[str, Any], profile_id: str) -> dict[str, Any]:
    """Counts that apply to one run spec (per-profile entry, falling back to the first one)."""
    per_profile = counts.get("_per_profile")
    if per_profile is None:
        return counts
    return per_profile.get(profile_id) or next(iter(per_profile.values()), {})


def render_overrides(*, look: str = "", routing: str = "", drc: str = "", design_rules: str = "") -> dict[str, Any]:
    """The ``render`` keys the GUI selectors override: ``{"look": ..., "board": {"routing": ..., "drc": ...,
    "design_rules": ...}}`` with only the chosen entries (``""`` / :data:`FROM_CONFIG` = keep the config's);
    empty when nothing is chosen. Unknown values raise ``ValueError`` (the config validation would too)."""
    def chosen(value: str, choices: tuple[str, ...], key: str) -> str:
        value = str(value or "").strip()
        if value in ("", FROM_CONFIG):
            return ""
        if value not in choices:
            raise ValueError(f"{key} must be one of {list(choices[1:])}, got {value!r}")
        return value

    out: dict[str, Any] = {}
    if v := chosen(look, LOOK_CHOICES, "render.look"):
        out["look"] = v
    board = {k: v for k, v in (("routing", chosen(routing, BOARD_ROUTING_CHOICES, "render.board.routing")),
                               ("drc", chosen(drc, BOARD_DRC_CHOICES, "render.board.drc")),
                               ("design_rules", chosen(design_rules, BOARD_RULES_CHOICES, "render.board.design_rules")))
             if v}
    if board:
        out["board"] = board
    return out


def apply_render_overrides(render: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """*render* (a run config's ``render`` section, modified in place) with :func:`render_overrides`
    applied: ``look`` replaced, ``board`` keys merged into the existing ``board`` mapping."""
    if overrides.get("look"):
        render["look"] = str(overrides["look"])
    if overrides.get("board"):
        render["board"] = {**(render.get("board") or {}), **dict(overrides["board"])}
    return render


def render_overrides_text(overrides: Mapping[str, Any]) -> str:
    """``"look=quins, board.routing=escape"`` for logs and captions (``""`` when empty)."""
    parts = [f"look={overrides['look']}"] if overrides.get("look") else []
    parts += [f"board.{k}={v}" for k, v in (overrides.get("board") or {}).items()]
    return ", ".join(parts)


def apply_precise_counts(cfg_data: dict[str, Any], counts: dict[str, Any]) -> dict[str, Any]:
    """Replace ``cfg_data["classes"]`` with *counts* (``_total`` applies to every existing class)."""
    existing_classes: dict[str, int] = {str(k): int(v) for k, v in (cfg_data.get("classes") or {}).items()}
    if "_total" in counts:
        total = int(counts["_total"])
        cfg_data["classes"] = {cls: total for cls in existing_classes}
    else:
        cfg_data["classes"] = {str(k): int(v) for k, v in counts.items()}
    return cfg_data


# ---------------------------------------------------------------------------
# Run specs
# ---------------------------------------------------------------------------

def mixed_profile_config(base_cfg: dict[str, Any], profile_ids: list[str]) -> dict[str, Any]:
    """Turn a run config into a mixed profile-classifier config over *profile_ids* (in place)."""
    run_block = dict(base_cfg.get("run") or {})
    run_block.pop("component_profile", None)
    run_block["mode"] = "profile_classifier"
    run_block["schema_version"] = int(run_block.get("schema_version", 2) or 2)
    run_block["component_profiles"] = list(profile_ids)
    if not run_block.get("run_id"):
        run_block["run_id"] = "run_profile_cls_selected"
    base_cfg["run"] = run_block
    base_cfg.setdefault("render", {})["backend"] = "opencv_2d"
    return base_cfg


def unique_timestamped_dir(path: Path, ts: str) -> Path:
    """``<path>_<ts>``, or ``<path>_<ts>_NN`` if that already exists."""
    cand = path.parent / f"{path.name}_{ts}"
    i = 1
    while cand.exists():
        cand = path.parent / f"{path.name}_{ts}_{i:02d}"
        i += 1
    return cand


def profile_run_targets(base_out: Path, base_model: Path, run_id: str, *, multi: bool, models_root: Path) -> tuple[Path, Path]:
    """``(out_dir, model_path)`` for one profile of a (possibly multi-profile) run.

    With several profiles each gets its own dataset dir/model named after its run_id next to the
    selected ones (or inside ``runs``/``versions`` if that root itself is selected).
    """
    if not multi:
        model = base_model if str(base_model).strip() else models_root / f"{run_id}.pt"
        return base_out, model
    if base_out.name in {"runs", "versions"}:
        out = base_out / run_id
    elif base_out.parent.name in {"runs", "versions"}:
        out = base_out.parent / run_id
    else:
        out = base_out.with_name(run_id)
    model = base_model.with_name(f"{run_id}.pt") if base_model.suffix == ".pt" else models_root / f"{run_id}.pt"
    return out, model


def profile_model_paths(sim_root: Path, preset: str, tag: str) -> tuple[Path, Path, Path]:
    """``(config, out_dataset, out_model)`` for building a profile classifier (``quick``/``full``).

    Existing outputs are not overwritten: *tag* is appended to their names.
    """
    full = preset == "full"
    cfg = sim_root / "configs" / ("run_profile_cls.yaml" if full else "run_profile_cls_quick.yaml")
    out_ds = sim_root / "outputs" / "sim_data" / "runs" / ("run_profile_cls" if full else "run_profile_cls_quick")
    out_model = model_root(sim_root) / ("profile_classifier_v1.pt" if full else "profile_classifier_quick.pt")
    if out_ds.exists():
        out_ds = out_ds.parent / f"{out_ds.name}_{tag}"
    if out_model.exists():
        out_model = out_model.with_name(f"{out_model.stem}_{tag}{out_model.suffix}")
    return cfg, out_ds, out_model


# ---------------------------------------------------------------------------
# Numeric UI fields
# ---------------------------------------------------------------------------

def parse_run_count(run_mode: str, raw: str) -> int:
    """Number of runs for a run mode (``-1`` = continuous); ``ValueError`` for a bad "multiple" count."""
    if run_mode == "multiple":
        if (n := int(raw)) < 1:
            raise ValueError(f"run count must be >= 1: {raw!r}")
        return n
    return -1 if run_mode == "continuous" else 1


def parse_snap_every(raw: str) -> int:
    """Snapshot interval (>= 1): blank -> 1, invalid -> 5."""
    try:
        n = int(s) if (s := raw.strip()) else 1
    except ValueError:
        n = 5
    return max(1, n)


def parse_snap_keep(raw: str) -> int:
    """Snapshots to keep (>= 0, 0 = all): blank -> 0, invalid -> 30."""
    try:
        k = int(s) if (s := raw.strip()) else 0
    except ValueError:
        k = 30
    return max(0, k)


# ---------------------------------------------------------------------------
# Display text
# ---------------------------------------------------------------------------

def sample_color_from_total(total: int) -> str:
    """Label colour for a dataset sample count."""
    if total >= 10000:
        return "green"
    if total >= 5000:
        return "blue"
    if total >= 1000:
        return "orange"
    return "black"


def samples_text(total: int, train: int | None, val: int | None, test: int | None) -> str:
    """``Samples: N`` plus ``(tX/vY/sZ)`` when all split counts are known."""
    parts = [f"Samples: {total}"]
    if train is not None and val is not None and test is not None:
        parts.append(f"(t{train}/v{val}/s{test})")
    return " ".join(parts)


def summarize_profile_ids(pids: list[str]) -> str:
    """Up to three profile ids, then ``(+N)``."""
    if len(pids) <= 3:
        return ", ".join(pids)
    return ", ".join(pids[:3]) + f" (+{len(pids) - 3})"


def summarize_labels(labels: list[str]) -> str:
    """``(none)``, the single label, or ``first (+N)``."""
    if not labels:
        return "(none)"
    if len(labels) == 1:
        return labels[0]
    return f"{labels[0]} (+{len(labels) - 1})"


def profile_compat_status(ds_profile_id: str, ds_profile_hash: Any, model_profile: dict[str, Any] | None) -> tuple[str, str]:
    """``(message, colour)`` comparing a model's component profile with a dataset's."""
    if not model_profile:
        return "⚠ Model has no profile (legacy)", "orange"
    model_profile_id, model_profile_hash = model_profile.get("profile_id"), model_profile.get("profile_hash")
    if isinstance(model_profile_id, str) and model_profile_id.lower().startswith("multi"):
        return "✓ Compatible (multi-model)", "green"
    if model_profile_id != ds_profile_id:
        return f"✗ INCOMPATIBLE: Model={model_profile_id}, Dataset={ds_profile_id}", "red"
    if model_profile_hash != ds_profile_hash:
        return "⚠ Profile hash mismatch (same ID, different version)", "orange"
    return f"✓ Compatible: {ds_profile_id}", "green"


# ---------------------------------------------------------------------------
# Serialized selections
# ---------------------------------------------------------------------------

def parse_profile_ids(raw: str, available: set[str]) -> list[str]:
    """Unique profile ids from a JSON list (or comma-separated) string, limited to *available* if non-empty."""
    raw = raw.strip() or "[]"
    items: list[str] = []
    try:
        if isinstance(v := json.loads(raw), list):
            items = [x.strip() for x in v if isinstance(x, str) and x.strip()]
    except ValueError:
        items = [x.strip() for x in raw.split(",") if x.strip()]
    out: list[str] = []
    for pid in items:
        if (pid in available or not available) and pid not in out:
            out.append(pid)
    return out


def decode_dataset_paths_json(raw: str, sim_root: Path) -> list[Path]:
    """Unique dataset paths from a JSON list; relative entries are resolved against *sim_root*."""
    try:
        arr = json.loads(raw or "[]")
    except ValueError:
        return []
    if not isinstance(arr, list):
        return []
    out: list[Path] = []
    seen: set[str] = set()
    for it in arr:
        if not isinstance(it, str):
            continue
        p = Path(it)
        if not p.is_absolute():
            p = (sim_root / p).resolve()
        if (key := str(p.resolve())) not in seen:
            seen.add(key)
            out.append(p)
    return out


def encode_dataset_paths_json(paths: list[Path], sim_root: Path) -> str:
    """JSON list of dataset paths (relative paths stored relative to *sim_root*)."""
    return json.dumps([str(p) if p.is_absolute() else str(p.resolve().relative_to(sim_root.resolve())) for p in paths])


# ---------------------------------------------------------------------------
# Command lines
# ---------------------------------------------------------------------------

def chain_commands(argvs: list[list[str]]) -> str:
    """Shell-quote each argv and join them with ``&&``."""
    return " && ".join(shlex.join(argv) for argv in argvs)


def validate_argv(ds: Path) -> list[str]:
    return [PYTHON, "tools/validate_dataset.py", "--data", str(ds)]


def train_argv(ds: Path, model_out: Path) -> list[str]:
    return [PYTHON, "scripts/train.py", "--data", str(ds), "--out", str(model_out)]


def eval_argv(ds: Path, model_in: Path, *, allow_profile_mismatch: bool = False) -> list[str]:
    argv = [PYTHON, "scripts/eval.py", "--data", str(ds), "--model", str(model_in)]
    if allow_profile_mismatch:
        argv.append("--allow-profile-mismatch")
    return argv


def continue_train_argv(ds: Path, model: Path, *, extra_epochs: int, out_mode: str, allow_profile_mismatch: bool = False) -> list[str]:
    argv = train_argv(ds, model) + ["--resume", str(model)]
    if allow_profile_mismatch:
        argv.append("--allow-profile-mismatch")
    return argv + ["--extra-epochs", str(extra_epochs), "--out-mode", out_mode]


def multi_train_argvs(
    dss: list[Path],
    model_out: Path,
    profile_ids: list[str | None],
    *,
    is_bundle: bool,
    extra_epochs: int,
    out_mode: str,
    allow_profile_mismatch: bool,
) -> list[list[str]]:
    """Sequential training over several datasets.

    A bundle trains each profile fresh once and resumes for repeated profiles; a single
    checkpoint trains on the first dataset and resumes (warm-starts) on every following one.
    """
    resume = ["--resume", str(model_out), "--extra-epochs", str(extra_epochs), "--out-mode", out_mode]
    argvs: list[list[str]] = []
    seen_pids: set[str] = set()
    for i, (ds, pid) in enumerate(zip(dss, profile_ids)):
        argv = train_argv(ds, model_out)
        if is_bundle:
            key = pid or f"__no_profile__:{ds}"
            if key in seen_pids:
                argv.extend(resume)
            else:
                seen_pids.add(key)
        elif i > 0:
            argv.extend(resume)
            if allow_profile_mismatch:
                argv.append("--allow-profile-mismatch")
        argvs.append(argv)
    return argvs
