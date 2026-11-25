"""Helpers shared by several test modules (import ``tests.helpers``; test modules never import each other)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from simple_sim.defects import sample_defect_params
from simple_sim.generators.opencv2d import render_roi, sample_nominal_geometry
from simple_sim.schema import MetaRow

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- photoreal 2D renders

TOL = {"ok_shift_px": 3.0, "ok_rotation_deg": 5.0, "tombstone_tilt_deg": 75.0}
CFG = {"component_color": [20, 20, 20], "substrate_color": [40, 90, 40], "copper_color": [60, 120, 180]}
FOOTPRINTS = {
    "chip_2pad": {},
    "sot23": {"pad_spacing_y": [36.0, 50.0]},
    "qfn_32": {"pad_width": [15.0, 18.0], "pad_spacing": [173.0, 187.0], "pad_spacing_y": [173.0, 187.0],
               "component_length": [162.0, 172.0], "component_width": [162.0, 172.0]},
    "soic_16": {"pad_width": [10.0, 12.0], "pad_spacing": [120.0, 130.0], "component_length": [90.0, 100.0],
                "component_width": [70.0, 80.0]},
}
CLASSES = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE", "SOLDER_BRIDGE", "CORNER_LIFT", "SOLDER_BALL", "EXCESS_SOLDER"]


def render_photoreal(footprint, defect_type, seed=1, **cfg):
    """One 128x128 photoreal 2D render of *footprint* with a sampled *defect_type*."""
    rng = np.random.default_rng(seed)
    nominal = sample_nominal_geometry({}, rng, FOOTPRINTS[footprint])
    defect = sample_defect_params(defect_type, rng, TOL)
    return render_roi(nominal, defect, {"rotation_deg": 0.0}, (128, 128), {**CFG, **cfg}, TOL, rng, footprint=footprint)


def bake_job(defect_type, footprint="chip_2pad", seed=3):
    """A minimal Blender bake job (64x64 ROI) with a sampled *defect_type*."""
    rng = np.random.default_rng(seed)
    return {"image_path": f"images/{defect_type}.png", "seed": seed, "mm_per_px": 0.01, "roi_width_px": 64,
            "roi_height_px": 64, "footprint": footprint, "component_height_mm": 0.45,
            "nominal": sample_nominal_geometry({}, rng, {"pad_spacing_y": [36.0, 50.0]}), "defect": sample_defect_params(defect_type, rng, TOL),
            "augment": {"rotation_deg": 0.0}, "render_3d": {}, "component_type": "chip_resistor", "tolerances": TOL}


# --------------------------------------------------------------------------- configs / golden files

def small_run_config(classes: dict[str, int], size_px: int) -> dict[str, Any]:
    """``configs/run_0001.yaml`` (loaded and validated) with *classes* and a square ROI of *size_px*."""
    from simple_sim.config import load_config

    cfg = load_config(ROOT / "configs" / "run_0001.yaml")
    cfg["classes"] = dict(classes)
    cfg["roi"].update(width_px=size_px, height_px=size_px)
    return cfg


def tiny_run_config(**classes: int) -> dict[str, Any]:
    """Raw ``configs/run_0001.yaml`` (not validated) with a 64x64 ROI and *classes* (default 4 OK + 4 MISSING)."""
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "run_0001.yaml").read_text(encoding="utf-8"))
    cfg["roi"] = {"width_px": 64, "height_px": 64, "mm_per_px": 0.04}
    cfg["classes"] = classes or {"OK": 4, "MISSING": 4}
    return cfg


def empty_dataset(root: Path) -> Path:
    """A dataset directory with every file present but no samples."""
    (root / "images").mkdir(parents=True)
    (root / "splits").mkdir()
    for name in ("meta.jsonl", "labels.jsonl"):
        (root / name).write_text("", encoding="utf-8")
    (root / "config.yaml").write_text("run: {seed: 1}\n", encoding="utf-8")
    for split in ("train", "val", "test"):
        (root / "splits" / f"{split}.txt").write_text("", encoding="utf-8")
    return root


def lib_versions() -> dict[str, str]:
    """Library versions recorded next to golden hashes (a mismatch skips the golden comparison)."""
    import cv2

    return {"opencv": cv2.__version__, "numpy": np.__version__}


# --------------------------------------------------------------------------- dataset rows

NOMINAL = {"pad_width": 1.0, "pad_height": 1.0, "pad_spacing": 1.0, "component_length": 1.0, "component_width": 1.0}
DEFECT_OK = {"type": "OK", "shift_x": 0.0, "shift_y": 0.0, "rotation_deg": 0.0, "tilt_deg": 0.0}
AUGMENT_NONE = {"blur_sigma": 0.0, "noise_stddev": 0.0, "brightness_factor": 1.0, "contrast_factor": 1.0,
                "rotation_deg": 0.0}


def meta_row(sample_id: str, image_path: str, *, split: str = "train", run_id: str = "r", domain: str = "domain_A",
             footprint: str = "0603", seed: int = 1, nominal: dict[str, float] | None = None, **extra: Any) -> MetaRow:
    """A schema-2 ``MetaRow`` of an OK sample (geometry values do not matter for the callers)."""
    return MetaRow(schema_version=2, id=sample_id, run_id=run_id, domain=domain, split=split, seed=seed,
                   image_path=image_path, render_backend="opencv_2d", footprint=footprint,
                   nominal=dict(nominal or NOMINAL), defect=dict(DEFECT_OK), augment=dict(AUGMENT_NONE), **extra)


# --------------------------------------------------------------------------- reports / fake sim roots

def write_report(path: Path, *, model: str, dataset: str, acc: float, f1: float, split: str = "test", ts: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "timestamp": ts,
                "model_path": model,
                "dataset_path": dataset,
                "split": split,
                "seen_samples": 40,
                "checkpoint_epoch": 3,
                "checkpoint_val_accuracy": 0.9,
                "img_per_s": 123.4,
                "metrics": {"accuracy": acc, "macro_f1": f1, "critical_fn_rates": {"MISSING": 0.1}},
            }
        ),
        encoding="utf-8",
    )


def build_sim_root(root: Path) -> Path:
    """Create a small fake Simple-Sim tree with tracked models, a bundle and reports."""
    models = root / "outputs" / "models"
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    (models / "a.pt").write_bytes(b"x" * 3000)
    (models / "b.pt").write_bytes(b"x" * 1500)
    bundle = models / "multi.bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "chip@1.pt").write_bytes(b"x" * 500)
    (bundle / "bundle.json").write_text(
        json.dumps({"bundle_version": 1, "created_at": "2026-01-01", "checkpoints": {"chip@1": "chip@1.pt"}}),
        encoding="utf-8",
    )
    for name in ("ds_one", "ds_two"):
        ds = root / "outputs" / "sim_data" / "runs" / name
        (ds / "predictions").mkdir(parents=True, exist_ok=True)
        (ds / "meta.jsonl").write_text("{}\n" * 10, encoding="utf-8")
    (models / "arena.json").write_text(
        json.dumps(
            {
                "arena_models": [
                    {"path": "outputs/models/a.pt"},
                    {"path": "outputs/models/b.pt"},
                    {"path": "outputs/models/multi.bundle"},
                    {"path": "outputs/models/gone.pt"},
                ]
            }
        ),
        encoding="utf-8",
    )
    ds1 = "outputs/sim_data/runs/ds_one"
    ds2 = "outputs/sim_data/runs/ds_two"
    hist = models / "history"
    write_report(models / "report_a.json", model="outputs/models/a.pt", dataset=ds1, acc=0.91, f1=0.90, ts="2026-01-02T10:00:00")
    write_report(hist / "batch_report_a_test_1.json", model="outputs/models/a.pt", dataset=ds2, acc=0.71, f1=0.70, ts="2026-01-03T10:00:00")
    # Train-split report must not win over the test one.
    write_report(hist / "batch_report_a_train_1.json", model="outputs/models/a.pt", dataset=ds2, acc=0.99, f1=0.99, split="train", ts="2026-01-03T11:00:00")
    write_report(hist / "batch_report_b_test_1.json", model="outputs/models/b.pt", dataset=ds1, acc=0.95, f1=0.94, ts="2026-01-04T10:00:00")
    write_report(
        root / "outputs" / "sim_data" / "runs" / "ds_one" / "predictions" / "batch_report_chip_test_1.json",
        model="outputs/models/multi.bundle/chip@1.pt", dataset=ds1, acc=0.5, f1=0.4, ts="2026-01-05T10:00:00",
    )
    # Untracked model: ignored.
    write_report(hist / "batch_report_z_test_1.json", model="outputs/models/z.pt", dataset=ds1, acc=1.0, f1=1.0, ts="2026-01-05T10:00:00")
    return root
