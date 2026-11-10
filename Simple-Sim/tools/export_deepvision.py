#!/usr/bin/env python3
"""Export rendered reference/probe pairs for the DeepVision ``aoi-judge`` model.

The judge is a Siamese network: it compares the crop of a part with a good reference crop of the *same*
part. This tool renders, per seed, one **reference** (an OK render) and several **probes** of the same
board: one OK probe (same board under another exposure: other sensor noise, illumination gradient, lens
blur draw, JPEG quality, brightness/contrast) and one probe per defect class (same board, same pose, same
neighbours, same joint light and the *same* sensor draws as the reference; only the inspected part and
its joints differ). Everything is a plain 2D render through :func:`simple_sim.generators.opencv2d.render_roi`
with its ``camera_rng`` argument, so the output is reproducible from the run config and ``--seed``.

What is constant within a pair and what is not:

* constant: nominal footprint geometry, crop orientation (``augment.rotation_deg``), the camera pose
  (tilt, zoom = field of view), the board (mask, copper, silkscreen, residue, neighbour parts), the joint
  lighting (meniscus side stream), the sensor draws of reference and defect probes (shared ``camera_rng``);
* varies with the class: the inspected part's placement / absence / tilt, its joints' surface relief,
  solder balls and bulges (the defect draws come after the board's in the renderer, so the board does not
  move; see ``render_roi``);
* varies for the OK probe only: the sensor draws and the brightness / contrast factors (another photo of
  the same board); its placement is a fresh OK draw within the tolerance.

Output (``<out>/``):

* ``manifest.json``: ``{run, side, profile, look, renderer_version, classes: {name: renders}, pairs,
  created, config, ...}``,
* ``records.ndjson``: one line per render, the DeepVision label schema plus ``sim``:
  ``{id, pairId, role: "ref" | "probe", board, side, label: "ok" | "fault", fault, form, size, fieldMm,
  crop: "crops/<id>.png", cropColor: "crops/rgb/<id>.png", sim: {class, seed, profile, look, defect_params,
  derived_class, camera_seed}}``,
* ``crops/<id>.png`` grey and ``crops/rgb/<id>.png`` colour crops at the app's scale (``--px-per-mm``,
  default 16.1: the ROI is resampled (area) from the profile scale to ``round(fieldMm x px/mm)`` pixels,
  49 px for a 0402 at 3.04 body lengths; ``--max-size`` caps it like the TOP app crops, which stay at 49 px
  for larger parts).

Class mapping (Simple-Sim -> DeepVision ``fault``, the spelling of the app's reviews): MISSING -> fehlt,
TOMBSTONE -> grabstein, SOLDER_BRIDGE / EXCESS_SOLDER -> loetbruecke, SOLDER_BALL -> fremdkoerper,
CORNER_LIFT -> loetfehler (a lifted corner is an open joint, a solder fault in the app's vocabulary), and
MISALIGNED -> verdreht when the rotation dominates (|rotation_deg| above the profile's ``ok_rotation_deg``
while the shift stays within ``ok_shift_px``), else versetzt (:data:`FAULTS`, :func:`fault_of`; the sampled
``defect_params`` are written with every record, so the rule can be re-applied). ``board`` is
``sim-<profile>-<look>-<run>``, ``id`` is ``<seed>-<role>-<class>``, ``pairId`` the seed; ``form`` the
package code of the profile (``component.package``, e.g. ``CAPC1005``). ``manifest.json`` carries the table
as ``faultMapping``.

Which defect classes a pair gets: the run config's ``classes`` counts are weights; every pair renders
``--faults-per-pair`` classes drawn without replacement by those weights (default 2; ``all`` renders
every class with a count > 0).

Usage:
    python tools/export_deepvision.py --config configs/run_0402_c_dv_top.yaml --out outputs/deepvision-export/run_0402_c_dv_top \\
        --pairs 120 [--side top] [--faults-per-pair 2] [--px-per-mm 16.1] [--max-size 0] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.classes import DefectClass  # noqa: E402
from simple_sim.config import defect_units, load_config, merge_profile_render, placement_sampling, validate_config  # noqa: E402
from simple_sim.defects import classify_defect, sample_defect_params  # noqa: E402
from simple_sim.generation.common import project_root  # noqa: E402
from simple_sim.generators.backends import backend_of  # noqa: E402
from simple_sim.generators.opencv2d import render_roi, sample_augment_params, sample_nominal_geometry  # noqa: E402
from simple_sim.generators.opencv2d.photoreal import camera_config, camera_settings  # noqa: E402
from simple_sim.generators.opencv2d.photoreal.look import look_name  # noqa: E402
from simple_sim.generators.version import RENDERER_VERSION  # noqa: E402
from simple_sim.io_utils import now_iso  # noqa: E402
from simple_sim.parallel import imap_ordered, render_workers  # noqa: E402
from simple_sim.profile_hash import load_profile  # noqa: E402
from simple_sim.rng import derive_sample_seed  # noqa: E402

#: Simple-Sim class -> DeepVision ``fault`` (spelling of the app's labels; ``None`` = good part). MISALIGNED is
#: decided per sample by :func:`fault_of` (``versetzt`` = shifted, ``verdreht`` = rotated).
FAULTS: dict[str, str | None] = {
    DefectClass.OK.value: None,
    DefectClass.MISSING.value: "fehlt",
    DefectClass.MISALIGNED.value: "versetzt",
    DefectClass.TOMBSTONE.value: "grabstein",
    DefectClass.SOLDER_BRIDGE.value: "loetbruecke",
    DefectClass.EXCESS_SOLDER.value: "loetbruecke",
    DefectClass.SOLDER_BALL.value: "fremdkoerper",
    DefectClass.CORNER_LIFT.value: "loetfehler",
}
MISALIGNED_ROTATED = "verdreht"
#: The mapping table written to ``manifest.json`` (``faultMapping``): class -> fault, MISALIGNED with its rule.
FAULT_MAPPING: dict[str, str | None] = {
    **FAULTS,
    DefectClass.MISALIGNED.value: "verdreht if |rotation_deg| > ok_rotation_deg and shift <= ok_shift_px, else versetzt",
}


def fault_of(derived_class: str, defect: dict[str, Any], tolerances: dict[str, Any] | None) -> str | None:
    """DeepVision ``fault`` of a render: :data:`FAULTS` by class; a MISALIGNED part is ``verdreht`` when its
    rotation alone breaks the tolerance (|rotation_deg| > ``ok_rotation_deg`` and the shift magnitude is within
    ``ok_shift_px``), otherwise ``versetzt`` (shift, or shift and rotation). Tolerances default as in
    :func:`simple_sim.defects.classify_defect`."""
    fault = FAULTS[derived_class]
    if derived_class != DefectClass.MISALIGNED.value:
        return fault
    tol = tolerances or {}
    ok_shift = float(tol.get("ok_shift_px", 2.0))
    ok_rot = float(tol.get("ok_rotation_deg", 3.0))
    shift = float(np.hypot(float(defect.get("shift_x", 0.0)), float(defect.get("shift_y", 0.0))))
    rotated = abs(float(defect.get("rotation_deg", 0.0))) > ok_rot
    return MISALIGNED_ROTATED if rotated and shift <= ok_shift else fault
ROLE_REF, ROLE_PROBE = "ref", "probe"
DEFAULT_PX_PER_MM = 16.1        # the app's crop scale (0402: 49 px for 3.037 mm)
SIDES = ("top", "bottom")
# salts of the per-pair generators (all derived from the pair seed)
_SALT_DEFECT, _SALT_CAMERA_REF, _SALT_CAMERA_PROBE, _SALT_PROBE_AUGMENT, _SALT_CLASSES, _SALT_RENDER = 11, 1, 2, 3, 4, 5


def package_code(profile: dict[str, Any]) -> str:
    """``form`` of the export: the package code in parentheses of ``component.package`` (``0402 (CAPC1005)`` ->
    ``CAPC1005``), else the whole package string, else the profile id."""
    comp = profile.get("component") or {}
    package = str(comp.get("package") or "").strip()
    m = re.search(r"\(([^()]+)\)\s*$", package)
    if m:
        return m.group(1).strip()
    return package or str((profile.get("profile") or {}).get("profile_id") or "")


def side_of(config: dict[str, Any], side: str | None) -> str:
    """``--side`` or the side implied by the look (``dv_top`` / ``dv_bottom``)."""
    if side:
        return side
    look = look_name(config.get("render") or {})
    for s in SIDES:
        if look.endswith(s):
            return s
    raise ValueError("--side is required unless render.look names a side (dv_top / dv_bottom)")


def fault_classes(config: dict[str, Any]) -> dict[str, int]:
    """Defect classes of the run config with a positive count (the weights), in config order."""
    out = {}
    for name, count in (config.get("classes") or {}).items():
        if name == DefectClass.OK.value or int(count) <= 0:
            continue
        if name not in FAULTS:
            raise ValueError(f"no DeepVision fault mapping for class {name!r}")
        out[str(name)] = int(count)
    if not out:
        raise ValueError("the run config has no defect class with a positive count")
    return out


def choose_classes(weights: dict[str, int], k: int | None, rng: np.random.Generator) -> list[str]:
    """*k* classes drawn without replacement with probability ~ weight (``None``: all, config order)."""
    names = list(weights)
    if k is None or k >= len(names):
        return names
    p = np.array([weights[n] for n in names], dtype=np.float64)
    picked = rng.choice(len(names), size=k, replace=False, p=p / p.sum())
    return [names[i] for i in sorted(picked)]


def field_mm(render: dict[str, Any], nominal: dict[str, float], roi_px: int) -> float:
    """Field of view of a render in mm: ``crop_part_lengths`` x the body length (mid of the range, the drawn
    factor is within the range) when the camera crops by body length, else the ROI at the profile scale."""
    mm_per_px = float(render["mm_per_px"])
    cam = camera_settings(camera_config(render))
    lo, hi = (float(v) for v in cam["crop_part_lengths"])
    if hi > 0.0:
        return 0.5 * (lo + hi) * float(nominal["component_length"]) * mm_per_px
    return roi_px * mm_per_px


def sample_pair_defect(class_name: str, seed: int, tolerances: dict[str, Any] | None, config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Defect params of *class_name* for pair *seed* whose derived label equals the class (re-drawn up to 8x)."""
    mm_per_px = (config.get("roi") or {}).get("mm_per_px")
    derived = ""
    for attempt in range(8):
        rng = np.random.default_rng([seed, _SALT_DEFECT, attempt, sorted(FAULTS).index(class_name)])
        defect = sample_defect_params(class_name, rng, tolerances=tolerances, placement=placement_sampling(config),
                                      units=defect_units(config), mm_per_px=mm_per_px)
        derived = classify_defect({}, defect, tolerances)
        if derived == class_name:
            return defect, derived
    return defect, derived


class Exporter:
    """Plans and renders the pairs of one run config (picklable: workers get the plan rows only)."""

    def __init__(self, config: dict[str, Any], profile: dict[str, Any], *, side: str, px_per_mm: float, max_size: int,
                 faults_per_pair: int | None, seed: int | None):
        validate_config(config)
        render = merge_profile_render(dict(config["render"]), dict(profile), config.get("roi"))
        if backend_of(render).needs_blender:
            raise ValueError("export_deepvision renders with the 2D backend only (render.backend: opencv_2d)")
        self.config, self.profile, self.render_cfg = config, profile, render
        self.side, self.px_per_mm, self.max_size = side, float(px_per_mm), int(max_size)
        self.roi = (int(config["roi"]["width_px"]), int(config["roi"]["height_px"]))
        self.footprint = str(profile["component"]["footprint"])
        self.tolerances = profile.get("tolerances")
        self.geometry_ranges = profile.get("geometry_ranges")
        self.domain = str(next(iter(config["domains"])))
        self.run_id = str(config["run"]["run_id"])
        self.profile_id = str(config["run"]["component_profile"])
        self.look = look_name(render)
        self.form = package_code(profile)
        self.board = f"sim-{self.profile_id}-{self.look}-{self.run_id}"
        self.run_seed = int(config["run"]["seed"]) if seed is None else int(seed)
        self.weights = fault_classes(config)
        self.faults_per_pair = faults_per_pair

    # -- planning ---------------------------------------------------------------------------------------------
    def plan_pair(self, index: int) -> list[dict[str, Any]]:
        """The renders of pair *index*: reference, OK probe, one probe per chosen defect class."""
        seed = derive_sample_seed(self.run_seed, self.domain, index)
        rng = np.random.default_rng(seed)
        nominal = sample_nominal_geometry(self.config["roi"], rng, geometry_ranges=self.geometry_ranges)
        augment = sample_augment_params(self.config["augment"], self.config["domains"][self.domain], rng)
        classes = choose_classes(self.weights, self.faults_per_pair, np.random.default_rng([seed, _SALT_CLASSES]))
        pair_id = f"{seed:019d}"
        rows: list[dict[str, Any]] = []

        def row(role: str, class_name: str, defect: dict[str, Any], derived: str, aug: dict[str, Any], camera_salt: int) -> dict[str, Any]:
            fault = fault_of(derived, defect, self.tolerances)
            rid = f"{pair_id}-{role}-{derived.lower()}" if role == ROLE_PROBE else f"{pair_id}-{role}"
            return {"id": rid, "pairId": pair_id, "role": role, "seed": seed, "nominal": nominal, "defect": defect,
                    "augment": aug, "class": class_name, "derived": derived, "label": "ok" if fault is None else "fault",
                    "fault": fault, "camera_salt": camera_salt}

        ref_defect, ref_derived = sample_pair_defect(DefectClass.OK.value, seed, self.tolerances, self.config)
        rows.append(row(ROLE_REF, DefectClass.OK.value, ref_defect, ref_derived, augment, _SALT_CAMERA_REF))
        # OK probe: another exposure of the same board (other sensor draws, other brightness / contrast).
        probe_rng = np.random.default_rng([seed, _SALT_PROBE_AUGMENT])
        other = sample_augment_params(self.config["augment"], self.config["domains"][self.domain], probe_rng)
        probe_aug = {**augment, "brightness_factor": other["brightness_factor"], "contrast_factor": other["contrast_factor"]}
        ok_defect, ok_derived = sample_pair_defect(DefectClass.OK.value, seed + 1, self.tolerances, self.config)
        rows.append(row(ROLE_PROBE, DefectClass.OK.value, ok_defect, ok_derived, probe_aug, _SALT_CAMERA_PROBE))
        for cls in classes:
            defect, derived = sample_pair_defect(cls, seed, self.tolerances, self.config)
            rows.append(row(ROLE_PROBE, cls, defect, derived, augment, _SALT_CAMERA_REF))
        return rows

    # -- rendering ---------------------------------------------------------------------------------------------
    def render(self, plan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """``(colour crop, grey crop, record)`` of one planned render."""
        img = render_roi(nominal=plan["nominal"], defect_params=plan["defect"], augment=plan["augment"], roi_size=self.roi,
                         config=dict(self.render_cfg), tolerances=None if self.tolerances is None else dict(self.tolerances),
                         rng=np.random.default_rng([plan["seed"], _SALT_RENDER]), footprint=self.footprint,
                         camera_rng=np.random.default_rng([plan["seed"], plan["camera_salt"]]))
        fov = field_mm(self.render_cfg, plan["nominal"], min(self.roi))
        size = max(8, int(round(fov * self.px_per_mm)))
        if self.max_size > 0:
            size = min(size, self.max_size)
        color = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        record = {
            "id": plan["id"], "pairId": plan["pairId"], "role": plan["role"], "board": self.board, "side": self.side,
            "label": plan["label"], "fault": plan["fault"], "form": self.form, "size": size, "fieldMm": round(fov, 4),
            "crop": f"crops/{plan['id']}.png", "cropColor": f"crops/rgb/{plan['id']}.png",
            "sim": {"class": plan["class"], "seed": plan["seed"], "profile": self.profile_id, "look": self.look,
                    "defect_params": plan["defect"], "derived_class": plan["derived"],
                    "camera_seed": [plan["seed"], plan["camera_salt"]], "renderer_version": RENDERER_VERSION},
        }
        return color, grey, record


def _render_task(task: tuple[Exporter, dict[str, Any], str]) -> dict[str, Any]:
    exporter, plan, out = task
    color, grey, record = exporter.render(plan)
    root = Path(out)
    (root / "crops" / "rgb").mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(root / record["cropColor"]), color) or not cv2.imwrite(str(root / record["crop"]), grey):
        raise OSError(f"failed to write crops of {record['id']}")
    return record


def export(config_path: Path, out: Path, *, pairs: int, side: str | None, px_per_mm: float, max_size: int,
           faults_per_pair: int | None, seed: int | None, workers: int | None, profiles_dir: Path | None = None,
           progress=None) -> dict[str, Any]:
    """Render *pairs* pairs of *config_path* into *out*; returns the manifest."""
    config = load_config(config_path)
    profile = load_profile(config["run"]["component_profile"], profiles_dir or project_root() / "configs" / "profiles")
    exporter = Exporter(config, profile, side=side_of(config, side), px_per_mm=px_per_mm, max_size=max_size,
                        faults_per_pair=faults_per_pair, seed=seed)
    plans = [p for i in range(int(pairs)) for p in exporter.plan_pair(i)]
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for done, rec in enumerate(imap_ordered(_render_task, [(exporter, p, str(out)) for p in plans], workers=render_workers(workers)), 1):
        records.append(rec)
        if progress is not None:
            progress(done, len(plans))
    with (out / "records.ndjson").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    classes = Counter(r["sim"]["class"] for r in records)
    manifest = {
        "run": exporter.run_id, "side": exporter.side, "profile": exporter.profile_id, "look": exporter.look,
        "renderer_version": RENDERER_VERSION, "classes": dict(classes), "pairs": int(pairs), "renders": len(records),
        "created": now_iso(), "config": config, "config_path": str(config_path), "form": exporter.form,
        "board": exporter.board, "px_per_mm": px_per_mm, "max_size": max_size,
        "faults_per_pair": "all" if faults_per_pair is None else faults_per_pair, "class_weights": exporter.weights,
        "seed": exporter.run_seed, "faultMapping": FAULT_MAPPING, "faults": dict(Counter(r["fault"] or "ok" for r in records)),
        "sizes": dict(Counter(r["size"] for r in records)),
        "roles": dict(Counter(r["role"] for r in records)), "labels": dict(Counter(r["label"] for r in records)),
        "constant_per_pair": ["nominal geometry", "crop orientation", "camera pose / field of view", "board (mask, copper, "
                              "silkscreen, residue, neighbours)", "joint lighting", "sensor draws of ref and defect probes"],
        "varies_per_pair": ["inspected part and its joints (class)", "OK probe: sensor draws, brightness, contrast, placement"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True, help="run config (2D backend)")
    ap.add_argument("--out", type=Path, required=True, help="export folder (manifest.json, records.ndjson, crops/)")
    ap.add_argument("--pairs", type=int, default=100, help="number of reference/probe pairs")
    ap.add_argument("--side", choices=SIDES, help="board side (default: from render.look dv_top / dv_bottom)")
    ap.add_argument("--faults-per-pair", default="2", help="defect probes per pair, drawn by class weight, or 'all'")
    ap.add_argument("--px-per-mm", type=float, default=DEFAULT_PX_PER_MM, help="crop scale of the app (default 16.1)")
    ap.add_argument("--max-size", type=int, default=0, help="cap the crop size in px (TOP app crops stay at 49 px); 0 = none")
    ap.add_argument("--seed", type=int, help="pair seed base (default: run.seed)")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args(argv)
    from simple_sim.cli_utils import configure_cli_logging

    configure_cli_logging()
    k = None if str(args.faults_per_pair).lower() == "all" else int(args.faults_per_pair)
    manifest = export(args.config, args.out, pairs=args.pairs, side=args.side, px_per_mm=args.px_per_mm, max_size=args.max_size,
                      faults_per_pair=k, seed=args.seed, workers=args.workers)
    print(f"{manifest['renders']} renders in {manifest['pairs']} pairs -> {args.out}")
    print("classes: " + ", ".join(f"{k} {v}" for k, v in manifest["classes"].items()))
    print("crop sizes: " + ", ".join(f"{k} px x{v}" for k, v in sorted(manifest["sizes"].items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
