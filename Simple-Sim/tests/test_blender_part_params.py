"""The Blender part geometry is pinned (D35): ``render_batch.py`` builds every family exactly as before.

``render_batch.py`` reads its per-family decisions (height ratio, standoff, end caps, lead width,
marking vocabulary and layout) from the baked scene's ``part`` entry (``board.footprints.scene_part``)
instead of its own tables. Without ``bpy`` (not installed) the script is loaded against a recording
stub: :func:`trace` records every body, end cap, lead segment, pin-1 mark and marking text that
``_build_part`` creates, the contacts, the solder height and the rng state, for every family and
several bodies / footprints. ``tests/golden/blender_part_params.json`` holds those traces as captured
on the script before D35 (HEAD ab592ad, which ignored ``part``).

Regenerate only for an intended Blender geometry change:
``.venv/bin/python -m tests.test_blender_part_params --update [--script PATH]``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "simple_sim" / "blender" / "render_batch.py"
GOLDEN = ROOT / "tests" / "golden" / "blender_part_params.json"


class Vector:
    """The part of ``mathutils.Vector`` that ``_build_part`` uses (3D, float arithmetic)."""

    __slots__ = ("x", "y", "z")

    def __init__(self, v=(0.0, 0.0, 0.0)):
        self.x, self.y, self.z = (float(c) for c in v)

    def __add__(self, o):
        return Vector((self.x + o.x, self.y + o.y, self.z + o.z))

    def __sub__(self, o):
        return Vector((self.x - o.x, self.y - o.y, self.z - o.z))

    def __mul__(self, k):
        return Vector((self.x * k, self.y * k, self.z * k))

    __rmul__ = __mul__

    def __truediv__(self, k):
        return Vector((self.x / k, self.y / k, self.z / k))

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    @property
    def length(self):
        return math.sqrt(self.dot(self))

    def as_list(self):
        return [self.x, self.y, self.z]


def load_script(path: Path = SCRIPT) -> types.ModuleType:
    """``render_batch.py`` imported with stub ``bpy`` / ``mathutils`` modules."""
    bpy, mathutils = types.ModuleType("bpy"), types.ModuleType("mathutils")
    mathutils.Vector = Vector
    with mock.patch.dict(sys.modules, {"bpy": bpy, "mathutils": mathutils}):
        spec = importlib.util.spec_from_file_location(f"_render_batch_{abs(hash(str(path)))}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def _plain(v: Any) -> Any:
    if isinstance(v, Vector):
        return v.as_list()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


class _Obj:
    def __init__(self, name: str):
        self.name, self.parent = name, None
        self.data = SimpleNamespace(materials=[])
        self.location = self.rotation_euler = None


def trace(mod: types.ModuleType, case: dict[str, Any]) -> dict[str, Any]:
    """Everything ``mod._build_part`` creates for *case*, as plain JSON values."""
    log: list[list[Any]] = []
    ctx = SimpleNamespace(collection=SimpleNamespace(objects=SimpleNamespace(link=lambda o: None)), active_object=None)

    def new_empty(name, data):
        log.append(["empty", name])
        return _Obj(name)

    def cylinder(**kw):
        log.append(["cylinder", kw["radius"], kw["depth"], _plain(kw["location"])])
        ctx.active_object = _Obj("cylinder")

    def material(name, color, **kw):
        log.append(["material", name, _plain(color), kw])
        return name

    def cfg_material(name, cfg, default_color, roughness, metallic, bump=0.0):
        log.append(["cfg_material", name, _plain(default_color), roughness, metallic, bump])
        return name

    def box(name, size, loc, mat, *, parent=None, bevel=0.0):
        log.append(["box", name, _plain(size), _plain(loc), mat, getattr(parent, "name", None), bevel])
        return _Obj(name)

    def box_between(name, p0, p1, width, thickness, mat):
        log.append(["lead", name, _plain(p0), _plain(p1), width, thickness, mat])
        return _Obj(name)

    def text(name, body, size, loc, mat, parent):
        log.append(["text", name, body, size, _plain(loc), mat, parent.name])

    mod.bpy.data = SimpleNamespace(objects=SimpleNamespace(new=new_empty))
    mod.bpy.context = ctx
    mod.bpy.ops = SimpleNamespace(mesh=SimpleNamespace(primitive_cylinder_add=cylinder))
    with mock.patch.multiple(mod, _material=material, _cfg_material=cfg_material, _box=box,
                             _box_between=box_between, _text=text):
        mmpp = case["mm_per_px"]
        pads = [(*mod._mm_xy(p[0], p[1], mmpp), mod._mm(p[2], mmpp), mod._mm(p[3], mmpp)) for p in case["board"]["pads"]]
        rng = random.Random(case["seed"])
        objs, contacts, h_solder = mod._build_part(pads, case["board"], mmpp, case["component_height_mm"], {}, rng)
    root = objs[0] if objs else None
    return {"log": log, "contacts": _plain(contacts), "solder_height": h_solder,
            "root": None if root is None else [_plain(root.location), _plain(root.rotation_euler)],
            "rng_after": rng.random()}


_PADS = {
    "chip_2pad": [[-30.0, 0.0, 20.0, 28.0], [30.0, 0.0, 20.0, 28.0]],
    "sot23": [[-25.0, -19.0, 14.0, 18.0], [-25.0, 19.0, 14.0, 18.0], [25.0, 0.0, 14.0, 22.0]],
    "soic_16": [[s * 52.0, -35.0 + 10.0 * i, 16.0, 6.0] for i in range(8) for s in (-1.0, 1.0)],
    "qfn_16": ([[s * 40.0, -12.0 + 8.0 * i, 10.0, 4.0] for i in range(4) for s in (-1.0, 1.0)]
               + [[-12.0 + 8.0 * i, s * 40.0, 4.0, 10.0] for i in range(4) for s in (-1.0, 1.0)]
               + [[0.0, 0.0, 36.0, 36.0]]),
}

# (footprint, body, mm_per_px, component_height_mm, seed): normal, rotated/shifted, tombstone, small.
_SCENARIOS = [
    ("chip_2pad", {"center": [3.0, -2.0], "length": 60.0, "width": 32.0, "rotation_deg": 7.5, "lead_frame_deg": 0.0,
                   "tilt_deg": 0.0, "is_tombstone": False}, 0.01, 0.45, 3),
    ("sot23", {"center": [-4.0, 5.0], "length": 58.0, "width": 26.0, "rotation_deg": 93.0, "lead_frame_deg": 90.0,
               "tilt_deg": 2.0, "is_tombstone": False}, 0.0137, 1.2, 11),
    ("soic_16", {"center": [1.5, 0.5], "length": 90.0, "width": 70.0, "rotation_deg": -4.0, "lead_frame_deg": 0.0,
                 "tilt_deg": 35.0, "is_tombstone": True, "raised_sign": -1.0}, 0.02, 1.75, 5),
    ("qfn_16", {"center": [0.0, 0.0], "length": 20.0, "width": 10.0, "rotation_deg": 0.0, "lead_frame_deg": 0.0,
                "tilt_deg": 0.0, "is_tombstone": False}, 0.01, 0.45, 7),
    ("soic_16", {"center": [6.0, -3.0], "length": 90.0, "width": 70.0, "rotation_deg": 12.0, "lead_frame_deg": 0.0,
                 "tilt_deg": 0.0, "is_tombstone": False}, 0.015, 1.75, 13),
]


def cases() -> dict[str, dict[str, Any]]:
    """Every family (plus an unknown name) on every scenario, and MISSING."""
    from simple_sim.board.footprints import FAMILIES, scene_part

    out: dict[str, dict[str, Any]] = {}
    for family in [*FAMILIES, "unknown_family"]:
        for i, (fp, body, mmpp, comp_h, seed) in enumerate(_SCENARIOS):
            board = {"family": family, "pads": _PADS[fp], "body": dict(body), "part": scene_part(family)}
            out[f"{family}/{i}_{fp}"] = {"board": board, "mm_per_px": mmpp, "component_height_mm": comp_h, "seed": seed}
        out[f"{family}/missing"] = {"board": {"family": family, "pads": _PADS["chip_2pad"], "body": None,
                                              "part": scene_part(family)},
                                    "mm_per_px": 0.01, "component_height_mm": 0.45, "seed": 1}
    return out


def all_traces(path: Path = SCRIPT) -> dict[str, Any]:
    mod = load_script(path)
    return json.loads(json.dumps({name: trace(mod, case) for name, case in cases().items()}))


def test_blender_part_geometry_matches_pre_d35_script():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    current = all_traces()
    assert set(current) == set(golden)
    changed = [name for name in golden if current[name] != golden[name]]
    assert not changed, f"Blender part geometry changed for {changed[:5]} ({len(changed)} cases)"


def test_part_params_reads_the_scene_not_the_family_name():
    """The family name is only a label: the scene's ``part`` decides the geometry."""
    from simple_sim.board.footprints import scene_part

    mod = load_script()
    case = cases()["resistor/0_chip_2pad"]
    board = dict(case["board"], family="soic")  # wrong name, resistor part: still a resistor
    p = mod.part_params([(-0.3, 0.0, 0.2, 0.28), (0.3, 0.0, 0.2, 0.28)], board, 0.01, 0.45, random.Random(3))
    assert p["end_caps"] and p["cap"] == 0.22 * 0.6 and p["marking"][0] in scene_part("resistor")["marking"]["codes"]


def test_bake_scene_carries_the_part_table():
    import numpy as np

    from simple_sim.board.footprints import scene_part
    from simple_sim.defects import sample_defect_params
    from simple_sim.generators.opencv2d import sample_nominal_geometry
    from simple_sim.generators.opencv2d.photoreal import render_board

    tol = {"ok_shift_px": 3.0, "ok_rotation_deg": 5.0, "tombstone_tilt_deg": 75.0}
    for footprint, component_type, family in (("chip_2pad", "chip_capacitor", "capacitor"), ("sot23", "", "sot")):
        rng = np.random.default_rng(3)
        nominal = sample_nominal_geometry({}, rng, {"pad_spacing_y": [36.0, 50.0]})
        defect = sample_defect_params("OK", rng, tol)
        _, scene = render_board(nominal, defect, {"component_type": component_type}, tol, np.random.default_rng(3),
                                footprint, (160, 160))
        assert scene["family"] == family and scene["part"] == scene_part(family)
        assert json.loads(json.dumps(scene["part"])) == scene["part"]  # survives the JSONL job file


def main(argv: list[str]) -> int:
    if "--update" not in argv:
        print(__doc__)
        return 2
    path = Path(argv[argv.index("--script") + 1]) if "--script" in argv else SCRIPT
    GOLDEN.write_text(json.dumps(all_traces(path), indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN} from {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
