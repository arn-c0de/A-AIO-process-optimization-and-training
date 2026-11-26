"""Recording stub for ``simple_sim/blender/render_batch.py`` without Blender (shared by the Blender tests).

:func:`load_script` imports the script against stub ``bpy`` / ``mathutils`` modules; :func:`trace` records
everything ``_build_part`` creates for one case as plain JSON values.
"""

from __future__ import annotations

import importlib.util
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

    def body(name, shape, loc, mat, *, parent=None):
        log.append(["body", name, [round(shape.l, 9), round(shape.w, 9), round(shape.h, 9), round(shape.r_corner, 9),
                                   round(shape.r_top, 9), shape.draft_deg, shape.parting], _plain(loc), mat,
                    getattr(parent, "name", None)])
        return _Obj(name)

    def strip(name, path, thickness, width, origin, direction, mat):
        log.append(["strip", name, [[round(a, 9), round(b, 9)] for a, b in path], thickness, width, _plain(list(origin)),
                    _plain(list(direction)), mat])
        return _Obj(name)

    def box_between(name, p0, p1, width, thickness, mat):
        log.append(["lead", name, _plain(p0), _plain(p1), width, thickness, mat])
        return _Obj(name)

    def text(name, body, size, loc, mat, parent):
        log.append(["text", name, body, size, _plain(loc), mat, parent.name])

    mod.bpy.data = SimpleNamespace(objects=SimpleNamespace(new=new_empty))
    mod.bpy.context = ctx
    mod.bpy.ops = SimpleNamespace(mesh=SimpleNamespace(primitive_cylinder_add=cylinder))
    with mock.patch.multiple(mod, _material=material, _cfg_material=cfg_material, _box=box, _body=body, _strip=strip,
                             _box_between=box_between, _text=text):
        mmpp = case["mm_per_px"]
        pads = [(*mod._mm_xy(p[0], p[1], mmpp), mod._mm(p[2], mmpp), mod._mm(p[3], mmpp)) for p in case["board"]["pads"]]
        rng = random.Random(case["seed"])
        objs, contacts, h_solder = mod._build_part(pads, case["board"], mmpp, case["component_height_mm"], {}, rng)
    root = objs[0] if objs else None
    return {"log": log, "contacts": _plain(contacts), "solder_height": h_solder,
            "root": None if root is None else [_plain(root.location), _plain(root.rotation_euler)],
            "rng_after": rng.random()}
