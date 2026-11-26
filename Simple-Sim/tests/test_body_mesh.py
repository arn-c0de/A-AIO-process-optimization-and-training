"""Part body meshes of the 3D-Board-Renderer (``simple_sim.blender.body_mesh``, no Blender needed)."""

import math

import pytest

from simple_sim.blender import body_mesh as bm


def _volume(m: bm.BodyMesh) -> float:
    vol = 0.0
    for f in m.faces:
        p = m.verts[f[0]]
        for i in range(1, len(f) - 1):
            a, b = m.verts[f[i]], m.verts[f[i + 1]]
            vol += (p[0] * (a[1] * b[2] - a[2] * b[1]) - p[1] * (a[0] * b[2] - a[2] * b[0]) + p[2] * (a[0] * b[1] - a[1] * b[0])) / 6
    return vol


def _ring_at(m: bm.BodyMesh, z: float):
    return [v for v in m.verts if abs(v[2] - z) < 1e-9]


SHAPES = {
    "chip": bm.chip_body_shape(1.6, 0.8, 0.45),
    "cap": bm.chip_cap_shape(0.33, 0.82, 0.46),
    "gullwing": bm.molded_shape(4.9, 3.9, 1.5, leads=True),
    "qfn": bm.molded_shape(5.0, 5.0, 0.9, leads=False),
    "base": bm.can_base_shape(6.6, 6.6, 0.86, (1, 2)),
    "can": bm.can_shape(6.3, 4.5),
}


@pytest.mark.parametrize("kind", SHAPES)
def test_closed_outward_mesh_with_sharp_flat_bottom(kind):
    shape = SHAPES[kind]
    m = bm.body_mesh(shape)
    zs = [v[2] for v in m.verts]
    assert min(zs) == pytest.approx(-shape.h / 2) and max(zs) == pytest.approx(shape.h / 2)
    assert 0.9 * shape.l * shape.w * shape.h * (math.pi / 4 if kind == "can" else 1) < _volume(m)   # outward, near full
    assert len(m.smooth) == len(m.faces)
    assert m.smooth[-2:] == [False, False]                    # flat bottom and top faces
    # the bottom ring is followed by a vertical / drafted wall, never by a rounding: its edge stays sharp
    bottom = _ring_at(m, -shape.h / 2)
    n = len(bottom)
    assert n == len(_ring_at(m, shape.h / 2))
    if kind not in ("gullwing", "can"):
        assert max(abs(v[0]) for v in bottom) == pytest.approx(shape.l / 2)


def test_edge_radii_are_small_and_absolute():
    # a 10 mm body keeps a sub-0.2 mm corner: the radius does not grow with the length any more
    for shape in (bm.molded_shape(10, 10, 1.4, leads=True), bm.chip_body_shape(6.4, 3.2, 0.6), bm.box_shape((10, 4, 2), 0.24)):
        assert shape.r_corner <= 0.15 and shape.r_top <= 0.06


def test_gullwing_draft_top_smaller_than_parting_line():
    shape = SHAPES["gullwing"]
    m = bm.body_mesh(shape, center_z=False)
    top = max(abs(v[0]) for v in _ring_at(m, shape.h))
    parting = max(abs(v[0]) for v in _ring_at(m, shape.parting * shape.h))
    bottom = max(abs(v[0]) for v in _ring_at(m, 0.0))
    assert parting == pytest.approx(shape.l / 2)
    assert top < parting - 0.1 and bottom < parting


def test_flat_walls_are_flat_shaded():
    m = bm.body_mesh(SHAPES["qfn"])
    flat = sum(1 for s in m.smooth if not s)
    assert flat >= 4 + 2          # four straight walls + top + bottom


def test_can_base_chamfers_on_pin1_side():
    assert bm.plus_corners([[-2.6, 0, 1, 2], [2.6, 0, 1, 2]], 6.6, 6.6) == (1, 2)
    assert bm.plus_corners([[2.6, 0, 1, 2], [-2.6, 0, 1, 2]], 6.6, 6.6) == (0, 3)
    m = bm.body_mesh(bm.can_base_shape(6.6, 6.6, 0.86, (1, 2)), center_z=False)
    bottom = _ring_at(m, 0.0)
    # chamfered corners (-x): no point near the (-a, +-b) corner; the +x corners keep their small radius
    assert min(math.hypot(x + 3.3, abs(y) - 3.3) for x, y, _ in bottom) > 0.5
    assert min(math.hypot(x - 3.3, abs(y) - 3.3) for x, y, _ in bottom) < 0.1


# --------------------------------------------------------------------------- assemblies (board3d)

from simple_sim.board3d.footprints import footprint_for, gullwing_pads, qfn_pads  # noqa: E402


def _comp(fam, pkg, l, w, h):
    return {"refdes": "X1", "family": fam, "package": pkg, "body": {"l": l, "w": w, "h": h}, "dnp": False}


@pytest.mark.parametrize("fam,pkg,kind", [
    ("resistor", "RESC1608", "chip"), ("inductor", "INDC2012", "chip"), ("led", "LEDC1608", "chip"),
    ("diode", "DIOC1608", "chip"), ("diode", "SOD123", "tabs"), ("power_inductor", "INDM4040", "tabs"),
    ("sot6", "SOT95P280-6", "gullwing"), ("soic", "SOIC127P600-8", "gullwing"), ("qfn", "QFN50P500X500-32", "qfn"),
    ("array_r", "RESCAV50P200X100-8", "array"), ("crystal", "OSC3225-4", "crystal"), ("bga", "BGA64C80P8X8_700X700", "bga"),
    ("electrolytic", "CAPAE660", "can"), ("connector", "CONX", "box"),
])
def test_every_family_gets_its_body(fam, pkg, kind):
    part = footprint_for(_comp(fam, pkg, 2.0, 1.25, 1.0))["part"]
    assert part["board"] == kind
    if kind == "chip":
        assert part["body"] == "chip" and part["cap_len"]


def _closed_ok(m):
    assert _volume(m) > 0


def test_gullwing_leads_reach_the_lands_and_sit_on_the_board():
    l, w, h, so = 2.9, 1.6, 1.0, 0.08
    pads = [tuple(p) for p in gullwing_pads(l, w, "SOT95P280-6", "sot6")]
    asm = bm.gullwing(l, w, h, so, pads)
    leads = [p for p in asm.pieces if p.name.startswith("lead_")]
    assert len(leads) == 6 and all(c is not None for c in asm.contacts)
    for p in leads:
        _closed_ok(p.mesh)
        assert min(v[2] for v in p.mesh.verts) == pytest.approx(-so + 0.005, abs=0.01)   # foot on the land
    for (px, py, pw, ph), c in zip(pads, asm.contacts):
        assert abs(c["cx"] - px) <= pw / 2 + 1e-6 and abs(c["cy"] - py) <= ph / 2 + 1e-6   # foot over its land


def test_bend_path_keeps_straight_runs():
    path = bm.bend_path([(0, 1), (0.3, 1), (0.5, 0), (1.0, 0)], 0.15)
    assert path[0] == (0, 1) and path[-1] == (1.0, 0)
    assert any(abs(z) < 1e-9 and 0.6 < x < 1.0 for x, z in path[:-1])         # flat foot left before the end


def test_qfn_terminals_on_the_walls_and_assemblies_closed():
    pads, thermal = qfn_pads(5.0, 5.0, "QFN50P500X500-32_33")
    asm = bm.qfn(5.0, 5.0, 0.9, 0.01, [tuple(p) for p in pads], thermal)
    assert len([p for p in asm.pieces if p.name.startswith("term_")]) == 32
    for a in (asm, bm.crystal(3.2, 2.5, 0.8), bm.bga(7, 7, 1.2, 0.3, 0.8),
              bm.molded_tabs(2.7, 1.6, 1.0, 0.1, [(-1.6, 0, 0.9, 1.0), (1.6, 0, 0.9, 1.0)], draft=True),
              bm.electrolytic(6.6, 6.6, 5.4, [(-2.4, 0, 2.3, 2.0), (2.4, 0, 2.3, 2.0)])):
        for p in a.pieces:
            assert p.role in bm.ROLES
            if p.name != "balls":
                _closed_ok(p.mesh)


def test_electrolytic_stands_at_catalog_height_with_bead():
    asm = bm.electrolytic(6.6, 6.6, 5.4, [(-2.4, 0, 2.3, 2.0), (2.4, 0, 2.3, 2.0)])
    can = next(p for p in asm.pieces if p.name == "body_can")
    assert max(v[2] for v in can.mesh.verts) == pytest.approx(5.4)
    r = [math.hypot(x, y) for x, y, z in can.mesh.verts if 1.0 < z < 1.6]
    assert min(r) < 0.955 * 6.6 / 2 - 0.08                                     # the bead dents the wall


def test_tht_joint_and_dsub_standoffs():
    pieces = bm.tht_joint(10.0, 5.0, 0.5)
    solder = next(p for p in pieces if p.name == "tht_solder")
    zs = [v[2] for v in solder.mesh.verts]
    assert min(zs) < 0.01 and max(zs) == pytest.approx(0.002 + 0.45 * 0.5)              # meniscus from ring to pin
    pin = next(p for p in pieces if p.name == "tht_pin")
    assert max(v[2] for v in pin.mesh.verts) > max(zs)                                 # the pin end stands out
    a = bm.connector("dsub", "right_angle", 21.98, 12.9, 12.5, pins=9, rows=1, pitch=2.77, total_h=None, front=(0.0, -1.0))
    bolts = [p for p in a.pieces if p.name.startswith("bolt_") and not p.name.startswith("bolt_hole")]
    xs = sorted(sum(v[0] for v in p.mesh.verts) / len(p.mesh.verts) for p in bolts)
    assert xs == pytest.approx([-12.5, 12.5], abs=0.01)                                # DE-9 jackscrews, 25 mm apart
    assert len([p for p in a.pieces if p.name.startswith("socket_")]) == 9


def test_battery_holder_is_empty_unless_the_catalog_says_cell():
    # holders are placed empty (the cell goes in at final assembly): an always-present cell rendered as a saturated disc
    kw = dict(pins=3, rows=1, pitch=2.54, total_h=None, front=(0.0, -1.0))
    empty = bm.connector("battery_holder", "horizontal", 21.9, 18.2, 5.0, **kw)
    assert [p.name for p in empty.pieces] == ["body"]
    full = bm.connector("battery_holder", "horizontal", 21.9, 18.2, 5.0, cell=True, **kw)
    assert {"body", "cell", "clip"} <= {p.name for p in full.pieces}


def test_dsub_top_cover_variant():
    kw = dict(pins=9, rows=1, pitch=2.77, total_h=None, front=(0.0, -1.0))
    plain = bm.connector("dsub", "right_angle", 21.98, 12.9, 12.5, **kw)
    cover = bm.connector("dsub", "right_angle", 21.98, 12.9, 12.5, top_cover="metal", **kw)
    assert "cover" not in {p.name for p in plain.pieces}
    lid = next(p for p in cover.pieces if p.name == "cover")
    assert lid.role == "lid" and min(v[2] for v in lid.mesh.verts) == pytest.approx(0.75 * 12.5)
