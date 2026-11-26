"""simple_sim.board3d.collide: body collisions and the outline fix (synthetic parts only)."""

from __future__ import annotations

import numpy as np

from simple_sim.board3d.collide import apply_collide, find_collisions, overlap, resolve


def body(ref, x, y, l, w, *, rot=0.0, h=5.0, kind="connector", pnp=None):
    return {"refdes": ref, "pose": {"x": x, "y": y, "rotDeg": rot, "z": 0.0}, "l": l, "w": w, "height": h, "pads": [],
            "part": {"board": kind}, "missing": False, "pnp": pnp or [x, y]}


def test_apart_and_touching_bodies_do_not_collide():
    a, b = body("A", 0, 0, 4, 2), body("B", 4.05, 0, 4, 2)
    assert overlap(a, b) is None
    assert find_collisions([a, body("C", 4.0 - 0.05, 0, 4, 2)]) == []      # 0.05 mm contact


def test_z_ranges_must_overlap():
    a = body("A", 0, 0, 4, 4, h=1.0)
    b = body("B", 1, 0, 4, 4, h=1.0)
    b["pose"]["z"] = 2.0                                                    # floats above A
    assert find_collisions([a, b]) == []


def test_depth_axis_and_rotation():
    a, b = body("A", 0, 0, 10, 4), body("B", 0, 3, 4, 10, rot=90)        # B rotated: 10 x 4 along x
    (col,) = find_collisions([a, b])
    assert abs(col["depthMm"] - 1.0) < 1e-6 and col["axis"] == [0.0, 1.0]


def test_catalog_size_slides_onto_the_pins():
    # two vertical connectors whose app outline (11 mm deep) is far deeper than the datasheet (4.5 mm)
    a = body("J1", 5.0, 30, 14.0, 11.0, rot=90)
    b = body("J2", 11.0, 30, 14.0, 11.0, rot=90)
    holes = [(5.0, 30 + d, 0.4) for d in (-5, -3, -1, 1, 3, 5)] + [(11.0, 30 + d, 0.4) for d in (-5, -3, -1, 1, 3, 5)]
    res = resolve([a, b], catalog={"J1": {"w": 4.5}, "J2": {"w": 4.5}}, holes=holes)
    assert res["left"] == []
    for c, px in ((a, 5.0), (b, 11.0)):
        assert abs(c["w"] - 4.5) < 1e-6
        lo, hi = c["pose"]["x"] - 2.25, c["pose"]["x"] + 2.25                 # rot 90: w along board x
        assert lo <= px <= hi                                               # still around its own pins
    assert set(res["fixed"]) == {"J1", "J2"}


def test_small_hard_part_inside_a_large_outline_is_cut_out_sideways():
    usb = body("J", 0, 0, 16, 20, h=15)
    led = body("D", -7.6, 2, 0.8, 1.6, h=0.6, kind="chip")                 # near the left edge, mid height
    res = resolve([usb, led])
    assert res["left"] == [] and "J" in res["fixed"] and "D" not in res["fixed"]
    assert abs(usb["w"] - 20) < 1e-6 and usb["l"] < 16                    # lost a strip on the left, kept its depth
    assert usb["pose"]["x"] - usb["l"] / 2 >= -7.2 + 0.05 - 1e-6


def test_two_hard_bodies_are_only_reported():
    a, b = body("U", 0, 0, 10, 10, kind="bga"), body("C", 5.2, 0, 1.0, 0.5, kind="chip")
    res = resolve([a, b])
    assert res["fixed"] == {} and len(res["left"]) == 1


def test_fit_grown_body_flagged_soft_takes_its_name_size():
    u = body("U", 0, 0, 11.0, 15.8, kind="bga")
    u["soft"] = True
    c = body("C", 0.0, 7.4, 1.0, 0.5, kind="chip")
    res = resolve([u, c], catalog={"U": {"l": 11.5, "w": 13.0}})
    assert res["left"] == [] and abs(u["w"] - 13.0) < 1e-6


def test_apply_collide_sets_pose_and_size():
    b3 = {"components": [{"refdes": "J1", "pose": {"x": 0, "y": 0, "rotDeg": 0}, "body": {"l": 5, "w": 9, "h": 6}}]}
    apply_collide(b3, {"J1": {"x": 1.0, "y": -2.0, "l": 5.0, "w": 4.5}})
    c = b3["components"][0]
    assert c["pose"]["x"] == 1.0 and c["pose"]["y"] == -2.0 and c["body"]["w"] == 4.5 and c["body"]["h"] == 6
    assert np.isclose(c["body"]["l"], 5.0)


def test_stacked_headers_stay_on_their_own_pins():
    # three 1x2 headers at 2.54 mm pitch whose app outlines are 5.6 mm high: each must end up centred on its own
    # pin row, not on the neighbour's (overlapping outlines contain each other's pins)
    ys = (10.0, 7.46, 4.92)
    parts = [body(f"J{i}", 20.0, y, 4.2, 5.6) for i, y in enumerate(ys)]
    holes = [(x, y, 0.4) for y in ys for x in (18.73, 21.27)]
    res = resolve(parts, catalog={p["refdes"]: {"w": 2.54} for p in parts}, holes=holes)
    assert res["left"] == []
    for p, y in zip(parts, ys):
        assert abs(p["w"] - 2.54) < 1e-6 and abs(p["pose"]["y"] - y) < 0.06
