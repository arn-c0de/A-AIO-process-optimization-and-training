"""Board model: footprints match the legacy pads, the escape router passes the DRC, legacy stays bit-identical."""

import logging

import numpy as np
import pytest

from simple_sim.board import (BOARD_ROUTERS, STREAM_PLACE, BoardDrcError, BoardOptions, board_options, build_window_board,
                              family_of, footprint_pads, neighbour_layout, validate_board_options)
from simple_sim.board.drc import UNCONNECTED, check
from simple_sim.board.footprints import pads_px, part_from_pads
from simple_sim.board.geometry import Box, Path, box, gap, track
from simple_sim.board.model import GND, Board, point_in_polygon
from simple_sim.board.occupancy import CopperMap, Grid
from simple_sim.board.raster import rasterize
from simple_sim.board.routing.base import runs_along_border, seed_occupancy
from simple_sim.board.routing.escape import EscapeRouter
from simple_sim.board.rules import PRESETS, fit_to_pitch, sample_rules
from simple_sim.board.zones import zone_fill
from simple_sim.generators.opencv2d import sample_nominal_geometry
from simple_sim.generators.opencv2d.geometry import pad_positions
from tests.test_photoreal import FOOTPRINTS, _render

WINDOW = (512, 512)   # 256 px ROI at 2x supersampling, as in the run configs
PX_TO_MM = 0.01 / 2
# The renderer's footprints plus the newer packages (geometry ranges of their @1 profiles).
BOARD_FOOTPRINTS = {
    **FOOTPRINTS,
    "sot_6": {"pad_width": [18.0, 21.0], "pad_height": [30.0, 34.0], "pad_spacing": [80.0, 85.0],
              "pad_spacing_y": [60.0, 62.0], "component_length": [50.0, 53.0], "component_width": [91.0, 95.0]},
    "chip_array_8": {"pad_width": [12.0, 15.0], "pad_height": [19.0, 24.0], "pad_spacing": [44.0, 50.0],
                     "pad_spacing_y": [66.0, 69.0], "component_length": [44.0, 47.0], "component_width": [88.0, 92.0]},
    "tssop_20": {"pad_width": [5.5, 6.5], "pad_height": [21.0, 24.0], "pad_spacing": [85.0, 89.0],
                 "pad_spacing_y": [87.0, 89.0], "component_length": [64.0, 67.0], "component_width": [96.0, 99.0]},
}
PRESET_NAMES = sorted(PRESETS)


def _window(footprint: str, seed: int, options: BoardOptions = BoardOptions(routing="escape", drc="warn"), *,
            shape=WINDOW, scale: float = 1.0, target_rotation: float = 0.0, keep: list | None = None):
    rng = np.random.default_rng(seed)
    nom = {k: float(v) * 2 for k, v in sample_nominal_geometry({}, rng, BOARD_FOOTPRINTS[footprint]).items()}
    specs = footprint_pads(footprint, nom, shape)
    pads = [p.as_dict() for p in specs]
    xs = [p["x"] for p in pads] + [p["x"] + p["w"] for p in pads]
    ys = [p["y"] for p in pads] + [p["y"] + p["h"] for p in pads]
    neighbours = neighbour_layout(pads, (min(xs), min(ys), max(xs), max(ys)), shape, np.random.default_rng([seed, STREAM_PLACE]))
    before = [dict(n) for n in neighbours]
    built = build_window_board(footprint=footprint, family=family_of(footprint), target_pads=specs,
                              target_body_px=(nom["component_length"], nom["component_width"]), shape=shape,
                              neighbours=neighbours, px_to_mm=PX_TO_MM * scale, options=options, seed=seed, scale=scale,
                              target_rotation=target_rotation)
    assert neighbours == before  # the caller's list is not modified
    if keep is not None:
        keep.append(built.neighbours)
    return built.board, built.report


# ---- footprints ----------------------------------------------------------------------------

@pytest.mark.parametrize("footprint", list(BOARD_FOOTPRINTS))
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("shape", [(128, 128), (256, 256), (257, 255)])
def test_footprints_match_legacy_pads(footprint, seed, shape):
    nominal = sample_nominal_geometry({}, np.random.default_rng(seed), BOARD_FOOTPRINTS[footprint])
    specs = footprint_pads(footprint, nominal, shape)
    assert [p.as_dict() for p in specs] == pad_positions(footprint, nominal, shape)
    part = part_from_pads("U1", footprint, family_of(footprint), specs, (shape[1] / 2, shape[0] / 2), (10.0, 5.0), 0.0, 0.0137)
    assert pads_px(part, 0.0137) == [p.as_dict() for p in specs]  # mm round trip keeps the integers


def test_footprint_roles_and_escape_directions():
    specs = footprint_pads("qfn_32", {"pad_width": 16, "pad_height": 40, "pad_spacing": 180, "pad_spacing_y": 180,
                                      "component_length": 168, "component_width": 168}, (256, 256))
    assert specs[-1].pin == "EP" and specs[-1].role == "thermal" and specs[-1].escape == (0.0, 0.0)
    assert {p.escape for p in specs[:-1]} == {(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)}
    assert len({p.pin for p in specs}) == 33
    chip = footprint_pads("chip_2pad", {"pad_width": 30, "pad_height": 35, "pad_spacing": 60}, (128, 128))
    assert [p.escape for p in chip] == [(-1.0, 0.0), (1.0, 0.0)]
    assert family_of("sot23") == "sot" and family_of("chip_2pad", "chip_capacitor") == "capacitor"


@pytest.mark.parametrize("footprint, pins, family", [("sot_6", 6, "sot6"), ("chip_array_8", 8, "array_r"), ("tssop_20", 20, "soic")])
def test_new_footprints_in_the_board_model(footprint, pins, family):
    """The newer packages have their pin count, family, two escaping rows and a refdes."""
    nominal = sample_nominal_geometry({}, np.random.default_rng(0), BOARD_FOOTPRINTS[footprint])
    specs = footprint_pads(footprint, nominal, (256, 256))
    assert sorted(int(p.pin) for p in specs) == list(range(1, pins + 1))
    assert {p.escape for p in specs} == {(-1.0, 0.0), (1.0, 0.0)}
    assert family_of(footprint) == family
    assert family_of("chip_array_8", "chip_capacitor_array") == "array_c"
    board, report = _window(footprint, 1, BoardOptions(routing="escape", drc="warn"))
    assert report.ok, report.violations
    assert board.target.ref[0] in "UR" and len(board.target.pads) == pins


# ---- model + router ------------------------------------------------------------------------

@pytest.mark.parametrize("footprint", list(BOARD_FOOTPRINTS))
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_window_board_drc_clean(footprint, preset):
    """Every footprint, every preset, a pour on every board: no violation at all (incl. connectivity)."""
    for seed in range(12):
        board, report = _window(footprint, seed, BoardOptions(routing="escape", drc="warn", design_rules=preset,
                                                              pour_probability=1.0))
        assert isinstance(board, Board) and report is not None
        assert report.ok, (seed, report.violations)


@pytest.mark.parametrize("footprint", ["chip_2pad", "sot23", "qfn_32", "soic_16"])
def test_window_board_drc_clean_at_other_scales(footprint):
    """Physical-scale profiles pass ``scale`` (rules follow mm_per_px) and the SOT body may stand upright."""
    for seed in range(6):
        for scale in (0.5, 3.0):
            board, report = _window(footprint, seed, BoardOptions(routing="escape", drc="warn", design_rules=PRESET_NAMES[seed % 3]),
                                    scale=scale, target_rotation=90.0 if footprint == "sot23" else 0.0)
            assert report.ok, (seed, scale, report.violations)


def test_target_pads_connected():
    """Each connected pin of the inspected part is reached by a track or via of its net or by its pour;
    a ground pad inside a pour is joined to the pour copper (relief spoke or track), never isolated."""
    in_pour = reliefs = 0
    for footprint in ("chip_2pad", "sot23", "qfn_32"):
        for seed in range(10):
            board, report = _window(footprint, seed, BoardOptions(routing="escape", drc="warn", pour_probability=1.0))
            assert not any(v.kind == UNCONNECTED for v in report.violations), (footprint, seed, report.violations)
            grid = Grid.for_window(board.bbox, board.rules.clearance / 8.0)
            fills = [zone_fill(board, z, grid) for z in board.zones]
            for pad in board.target.pads:
                if pad.net != GND or pad.role == "thermal" or not any(point_in_polygon(pad.center, z.outline) for z in board.zones):
                    continue
                in_pour += 1
                own = [t for t in board.tracks if t.net == GND and gap(track(t.points, t.width), box(pad.bbox)) == 0.0]
                assert own, (footprint, seed, pad)
                if footprint == "qfn_32":
                    continue  # fine-pitch ground pins fan out like any pin (DRC: no dangling end, connected)
                # Chip-sized parts get a relief spoke (a lone track that ends in the pour) unless it does not fit
                # (then the pad fans out to a via or the border like any other); a spoke reaches pour copper.
                for t in own:
                    tip = t.points[-1]
                    continued = any(o is not t and o.net == GND and o.points[0] == tip for o in board.tracks)
                    via = any(v.net == GND and v.center == tip for v in board.vias)
                    if continued or via or board.target.ref != t.part:
                        continue
                    reliefs += 1
                    reach = grid.polyline(t.points, t.width + 2 * grid.res)
                    assert any(np.any((reach > 0) & (f > 0)) for f in fills), (footprint, seed, pad)
    assert in_pour >= 5 and reliefs >= 3


def test_thermal_pad_has_vias_and_no_track():
    for seed in range(4):
        board, _ = _window("qfn_32", seed)
        ep = [p for p in board.target.pads if p.role == "thermal"][0]
        assert ep.net == GND
        x0, y0, x1, y1 = ep.bbox
        inside = [v for v in board.vias if x0 <= v.center[0] <= x1 and y0 <= v.center[1] <= y1]
        assert len(inside) >= 4 and all(v.net == GND for v in inside)
        for v in inside:  # every via disc lies on the pad, not only its centre
            r = v.diameter / 2
            assert x0 <= v.center[0] - r and v.center[0] + r <= x1 and y0 <= v.center[1] - r and v.center[1] + r <= y1
        for t in board.tracks:
            assert gap(track(t.points, t.width), box(ep.bbox)) > 0, "track on the thermal pad"


def test_no_foreign_copper_in_target_keepout():
    for footprint in ("sot23", "soic_16"):
        for seed in range(8):
            board, _ = _window(footprint, seed)
            own = {p.net for p in board.target.pads if p.net}
            keep = box(board.target.keepout)
            for t in board.tracks:
                if t.net not in own:
                    assert gap(track(t.points, t.width), keep) > 0, (seed, t)


def test_no_track_runs_along_the_window_border():
    for footprint in ("chip_2pad", "sot23", "qfn_32", "soic_16"):
        for seed in range(15):
            board, _ = _window(footprint, seed)
            for t in board.tracks:
                assert not (len(t.points) > 2 and runs_along_border(board.bbox, t.points, t.width / 2)), (footprint, seed, t)


def test_nets_are_isolated_from_routing_stream():
    a, _ = _window("chip_2pad", 3, BoardOptions(routing="escape"))
    b, _ = _window("chip_2pad", 3, BoardOptions(routing="escape", pour_probability=0.0))
    assert [p.net for p in a.target.pads] == [p.net for p in b.target.pads]
    assert [part.ref for part in a.parts] == [part.ref for part in b.parts]
    assert a.zones and not b.zones


def test_drc_strict_rerolls_until_clean(caplog):
    with caplog.at_level(logging.WARNING, logger="simple_sim.board"):
        for seed in range(4):
            board, report = _window("chip_2pad", seed, BoardOptions(routing="escape", drc="strict"))
            assert report is not None and report.ok
    assert not caplog.records


def test_drc_strict_raises_when_every_attempt_fails(monkeypatch):
    import simple_sim.board.pipeline as board_pkg
    from simple_sim.board.drc import DrcReport

    def always_bad(board):
        report = DrcReport()
        report.add("clearance", "forced")
        return report

    monkeypatch.setattr(board_pkg, "check", always_bad)
    with pytest.raises(BoardDrcError) as err:
        _window("chip_2pad", 1, BoardOptions(routing="escape", drc="strict"))
    assert err.value.report.counts() == {"clearance": 1}
    board, report = _window("chip_2pad", 1, BoardOptions(routing="escape", drc="warn"))
    assert not report.ok  # warn returns the board with its report


def test_drc_catches_short_and_dangling():
    board, report = _window("chip_2pad", 1, BoardOptions(routing="escape", drc="warn"))
    assert report.ok
    from simple_sim.board.model import TOP, Track

    pad_a, pad_b = board.target.pads
    board.tracks.append(Track("SHORT", TOP, [pad_a.center, pad_b.center], 0.05))
    kinds = check(board).counts()
    assert kinds.get("clearance") and kinds.get("dangling") and kinds.get("keepout")


def test_drc_flags_a_through_net_next_to_a_foreign_pad():
    """No per-net exemption: a GND track passing a signal pad of the target closer than the clearance is caught,
    even though GND pads exist elsewhere on the board."""
    from simple_sim.board.model import TOP, Track

    board, report = _window("sot23", 3, BoardOptions(routing="escape", drc="warn"))
    assert report.ok
    pad = next(p for p in board.target.pads if p.net and p.net != GND)
    x0, y0, x1, y1 = pad.bbox
    y = y0 - board.rules.clearance * 0.5 - 0.01
    board.tracks.append(Track(GND, TOP, [(0.0, y), (board.size[0], y)], 0.02))
    assert check(board).counts().get("clearance")


def test_drc_flags_isolated_target_pad():
    board, report = _window("chip_2pad", 2, BoardOptions(routing="escape", drc="warn", pour_probability=0.0))
    assert report.ok
    pad = board.target.pads[0]
    board.tracks = [t for t in board.tracks if gap(track(t.points, t.width), box(pad.bbox)) > 0]
    assert check(board).counts().get(UNCONNECTED)


def test_board_serialises():
    import json

    board, _ = _window("sot23", 2)
    d = board.to_dict()
    json.dumps(d)
    assert d["rules"]["clearance"] > 0 and len(d["parts"]) == len(board.parts)


def test_rasterize_matches_window():
    board, _ = _window("chip_2pad", 4)
    copper = rasterize(board, WINDOW, PX_TO_MM)
    assert copper.copper.shape == WINDOW and copper.copper.dtype == np.uint8 and copper.copper.max() == 255
    for via in copper.vias:
        assert 0 <= via.x < WINDOW[1] and 0 <= via.y < WINDOW[0] and via.radius >= 1


def test_rules_scale_and_pitch_fit():
    rules = sample_rules("industrial", np.random.default_rng(0))
    double = sample_rules("industrial", np.random.default_rng(0), 2.0)
    assert double.clearance == pytest.approx(2 * rules.clearance) and double.via_diameter == pytest.approx(2 * rules.via_diameter)
    assert fit_to_pitch(rules, 10.0) is rules
    fitted = fit_to_pitch(rules, 0.1)
    assert fitted.neck_down + fitted.clearance <= 0.85 * 0.1 + 1e-12
    assert fitted.via_diameter / fitted.clearance == pytest.approx(rules.via_diameter / rules.clearance)


def test_rules_presets_sample_within_ranges():
    for name, preset in PRESETS.items():
        rules = sample_rules(name, np.random.default_rng(0))
        assert preset.clearance[0] <= rules.clearance <= preset.clearance[1]
        assert rules.via_diameter > rules.via_drill
        assert rules.width("power") > rules.width("signal")


# ---- config + renderer ---------------------------------------------------------------------

def test_board_options_validation():
    assert board_options(None).legacy and board_options({"board": None}).routing == "legacy"
    assert board_options({"board": {"routing": "escape", "drc": "strict"}}).drc == "strict"
    for bad in ({"routing": "astar"}, {"design_rules": "none"}, {"pour_probability": 2}, {"drc": "maybe"}, {"foo": 1}, "escape"):
        with pytest.raises(ValueError):
            validate_board_options(bad)
    assert set(BOARD_ROUTERS) == {"legacy", "none", "escape"}


def test_legacy_routing_is_the_default_and_unchanged(monkeypatch):
    a = _render("chip_2pad", "OK", seed=5)
    b = _render("chip_2pad", "OK", seed=5, board={"routing": "legacy"})
    assert np.array_equal(a, b)
    # Legacy never builds a board model (router, zones, DRC changes cannot reach it).
    from simple_sim.generators.opencv2d.photoreal import copper

    def boom(**_kw):
        raise AssertionError("legacy rendering called build_window_board")

    monkeypatch.setattr(copper, "build_window_board", boom)
    for footprint in FOOTPRINTS:
        _render(footprint, "OK", seed=5)


@pytest.mark.parametrize("footprint", list(FOOTPRINTS))
def test_escape_routing_renders_every_footprint(footprint):
    img = _render(footprint, "MISSING", seed=2, board={"routing": "escape"}, mm_per_px=0.01)
    assert img.shape == (128, 128, 3) and 10 < img.mean() < 200
    assert not np.array_equal(img, _render(footprint, "MISSING", seed=2))
    assert np.array_equal(img, _render(footprint, "MISSING", seed=2, board={"routing": "escape"}, mm_per_px=0.01))


def test_routing_none_has_no_copper_but_keeps_the_rest():
    none = _render("chip_2pad", "OK", seed=9, board={"routing": "none"})
    legacy = _render("chip_2pad", "OK", seed=9)
    assert not np.array_equal(none, legacy)


# ---- geometry, copper map, raster ------------------------------------------------------------

def _dense(shape):
    """Points densely covering *shape* (brute force reference for the exact gap)."""
    if isinstance(shape, Box):
        xs, ys = np.meshgrid(np.linspace(shape.x0, shape.x1, 40), np.linspace(shape.y0, shape.y1, 40))
        return np.stack([xs.ravel(), ys.ravel()], axis=1), 0.0
    pts = []
    for a, b in shape.segments():
        t = np.linspace(0.0, 1.0, 250)[:, None]
        pts.append(np.array(a) * (1 - t) + np.array(b) * t)
    return np.concatenate(pts), shape.r


def _random_shape(rng):
    if rng.random() < 0.4:
        x, y = rng.uniform(0, 1, 2)
        return Box(x, y, x + rng.uniform(0.02, 0.3), y + rng.uniform(0.02, 0.3))
    pts = [tuple(rng.uniform(0, 1.3, 2)) for _ in range(int(rng.integers(1, 4)))]
    return Path(tuple(pts), float(rng.uniform(0.005, 0.06)))


def test_exact_gap_matches_brute_force():
    """Router and DRC share :func:`gap`: compare it with dense sampling (no false conflicts, no misses)."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        s, t = _random_shape(rng), _random_shape(rng)
        ps, rs = _dense(s)
        pt, rt = _dense(t)
        d = np.sqrt(((ps[:, None, :] - pt[None, :, :]) ** 2).sum(-1)).min() - rs - rt
        # Sampling only over-estimates the distance (by < one sample step) and cannot see overlap depth.
        g = gap(s, t)
        assert g <= max(d, 0.0) + 1e-9
        assert g >= max(d, 0.0) - 0.015
        # Clearance decisions are exact: flagged iff the gap is below the clearance (no raster slack).
        clearance = 0.05
        cm = CopperMap((0, 0, 2, 2))
        cm.add(t, "B")
        assert cm.conflicts(s, "A", clearance) == (g < clearance or g <= 1e-9)


def test_copper_map_groups_and_removal():
    cm = CopperMap((0, 0, 1, 1))
    g = cm.group("U1")
    cm.add(box((0.0, 0.0, 0.5, 0.5)), None, group=g, blocked=True)        # own keep-out
    cm.add(box((0.10, 0.10, 0.20, 0.20)), "A", group=g)                     # own pad, other net
    stub = track([(0.15, 0.22), (0.15, 0.6)], 0.02)
    assert cm.conflicts(stub, "B", 0.05)                                    # full clearance: too close to pad A
    assert not cm.conflicts(stub, "B", 0.05, group=g, group_clearance=0.005)  # own pin row: pitch clearance
    other = cm.group("R2")
    t = cm.add(box((0.17, 0.62, 0.3, 0.7)), None, group=other, blocked=True)
    assert cm.conflicts(stub, "B", 0.05, group=g, group_clearance=0.005)    # another part's keep-out: full clearance
    cm.remove(t)
    assert not cm.conflicts(stub, "B", 0.05, group=g, group_clearance=0.005)


def test_failed_fanout_is_taken_back_exactly():
    """A stub whose continuation finds no route is removed; nothing else in the map changes (no leaked via)."""
    from simple_sim.board.routing.base import NetWidths, unit

    board, _r = _window("chip_2pad", 4, BoardOptions(routing="escape", pour_probability=0.0))
    board.tracks, board.vias = [], []
    occ = seed_occupancy(board)
    part = board.target
    pad = part.pads[0]
    d = unit(pad.escape_dir)
    rules = board.rules
    router = EscapeRouter()
    stub, _w = router._stub(board, part, pad, NetWidths(board, np.random.default_rng(0)), np.random.default_rng(1))
    end = stub[-1]
    # Wall of foreign copper just beyond the stub end (the stub fits, nothing after it does).
    x = end[0] + d[0] * (1.01 * rules.clearance)
    wall = (x - 5.0, 0.0, x, board.size[1]) if d[0] < 0 else (x, 0.0, x + 5.0, board.size[1])
    occ.add(box(wall), "WALL")
    before = [i for i in occ.items if i is not None]
    res = router._fanout(board, occ, part, pad, NetWidths(board, np.random.default_rng(0)), np.random.default_rng(1))
    assert res.unrouted and not res.tracks and not res.vias
    assert [i for i in occ.items if i is not None] == before


def test_grid_rect_covers_exactly_its_cells():
    g = Grid(origin=(0.0, 0.0), res=1.0, shape=(20, 20))
    m = g.rect((2.0, 3.0, 7.0, 5.0))
    assert int((m > 0).sum()) == 5 * 2
    assert Grid.for_window((0, 0, 1, 1), 0.1).shape == (11, 11)


def _rows_mask(grid, rows):
    mask = grid.scratch()
    for y, x0, x1 in rows:
        mask[y, x0:x1] = 255
    return mask


def test_inflated_rect_rows_equal_inflate_of_rect():
    """Fast span path == dilating the drawn rect, incl. boxes partly or wholly outside the grid."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        res = float(rng.uniform(0.01, 0.2))
        rows, cols = (int(v) for v in rng.integers(4, 60, size=2))
        g = Grid(origin=(float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))), res=res, shape=(rows, cols))
        x0, y0 = g.origin[0] + rng.uniform(-0.3, cols * res + 0.3), g.origin[1] + rng.uniform(-0.3, rows * res + 0.3)
        bbox = (x0, y0, x0 + rng.uniform(0, 12) * res, y0 + rng.uniform(0, 12) * res)
        mm = float(rng.choice([0.0, rng.uniform(0, 15) * res]))
        want = g.inflate(g.rect(bbox), mm)
        got = g.inflated_rect_rows(bbox, mm)
        assert isinstance(got, list)
        assert np.array_equal(_rows_mask(g, got), want)
        assert np.array_equal(g.inflated_rect(bbox, mm), want)


def test_inflated_rect_rows_fallback_without_kernel_spans(monkeypatch):
    from simple_sim.board import occupancy

    monkeypatch.setattr(occupancy, "_ellipse_spans", lambda n: None)
    g = Grid(origin=(0.0, 0.0), res=0.1, shape=(30, 40))
    bbox, mm = (1.0, 1.0, 2.0, 1.6), 0.45
    rows = g.inflated_rect_rows(bbox, mm)
    assert rows and np.array_equal(_rows_mask(g, rows), g.inflate(g.rect(bbox), mm))
    assert occupancy._mask_rows(np.array([[0, 255, 255, 0, 255]], dtype=np.uint8)) == [(0, 1, 3), (0, 4, 5)]


def test_neighbours_in_the_fanout_band_are_left_out():
    for seed in range(10):
        keep: list = []
        board, _ = _window("qfn_32", seed, keep=keep)
        assert len(keep[0]) == len(board.parts) - 1   # renderer list == model neighbours
        for n in keep[0]:
            vertical = abs(float(n["angle"]) % 180.0 - 90.0) < 45.0
            part = next(p for p in board.parts if not p.is_target and abs(p.center[0] - n["center"][0] * PX_TO_MM) < 1e-9
                        and abs(p.center[1] - n["center"][1] * PX_TO_MM) < 1e-9)
            assert (part.pose[2] == 90.0) == vertical


def test_target_pose_follows_the_drawn_part():
    board, _ = _window("sot23", 0, shape=(255, 257), target_rotation=90.0)
    assert board.target.pose == (257 // 2 * PX_TO_MM, 255 // 2 * PX_TO_MM, 90.0)
    bl, bw = board.target.body
    x0, y0, x1, y1 = board.target.keepout
    assert y1 - y0 >= bl - 1e-9   # upright body: its length runs along y


def test_pour_has_no_floating_islands():
    for seed in range(10):
        board, _ = _window("chip_2pad", seed, BoardOptions(routing="escape", pour_probability=1.0))
        grid = Grid.for_window(board.bbox, board.rules.clearance / 4.0)
        import cv2

        for z in board.zones:
            fill = zone_fill(board, z, grid)
            n, labels = cv2.connectedComponents((fill > 0).astype(np.uint8), connectivity=8)
            border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
            own = grid.scratch()
            for t in board.tracks:
                if t.net == z.net:
                    grid.polyline(t.points, t.width, mask=own)
            for v in board.vias:
                if v.net == z.net:
                    grid.circle(v.center, v.diameter, mask=own)
            for p in board.parts:
                for pad in p.pads:
                    if pad.net == z.net:
                        grid.rect(pad.bbox, mask=own)
            anchored = set(np.unique(labels[grid.inflate(own, grid.res) > 0]))
            for k in range(1, n):
                assert k in border or k in anchored, (seed, k)


def test_shared_geometry_helpers_keep_both_behaviours():
    """``polyline45`` / ``octagon`` serve the board model and the legacy copper generator (D27):
    integer pixel inputs give integral floats, the legacy path keeps a coinciding jog point."""
    from simple_sim.board.geometry import octagon, polyline45
    from simple_sim.board.routing.base import polyline45 as from_routing

    assert from_routing is polyline45
    assert polyline45((0.0, 0.0), (10.0, 4.0)) == [(0.0, 0.0), (4.0, 4.0), (10.0, 4.0)]
    assert polyline45((0.0, 0.0), (10.0, 0.0)) == [(0.0, 0.0), (10.0, 0.0)]             # jog of 0 dropped
    assert polyline45((0.0, 0.0), (10.0, 0.0), dedupe=False) == [(0.0, 0.0), (0.0, 0.0), (10.0, 0.0)]
    pts = np.array(polyline45((3, 7), (-5, 20), dedupe=False), dtype=np.int32)
    assert pts.tolist() == [[3, 7], [-5, 15], [-5, 20]]
    assert octagon(1.0, 2.0, 10.0, 6.0, 1.5) == [(2.5, 2.0), (9.5, 2.0), (11.0, 3.5), (11.0, 6.5), (9.5, 8.0), (2.5, 8.0),
                                                 (1.0, 6.5), (1.0, 3.5)]


def test_router_names_match_the_routers():
    from simple_sim.board.options import ROUTER_NAMES
    from simple_sim.board.routing import ROUTERS

    assert tuple(ROUTERS) == ROUTER_NAMES
    assert BOARD_ROUTERS == ("legacy", "none", *ROUTERS)


def test_package_exports_resolve():
    import simple_sim.board as board_pkg
    from simple_sim.board import drc, pipeline, zones

    for name in board_pkg.__all__:
        assert getattr(board_pkg, name) is not None, name
    assert board_pkg.build_window_board is pipeline.build_window_board and board_pkg.check is drc.check
    assert board_pkg.sample_zones is zones.sample_zones and board_pkg.zones is zones
    assert {"build_window_board", "drc", "ROUTERS"} <= set(dir(board_pkg))
    with pytest.raises(AttributeError):
        board_pkg.no_such_name


def test_fine_pitch_has_one_definition():
    from simple_sim.board.footprints import FINE_PITCH_FAMILIES, is_fine_pitch
    from simple_sim.board.routing import escape

    assert escape.FINE_PITCH_FAMILIES is FINE_PITCH_FAMILIES
    assert [f for f in ("resistor", "sot", "soic", "qfn", "sot6", "array_r", "array_c") if is_fine_pitch(f)] == \
        ["soic", "qfn", "sot6", "array_r", "array_c"]


def test_pour_raster_resolution_is_one_constant():
    from simple_sim.board import drc, zones

    assert drc.POUR_CELLS_PER_CLEARANCE is zones.POUR_CELLS_PER_CLEARANCE == 4
