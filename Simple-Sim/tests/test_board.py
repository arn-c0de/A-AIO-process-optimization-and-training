"""Board model: footprints match the legacy pads, the escape router passes the DRC, legacy stays bit-identical."""

import logging

import numpy as np
import pytest

from simple_sim.board import (BOARD_ROUTERS, STREAM_PLACE, BoardOptions, board_options, build_window_board, family_of,
                              footprint_pads, neighbour_layout, validate_board_options)
from simple_sim.board.drc import check
from simple_sim.board.footprints import pads_px, part_from_pads
from simple_sim.board.model import GND, Board
from simple_sim.board.raster import rasterize
from simple_sim.board.rules import PRESETS, sample_rules
from simple_sim.generators.opencv2d import sample_nominal_geometry
from simple_sim.generators.opencv2d.draw import _compute_pad_positions
from tests.test_photoreal import FOOTPRINTS, _render

WINDOW = (512, 512)   # 256 px ROI at 2x supersampling, as in the run configs
PX_TO_MM = 0.01 / 2


def _window(footprint: str, seed: int, options: BoardOptions = BoardOptions(routing="escape", drc="warn")):
    rng = np.random.default_rng(seed)
    nom = {k: float(v) * 2 for k, v in sample_nominal_geometry({}, rng, FOOTPRINTS[footprint]).items()}
    specs = footprint_pads(footprint, nom, WINDOW)
    pads = [p.as_dict() for p in specs]
    xs = [p["x"] for p in pads] + [p["x"] + p["w"] for p in pads]
    ys = [p["y"] for p in pads] + [p["y"] + p["h"] for p in pads]
    neighbours = neighbour_layout(pads, (min(xs), min(ys), max(xs), max(ys)), WINDOW, np.random.default_rng([seed, STREAM_PLACE]))
    return build_window_board(footprint=footprint, family=family_of(footprint), target_pads=specs,
                              target_body_px=(nom["component_length"], nom["component_width"]), shape=WINDOW,
                              neighbours=neighbours, px_to_mm=PX_TO_MM, options=options, seed=seed)


# ---- footprints ----------------------------------------------------------------------------

@pytest.mark.parametrize("footprint", list(FOOTPRINTS))
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("shape", [(128, 128), (256, 256), (257, 255)])
def test_footprints_match_legacy_pads(footprint, seed, shape):
    nominal = sample_nominal_geometry({}, np.random.default_rng(seed), FOOTPRINTS[footprint])
    specs = footprint_pads(footprint, nominal, shape)
    assert [p.as_dict() for p in specs] == _compute_pad_positions(footprint, nominal, shape)
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


# ---- model + router ------------------------------------------------------------------------

@pytest.mark.parametrize("footprint", ["chip_2pad", "sot23", "qfn_32"])
@pytest.mark.parametrize("seed", range(6))
def test_window_board_drc_clean(footprint, seed):
    board, report = _window(footprint, seed)
    assert isinstance(board, Board) and report is not None
    assert report.ok, report.violations


def test_target_pads_connected():
    for seed in range(8):
        board, _ = _window("chip_2pad", seed)
        nets = {t.net for t in board.tracks} | {v.net for v in board.vias} | {z.net for z in board.zones}
        for pad in board.target.pads:
            assert pad.net and pad.net in nets, (seed, pad)


def test_thermal_pad_has_vias_and_no_track():
    for seed in range(4):
        board, _ = _window("qfn_32", seed)
        ep = [p for p in board.target.pads if p.role == "thermal"][0]
        assert ep.net == GND
        x0, y0, x1, y1 = ep.bbox
        inside = [v for v in board.vias if x0 <= v.center[0] <= x1 and y0 <= v.center[1] <= y1]
        assert len(inside) >= 4 and all(v.net == GND for v in inside)
        for track in board.tracks:
            for px, py in track.points:
                assert not (x0 < px < x1 and y0 < py < y1), "track on the thermal pad"


def test_no_foreign_copper_in_target_keepout():
    for seed in range(8):
        board, _ = _window("sot23", seed)
        own = {p.net for p in board.target.pads if p.net}
        x0, y0, x1, y1 = board.target.keepout
        for track in board.tracks:
            if track.net in own:
                continue
            for px, py in track.points:
                assert not (x0 < px < x1 and y0 < py < y1), (seed, track)


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


def test_drc_catches_short_and_dangling():
    board, report = _window("chip_2pad", 1, BoardOptions(routing="escape", drc="warn"))
    assert report.ok
    from simple_sim.board.model import TOP, Track

    pad_a, pad_b = board.target.pads
    board.tracks.append(Track("SHORT", TOP, [pad_a.center, pad_b.center], 0.05))
    kinds = check(board).counts()
    assert kinds.get("clearance") and kinds.get("dangling") and kinds.get("keepout")


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


def test_legacy_routing_is_the_default_and_unchanged():
    a = _render("chip_2pad", "OK", seed=5)
    b = _render("chip_2pad", "OK", seed=5, board={"routing": "legacy"})
    assert np.array_equal(a, b)


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
