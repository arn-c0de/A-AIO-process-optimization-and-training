"""Per-part face textures (``simple_sim.board3d.parttex``): synthetic data only."""

import numpy as np
import pytest

from simple_sim.board3d import parttex as pt


def test_part_faces_cover_body_and_lands():
    faces = {f.name: f for f in pt.part_faces(1.6, 0.8, 0.45, [(-0.8, 0.0, 0.8, 1.1), (0.8, 0.0, 0.8, 1.1)])}
    assert set(faces) == set(pt.FACES)
    top = faces["top"]
    u0, u1, v0, v1 = top.bounds
    assert u1 == pytest.approx(0.8 + 0.4 + pt.TOP_MARGIN_MM) and v1 == pytest.approx(0.55 + pt.TOP_MARGIN_MM)
    z = top.local[..., 2]
    assert z.max() == pytest.approx(0.45) and z.min() == pytest.approx(pt.LOW_Z)      # body top vs lands
    px = faces["+x"].local
    assert np.allclose(px[..., 0], 0.8) and px[..., 2].min() > 0 and px[..., 2].max() < 0.45
    assert faces["+x"].local.shape[1] == round(0.8 * pt.TEX_PX_PER_MM)
    # a flat part (test point) has no walls
    assert [f.name for f in pt.part_faces(1.0, 1.0, 0.02, [])] == ["top"]


def test_select_prefers_sharp_within_tolerance():
    assert pt.select({"top": {"sharp": 10.3, "blur0.7": 10.1, "blur1.5": 10.0, "mosaic": 11.0}}) == {"top": "sharp"}
    assert pt.select({"top": {"sharp": 12.0, "blur0.7": 10.6, "blur1.5": 10.0}}) == {"top": "blur1.5"}
    assert pt.select({"walls": {"sharp": 30.0, "flat": 20.0, "current": 25.0}}) == {"walls": "flat"}


def test_variants():
    tile = np.zeros((10, 20, 3), np.uint8)
    tile[:, 10:] = 200
    assert pt.variant(tile, "sharp") is tile
    assert pt.variant(tile, "blur1.5")[:, 9:11].std() < tile[:, 9:11].std()
    rows = pt.variant(tile, "rows")
    assert (rows == rows[:, :1]).all()
    assert len(np.unique(pt.variant(tile, "flat").reshape(-1, 3), axis=0)) == 1
    cur = np.full_like(tile, 7)
    assert pt.variant(tile, "current", cur) is cur


def test_clean_wall_drops_board_and_fills_specks():
    green = np.array([60, 120, 40], np.uint8)
    wall = np.full((20, 30, 3), 230, np.uint8)
    wall[:, :5] = green                              # a strip of board seen past the part's edge
    out = pt.clean_wall(wall.copy(), green)
    assert out is not None and np.abs(out[:, :5].astype(int) - 230).max() < 40
    assert pt.clean_wall(np.broadcast_to(green, (20, 30, 3)).copy(), green) is None


def test_align_recovers_shift():
    rng = np.random.default_rng(0)
    ref = (rng.random((40, 60, 3)) * 255).astype(np.float32)
    ref = np.stack([np.convolve(r, np.ones(3) / 3, "same") for r in ref.reshape(40, -1)]).reshape(40, 60, 3)
    shifted = np.roll(ref, (2, -3), axis=(0, 1))
    out, mask, (dx, dy) = pt.align(shifted, np.ones((40, 60), bool), ref)
    assert dx == pytest.approx(-3, abs=0.3) and dy == pytest.approx(2, abs=0.3)
    assert np.abs(out[5:-5, 5:-5] - ref[5:-5, 5:-5]).mean() < np.abs(shifted[5:-5, 5:-5] - ref[5:-5, 5:-5]).mean()


def test_fuse_face_rejects_a_glare_photo():
    rows, cols = 12, 12
    base = np.full((rows * cols, 3), 80.0, np.float32)
    base[::7] = 160.0
    good = [base + np.random.default_rng(i).normal(0, 2, base.shape).astype(np.float32) for i in range(4)]
    glare = base.copy()
    glare[:] = 250.0
    vals = np.stack([*good, glare])
    wts = np.ones(vals.shape[:2], np.float32)
    tile, mask = pt.fuse_face(vals, wts, (rows, cols))
    assert mask.all()
    assert abs(float(tile.reshape(-1, 3).mean()) - float(base.mean())) < 5


def test_pack_places_every_tile_without_overlap():
    tiles = {"A": {"top": np.full((30, 50, 3), 1, np.uint8), "+x": np.full((8, 20, 3), 2, np.uint8)},
             "B": {"top": np.full((40, 40, 3), 3, np.uint8)}}
    atlas, rects = pt.pack(tiles, width=128)
    for r, fs in rects.items():
        for f, (x, y, w, h) in fs.items():
            assert (atlas[y:y + h, x:x + w] == tiles[r][f][0, 0]).all()


def test_exposure_gains_match_a_darker_photo():
    top = np.full((3, 50, 3), 100.0, np.float32)
    top[1] *= 0.5
    g = pt.exposure_gains(top)
    assert g[1] == pytest.approx(np.full(3, 2.0), rel=0.05) and g[0] == pytest.approx(np.ones(3), rel=0.05)


def _top_tile(l, w, real_l, shift_x, board=(60, 120, 40), body=(30, 30, 30)):
    """Fused top of a part whose real body is *real_l* long, offset by *shift_x* (mm), drawn into the model's tile."""
    faces = pt.part_faces(l, w, 1.0, [])
    top = next(f for f in faces if f.name == "top")
    x, y = top.local[..., 0], top.local[..., 1]
    inside = (np.abs(x - shift_x) <= real_l / 2) & (np.abs(y) <= w / 2)
    tile = np.where(inside[..., None], np.array(body, np.uint8), np.array(board, np.uint8)).astype(np.uint8)
    return pt.PartTex("X1", (10.0, 20.0, 90.0), (l, w, 1.0), faces), tile, top.bounds


def test_photo_fit_trims_an_oversized_edge():
    # model 6 mm long, the real body 5.5 mm, flush with the -x end: the +x edge is 0.5 mm too far out
    part, tile, bounds = _top_tile(6.0, 4.0, 5.5, -0.25)
    fit = pt.photo_fit(part, tile, bounds)
    assert set(fit["trimMm"]) == {"+x"} and fit["trimMm"]["+x"] == pytest.approx(0.5, abs=0.06)
    assert fit["l"] == pytest.approx(5.5, abs=0.06) and fit["w"] == 4.0
    # the part is rotated 90 deg: its -x shift of 0.25 mm moves the board centre along -y
    assert fit["x"] == pytest.approx(10.0, abs=1e-3) and fit["y"] == pytest.approx(20.0 - 0.25, abs=0.04)


def test_photo_fit_keeps_a_good_outline_flags_an_absent_part_and_respects_its_limits():
    part, tile, bounds = _top_tile(6.0, 4.0, 6.0, 0.0)
    assert pt.photo_fit(part, tile, bounds) is None
    part, tile, bounds = _top_tile(6.0, 4.0, 0.0, 0.0)
    assert pt.photo_fit(part, tile, bounds) == {"absent": True}
    # the real body is half as long: the trim stops at PHOTO_FIT_MAX_TOTAL of the length
    part, tile, bounds = _top_tile(6.0, 4.0, 3.0, 0.0)
    assert pt.photo_fit(part, tile, bounds)["l"] == pytest.approx(6.0 * (1 - pt.PHOTO_FIT_MAX_TOTAL), abs=0.01)
    # small parts and chips keep their (IPC) size
    part, tile, bounds = _top_tile(2.0, 1.2, 1.5, 0.0)
    assert pt.photo_fit(part, tile, bounds) is None
    part, tile, bounds = _top_tile(6.0, 4.0, 5.0, 0.0)
    part.kind = "chip"
    assert pt.photo_fit(part, tile, bounds) is None


def test_gate_profiles_gullwing_and_untested_walls_and_drops_tall_walls():
    faces = pt.part_faces(4.9, 3.9, 1.5, [])
    soic = pt.PartTex("U1", (0, 0, 0), (4.9, 3.9, 1.5), faces, kind="gullwing")
    assert pt.gate(soic, {"top": "sharp", "walls": "sharp"}, {})[0]["walls"] == "rows"
    chip = pt.PartTex("C1", (0, 0, 0), (1.6, 0.8, 0.8), pt.part_faces(1.6, 0.8, 0.8, []), kind="chip")
    assert pt.gate(chip, {"top": "sharp"}, {"top": {"sharp": 15.0}})[0] == {"top": "sharp", "walls": "rows"}
    tall = pt.PartTex("J1", (0, 0, 0), (16.0, 20.0, 16.0), pt.part_faces(16.0, 20.0, 16.0, []), kind="connector")
    choice, drop = pt.gate(tall, {"top": "sharp", "walls": "sharp"}, {"top": {"sharp": 30.0}})
    assert set(drop) == {"+x", "-x", "+y", "-y", "top"}
    assert set(pt.gate(tall, {"top": "sharp"}, {"top": {"sharp": 15.0}})[1]) == {"+x", "-x", "+y", "-y"}


def _marked_body(text=True, body=(40, 40, 45), ink=(150, 150, 150)):
    img = np.zeros((80, 120, 3), np.uint8)
    img[:] = body
    if text:
        import cv2
        cv2.putText(img, "AB12", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ink, 2, cv2.LINE_AA)
    rng = np.random.default_rng(1)
    return np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16), 0, 255).astype(np.uint8)


def test_print_mask_finds_marking_and_ignores_a_plain_body():
    mask, base, ink = pt.print_mask(_marked_body())
    assert 0.02 < mask.mean() < 0.3 and ink.mean() > base.mean() + 40      # light marking, contrast boosted
    mask, base, ink = pt.print_mask(_marked_body(text=False))
    assert mask.sum() == 0 and (ink == base).all()


def test_clean_tile_is_flat_with_crisp_print_at_double_size():
    body = _marked_body()
    out = pt.clean_tile(body, (0, 0, 120, 80))
    assert out.shape == (160, 240, 3)
    _vals, counts = np.unique(out.reshape(-1, 3), axis=0, return_counts=True)
    assert counts.max() > 0.7 * out.shape[0] * out.shape[1]          # the body itself is one flat colour
    flat = pt.clean_tile(body, (0, 0, 120, 80), with_print=False)
    assert len(np.unique(flat.reshape(-1, 3), axis=0)) == 1


def test_clean_board_smooths_mask_and_draws_silk_and_solder_crisp():
    import cv2
    from simple_sim.board3d.substrate import clean_board
    rng = np.random.default_rng(3)
    bare = np.zeros((120, 160, 3), np.uint8)
    bare[:] = (90, 120, 40)                                        # solder mask green (BGR)
    cv2.putText(bare, "R12", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (225, 230, 228), 2, cv2.LINE_AA)   # silkscreen
    bare[80:100, 110:140] = (170, 175, 172)                        # a land with solder
    noisy = np.clip(bare.astype(np.int16) + rng.normal(0, 6, bare.shape).astype(np.int16), 0, 255).astype(np.uint8)
    pads = np.zeros((120, 160), np.uint8)
    pads[78:102, 108:142] = 255
    out, st = clean_board(noisy, pads)
    assert out.shape == (240, 320, 3)
    assert st["silk"]["share"] > 0.005 and st["solder"]["share"] > 0.01
    green = out[200:230, 20:80].reshape(-1, 3).astype(float)
    assert green.std(0).max() < noisy[100:115, 10:40].reshape(-1, 3).std(0).max()   # the photo noise is gone


def test_resample_keeps_board_positions_and_match_colours_transfers():
    import cv2
    from simple_sim.board3d.substrate import BoardRaster, match_colours, resample
    src = BoardRaster(0.0, 0.0, 10.0, 8.0, 24.0)
    img = np.zeros((src.rows, src.cols, 3), np.uint8)
    cx, cy = src.to_px(np.array([[4.0, 3.0]]))[0]
    cv2.circle(img, (int(round(cx)), int(round(cy))), 6, (255, 255, 255), -1)
    dst = BoardRaster(-1.0, -2.0, 14.0, 12.0, 40.0)
    out = resample(img, src, dst)
    ys, xs = np.nonzero(out[..., 0] > 128)
    bx, by = dst.x0 + (xs.mean() + 0.5) / dst.s, dst.y0 + dst.h - (ys.mean() + 0.5) / dst.s
    assert bx == pytest.approx(4.0, abs=0.05) and by == pytest.approx(3.0, abs=0.05)
    a = np.full((20, 20, 3), 60, np.uint8)
    a[:10] = 120
    b = (a.astype(int) * 2 // 1 - 20).clip(0, 255).astype(np.uint8)
    m = match_colours(a, b, np.ones((20, 20), bool))
    assert np.abs(m.astype(int) - b.astype(int)).max() <= 2
