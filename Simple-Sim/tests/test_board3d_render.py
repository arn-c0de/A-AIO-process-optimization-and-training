"""3D-Board-Renderer T1-T6 pure parts: substrate, footprints, defect ops, Blender job, comparison, viewer server (no Blender)."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from simple_sim.board3d.footprints import CHIP_PAD_OUT, chip_pads, footprint_for, gullwing_pads, qfn_pads
from simple_sim.board3d.render import compare, crop_real, frame_to_board_matrix, prepare_substrate
from simple_sim.board3d.scene import apply_defects, build_job, job_component
from simple_sim.board3d.substrate import BoardRaster, bare_board, part_mask, rectify


def _comp(ref, fam, pkg, x, y, rot=0.0, l=1.0, w=0.5, h=0.5, stage="smd", dnp=False):
    return {"refdes": ref, "family": fam, "familyStatus": "builder", "package": pkg, "stage": stage, "dnp": dnp,
            "pose": {"x": x, "y": y, "rotDeg": rot, "z": 0.0, "tiltDeg": [0, 0]}, "body": {"l": l, "w": w, "h": h}}


def _board3d(tmp_path: Path) -> dict:
    scan = np.full((200, 300, 3), (40, 120, 40), np.uint8)          # green mask
    h = [10, 0, 0, 0, -10, 200, 0, 0, 1]                              # board mm -> px, 10 px/mm, y up
    for x, y in ((10, 10), (20, 10)):                                 # two bright "parts"
        cv2.rectangle(scan, (x * 10 - 5, 200 - y * 10 - 3), (x * 10 + 5, 200 - y * 10 + 3), (200, 200, 200), -1)
    path = tmp_path / "scan.png"
    cv2.imwrite(str(path), scan)
    return {"widthMm": 30, "heightMm": 20, "product": "demo", "side": "top",
            "sources": {"scan": {"path": str(path), "boardToImage": h}},
            "components": [_comp("C1", "capacitor", "CAPC1005", 10, 10), _comp("C2", "capacitor", "CAPC1005", 20, 10, dnp=True),
                           _comp("J1", "connector", "CONX", 5, 5, l=4, w=2, h=None, stage="tht"),
                           _comp("U1", "sot", "SOT95P280-3", 25, 15, l=1.3, w=2.9, h=1.1)]}


def test_tall_parts_mask_keeps_low_parts_for_the_colour_fit(tmp_path):
    from simple_sim.board3d.render import tall_parts_mask
    board3d = _board3d(tmp_path)
    raster = BoardRaster(0, 0, 30, 20, 10.0)
    m = tall_parts_mask(board3d, raster)
    at = lambda x, y: m[int(round((20 - y) * 10)), int(round(x * 10))]  # noqa: E731
    assert at(5, 5) == 255                                  # J1: height unknown -> tall
    assert at(10, 10) == 0 and at(25, 15) == 0              # C1 0.5 mm, U1 1.1 mm: low
    assert at(20, 10) == 0                                  # DNP
    board3d["components"][3]["body"]["h"] = 2.0
    assert tall_parts_mask(board3d, raster)[50, 250] == 255


def test_corrected_textures_point_the_job_at_colour_corrected_copies(tmp_path):
    from simple_sim.board3d.render import _corrected_textures
    img = np.full((8, 8, 3), 100, np.uint8)
    a, b = tmp_path / "board.png", tmp_path / "atlas.png"
    cv2.imwrite(str(a), img)
    cv2.imwrite(str(b), img)
    job = {"substrate": {"path": str(a)}, "photo": {"path": str(a)}, "partAtlas": {"path": str(b)}}
    m = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0, 0, 0]], np.float32)
    out = _corrected_textures(job, tmp_path / "x.tex", m)
    assert job["substrate"]["path"] == job["photo"]["path"] != str(a)          # one copy for the image used twice
    assert len(list(out.iterdir())) == 2
    assert cv2.imread(job["partAtlas"]["path"])[0, 0, 0] < 80                    # half the light, darker
    assert cv2.imread(str(a))[0, 0, 0] == 100                                    # the source stays as it was


def test_rectify_and_raster_round_trip(tmp_path):
    board3d = _board3d(tmp_path)
    img = cv2.imread(board3d["sources"]["scan"]["path"])
    r = BoardRaster(0, 0, 30, 20, 10)
    rect = rectify(img, board3d["sources"]["scan"]["boardToImage"], r)
    assert rect.shape == (200, 300, 3)
    u, v = r.to_px(np.array([[10.0, 10.0]]))[0]
    assert rect[round(v), round(u)].tolist() == [200, 200, 200]      # the part lands where it is on the board
    assert rect[5, 5].tolist() == [40, 120, 40]


def test_part_mask_skips_dnp_and_other_stages(tmp_path):
    board3d = _board3d(tmp_path)
    r = BoardRaster(0, 0, 30, 20, 10)
    parts, pads = part_mask(board3d, r, {"smd"})
    at = lambda x, y: parts[tuple(np.round(r.to_px(np.array([[x, y]]))[0][::-1]).astype(int))]  # noqa: E731
    assert at(10, 10) == 255 and at(20, 10) == 0 and at(5, 5) == 0  # DNP lands stay real, THT not in the SMD stage
    assert pads.sum() > 0
    rect = rectify(cv2.imread(board3d["sources"]["scan"]["path"]), board3d["sources"]["scan"]["boardToImage"], r)
    bare = bare_board(rect, parts, pads)
    u, v = np.round(r.to_px(np.array([[10.0, 10.0]]))[0]).astype(int)
    assert bare[v, u, 1] > bare[v, u, 0] + 40                          # part inpainted with green mask


def test_footprints():
    pads = chip_pads(1.0, 0.5)
    assert len(pads) == 2 and pads[1][0] + pads[1][2] / 2 == pytest.approx(0.5 + CHIP_PAD_OUT)
    sot = gullwing_pads(1.3, 2.9, "SOT95P280-3", "sot")               # long axis y: rows across x
    assert len(sot) == 3 and sorted({round(p[0], 3) for p in sot}) == [-round(sot[2][0], 3), round(sot[2][0], 3)]
    assert sorted(p[1] for p in sot if p[0] < 0) == pytest.approx([-0.95, 0.95])
    soic = gullwing_pads(4.9, 3.9, "SOIC127P600-8", "soic")
    assert len(soic) == 8
    qfn, thermal = qfn_pads(5.0, 5.0, "QFN50P500X500RHB-32_33")
    assert len(qfn) == 33 and thermal == 32
    fp = footprint_for(_comp("J1", "connector", "CONX", 0, 0, h=None))
    assert fp["heightGuess"] and fp["pads"] == [] and fp["part"]["body"] == "ic"
    fp = footprint_for(_comp("C1", "capacitor", "CAPC1005", 0, 0))
    assert fp["part"]["marking"] is None and fp["part"]["height_ratio"] >= 100 and fp["height"] == 0.5


def test_apply_defects_and_job(tmp_path):
    board3d = _board3d(tmp_path)
    out = apply_defects(board3d, [{"op": "move", "refdes": "C1", "dx": 0.2, "dy": -0.1, "dRot": 370},
                               {"op": "tombstone", "refdes": "U1", "raisedSign": -1}, {"op": "excess", "refdes": "C1", "pads": [1]}])
    c1 = next(c for c in out["components"] if c["refdes"] == "C1")
    assert c1["pose"]["x"] == pytest.approx(10.2) and c1["pose"]["rotDeg"] == pytest.approx(10.0)
    assert c1["nominalPose"]["x"] == 10 and board3d["components"][0]["pose"]["x"] == 10     # original untouched
    jc = job_component(c1)
    assert jc["nominal"]["x"] == 10 and jc["excess"] == [1] and not jc["missing"]
    assert job_component(next(c for c in out["components"] if c["refdes"] == "U1"))["tombstone"] == {"tiltDeg": 70.0, "raisedSign": -1.0}
    dnp = job_component(board3d["components"][1])
    assert dnp["missing"] and dnp["pads"] == []
    with pytest.raises(ValueError, match="unknown defect op"):
        apply_defects(board3d, [{"op": "zap", "refdes": "C1"}])
    with pytest.raises(ValueError, match="no placed component"):
        apply_defects(board3d, [{"op": "move", "refdes": "NOPE"}])
    job = build_job(out, BoardRaster(8, 8, 5, 5, 10), out="x.png", substrate={}, photo=None, stages={"smd"})
    assert [c["refdes"] for c in job["components"]] == ["C1"] and (job["widthPx"], job["heightPx"]) == (50, 50)


def test_frame_matrix_matches_overlay_mapper():
    from simple_sim.board3d.overlay import frame_mapper
    f = {"x0": -3, "y0": -4, "pxPerMm": 16, "mirrored": True}
    m = frame_to_board_matrix(f, 2256)
    p = np.array([[12.5, 30.0]])
    q = m @ [12.5, 30.0, 1]
    assert q[:2] == pytest.approx(frame_mapper(f, 2256)(p)[0])


def test_substrate_crop_and_compare(tmp_path):
    board3d = _board3d(tmp_path)
    sub = prepare_substrate(board3d, tmp_path, px_per_mm=10)
    assert sub["source"] == "scan" and Path(sub["maps"]["bare_rgb"]).is_file()
    again = prepare_substrate(board3d, tmp_path, px_per_mm=10)                  # cached
    assert again["raster"] == sub["raster"]
    win = BoardRaster(5, 5, 10, 8, 10)
    real = crop_real(board3d, win)
    assert real.shape == (80, 100, 3)
    full = cv2.imread(sub["maps"]["scan_rgb"])
    assert real[0, 0].tolist() == full[(20 - 13) * 10, 5 * 10].tolist()     # top-left pixel = board (5, 13)
    res = compare(real, real, {"all": np.full(real.shape[:2], 255, np.uint8)})
    assert res["ssim"] == pytest.approx(1.0) and res["deltaE"]["mean"] == 0 and res["deltaE_all"]["share"] == 1.0
    worse = compare(real, cv2.GaussianBlur(real, (0, 0), 2))
    assert worse["ssim"] < 1.0 and worse["deltaE"]["mean"] > 0


@pytest.fixture
def viewer(tmp_path):
    import tools.board3d_viewer as tv
    board3d = _board3d(tmp_path)
    (tmp_path / "viewer").mkdir()
    (tmp_path / "viewer" / "board.glb").write_bytes(b"glTF")
    tv.Handler.board3d, tv.Handler.folder, tv.Handler.source = board3d, tmp_path, "scan"
    httpd = tv.ThreadingHTTPServer(("127.0.0.1", 0), tv.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", tmp_path
    httpd.shutdown()


def _req(url, data=None, headers=None):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_viewer_server(viewer):
    base, folder = viewer
    status, body = _req(f"{base}/api/board3d")
    data = json.loads(body)
    assert status == 200 and data["glb"] == "/data/viewer/board.glb" and len(data["components"]) == 4
    assert _req(f"{base}/data/viewer/board.glb")[1] == b"glTF"
    assert _req(f"{base}/")[0] == 200 and _req(f"{base}/app.js")[0] == 200
    assert _req(f"{base}/data/../../etc/passwd")[0] == 404
    assert _req(f"{base}/../tools/board3d_viewer.py")[0] == 404
    assert _req(f"{base}/data/%2e%2e/%2e%2e/etc/passwd")[0] == 404          # encoded ..
    assert _req(f"{base}/data/%2Fetc%2Fpasswd")[0] == 404                   # absolute path joins away from the base
    assert _req(f"{base}/data/viewer/board.glb%00")[0] == 404               # NUL
    (folder / "x.png\r\nX-Evil: 1").write_bytes(b"png")
    with urllib.request.urlopen(f"{base}/data/x.png%0D%0AX-Evil:%201") as r:   # CR/LF in a file name stays out of the headers
        assert r.headers["Content-Type"] == "application/octet-stream" and r.headers.get("X-Evil") is None
    with urllib.request.urlopen(f"{base}/data/viewer/board.glb") as r:
        assert r.headers["Content-Type"] == "model/gltf-binary"
    with urllib.request.urlopen(f"{base}/app.js") as r:
        assert r.headers["Content-Type"] == "text/javascript"
    status, body = _req(f"{base}/api/defects", {"name": "t1", "ops": [{"op": "move", "refdes": "C1", "dx": 0.1}]})
    assert status == 200 and json.loads((folder / "defects" / "t1.json").read_text())[0]["dx"] == 0.1
    assert _req(f"{base}/api/defects", {"name": "../x", "ops": []})[0] == 400
    assert _req(f"{base}/api/defects", {"name": "x", "ops": [{"op": "zap", "refdes": "C1"}]})[0] == 400
    assert _req(f"{base}/api/defects", {"name": "x", "ops": []}, {"Origin": "http://evil.example"})[0] == 403
    assert _req(f"{base}/api/render", {"name": "x", "ops": [], "window": [0, 0, 100, 100]})[0] == 400



def test_dataset_sampler_and_crop(tmp_path):
    import random

    from simple_sim.classes import DefectClass
    from simple_sim.board3d.dataset import FIELD_BODY_LENGTHS, crop_part, plan_sheet, sample_op
    board3d = {"components": [_comp(f"C{i}", "capacitor", "CAPC1005", 5 + 4 * i, 10) for i in range(6)]
            + [_comp("J1", "connector", "CONX", 10, 15, stage="tht"), _comp("C9", "capacitor", "CAPC1005", 6, 10, dnp=True)]}
    rng = random.Random(1)
    win = BoardRaster(0, 0, 30, 20, 10)
    plan = plan_sheet(board3d, win, per_sheet=10, spacing_mm=4.0, rng=rng)
    refs = [p["component"]["refdes"] for p in plan]
    assert "J1" not in refs and "C9" not in refs and len(refs) == len(set(refs)) >= 4
    xs = sorted(p["component"]["pose"]["x"] for p in plan)
    assert all(b - a >= 4.0 for a, b in zip(xs, xs[1:]))
    c = board3d["components"][0]
    mis = sample_op(c, DefectClass.MISALIGNED, random.Random(3))
    assert mis["op"] == "move" and (abs(mis["dx"]) > 0.1 or abs(mis["dy"]) > 0.1 or abs(mis["dRot"]) >= 8)
    assert sample_op(c, DefectClass.OK, rng) is None and sample_op(c, DefectClass.MISSING, rng)["op"] == "remove"
    with pytest.raises(ValueError):
        sample_op(c, DefectClass.SOLDER_BALL, rng)
    img = np.zeros((200, 300, 3), np.uint8)
    u, v = np.round(win.to_px(np.array([[5.0, 10.0]]))[0]).astype(int)
    img[v - 2:v + 3, u - 2:u + 3] = 255
    crop, geo = crop_part(img, win, c, 64)
    assert crop.shape == (64, 64, 3) and geo["fieldMm"] == pytest.approx(FIELD_BODY_LENGTHS)
    ys, xs_ = np.nonzero(crop[..., 0] > 100)
    assert abs(xs_.mean() - 32) < 3 and abs(ys.mean() - 32) < 3          # part in the middle of its crop


def test_bridge_op_and_class_rules(tmp_path):
    import random

    from simple_sim.classes import DefectClass
    from simple_sim.board3d.dataset import NotApplicable, sample_op
    board3d = {"components": [_comp("C1", "capacitor", "CAPC1005", 10, 10), _comp("C2", "capacitor", "CAPC1005", 11.3, 10),
                           _comp("C3", "capacitor", "CAPC1005", 30, 30),
                           _comp("U1", "soic", "SOIC127P600-8", 20, 10, l=4.9, w=3.9, h=1.5)]}
    out = apply_defects(board3d, [{"op": "bridge", "refdes": "C1", "pad": 1, "to": "C2", "toPad": 0}])
    (b,) = out["bridges"]
    assert b["from"] == ["C1", 1] and b["a"][0] == pytest.approx(10.55) and b["b"][0] == pytest.approx(11.3 - 0.55)
    with pytest.raises(ValueError, match="two different lands"):
        apply_defects(board3d, [{"op": "bridge", "refdes": "C1", "pad": 0, "toPad": 0}])
    with pytest.raises(ValueError, match="no land 5"):
        apply_defects(board3d, [{"op": "bridge", "refdes": "C1", "pad": 5}])
    job = build_job(out, BoardRaster(5, 5, 10, 10, 10), out="x.png", substrate={}, photo=None, stages={"smd"})
    assert len(job["bridges"]) == 1
    assert build_job(out, BoardRaster(40, 40, 5, 5, 10), out="x.png", substrate={}, photo=None, stages={"smd"})["bridges"] == []
    rng = random.Random(0)
    c1, c3, u1 = board3d["components"][0], board3d["components"][2], board3d["components"][3]
    op = sample_op(c1, DefectClass.SOLDER_BRIDGE, rng, board3d)
    assert op["to"] == "C2" and (op["pad"], op["toPad"]) == (1, 0)            # the facing lands of the neighbours
    with pytest.raises(NotApplicable):
        sample_op(c3, DefectClass.SOLDER_BRIDGE, rng, board3d)                  # nobody close enough
    ic = sample_op(u1, DefectClass.SOLDER_BRIDGE, rng, board3d)
    assert ic["refdes"] == "U1" and abs(ic["pad"] - ic["toPad"]) == 1        # adjacent pins of one row
    with pytest.raises(NotApplicable):
        sample_op(u1, DefectClass.TOMBSTONE, rng, board3d)


def test_electrolytic_footprint():
    fp = footprint_for(_comp("C9", "electrolytic", "CAPAE660", 0, 0, l=6.6, w=6.6, h=5.41))
    assert fp["part"]["body"] == "can" and len(fp["pads"]) == 2 and fp["height"] == 5.41


def test_fit_recovers_pose_and_size_and_apply_fit():
    from simple_sim.board3d.fit import fit_part, fit_board3d
    from simple_sim.board3d.overlay import corners
    from simple_sim.board3d.scene import apply_fit
    r = BoardRaster(0, 0, 20, 20, 24)
    img = np.full((r.rows, r.cols, 3), (40, 110, 40), np.uint8)        # green mask
    true = {"x": 10.08, "y": 9.94, "rotDeg": 92.0}
    body_true = {"l": 3.15, "w": 1.65}
    q = r.to_px(corners(true, body_true)).round().astype(np.int32)
    cv2.fillPoly(img, [q], (30, 30, 30))                                  # black IC body
    c = _comp("U1", "soic", "SOIC127P600-8", 10.0, 10.0, rot=90.0, l=3.0, w=1.5, h=1.5)
    from simple_sim.board3d.fit import edge_image
    f = fit_part(edge_image(img), r, c)
    assert f["accepted"] and f["gain"] > 1.2
    assert (f["x"], f["y"]) == pytest.approx((true["x"], true["y"]), abs=0.03)
    assert f["rotDeg"] == pytest.approx(92.0, abs=0.5)
    assert (f["l"], f["w"]) == pytest.approx((3.15, 1.65), abs=0.08)
    board3d = {"components": [c, _comp("C1", "capacitor", "CAPC1005", 3, 3)]}
    summary = fit_board3d(board3d, img, r)
    assert summary["fitted"] == 1 and "fit" not in board3d["components"][1]   # 0402 chips are not fitted
    applied = apply_fit(board3d)
    assert applied is not board3d and applied["fitApplied"]
    assert applied["components"][0]["pose"]["x"] == pytest.approx(10.08, abs=0.03) and board3d["components"][0]["pose"]["x"] == 10.0
    assert apply_fit(applied) is applied                                  # idempotent
    moved = apply_defects(board3d, [{"op": "move", "refdes": "U1", "dx": 0.5}])
    assert moved["components"][0]["pose"]["x"] == pytest.approx(10.58, abs=0.03)   # defects are relative to the real pose


def test_part_mask_keeps_lands_by_default(tmp_path):
    board3d = _board3d(tmp_path)
    r = BoardRaster(0, 0, 30, 20, 10)
    parts, pads = part_mask(board3d, r, {"smd"})
    parts_all, _ = part_mask(board3d, r, {"smd"}, keep_lands=False)
    assert (parts_all > 0).sum() > (parts > 0).sum() > 0
    assert ((pads > 0) & (parts == 0)).any()                              # some land area is kept as real pixels


def test_board_texture_scan_under_nominal_parts_bare_where_a_defect_uncovers(tmp_path):
    from simple_sim.board3d.render import board_texture
    board3d = _board3d(tmp_path)
    sub = prepare_substrate(board3d, tmp_path, px_per_mm=10)
    board3d["substrate"] = sub
    assert board_texture(board3d, tmp_path, sub, "t") == sub["maps"]["scan_rgb"]                  # nothing uncovered
    assert board_texture(board3d, tmp_path, sub, "t", under_parts="bare") == sub["maps"]["bare_rgb"]
    assert board_texture(board3d, tmp_path, sub, "t", camera={"f": 1}) == sub["maps"]["bare_rgb"]  # oblique view
    excess = apply_defects(board3d, [{"op": "excess", "refdes": "C1"}])
    assert board_texture(excess, tmp_path, sub, "t") == sub["maps"]["scan_rgb"]                   # part still in place
    moved = apply_defects(board3d, [{"op": "move", "refdes": "C1", "dx": 3.0}])
    path = board_texture(moved, tmp_path, sub, "t")
    assert path.endswith("t.substrate.png")
    mix, scan, bare = (cv2.imread(p) for p in (path, sub["maps"]["scan_rgb"], sub["maps"]["bare_rgb"]))
    r = sub["raster"]
    u, v = np.round(BoardRaster.from_meta(r).to_px(np.array([[10.0, 10.0]]))[0]).astype(int)
    assert mix[v, u].tolist() == bare[v, u].tolist() != scan[v, u].tolist()   # the nominal place goes bare
    assert mix[5, 5].tolist() == scan[5, 5].tolist()                            # the rest stays the scan
    with pytest.raises(ValueError):
        board_texture(board3d, tmp_path, sub, "t", under_parts="nope")


def test_bracket_job_cut_outs_and_mount_stage_only(tmp_path):
    from simple_sim.board3d.scene import bracket_job
    board3d = _board3d(tmp_path)
    board3d["components"].append(_comp("J9", "connector", "CONX", 15, 18, rot=180.0, l=10.0, w=6.0, h=4.0, stage="tht"))
    br = {"edge": "y+", "innerMm": 20.1, "thicknessMm": 1.0, "fromMm": -2, "toMm": 32, "bottomMm": -5, "topMm": 12,
          "holesFor": ["J9"]}
    job = bracket_job(br, board3d)
    (h,) = job["holes"]
    assert h[0] == pytest.approx(10.0 - 0.3) and h[1] == pytest.approx(20.0 + 0.3)     # body extent along the edge
    assert h[2] == pytest.approx(-0.3) and h[3] > 0.3                                  # from the board up
    with pytest.raises(ValueError):
        bracket_job({**br, "holesFor": ["NOPE"]}, board3d)
    with pytest.raises(ValueError):
        bracket_job({**br, "edge": "z+"}, board3d)
    board3d["brackets"] = [br]
    sub = {"path": "x", "raster": {}}
    smd = build_job(board3d, BoardRaster(0, 0, 30, 20, 10), out="o.png", substrate=sub, photo=None, stages={"smd"})
    mounted = build_job(board3d, BoardRaster(0, 0, 30, 20, 10), out="o.png", substrate=sub, photo=None, stages={"smd", "tht", "mount"})
    assert smd["brackets"] == [] and len(mounted["brackets"]) == 1


def test_battery_cell_in_the_holder_with_the_mount_stage(tmp_path):
    board3d = _board3d(tmp_path)
    c = _comp("X9", "connector", "CONX", 15, 10, l=8.0, w=6.0, h=4.0)
    board3d["components"].append(c)
    win, sub = BoardRaster(0, 0, 30, 20, 10), {"path": "x", "raster": {}}
    import simple_sim.board3d.scene as scene
    orig = scene.job_component

    def holder(comp):
        jc = orig(comp)
        if comp["refdes"] == "X9":
            jc["part"] = {**jc["part"], "connector": {"kind": "battery_holder", "pins": 3, "rows": 1, "pitch": 2.54}}
        return jc
    scene.job_component = holder
    try:
        cell = lambda stages: next(x for x in build_job(board3d, win, out="o", substrate=sub, photo=None, stages=stages)["components"]  # noqa: E731
                                   if x["refdes"] == "X9")["part"]["connector"]["cell"]
        assert cell({"smd"}) is False and cell({"smd", "mount"}) is True
    finally:
        scene.job_component = orig


def test_trim_to_brackets_keeps_the_lands_in_place():
    from simple_sim.board3d.scene import trim_to_brackets
    pose = {"x": 10.0, "y": 18.0, "rotDeg": 90.0, "z": 0.0, "tiltDeg": [0, 0]}
    jc = {"refdes": "X1", "l": 6.0, "w": 4.0, "height": 5.0, "pose": dict(pose), "nominal": dict(pose), "pads": [[1.0, 0.5, 1, 1]]}
    keep = {"refdes": "J1", "l": 6.0, "w": 4.0, "height": 5.0, "pose": dict(pose), "nominal": dict(pose), "pads": []}
    br = {"edge": "y+", "innerMm": 20.0, "fromMm": 0, "toMm": 30, "bottomMm": -5, "topMm": 12}
    land = lambda c: (c["pose"]["x"] - c["pads"][0][1], c["pose"]["y"] + c["pads"][0][0])   # noqa: E731  part -> board at 90 deg
    before = land(jc)
    assert trim_to_brackets([jc, keep], [br], {"J1"}) == ["X1"]
    assert jc["pose"]["y"] + jc["l"] / 2 == pytest.approx(20.0)                 # rotated 90: l lies along y, ends at the face
    assert jc["pose"]["y"] - jc["l"] / 2 == pytest.approx(15.0)                 # far side unchanged
    assert jc["nominal"] == jc["pose"] and land(jc) == pytest.approx(before)
    assert keep["pose"]["y"] == 18.0 and keep["l"] == 6.0                       # goes through its cut-out


def test_sheet_job_and_mount_stage_only(tmp_path):
    from simple_sim.board3d.scene import sheet_job
    sh = sheet_job({"x0": 5, "x1": 25, "y0": -3, "y1": 10, "zTopMm": -1.6, "cornerMm": 2, "rgb": [0.1, 0.2, 0.1]})
    assert sh["thicknessMm"] == 0.3 and sh["specular"] == 0.5 and sh["rgb"] == [0.1, 0.2, 0.1]
    with pytest.raises(ValueError):
        sheet_job({"x0": 5, "x1": 5, "y0": 0, "y1": 1, "zTopMm": -1.6})
    board3d = _board3d(tmp_path)
    board3d["sheets"] = [{"x0": 5, "x1": 25, "y0": -3, "y1": 10, "zTopMm": -1.6}]
    win, sub = BoardRaster(0, 0, 30, 20, 10), {"path": "x", "raster": {}}
    assert build_job(board3d, win, out="o", substrate=sub, photo=None, stages={"smd"})["sheets"] == []
    assert len(build_job(board3d, win, out="o", substrate=sub, photo=None, stages={"smd", "mount"})["sheets"]) == 1


def test_gpu_out_of_memory_messages():
    from simple_sim.board3d.render import gpu_out_of_memory
    assert gpu_out_of_memory("Error: Out of memory in CUDA queue enqueue (integrator_shade_surface)")
    assert gpu_out_of_memory("CUDA error: out of GPU memory")
    assert not gpu_out_of_memory("Fra:1 Mem:120M | Rendering 64 / 64 samples")


def test_residual_homography_recovers_a_shift():
    from simple_sim.board3d.photofx import residual_homography
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur((rng.random((240, 320)) * 255).astype(np.uint8), (0, 0), 3)
    img = cv2.cvtColor(cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
    shifted = cv2.warpAffine(img, np.float32([[1, 0, 4], [0, 1, -3]]), (320, 240), borderMode=cv2.BORDER_REFLECT)
    mask = np.zeros((240, 320), np.uint8); mask[30:210, 40:280] = 255
    warp, cc = residual_homography(img, shifted, mask)        # render(W x) ~ real(x): the render is the real one moved by (4, -3)
    assert cc > 0.9
    assert warp[0, 2] / warp[2, 2] == pytest.approx(4.0, abs=0.3) and warp[1, 2] / warp[2, 2] == pytest.approx(-3.0, abs=0.3)


def test_sample_variants_picks_units(tmp_path):
    import random
    from simple_sim.board3d.footprints import connector_spec
    from simple_sim.board3d.scene import sample_variants
    board3d = _board3d(tmp_path)
    c = board3d["components"][0]
    c["body"]["catalog"] = {"kind": "dsub", "variants": [{"weight": 1, "set": {"topCover": "metal"}}, {"weight": 1, "set": {}}]}
    board3d["sheets"] = [{"x0": 0, "x1": 1, "y0": 0, "y1": 1, "zTopMm": -1.6, "p": 0.5}]
    rng = random.Random(3)
    units = [sample_variants(board3d, rng) for _ in range(200)]
    covers = sum(u["components"][0]["body"]["catalog"].get("topCover") == "metal" for u in units)
    sheets = sum(len(u["sheets"]) for u in units)
    assert 70 < covers < 130 and 70 < sheets < 130                      # both about half
    assert all("variants" not in u["components"][0]["body"]["catalog"] for u in units)
    assert "variants" in c["body"]["catalog"] and len(board3d["sheets"]) == 1   # the input is untouched
    assert connector_spec("CONTDSU277P-9", {"kind": "dsub", "topCover": "metal"})["topCover"] == "metal"
