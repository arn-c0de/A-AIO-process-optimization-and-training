"""3D-Board-Renderer T0: DeepVision store -> board3d.json, inventory and overlays (synthetic store, no real data)."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from simple_sim.board3d import SCHEMA
from simple_sim.board3d.dv_store import ProductStore, read_pnp
from simple_sim.board3d.importer import body_from_polygon, build_board3d
from simple_sim.board3d.overlay import corners, frame_mapper, homography_mapper, write_overlays
from simple_sim.board3d.packages import default_height_mm, family_of, family_status, ipc_dimensions
from tools.board3d_import import main


def _rect(x, y, w, h, rot):
    """The app's body rectangle (package-library/src/shapes.js)."""
    a = math.radians(rot)
    c, s = math.cos(a), math.sin(a)
    return [[x + px * c - py * s, y + px * s + py * c] for px, py in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]]


def _store(root: Path, with_images: bool = True) -> ProductStore:
    d = root / "store" / "products" / "demo"
    (d / "simple-aoi" / "top").mkdir(parents=True)
    (d / "compare").mkdir()
    (d / "pnp.csv").write_text(
        "R1,RES,10k,,RESC1005,10.000,20.000,90.000,NO\n"
        'C1,CAP,"1uF, 16V",,CAPC1608,30.000,20.000,0.000,NO\n'
        "J1,CON,,,CONTEST250P-4,50.000,10.000,0.000,NO\n"
        "MT1,MTG,,,MTGP640H320Z640,5.000,5.000,0.000,NO\n"
        "TP1,TP,,,TP80B,60.000,30.000,0.000,NO\n"
        "X1,ODD,,,WEIRDPKG,70.000,30.000,0.000,NO\n"
        "R2,RES,1k,,RESC1005,10.000,40.000,0.000,NO\n"
        "R9,RES,1k,,RESC1005,10.000,40.000,0.000,YES\n", encoding="utf-8")
    parts = [
        {"name": "R1", "id": "a1", "package": "RESC1005", "mount": "smd", "dnp": False, "outline": _rect(10, 20, 1.0, 0.5, 90), "centerMm": [10, 20]},
        {"name": "C1", "id": "a2", "package": "CAPC1608", "mount": "smd", "dnp": False, "outline": _rect(30, 20, 1.6, 0.8, 0), "centerMm": [30, 20]},
        {"name": "J1", "id": "a3", "package": "CONTEST250P-4", "mount": "tht", "dnp": True, "outline": _rect(50, 10, 10, 6, 0), "centerMm": [50, 10]},
        {"name": "MT1", "id": "a4", "package": "MTGP640H320Z640", "mount": "tht", "dnp": False,
         "outline": [[5 + 3.2 * math.cos(i * math.pi / 12), 5 + 3.2 * math.sin(i * math.pi / 12)] for i in range(24)], "centerMm": [5, 5]},
        {"name": "R2", "id": "a5", "package": "RESC1005", "mount": "smd", "dnp": False, "outline": _rect(10, 40, 1.0, 0.5, 0), "centerMm": [10, 40]},
    ]
    (d / "simple-aoi" / "top" / "parts.json").write_text(json.dumps({
        "board": {"widthMm": 80, "heightMm": 50}, "frame": {"x0": -2, "y0": -2, "width": 84, "height": 54, "pxPerMm": 4, "mirrored": True},
        "photos": 3, "parts": parts}))
    (d / "dnp.json").write_text(json.dumps({"sides": {"top": ["a5", "a3"]}}))
    if with_images:
        h = [10, 0, 5, 0, -10, 520, 0, 0, 1]                       # board mm -> scan px (y up -> rows down)
        (d / "compare" / "top.json").write_text(json.dumps({"boardToImage": h, "width": 810, "height": 530, "source": "x.tif",
                                                            "fit": {"residualMm": {"median": 0.05}}}))
        cv2.imwrite(str(d / "compare" / "top.png"), np.full((530, 810, 3), 60, np.uint8))
        cv2.imwrite(str(d / "simple-aoi" / "top" / "render.jpg"), np.full((216, 336, 3), 90, np.uint8))
    return ProductStore(root, "demo")


def test_ipc_dimensions_match_package_library():
    assert ipc_dimensions("RESC1005") == {"bodyW": 1.0, "bodyH": 0.5, "compW": 1.0, "compH": 0.5, "estimated": False}
    assert ipc_dimensions("QFN50P500X500RHB-32_33")["bodyW"] == 5.0
    assert ipc_dimensions("BGA96C80P9X16_1000X1400")["bodyH"] == 14.0
    sot = ipc_dimensions("SOT95P280-3")
    assert sot["estimated"] is True and sot["compW"] == 2.8 and sot["bodyW"] == 1.4 and sot["bodyH"] == 2.9
    assert ipc_dimensions("SOP65P640-20")["bodyH"] == pytest.approx(6.85)
    assert ipc_dimensions("CAPAE1000B")["bodyW"] == 10.0          # four digits: 10.00 mm, not 1.00 mm
    assert ipc_dimensions("MTGP640H320Z640")["bodyW"] == 6.4
    assert ipc_dimensions("TP80B")["bodyW"] == 0.8
    assert ipc_dimensions("INDM1365") is None                      # moulded inductor stays open
    assert ipc_dimensions("WEIRDPKG") is None


@pytest.mark.parametrize("pkg,family", [
    ("RESC1005", "resistor"), ("RESCAXS50P100X200-8", "array_r"), ("CAPC1005", "capacitor"), ("CAPCAF50P200X125-8", "array_c"),
    ("CAPAE660", "electrolytic"), ("INDC1608", "inductor"), ("LEDSC1608-2", "led"), ("SOD2512", "diode"),
    ("SOT95P280-3", "sot"), ("SOT65P210-6", "sot6"), ("SOT65P210-5", "sot6"), ("SOP65P640-20", "soic"),
    ("SOIC127P600-8B", "soic"), ("QFN50P500X500RHB-32_33", "qfn"), ("BGA96C80P9X16_1000X1400", "bga"),
    ("OSCCC320X150-2", "crystal"), ("CONTEST250P-4", "connector"), ("KEYTEST100", "connector"),
    ("TC2050-IDC-NL", "pads_only"), ("TP150", "testpoint"), ("FID150X330", "fiducial"), ("MTGP640H320Z640", "mounting_hole"),
    ("SOCSTEST-3", "generic"),
])
def test_family_of(pkg, family):
    assert family_of(pkg) == family


def test_family_status_and_heights():
    assert family_status("resistor") == "builder" and family_status("bga") == "planned" and family_status("generic") == "generic"
    assert default_height_mm("CAPC1608", "capacitor", 1.6, 0.8) == 0.8
    assert default_height_mm("RESC1005", "resistor", 1.0, 0.5) == 0.35
    assert default_height_mm("SOP65P640-20", "soic", 6.4, 6.85) == 1.1
    assert default_height_mm("CAPAE660", "electrolytic", 6.6, 6.6) == pytest.approx(5.41)
    assert default_height_mm("CONTEST250P-4", "connector", 10, 6) is None


def test_body_from_polygon_round_trip():
    b = body_from_polygon(_rect(12.0, 7.0, 2.0, 1.25, 270.0))
    assert (b["x"], b["y"], b["l"], b["w"]) == pytest.approx((12.0, 7.0, 2.0, 1.25))
    assert b["rotDeg"] == pytest.approx(270.0)
    assert corners({"x": b["x"], "y": b["y"], "rotDeg": b["rotDeg"]}, b) == pytest.approx(np.array(_rect(12.0, 7.0, 2.0, 1.25, 270.0)))
    circle = body_from_polygon([[5 + 3 * math.cos(i * math.pi / 12), 5 + 3 * math.sin(i * math.pi / 12)] for i in range(24)])
    assert circle["l"] == pytest.approx(6.0) and circle["rotDeg"] == 0.0


def test_read_pnp_quoted_comma_and_sides(tmp_path):
    rows = read_pnp(_store(tmp_path, with_images=False).dir / "pnp.csv")
    c1 = next(r for r in rows if r.refdes == "C1")
    assert c1.value == "1uF, 16V" and c1.package == "CAPC1608"
    assert [r.side for r in rows].count("bottom") == 1


def test_build_board3d(tmp_path):
    board3d, inv = build_board3d(_store(tmp_path), "top")
    assert board3d["schema"] == SCHEMA and (board3d["widthMm"], board3d["heightMm"]) == (80, 50)
    by = {c["refdes"]: c for c in board3d["components"]}
    assert "R9" not in by and len(by) == 7                                  # bottom placement left out
    assert by["R1"]["shapeSource"] == "app" and by["R1"]["pose"]["rotDeg"] == pytest.approx(90.0)
    assert by["R1"]["body"] == {"l": 1.0, "w": 0.5, "h": 0.35, "heightSource": "default"}
    assert by["J1"]["stage"] == "tht" and by["MT1"]["stage"] == "mount" and by["R1"]["stage"] == "smd"
    # DNP only for SMD sites: J1 is listed (SMD-stage scan) but is a THT part
    assert by["R2"]["dnp"] is True and by["J1"]["dnp"] is False
    assert by["TP1"]["shapeSource"] == "ipc-name" and by["TP1"]["body"]["l"] == 0.8 and by["TP1"]["body"]["h"] == 0.0
    assert by["X1"]["body"] is None and by["X1"]["pose"] is None
    assert inv["withoutBody"] == [{"refdes": "X1", "package": "WEIRDPKG", "reason": "no body in parts.json and no size in the package name"}]
    assert inv["byStage"] == {"smd": 5, "tht": 1, "mount": 1} and inv["dnp"] == 1
    assert set(inv["familiesToBuild"]) == {"connector", "mounting_hole", "testpoint", "generic"}
    assert inv["withoutHeight"] == ["CONTEST250P-4"]
    assert board3d["sources"]["scan"]["sha256"] and board3d["sources"]["mountedRender"]["frame"]["mirrored"] is True


def test_mappers():
    to_scan = homography_mapper([10, 0, 5, 0, -10, 520, 0, 0, 1])
    assert to_scan(np.array([[1.0, 2.0]])) == pytest.approx(np.array([[15.0, 500.0]]))
    to_render = frame_mapper({"x0": -2, "y0": -2, "pxPerMm": 4, "mirrored": True}, 336)
    assert to_render(np.array([[0.0, 0.0]])) == pytest.approx(np.array([[335 - 7.5, 7.5]]))


def test_overlays_and_cli(tmp_path):
    store = _store(tmp_path)
    board3d, _ = build_board3d(store, "top")
    res = write_overlays(board3d, tmp_path / "ov")
    assert res["scan"]["drawn"] == 4 and res["scan"]["withoutBody"] == 1      # SMD stage only
    assert res["mounted"]["drawn"] == 6 and res["mounted"]["withoutBody"] == 1
    img = cv2.imread(res["scan"]["path"])
    assert img.shape == (530, 810, 3) and (img != 60).any()

    out = tmp_path / "out"
    assert main(["--dv", str(tmp_path), "--product", "demo", "--side", "top", "--out", str(out), "--no-hash", "--no-overlay"]) == 0
    saved = json.loads((out / "board3d.json").read_text())
    assert saved["schema"] == SCHEMA and saved["sources"]["pnp"]["sha256"] is None
    assert json.loads((out / "inventory.json").read_text())["components"] == 7


def test_rebase_dv_root(tmp_path, monkeypatch):
    from simple_sim.board3d import DV_ROOT_ENV, rebase_dv_root
    new = tmp_path / "dv"
    new.mkdir()
    monkeypatch.setenv(DV_ROOT_ENV, str(new))
    b = {"sources": {"dvRoot": "/elsewhere/dv", "scan": {"path": "/elsewhere/dv/store/x.png"}, "other": "/elsewhere/dvx/y"}}
    rebase_dv_root(b)
    assert b["sources"]["dvRoot"] == str(new.resolve())
    assert b["sources"]["scan"]["path"] == f"{new.resolve()}/store/x.png"
    assert b["sources"]["other"] == "/elsewhere/dvx/y"            # only whole path components
