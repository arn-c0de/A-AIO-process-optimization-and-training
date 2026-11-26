"""Prepare the board3d's textures (T1), render a window in Blender (T2/T3a) and compare it with the real scan (T5).

Files in the board3d folder (``outputs/board3d/<product>/<side>/``):

* ``substrate/scan_rgb.png``   rectified SMD scan (parts included) = photo texture of the body tops
* ``substrate/bare_rgb.png``   the same with the parts removed (part bodies filled from the board around them) = board texture
* ``substrate/parts_mask.png`` / ``pads_mask.png``
* ``renders/<name>.png``, ``renders/<name>_compare.png`` (real | render | difference), ``renders/<name>.json``
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from simple_sim.blender.job_schema import BLENDER_ENV_VAR, DEFAULT_BLENDER_EXECUTABLE
from simple_sim.io_utils import atomic_write_json, load_json
from simple_sim.board3d.scene import apply_defects, build_job
from simple_sim.board3d.substrate import BoardRaster, bare_board, pad_colour, part_mask, rectify

SCRIPT = Path(__file__).resolve().parents[1] / "blender" / "render_board3d.py"
DEFAULT_PX_PER_MM = 24.0     # the 600 dpi scan has 23.6 px/mm
#: how the bare board under the bodies is made (a cached substrate with another method is rebuilt)
BARE_METHOD = "push-pull fill + board grain, DNP lands kept"


#: substrate sources: which real image, which stages it shows, default resolution
SOURCES = {"scan": {"stages": ("smd",), "pxPerMm": DEFAULT_PX_PER_MM, "folder": "substrate"},
           "mounted": {"stages": ("smd", "tht", "mount"), "pxPerMm": 16.0, "folder": "substrate_mounted"},
           # fused from the single DeepVision stills (photomosaic.py): sharp, no lamp glare, prints not mirrored
           "photos": {"stages": ("smd", "tht", "mount"), "pxPerMm": 20.0, "folder": "substrate_photos"}}
#: the photo mosaic covers the mounted frame (sockets overhang the board edge): board mm
MOSAIC_FRAME = (-3.0, -4.0, 141.0, 112.0)
#: runs left out of the mosaic, one DeepVision run id per line (e.g. the photos a phone comparison is measured on: the
#: mosaic must not have seen them). Kept next to the data in the git-ignored output folder, never in the code.
MOSAIC_EXCLUDE_FILE = "photos/exclude.txt"


#: stages whose bodies come from SimpleAOI's render of one fully mounted board instead of the fused stills: the stills
#: are of many units (most without coin cell and standoffs) and tall parts look different from every viewpoint
MOSAIC_FROM_MOUNTED = ("tht", "mount")


def _tall_parts_from_mounted(board3d: dict[str, Any], folder: Path, raster: BoardRaster, img: np.ndarray) -> tuple[np.ndarray, Any]:
    """*img* with the bodies of :data:`MOSAIC_FROM_MOUNTED` (+0.5 mm, feathered) pasted from the ``mounted`` source,
    its colours quantile-matched to the mosaic over the board."""
    from simple_sim.board3d.photofx import apply_tone_lut, fit_tone_lut
    from simple_sim.board3d.substrate import component_polygons
    try:
        ref_img, ref_h, _r = source_image(board3d, "mounted", folder)
    except (ValueError, FileNotFoundError) as e:
        return img, {"skipped": str(e)}
    ref = rectify(ref_img, ref_h.ravel().tolist(), raster)
    board = np.zeros((raster.rows, raster.cols), np.uint8)
    q = raster.to_px(np.array([[0, 0], [board3d["widthMm"], 0], [board3d["widthMm"], board3d["heightMm"]], [0, board3d["heightMm"]]]))
    cv2.fillPoly(board, [q.round().astype(np.int32)], 255)
    ref = apply_tone_lut(ref, fit_tone_lut(ref, img, board))
    mask = np.zeros_like(board)
    for c in board3d["components"]:
        if c["stage"] in MOSAIC_FROM_MOUNTED and not c["dnp"]:
            for poly in component_polygons(c, 0.5)[0]:
                cv2.fillPoly(mask, [raster.to_px(poly).round().astype(np.int32)], 255)
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 0.25 * raster.s)[..., None]
    out = np.clip(img * (1 - a) + ref * a + 0.5, 0, 255).astype(np.uint8)
    return out, {"stages": list(MOSAIC_FROM_MOUNTED), "pixels": int((a[..., 0] > 0.5).sum())}


def photo_mosaic(board3d: dict[str, Any], folder: Path, *, limit: int = 40, force: bool = False, log=print) -> Path:
    """``<folder>/photos/mosaic.png`` (+ ``mosaic.json``), built from the DeepVision stills on first use."""
    from simple_sim.board3d.photomosaic import build_mosaic, select_photos
    from simple_sim.io_utils import atomic_write_json as write
    out = folder / "photos" / "mosaic.png"
    if out.is_file() and not force:
        return out
    dv = Path(board3d["sources"].get("dvRoot") or "")
    excl = folder / MOSAIC_EXCLUDE_FILE
    exclude = {ln.strip() for ln in excl.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")} if excl.is_file() else set()
    geos = select_photos(dv, side=board3d.get("side", "top"), limit=limit, exclude=exclude)
    if not geos:
        raise ValueError(f"no DeepVision stills with a registration under {dv}")
    log(f"photo mosaic: fusing {len(geos)} stills from {dv} ...")
    raster = BoardRaster(*MOSAIC_FRAME, SOURCES["photos"]["pxPerMm"])
    img, rep = build_mosaic(board3d, raster, geos, stages=set(SOURCES["photos"]["stages"]), log=lambda *_a: None)
    img, rep["tallParts"] = _tall_parts_from_mounted(board3d, folder, raster, img)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    write(out.with_suffix(".json"), rep, indent=1)
    return out


def board_raster(board3d: dict[str, Any], px_per_mm: float) -> BoardRaster:
    return BoardRaster(0.0, 0.0, float(board3d["widthMm"]), float(board3d["heightMm"]), px_per_mm)


def frame_to_board_matrix(frame: dict[str, Any], width: int) -> np.ndarray:
    """3x3 board mm -> pixels of the SimpleAOI median render (``render.py``: ``u = (x - x0) px - 0.5``,
    ``v = (y - y0) px - 0.5``, mirrored: ``u' = W - 1 - u``)."""
    px, x0, y0 = float(frame["pxPerMm"]), float(frame["x0"]), float(frame["y0"])
    m = np.array([[px, 0, -x0 * px - 0.5], [0, px, -y0 * px - 0.5], [0, 0, 1]])
    if frame.get("mirrored"):
        m = np.array([[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]]) @ m
    return m


def source_image(board3d: dict[str, Any], source: str, folder: Path | None = None) -> tuple[np.ndarray, np.ndarray, BoardRaster]:
    """(image, board mm -> image px matrix, raster to build) of a substrate source (*folder*: the board3d folder,
    where the ``photos`` mosaic is kept)."""
    cfg = SOURCES[source]
    if source == "scan":
        scan = board3d["sources"].get("scan")
        if not scan or not scan.get("boardToImage"):
            raise ValueError("the board3d has no flatbed scan (sources.scan)")
        image = cv2.imread(scan["path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(scan["path"])
        return image, np.asarray(scan["boardToImage"], float).reshape(3, 3), board_raster(board3d, cfg["pxPerMm"])
    if source == "photos":
        if folder is None:
            raise ValueError("source 'photos' needs the board3d folder")
        path = photo_mosaic(board3d, folder)
        raster = BoardRaster(*MOSAIC_FRAME, cfg["pxPerMm"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return image, np.linalg.inv(raster.pixel_to_board()), raster
    mounted = board3d["sources"].get("mountedRender")
    if not mounted or not mounted.get("frame"):
        raise ValueError("the board3d has no mounted median render (sources.mountedRender)")
    image = cv2.imread(mounted["path"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(mounted["path"])
    f = mounted["frame"]
    # the frame covers parts that overhang the board (sockets over the edge): the substrate plane does too
    raster = BoardRaster(f["x0"], f["y0"], f["width"], f["height"], cfg["pxPerMm"])
    return image, frame_to_board_matrix(f, image.shape[1]), raster


def prepare_substrate(board3d: dict[str, Any], folder: Path, px_per_mm: float | None = None, force: bool = False,
                      source: str = "scan") -> dict[str, Any]:
    """Write the substrate images of *source* (cached; *force* rebuilds) and set ``board3d["substrate"]``. Returns it."""
    cfg = SOURCES[source]
    d = folder / cfg["folder"]
    d.mkdir(parents=True, exist_ok=True)
    image, to_image, raster = source_image(board3d, source, folder)
    if px_per_mm:
        raster = BoardRaster(raster.x0, raster.y0, raster.w, raster.h, px_per_mm)
    paths = {k: d / f"{k}.png" for k in ("scan_rgb", "bare_rgb", "parts_mask", "pads_mask")}
    meta_path = d / "substrate.json"
    meta = load_json(meta_path) if meta_path.is_file() else None
    mosaic = folder / "photos" / "mosaic.png"
    stamp = int(mosaic.stat().st_mtime_ns) if source == "photos" and mosaic.is_file() else None   # rebuilt mosaic -> rebuild
    lands = folder / "bare" / "lands.json"
    lands_stamp = int(lands.stat().st_mtime_ns) if lands.is_file() else None      # measured lands -> pad mask changes
    if (force or meta is None or meta.get("raster") != raster.meta() or meta.get("method") != BARE_METHOD
            or meta.get("sourceStamp") != stamp or meta.get("landsStamp") != lands_stamp
            or not all(p.is_file() for p in paths.values())):
        rect = rectify(image, to_image.ravel().tolist(), raster)
        parts, pads = part_mask(board3d, raster, set(cfg["stages"]))
        colour = pad_colour(rect, pads, parts)
        bare = bare_board(rect, parts, pads, colour)
        for key, img in (("scan_rgb", rect), ("bare_rgb", bare), ("parts_mask", parts), ("pads_mask", pads)):
            cv2.imwrite(str(paths[key]), img)
        meta = {"raster": raster.meta(), "padColourBgr": list(colour), "source": source, "stages": list(cfg["stages"]),
                "method": BARE_METHOD, "sourceStamp": stamp, "landsStamp": lands_stamp}
        atomic_write_json(meta_path, meta, indent=1)
    maps = {k: str(p) for k, p in paths.items()}
    if source in ("photos", "mounted"):
        # the clean look's board (substrate.clean_board): built once from the bare board, rebuilt with it
        clean = d / "clean_rgb.png"
        scan_dir = folder / SOURCES["scan"]["folder"]
        sources = [paths["bare_rgb"], scan_dir / "bare_rgb.png"]
        newest = max(q.stat().st_mtime for q in sources if q.is_file())
        if not clean.is_file() or clean.stat().st_mtime < newest:
            from simple_sim.board3d.substrate import clean_board, scan_board
            bare = cv2.imread(str(paths["bare_rgb"]), cv2.IMREAD_COLOR)
            if (scan_dir / "bare_rgb.png").is_file() and (scan_dir / "substrate.json").is_file():
                # the flatbed scan is the sharpest view of the bare board: colour-matched to the photos
                sr = load_json(scan_dir / "substrate.json")["raster"]
                img, _st = scan_board(bare, raster, cv2.imread(str(scan_dir / "bare_rgb.png"), cv2.IMREAD_COLOR),
                                      BoardRaster(sr["x0"], sr["y0"], sr["widthMm"], sr["heightMm"], sr["pxPerMm"]),
                                      cv2.imread(str(paths["parts_mask"]), cv2.IMREAD_GRAYSCALE))
            else:
                img, _st = clean_board(bare, cv2.imread(str(paths["pads_mask"]), cv2.IMREAD_GRAYSCALE))
            cv2.imwrite(str(clean), img)
        maps["clean_rgb"] = str(clean)
    board3d["substrate"] = {"raster": meta["raster"], "maps": maps, "padColourBgr": meta["padColourBgr"],
                         "method": meta["method"], "source": source, "stages": list(cfg["stages"])}
    return board3d["substrate"]


def blender_executable() -> str:
    """``$SIMPLE_SIM_BLENDER``, else ``blender`` on the PATH (same variable as ``generation.blender.resolve_blender_executable``)."""
    exe = os.environ.get(BLENDER_ENV_VAR) or shutil.which(DEFAULT_BLENDER_EXECUTABLE)
    if not exe:
        raise FileNotFoundError(f"blender not found (set {BLENDER_ENV_VAR})")
    return exe


def _run_blender_job(job_path: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    """``render_board3d.py`` on *job_path* in a background Blender (output captured)."""
    return subprocess.run([blender_executable(), "--background", "--factory-startup", "--python", str(SCRIPT), "--", "--job",
                           str(job_path)], capture_output=True, text=True, timeout=timeout, check=False)


def _blender_failure(what: str, proc: subprocess.CompletedProcess[str]) -> RuntimeError:
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    return RuntimeError(f"{what} ({proc.returncode}):\n{tail}")


def texture_interpolation(window: BoardRaster, raster: dict[str, Any], camera: dict[str, Any] | None) -> str:
    """``Closest`` when the top-down render grid lies on the texture grid (same px/mm, window offset a whole number of
    texels): every pixel then shows its scan texel unchanged (probe 20 x 20 mm: SSIM 0.952 / Delta E 1.5 against
    0.940 / 2.8 with Blender's B-spline ``Cubic``). Otherwise ``Linear`` (0.948 / 2.1)."""
    if camera is None and abs(window.s - float(raster["pxPerMm"])) < 1e-6:
        offs = ((window.x0 - float(raster["x0"])) * window.s, (window.y0 - float(raster["y0"])) * window.s,
                window.w * window.s, window.h * window.s)
        if all(abs(o - round(o)) < 1e-3 for o in offs):
            return "Closest"
    return "Linear"


#: how parts are textured from the photos: ``clean`` (default: their own materials, the body colour per part and its
#: print as a crisp mask, ``atlas_clean.png``) or ``photo`` (the fused photo texture on every face, ``atlas.png``)
PART_LOOK = "clean"


def tht_joints(board3d: dict[str, Any], folder: Path, window: BoardRaster, stages: set[str], margin_mm: float = 3.0) -> list[dict[str, float]]:
    """Solder joints of the populated through-hole parts in *window*: the plated holes of the flatbed scan (cached in
    ``substrate/holes.json``, :func:`substrate.find_holes`) inside a THT part's outline (+1.5 mm: shield and latch
    pins), for parts the photos show (not ``absentInPhotos``) and not mounting holes. ``[{x, y, r, refdes}]``."""
    if "tht" not in stages:
        return []
    scan_dir = folder / SOURCES["scan"]["folder"]
    cache = scan_dir / "holes.json"
    if cache.is_file():
        holes = load_json(cache)["holes"]
    elif (scan_dir / "scan_rgb.png").is_file() and (scan_dir / "substrate.json").is_file():
        from simple_sim.board3d.substrate import find_holes
        sr = load_json(scan_dir / "substrate.json")["raster"]
        holes = find_holes(cv2.imread(str(scan_dir / "scan_rgb.png"), cv2.IMREAD_COLOR),
                           BoardRaster(sr["x0"], sr["y0"], sr["widthMm"], sr["heightMm"], sr["pxPerMm"]))
        atomic_write_json(cache, {"holes": holes}, indent=1)
    else:
        return []
    pf = folder / "parttex" / "photofit.json"
    absent = set(load_json(pf).get("absentInPhotos", [])) if pf.is_file() else set()
    parts = [c for c in board3d["components"] if c.get("stage") == "tht" and c.get("pose") and c.get("body") and not c.get("dnp")
             and c["family"] != "mounting_hole" and c["refdes"] not in absent]
    out = []
    for x, y, r in holes:
        if not (window.x0 - margin_mm <= x <= window.x0 + window.w + margin_mm and window.y0 - margin_mm <= y <= window.y0 + window.h + margin_mm):
            continue
        for c in parts:
            p = c["pose"]
            a = -math.radians(p["rotDeg"])
            dx, dy = x - p["x"], y - p["y"]
            lx, ly = dx * math.cos(a) - dy * math.sin(a), dx * math.sin(a) + dy * math.cos(a)
            if abs(lx) <= c["body"]["l"] / 2 + 1.5 and abs(ly) <= c["body"]["w"] / 2 + 1.5:
                out.append({"x": x, "y": y, "r": r, "refdes": c["refdes"]})
                break
    return out


def _substrate_path(sub: dict[str, Any], look: str | None = None) -> str:
    """The board texture: the clean board (substrate.clean_board) in the clean look when there is one, else the bare
    board."""
    if (look or PART_LOOK) == "clean" and sub["maps"].get("clean_rgb"):
        return sub["maps"]["clean_rgb"]
    return sub["maps"]["bare_rgb"]


def part_atlas(folder: Path, sub: dict[str, Any], look: str | None = None) -> dict[str, Any] | None:
    """The per-part face texture atlas (``tools/board3d_parttex.py``, :mod:`simple_sim.board3d.parttex`) for the job, when
    the substrate comes from the phone photos (the atlas is built from the same stills; the flatbed scan keeps its
    own pixels)."""
    if sub.get("source") not in ("photos", "mounted"):
        return None
    look = look or PART_LOOK
    atlas, rep = folder / "parttex" / "atlas.png", folder / "parttex" / "parttex.json"
    clean = folder / "parttex" / "atlas_clean.png"
    if not atlas.is_file() or not rep.is_file():
        return None
    r = load_json(rep)
    if look == "clean" and clean.is_file():
        from simple_sim.board3d.parttex import CLEAN_SCALE
        parts = {k: {"pose": v["pose"], "size": v["size"], "kind": v.get("kind"),
                     "faces": {f: {**i, "rect": [c * CLEAN_SCALE for c in i["rect"]]} for f, i in v["faces"].items() if "rect" in i}}
                 for k, v in r["parts"].items()}
        return {"path": str(clean), "size": [c * CLEAN_SCALE for c in r["atlasSize"]], "look": "clean", "parts": parts}
    return {"path": str(atlas), "size": r["atlasSize"], "look": "photo",
            "parts": {k: {"pose": v["pose"], "size": v["size"], "faces": v["faces"]} for k, v in r["parts"].items()}}


def render_window(board3d: dict[str, Any], folder: Path, window: BoardRaster, name: str, *, stages: set[str] | None = None,
                  samples: int = 64, device: str = "GPU", photo: bool = True, light: dict[str, Any] | None = None,
                  timeout: float = 1800.0, camera: dict[str, Any] | None = None, size_px: tuple[int, int] | None = None,
                  job_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render *window* of the board3d; returns ``{"image", "job", "seconds", "components"}``. *job_extra* is merged into
    the Blender job last (``board``, ``shadowCatcher``, ...)."""
    sub = board3d.get("substrate") or prepare_substrate(board3d, folder)
    stages = stages or set(sub.get("stages", ["smd"]))
    out = folder / "renders" / f"{name}.png"
    job = build_job(board3d, window, out=str(out), substrate={"path": _substrate_path(sub), "raster": sub["raster"]},
                    photo={"path": sub["maps"]["scan_rgb"], "raster": sub["raster"]} if photo else None,
                    stages=stages, light=light, samples=samples, device=device)
    if photo and (atlas := part_atlas(folder, sub)):
        job["partAtlas"] = atlas
    job["thtJoints"] = tht_joints(board3d, folder, window, stages)
    if camera is not None:
        job["camera"] = camera
    if size_px is not None:
        job["widthPx"], job["heightPx"] = size_px
    job["interpolation"] = texture_interpolation(window, sub["raster"], camera)
    for k, v in (job_extra or {}).items():
        job[k] = {**job[k], **v} if isinstance(v, dict) and isinstance(job.get(k), dict) else v
    job_path = folder / "renders" / f"{name}.job.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(job_path, job, indent=1)
    t0 = time.monotonic()
    out.unlink(missing_ok=True)
    proc = _run_blender_job(job_path, timeout)
    fallback = None
    if (not out.is_file()) and job["device"] != "CPU" and "out of GPU memory" in proc.stdout + proc.stderr:
        # the GPU is shared (DeepVision's browsers, other renders): finish on the CPU instead of failing
        fallback = "GPU out of memory -> CPU"
        job["device"] = "CPU"
        atomic_write_json(job_path, job, indent=1)
        proc = _run_blender_job(job_path, timeout)
    if proc.returncode != 0 or not out.is_file():
        raise _blender_failure("Blender failed", proc)
    timing = next((ln.split("|", 1)[1].strip() for ln in proc.stdout.splitlines() if ln.startswith("[render_board3d]") and "|" in ln), None)
    return {"image": str(out), "job": str(job_path), "seconds": round(time.monotonic() - t0, 1), "components": len(job["components"]),
            "blender": timing, "fallback": fallback}


def export_glb(board3d: dict[str, Any], folder: Path, name: str = "board", *, stages: set[str] | None = None,
               timeout: float = 1800.0) -> dict[str, Any]:
    """The whole 3D board scene as ``<folder>/viewer/<name>.glb`` (for ``web/board3d``); returns ``{"glb", "seconds", ...}``."""
    sub = board3d.get("substrate") or prepare_substrate(board3d, folder)
    r = sub["raster"]
    window = BoardRaster(r["x0"], r["y0"], r["widthMm"], r["heightMm"], 1.0)
    glb = folder / "viewer" / f"{name}.glb"
    job = build_job(board3d, window, out=str(folder / "viewer" / f"{name}.png"),
                    substrate={"path": _substrate_path(sub), "raster": r}, photo={"path": sub["maps"]["scan_rgb"], "raster": r},
                    stages=stages or set(sub.get("stages", ["smd"])), samples=1, device="CPU", margin_mm=50.0)
    job.update({"exportGlb": str(glb), "render": False})
    if atlas := part_atlas(folder, sub):
        job["partAtlas"] = atlas
    job["thtJoints"] = tht_joints(board3d, folder, window, stages or set(sub.get("stages", ["smd"])))
    job["board"]["clip"] = True          # the viewer shows the board as a laminate on a desk, not the photo's frame
    job_path = folder / "viewer" / f"{name}.job.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(job_path, job, indent=1)
    t0 = time.monotonic()
    proc = _run_blender_job(job_path, timeout)
    if proc.returncode != 0 or not glb.is_file():
        raise _blender_failure("Blender GLB export failed", proc)
    return {"glb": str(glb), "seconds": round(time.monotonic() - t0, 1), "components": len(job["components"]),
            "substrate": sub["source"] if "source" in sub else "scan"}


def probe(board3d: dict[str, Any], folder: Path, window: BoardRaster, name: str, *, ops: list[dict[str, Any]] | None = None,
          source: str = "scan", stages: set[str] | None = None, samples: int = 64, device: str = "GPU", photo: bool = True,
          light: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render *window* (with the defect *ops* applied) and compare it with the real image of the substrate *source*.

    Writes ``renders/<name>.png``, ``<name>_compare.png`` (real | render | Delta E) and ``<name>.json`` (the
    returned report: :func:`render_window` result, settings and :func:`compare` metrics for parts / board).
    *board3d* must hold its ``substrate`` (:func:`prepare_substrate`).
    """
    ops = list(ops or [])
    scene = apply_defects(board3d, ops) if ops else board3d
    res = render_window(scene, folder, window, name, stages=stages, samples=samples, device=device, photo=photo, light=light)
    render = cv2.imread(res["image"], cv2.IMREAD_COLOR)
    real = crop_real(board3d, window)
    masks = {"parts": crop_real(board3d, window, "parts_mask")[..., 0]}
    masks["board"] = 255 - masks["parts"]
    metrics = compare(real, render, masks)
    compare_image(real, render, folder / "renders" / f"{name}_compare.png")
    report = {**res, "source": source, "window": window.meta(), "defects": ops, "samples": samples, "light": light, "photo": photo,
              "metrics": metrics}
    atomic_write_json(folder / "renders" / f"{name}.json", report, indent=1)
    return report


def crop_real(board3d: dict[str, Any], window: BoardRaster, key: str = "scan_rgb") -> np.ndarray:
    """The rectified real image cut to *window* (same pixel grid as the render, when both use the same px/mm)."""
    sub = board3d["substrate"]
    full = cv2.imread(sub["maps"][key], cv2.IMREAD_COLOR)
    r = sub["raster"]
    raster = BoardRaster(r["x0"], r["y0"], r["widthMm"], r["heightMm"], r["pxPerMm"])
    # window pixel -> board point -> raster pixel (affine)
    x_off = (window.x0 - raster.x0) * raster.s
    y_off = ((raster.y0 + raster.h) - (window.y0 + window.h)) * raster.s
    inv = np.array([[raster.s / window.s, 0, x_off + 0.5 * raster.s / window.s - 0.5],
                    [0, raster.s / window.s, y_off + 0.5 * raster.s / window.s - 0.5]])
    return cv2.warpAffine(full, inv, (window.cols, window.rows), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM on grey (Gaussian window 1.5 px, the usual constants)."""
    a, b = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float64) for x in (a, b))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s.mean())


def _delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la, lb = (cv2.cvtColor(x.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB) for x in (a, b))
    return np.linalg.norm(la - lb, axis=2)       # CIE76 on L*a*b*


def compare(real: np.ndarray, render: np.ndarray, masks: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """SSIM (grey), mean / p90 Delta E (CIE76) overall and per region mask, mean colour per image."""
    de = _delta_e(real, render)
    res: dict[str, Any] = {"ssim": round(_ssim(real, render), 4), "deltaE": {"mean": round(float(de.mean()), 2), "p90": round(float(np.percentile(de, 90)), 2)},
                           "meanBgr": {"real": [round(float(v), 1) for v in real.reshape(-1, 3).mean(0)],
                                       "render": [round(float(v), 1) for v in render.reshape(-1, 3).mean(0)]}}
    for k, m in (masks or {}).items():
        sel = m > 0
        if sel.sum() > 100:
            res[f"deltaE_{k}"] = {"mean": round(float(de[sel].mean()), 2), "p90": round(float(np.percentile(de[sel], 90)), 2), "share": round(float(sel.mean()), 3)}
    return res


def compare_image(real: np.ndarray, render: np.ndarray, path: Path) -> None:
    de = _delta_e(real, render)
    heat = cv2.applyColorMap(np.clip(de * 255 / 30, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    gap = np.full((real.shape[0], 8, 3), 255, np.uint8)
    strip = np.hstack([real, gap, render, gap, heat])
    for text, x in (("real scan", 6), ("render", real.shape[1] + 14), ("Delta E (0-30)", 2 * real.shape[1] + 22)):
        cv2.putText(strip, text, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(strip, text, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), strip)


def photo_view(board3d: dict[str, Any], folder: Path, geometry_path: Path, name: str, *, scale: float = 0.5, samples: int = 64,
               device: str = "GPU", f_px: float | None = None, calibrate: bool = True, backplate: bool = True,
               camera_fx: bool = True, seed: int = 0, source: str = "photos") -> dict[str, Any]:
    """T4 ``phone``: render the board3d from the camera of a real photo (DeepVision ``geometry.json`` + its still) and
    compare inside the board outline.

    With *calibrate* the colour gain of this photo is measured first (:func:`photofx.colour_gain`, texture against
    the photo on the bare board) and rendered in; with *backplate* the board (clipped to its outline, 1.6 mm thick,
    over a shadow catcher) is laid over the real photo; with *camera_fx* lens blur and sensor noise fitted to the
    photo are added and the result goes through JPEG. Writes ``renders/<name>.png`` (final), ``<name>_raw.png``
    (Cycles, RGBA), ``<name>_compare.png``; returns the report (metrics of the final image, ``metricsRaw``)."""
    from simple_sim.board3d import photofx
    from simple_sim.board3d.camera import blender_camera
    geo = load_json(geometry_path)
    still = Path(geometry_path).parent / "still.jpg"
    if not still.is_file():
        still = Path(geometry_path).parent / geo.get("file", "photo.jpg")
    w, h = (int(v) for v in geo["sourceSize"])
    cam = blender_camera(geo["boardToSource"], w, h, f_px)
    sub = prepare_substrate(board3d, folder, source=source)    # phone photos show the mounted board
    r = sub["raster"]
    window = BoardRaster(r["x0"], r["y0"], r["widthMm"], r["heightMm"], 1.0)
    size = (round(w * scale), round(h * scale))
    still_full = cv2.imread(str(still), cv2.IMREAD_COLOR)
    if still_full is None:
        raise FileNotFoundError(still)
    real = cv2.resize(still_full, size, interpolation=cv2.INTER_AREA)
    raster = BoardRaster(r["x0"], r["y0"], r["widthMm"], r["heightMm"], r["pxPerMm"])
    board = np.zeros((raster.rows, raster.cols), np.uint8)
    q = raster.to_px(np.array([[0, 0], [board3d["widthMm"], 0], [board3d["widthMm"], board3d["heightMm"]], [0, board3d["heightMm"]]]))
    cv2.fillPoly(board, [q.round().astype(np.int32)], 255)
    parts_r = cv2.imread(sub["maps"]["parts_mask"], cv2.IMREAD_GRAYSCALE)
    light: dict[str, Any] = {"preset": "room"}
    if calibrate:
        # the photo in the texture's raster (board plane, z = 0): only bare board well away from parts (no parallax)
        to_photo_full = np.asarray(geo["boardToSource"], float).reshape(3, 3) @ raster.pixel_to_board()
        real_r = cv2.warpPerspective(still_full, to_photo_full, (raster.cols, raster.rows), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)
        flat = cv2.erode(board, np.ones((15, 15), np.uint8)) & ~cv2.dilate(parts_r, np.ones((25, 25), np.uint8))
        light["gainRgb"] = photofx.colour_gain(cv2.imread(sub["maps"]["scan_rgb"], cv2.IMREAD_COLOR), real_r, flat)
    extra: dict[str, Any] = {"board": {"clip": True}}
    if backplate:
        extra["shadowCatcher"] = True
    res = render_window(board3d, folder, window, name, samples=samples, device=device, light=light, camera=cam, size_px=size,
                        job_extra=extra)
    raw_path = folder / "renders" / f"{name}_raw.png"
    Path(res["image"]).replace(raw_path)
    raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
    # board outline and part areas in the scaled photo: raster px -> board mm -> photo px -> scaled
    hb = np.diag([scale, scale, 1.0]) @ np.asarray(geo["boardToSource"], float).reshape(3, 3)
    to_photo = hb @ raster.pixel_to_board()
    board_m = cv2.warpPerspective(board, to_photo, size, flags=cv2.INTER_NEAREST)
    parts_m = cv2.warpPerspective(parts_r, to_photo, size, flags=cv2.INTER_NEAREST)
    parts_m &= board_m
    masks = {"inBoard": board_m, "parts": parts_m, "board": board_m & ~parts_m}
    tone_lut = None
    if calibrate:
        # tone curve / black level of this photo: per-channel quantile match inside the board outline
        inner = cv2.erode(board_m, np.ones((9, 9), np.uint8))
        if raw.ndim == 3 and raw.shape[2] == 4:
            inner &= (raw[..., 3] > 250).astype(np.uint8) * 255
        tone_lut = photofx.fit_tone_lut(raw[..., :3], real, inner)
        raw = photofx.apply_tone_lut(raw, tone_lut)
    if raw.ndim == 3 and raw.shape[2] == 4:
        render = photofx.composite(raw, real) if backplate else photofx.composite(raw, np.full_like(real, 255))
    else:
        render = raw[..., :3]
    ys, xs = np.nonzero(board_m)
    crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    metrics_raw = compare(real, render, masks)
    metrics_raw["ssimBoardBox"] = round(_ssim(real[crop], render[crop]), 4)
    fx = None
    if camera_fx:
        flat_m = cv2.erode(masks["board"], np.ones((9, 9), np.uint8))
        fx = photofx.fit_camera_fx(real, render, masks["inBoard"], flat_m)
        render = photofx.camera_fx(render, blur_sigma=fx["blurSigma"], noise_std=fx["noiseStd"], jpeg_quality=90, seed=seed)
        fx["jpegQuality"] = 90
    cv2.imwrite(res["image"], render)
    metrics = compare(real, render, masks)
    metrics["ssimBoardBox"] = round(_ssim(real[crop], render[crop]), 4)
    compare_image(real, render, folder / "renders" / f"{name}_compare.png")
    report = {**res, "source": source, "raw": str(raw_path), "geometry": str(geometry_path), "still": str(still), "scale": scale, "camera": cam,
              "light": light,
              "toneLut": None if tone_lut is None else tone_lut[:, ::32].tolist(), "backplate": backplate, "cameraFx": fx, "metrics": metrics, "metricsRaw": metrics_raw}
    atomic_write_json(folder / "renders" / f"{name}.json", report, indent=1)
    return report
