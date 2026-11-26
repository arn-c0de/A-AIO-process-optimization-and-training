#!/usr/bin/env python3
"""3D-Board-Renderer viewer / defect editor: a local web server for ``web/board3d/`` (three.js).

Serves only on 127.0.0.1. Exports the board3d as GLB once (``<board3d>/viewer/board.glb``, Blender), then:

====================  =====================================================================================
``GET /``             the viewer (``web/board3d/index.html`` and its static files)
``GET /api/board3d``     the components (refdes, family, package, stage, nominal pose, body, DNP) + substrate
``GET /data/<p>``     a file below the board3d folder (GLB, renders); nothing outside it
``POST /api/defects`` ``{"name", "ops"}`` -> ``<board3d>/defects/<name>.json`` (ops checked by ``apply_defects``)
``POST /api/render``  ``{"ops", "window": [x0, y0, w, h], "name", "samples"}`` -> Cycles render + comparison
                      with the real image (``tools/board3d_render.py``); answers the report and image URLs
====================  =====================================================================================

Usage:
    python tools/board3d_viewer.py --board3d outputs/board3d/<product>/<side> [--source mounted] [--port 8765] [--reexport]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.io_utils import atomic_write_json  # noqa: E402
from simple_sim.board3d import load_board3d  # noqa: E402
from simple_sim.board3d.render import SOURCES, export_glb, prepare_substrate, probe  # noqa: E402
from simple_sim.board3d.scene import BOARD_THICKNESS_MM, apply_defects  # noqa: E402
from simple_sim.board3d.substrate import BoardRaster  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "board3d"
NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAX_BODY = 1 << 20
#: bump when the exported scene changes (2: board clipped to its outline + 1.6 mm laminate, push-pull bare board;
#: 3: datasheet-like part bodies (sharp bottom edges, mould draft, bent leads, SOT-23-6 leads, array / QFN
#: terminals, crystal lid, BGA, electrolytic bead and terminals), photo on top faces only, top at the catalog height)
GLB_VERSION = "glb3"


def board3d_summary(board3d: dict, source: str) -> dict:
    comps = [{"refdes": c["refdes"], "family": c["family"], "familyStatus": c["familyStatus"], "package": c["package"],
              "stage": c["stage"], "dnp": c["dnp"], "pose": c["pose"], "body": c["body"]} for c in board3d["components"]]
    return {"product": board3d["product"], "side": board3d["side"], "widthMm": board3d["widthMm"], "heightMm": board3d["heightMm"],
            "source": source, "stages": SOURCES[source]["stages"], "substrate": board3d.get("substrate", {}).get("raster"),
            "glb": "/data/viewer/board.glb", "components": comps, "board": {"thicknessMm": BOARD_THICKNESS_MM}}


def inside(base: Path, rel: str) -> Path | None:
    p = (base / rel).resolve()
    return p if p.is_file() and base.resolve() in p.parents else None


class Handler(BaseHTTPRequestHandler):
    server_version = "Board3DViewer/1"
    board3d: dict = {}
    folder: Path = Path()
    source: str = "scan"

    def log_message(self, fmt: str, *args) -> None:  # quiet: one line per request on stderr is enough
        sys.stderr.write(f"[viewer] {self.command} {self.path} {args[1] if len(args) > 1 else ''}\n")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _file(self, path: Path | None) -> None:
        if path is None:
            self._json({"error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(path.name)[0] or ("model/gltf-binary" if path.suffix == ".glb" else "application/octet-stream")
        if path.suffix == ".js":
            ctype = "text/javascript"
        self._send(200, path.read_bytes(), ctype)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        p = unquote(url.path)
        if p in ("/", "/index.html"):
            self._file(WEB / "index.html")
        elif p == "/api/board3d":
            self._json(board3d_summary(self.board3d, self.source))
        elif p.startswith("/data/"):
            self._file(inside(self.folder, p[len("/data/"):]))
        else:
            self._file(inside(WEB, p.lstrip("/")))

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            raise ValueError("bad body size")
        data = json.loads(self.rfile.read(n))
        if not isinstance(data, dict):
            raise ValueError("body must be an object")
        return data

    def do_POST(self) -> None:  # noqa: N802
        if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
            self._json({"error": "foreign origin"}, HTTPStatus.FORBIDDEN)
            return
        try:
            data = self._body()
            name = str(data.get("name") or "edit")
            if not NAME.match(name):
                raise ValueError("bad name")
            ops = data.get("ops") or []
            if not isinstance(ops, list):
                raise ValueError("ops must be a list")
            apply_defects(self.board3d, ops)                       # validates every op against the scene
            path = self.folder / "defects" / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, ops, indent=1)
            if urlparse(self.path).path == "/api/defects":
                self._json({"saved": str(path.relative_to(self.folder))})
                return
            if urlparse(self.path).path != "/api/render":
                self._json({"error": "not found"}, 404)
                return
            win = [float(v) for v in data.get("window") or []]
            if len(win) != 4 or win[2] <= 0 or win[3] <= 0 or win[2] * win[3] > 2500:
                raise ValueError("window must be [x0, y0, w, h] with w*h <= 2500 mm2")
            samples = max(8, min(256, int(data.get("samples") or 64)))
            # In process (the board3d and its substrate are loaded already); Blender itself runs as a subprocess.
            # Same settings as ``tools/board3d_render.py --window ... --defects`` with its defaults.
            raster = self.board3d["substrate"]["raster"]
            report = probe(self.board3d, self.folder, BoardRaster(*win, raster["pxPerMm"]), f"viewer_{name}", ops=ops,
                           source=self.source, samples=samples, light={"preset": "flatbed", "lineStrength": 0.0})
            self._json({"report": report, "render": f"/data/renders/viewer_{name}.png",
                        "compare": f"/data/renders/viewer_{name}_compare.png"})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001 - report the render failure to the page
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="3D-Board-Renderer viewer / defect editor (local, 127.0.0.1 only)")
    ap.add_argument("--board3d", type=Path, required=True)
    ap.add_argument("--source", default="photos", choices=sorted(SOURCES))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--reexport", action="store_true", help="rebuild board.glb")
    a = ap.parse_args(argv)
    folder = a.board3d.resolve()
    board3d = load_board3d(folder)
    prepare_substrate(board3d, folder, source=a.source)
    glb = folder / "viewer" / "board.glb"
    meta = folder / "viewer" / "board.source"
    # new part textures (board3d_parttex.py, also a --clean-only rebuild) -> re-export
    tex = max((int(f.stat().st_mtime) for f in [*(folder / "parttex").glob("atlas*.png"), *folder.glob("substrate*/clean_rgb.png")]),
              default=0)
    # outlines fixed where bodies collided (tools/board3d_collide.py) -> re-export
    cf = folder / "parttex" / "collide.json"
    collide = int(cf.stat().st_mtime) if cf.is_file() else 0
    stamp = f"{a.source} {GLB_VERSION} parts:{board3d['partCatalog']['digest']} tex:{tex} collide:{collide}"   # catalog edit -> re-export
    if a.reexport or not glb.is_file() or not meta.is_file() or meta.read_text() != stamp:
        print(f"exporting GLB ({a.source}) ...", flush=True)
        res = export_glb(board3d, folder)
        meta.write_text(stamp)
        print(f"  {res['components']} components, {res['seconds']} s -> {res['glb']}", flush=True)
    Handler.board3d, Handler.folder, Handler.source = board3d, folder, a.source
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"viewer: http://127.0.0.1:{a.port}/  (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
