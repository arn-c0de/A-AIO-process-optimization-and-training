from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from simple_sim.generators.blender_3d import render_blender_batch, write_jobs_jsonl


StatusCallback = Callable[[str, Optional[Path]], None]


@dataclass(frozen=True)
class RenderRequest:
    profile_data: Dict[str, Any]
    run_data: Dict[str, Any]


class BlenderLivePreviewWorker:
    """Serial background worker for live Blender HQ previews."""

    def __init__(self, sim_root: Path, on_status: StatusCallback):
        self.sim_root = Path(sim_root)
        self.on_status = on_status
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._running = True
        self._pending: Optional[RenderRequest] = None
        self._thread = threading.Thread(target=self._loop, name="blender-live-preview", daemon=True)
        self._thread.start()

    def submit(self, req: RenderRequest) -> None:
        with self._cv:
            self._pending = req
            self._cv.notify()

    def stop(self) -> None:
        with self._cv:
            self._running = False
            self._cv.notify_all()
        self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._running and self._pending is None:
                    self._cv.wait(timeout=0.4)
                if not self._running:
                    return
                req = self._pending
                self._pending = None
            if req is None:
                continue
            try:
                self.on_status("rendering", None)
                img_path = self._render(req)
                self.on_status("ready", img_path)
            except Exception as exc:
                self.on_status(f"error: {exc}", None)

    def _render(self, req: RenderRequest) -> Path:
        profile_data = req.profile_data or {}
        run_data = req.run_data or {}

        run = run_data.get("run") or {}
        roi = run_data.get("roi") or {}
        render = run_data.get("render") or {}
        blender = render.get("blender") or {}

        backend = str(render.get("backend") or "").strip()
        if backend != "blender_3d":
            raise ValueError("run.render.backend must be blender_3d for HQ preview")

        if not isinstance(profile_data.get("render_3d"), dict):
            raise ValueError("profile.render_3d missing")

        profile_id = str(((profile_data.get("profile") or {}).get("profile_id") or "profile")).strip() or "profile"
        footprint = str(((profile_data.get("component") or {}).get("footprint") or "chip_2pad")).strip() or "chip_2pad"
        nominal_dims = ((profile_data.get("component") or {}).get("nominal_dims_mm") or {})
        geometry_ranges = profile_data.get("geometry_ranges") or {}

        width_px = int(roi.get("width_px", 256))
        height_px = int(roi.get("height_px", 256))
        mm_per_px = float(roi.get("mm_per_px", 0.01))

        blender_executable = str(blender.get("executable", "blender"))
        cycles_samples = int(blender.get("samples", 48))
        device = str(blender.get("device", "CPU"))

        digest = hashlib.sha1(
            json.dumps({"p": profile_data, "r": run_data}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:14]

        out_root = self.sim_root / "outputs" / "profile_editor" / "hq_preview"
        out_root.mkdir(parents=True, exist_ok=True)
        jobs_path = out_root / f"job_{digest}.jsonl"
        image_rel = f"preview_{digest}.png"

        nominal = self._nominal_from_ranges(geometry_ranges)
        component_height_mm = float(nominal_dims.get("height", 0.6))

        job = {
            "image_path": image_rel,
            "seed": int(run.get("seed", 42)),
            "mm_per_px": mm_per_px,
            "roi_width_px": width_px,
            "roi_height_px": height_px,
            "footprint": footprint,
            "component_height_mm": component_height_mm,
            "nominal": nominal,
            "defect": {
                "type": "OK",
                "shift_x": 0.0,
                "shift_y": 0.0,
                "rotation_deg": 0.0,
                "tilt_deg": 0.0,
            },
            "augment": {"rotation_deg": 0.0},
            "render_3d": profile_data.get("render_3d") or {},
        }

        write_jobs_jsonl(jobs_path, [job])
        render_blender_batch(
            sim_root=self.sim_root,
            jobs_path=jobs_path,
            output_root=out_root,
            blender_executable=blender_executable,
            cycles_samples=cycles_samples,
            device=device,
        )
        img_path = out_root / image_rel
        if not img_path.exists():
            raise RuntimeError("Blender did not produce preview image")
        return img_path

    @staticmethod
    def _nominal_from_ranges(geometry_ranges: Dict[str, Any]) -> Dict[str, float]:
        def mid(name: str, fallback: float) -> float:
            raw = geometry_ranges.get(name, [fallback, fallback])
            if isinstance(raw, list) and len(raw) == 2:
                try:
                    return (float(raw[0]) + float(raw[1])) * 0.5
                except Exception:
                    return fallback
            return fallback

        out = {
            "pad_width": mid("pad_width", 30.0),
            "pad_height": mid("pad_height", 40.0),
            "pad_spacing": mid("pad_spacing", 60.0),
            "component_length": mid("component_length", 60.0),
            "component_width": mid("component_width", 30.0),
        }
        if "pad_spacing_y" in geometry_ranges:
            out["pad_spacing_y"] = mid("pad_spacing_y", 80.0)
        return out
