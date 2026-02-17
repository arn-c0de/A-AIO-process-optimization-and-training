from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import tkinter as tk
from tkinter import ttk


Vec3 = Tuple[float, float, float]
Face = Tuple[Vec3, Vec3, Vec3, Vec3]
GeometryUpdates = Dict[Tuple[str, ...], Any]


@dataclass
class BoxMesh:
    name: str
    center: Vec3
    size: Vec3
    movable: bool
    faces: List[Face]
    color: str
    outline: str


class Preview3D(ttk.Frame):
    """Interactive fast 3D preview with selection, movement, and live geometry updates."""

    def __init__(
        self,
        master: tk.Misc,
        on_settings_changed: Callable[[], None] | None = None,
        on_geometry_changed: Callable[[GeometryUpdates], None] | None = None,
    ):
        super().__init__(master)
        self._on_settings_changed = on_settings_changed
        self._on_geometry_changed = on_geometry_changed

        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=(6, 2))

        self.components_btn = ttk.Menubutton(top, text="Scene Components")
        self.components_btn.pack(side="left")
        self.components_menu = tk.Menu(self.components_btn, tearoff=False)
        self.components_btn.configure(menu=self.components_menu)

        self.auto_fit_on_profile_change = tk.BooleanVar(value=True)
        self.move_objects_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top,
            text="Auto-fit on profile change",
            variable=self.auto_fit_on_profile_change,
            command=self._emit_settings_changed,
        ).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(
            top,
            text="Move Objects",
            variable=self.move_objects_mode,
            command=self._emit_settings_changed,
        ).pack(side="left", padx=(8, 0))

        ttk.Button(top, text="Reset Camera", command=self._reset_camera).pack(side="left", padx=(8, 0))
        ttk.Label(top, text="").pack(side="left", expand=True)

        self.invert_orbit_x = tk.BooleanVar(value=True)
        self.invert_orbit_y = tk.BooleanVar(value=True)
        self.invert_pan_x = tk.BooleanVar(value=True)
        self.invert_pan_y = tk.BooleanVar(value=True)
        self.invert_zoom = tk.BooleanVar(value=True)

        self.settings_btn = ttk.Menubutton(top, text="Settings")
        self.settings_btn.pack(side="right")
        self.settings_menu = tk.Menu(self.settings_btn, tearoff=False)
        self.settings_menu.add_checkbutton(
            label="Invert Orbit X",
            variable=self.invert_orbit_x,
            command=self._emit_settings_changed,
        )
        self.settings_menu.add_checkbutton(
            label="Invert Orbit Y",
            variable=self.invert_orbit_y,
            command=self._emit_settings_changed,
        )
        self.settings_menu.add_separator()
        self.settings_menu.add_checkbutton(
            label="Invert Pan X",
            variable=self.invert_pan_x,
            command=self._emit_settings_changed,
        )
        self.settings_menu.add_checkbutton(
            label="Invert Pan Y",
            variable=self.invert_pan_y,
            command=self._emit_settings_changed,
        )
        self.settings_menu.add_separator()
        self.settings_menu.add_checkbutton(
            label="Invert Zoom",
            variable=self.invert_zoom,
            command=self._on_invert_zoom_changed,
        )
        self.settings_btn.configure(menu=self.settings_menu)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.zoom_var = tk.DoubleVar(value=50.0)
        self.zoom_scale = ttk.Scale(
            body,
            orient="vertical",
            from_=100.0,
            to=0.0,
            variable=self.zoom_var,
            command=self._on_zoom_scale,
        )
        self.zoom_scale.pack(side="left", fill="y", padx=(0, 6))

        self.canvas = tk.Canvas(body, bg="#0f1116", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self._dark_mode = True

        self.yaw = math.radians(35.0)
        self.pitch = math.radians(28.0)
        self.distance = 340.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self._last_xy: Tuple[int, int] | None = None
        self._drag_button = 1
        self._updating_zoom_scale = False

        self.meshes: List[BoxMesh] = []
        self._visibility_vars: Dict[str, tk.BooleanVar] = {}
        self._camera_initialized = False
        self._last_profile_id = ""
        self._selected_name: str | None = None
        self._active_move_name: str | None = None
        self._hit_regions: List[Tuple[str, List[Tuple[float, float]]]] = []
        self._scene_scale = 1.0
        self._scene_footprint = "chip_2pad"

        self.canvas.bind("<Configure>", lambda _e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", lambda e: self._start_drag(e, 1))
        self.canvas.bind("<ButtonPress-3>", lambda e: self._start_drag(e, 3))
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<B3-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._stop_drag)
        self.canvas.bind("<ButtonRelease-3>", self._stop_drag)
        self.canvas.bind("<MouseWheel>", self._on_wheel)

    def set_profile_data(self, profile_data: dict[str, Any]) -> None:
        self.meshes = self._build_scene(profile_data or {})
        self._rebuild_components_menu()

        profile_id = str(((profile_data.get("profile") or {}).get("profile_id") or "")).strip()
        profile_changed = bool(profile_id and profile_id != self._last_profile_id)

        if (not self._camera_initialized) or (profile_changed and self.auto_fit_on_profile_change.get()):
            self.pan_x = 0.0
            self.pan_y = 0.0
            self._fit_distance_to_scene()
            self._apply_camera_hint(profile_data or {})
            self._camera_initialized = True

        if profile_id:
            self._last_profile_id = profile_id
        self.redraw()

    def _reset_camera(self) -> None:
        self.yaw = math.radians(35.0)
        self.pitch = math.radians(28.0)
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._fit_distance_to_scene()
        self._camera_initialized = True
        self._sync_zoom_slider()
        self._emit_settings_changed()
        self.redraw()

    def _rebuild_components_menu(self) -> None:
        self.components_menu.delete(0, "end")
        keep: Dict[str, tk.BooleanVar] = {}
        for mesh in self.meshes:
            var = self._visibility_vars.get(mesh.name)
            if var is None:
                var = tk.BooleanVar(value=True)
            keep[mesh.name] = var
        self._visibility_vars = keep

        for mesh in self.meshes:
            self.components_menu.add_checkbutton(
                label=mesh.name,
                variable=self._visibility_vars[mesh.name],
                command=self._on_component_visibility_changed,
            )

    def _fit_distance_to_scene(self) -> None:
        if not self.meshes:
            self.distance = 340.0
            return
        max_abs = 1.0
        for mesh in self.meshes:
            for face in mesh.faces:
                for vx, vy, vz in face:
                    max_abs = max(max_abs, abs(vx), abs(vy), abs(vz))
        self.distance = max(120.0, min(1200.0, max_abs * 3.8))
        self._sync_zoom_slider()

    def _apply_camera_hint(self, profile_data: dict[str, Any]) -> None:
        render_3d = profile_data.get("render_3d") or {}
        camera = render_3d.get("camera") if isinstance(render_3d, dict) else {}
        if not isinstance(camera, dict):
            return
        loc = camera.get("location_mm")
        if not isinstance(loc, list) or len(loc) != 3:
            return
        try:
            x, y, z = float(loc[0]), float(loc[1]), float(loc[2])
        except Exception:
            return
        horizontal = math.sqrt(x * x + y * y)
        self.yaw = math.atan2(x, y + 1e-9)
        self.pitch = math.atan2(z, horizontal + 1e-9)

    def _start_drag(self, event: tk.Event, button: int) -> None:
        if button == 1:
            picked = self._pick_object(event.x, event.y)
            if picked:
                self._selected_name = picked
                mesh = self._mesh_by_name(picked)
                if self.move_objects_mode.get() and mesh is not None and mesh.movable:
                    self._active_move_name = picked
                    self._last_xy = (event.x, event.y)
                    self.redraw()
                    return
                self.redraw()
        self._last_xy = (event.x, event.y)
        self._drag_button = button

    def _stop_drag(self, _event: tk.Event) -> None:
        self._last_xy = None
        self._active_move_name = None

    def _on_drag(self, event: tk.Event) -> None:
        if not self._last_xy:
            return

        lx, ly = self._last_xy
        dx = event.x - lx
        dy = event.y - ly
        self._last_xy = (event.x, event.y)

        if self._active_move_name:
            self._move_selected_object(dx, dy)
            self._emit_settings_changed()
            self.redraw()
            return

        if self._drag_button == 1:
            sx = -1.0 if self.invert_orbit_x.get() else 1.0
            sy = -1.0 if self.invert_orbit_y.get() else 1.0
            self.yaw += dx * 0.01 * sx
            self.pitch -= dy * 0.01 * sy
            two_pi = 2.0 * math.pi
            self.yaw = (self.yaw + two_pi) % two_pi
            self.pitch = (self.pitch + two_pi) % two_pi
        else:
            sx = -1.0 if self.invert_pan_x.get() else 1.0
            sy = -1.0 if self.invert_pan_y.get() else 1.0
            self.pan_x += dx * 0.9 * sx
            self.pan_y += dy * 0.9 * sy

        self._emit_settings_changed()
        self.redraw()

    def _on_wheel(self, event: tk.Event) -> None:
        delta = -1.0 if event.delta < 0 else 1.0
        if self.invert_zoom.get():
            delta = -delta
        self.distance *= 0.92 if delta > 0 else 1.08
        self.distance = max(80.0, min(1600.0, self.distance))
        self._sync_zoom_slider()
        self._emit_settings_changed()
        self.redraw()

    def _on_zoom_scale(self, _value: str) -> None:
        if self._updating_zoom_scale:
            return
        z = float(self.zoom_var.get())
        if self.invert_zoom.get():
            z = 100.0 - z
        self.distance = self._zoom_to_distance(z)
        self._emit_settings_changed()
        self.redraw()

    def _on_invert_zoom_changed(self) -> None:
        self._sync_zoom_slider()
        self._emit_settings_changed()
        self.redraw()

    def _on_component_visibility_changed(self) -> None:
        self._emit_settings_changed()
        self.redraw()

    def _sync_zoom_slider(self) -> None:
        self._updating_zoom_scale = True
        try:
            z = self._distance_to_zoom(self.distance)
            if self.invert_zoom.get():
                z = 100.0 - z
            self.zoom_var.set(z)
        finally:
            self._updating_zoom_scale = False

    @staticmethod
    def _distance_to_zoom(distance: float) -> float:
        d_min = 80.0
        d_max = 1600.0
        distance = max(d_min, min(d_max, distance))
        return 100.0 * (d_max - distance) / (d_max - d_min)

    @staticmethod
    def _zoom_to_distance(zoom: float) -> float:
        d_min = 80.0
        d_max = 1600.0
        zoom = max(0.0, min(100.0, zoom))
        return d_max - (zoom / 100.0) * (d_max - d_min)

    def _move_selected_object(self, dx: float, dy: float) -> None:
        mesh = self._mesh_by_name(self._active_move_name or "")
        if mesh is None:
            return

        x_cam, y_cam, z_cam = self._to_camera(mesh.center)
        _ = y_cam
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        focal = min(width, height) * 0.95
        if focal <= 1e-6:
            return

        _forward, right, up = self._camera_basis()
        scale = max(1.0, z_cam) / focal
        dx_cam = dx * scale
        dy_cam = -dy * scale

        wx = right[0] * dx_cam + up[0] * dy_cam
        wy = right[1] * dx_cam + up[1] * dy_cam
        cx, cy, cz = mesh.center
        mesh.center = (cx + wx, cy + wy, cz)
        mesh.faces = self._box_faces(mesh.center, mesh.size)
        self._emit_geometry_updates_from_scene()

    def _build_scene(self, profile: dict[str, Any]) -> List[BoxMesh]:
        gr = profile.get("geometry_ranges") or {}
        if not isinstance(gr, dict):
            gr = {}

        def mid(name: str, fallback: float) -> float:
            raw = gr.get(name, [fallback, fallback])
            if isinstance(raw, list) and len(raw) == 2:
                try:
                    return (float(raw[0]) + float(raw[1])) * 0.5
                except Exception:
                    return fallback
            return fallback

        pad_w_raw = mid("pad_width", 30.0)
        pad_h_raw = mid("pad_height", 40.0)
        pad_sx_raw = mid("pad_spacing", 60.0)
        pad_sy_raw = mid("pad_spacing_y", 70.0)
        comp_l_raw = mid("component_length", 60.0)
        comp_w_raw = mid("component_width", 30.0)

        span_raw = max(comp_l_raw, comp_w_raw, pad_sx_raw + pad_w_raw, pad_sy_raw + pad_h_raw, 1.0)
        scale = 180.0 / span_raw
        self._scene_scale = scale

        pad_w = pad_w_raw * scale
        pad_h = pad_h_raw * scale
        pad_sx = pad_sx_raw * scale
        pad_sy = pad_sy_raw * scale
        comp_l = comp_l_raw * scale
        comp_w = comp_w_raw * scale

        nominal_dims = ((profile.get("component") or {}).get("nominal_dims_mm") or {})
        comp_h_mm = float(nominal_dims.get("height", 0.6))
        comp_l_mm = float(nominal_dims.get("length", 1.6))
        comp_w_mm = float(nominal_dims.get("width", 0.8))
        ref_mm = max(comp_l_mm, comp_w_mm, 1e-6)
        h_ratio = max(0.08, min(1.2, comp_h_mm / ref_mm))
        comp_h = max(8.0, min(42.0, h_ratio * max(comp_l, comp_w) * 0.65))
        pad_t = max(2.0, min(10.0, min(pad_w, pad_h) * 0.22))

        footprint = str(((profile.get("component") or {}).get("footprint") or "chip_2pad")).strip()
        self._scene_footprint = footprint

        body_len_x = comp_l
        body_len_y = comp_w
        if footprint == "soic_16":
            # Match blender/render_batch.py orientation for SOIC-16.
            body_len_x = comp_w
            body_len_y = comp_l

        board_w = max(260.0, body_len_x * 2.8, pad_sx + pad_w * 2.6)
        board_h = max(220.0, body_len_y * 2.8, pad_sy + pad_h * 2.8)

        substrate = self._box(
            "substrate",
            (0.0, 0.0, -pad_t),
            (board_w, board_h, pad_t * 1.8),
            "#124d23",
            "#2f7f4a",
            movable=False,
        )
        body = self._box(
            "component_body",
            (0.0, 0.0, comp_h * 0.5 + pad_t),
            (body_len_x, body_len_y, comp_h),
            "#2b2b2f",
            "#616161",
            movable=False,
        )

        pads: List[BoxMesh] = []
        if footprint == "soic_16":
            # 8 pads on the left + 8 pads on the right, matching blender/render_batch.py intent.
            # Keep pads distributed along package length and outside body width.
            row_x = max(pad_sx * 0.5, (body_len_x * 0.5) + (pad_h * 0.65))
            y_span = max(pad_w * 7.0, body_len_y * 0.82)
            pitch = y_span / 7.0
            y0 = -y_span * 0.5
            idx = 1
            for i in range(8):
                y = y0 + i * pitch
                pads.append(
                    self._box(
                        f"pad_{idx}",
                        (-row_x, y, pad_t * 0.5),
                        (pad_h, pad_w, pad_t),
                        "#bd7a33",
                        "#d79c5f",
                        movable=True,
                    )
                )
                idx += 1
                pads.append(
                    self._box(
                        f"pad_{idx}",
                        (row_x, y, pad_t * 0.5),
                        (pad_h, pad_w, pad_t),
                        "#bd7a33",
                        "#d79c5f",
                        movable=True,
                    )
                )
                idx += 1
        elif footprint in {"sot23", "qfn_32", "qfn"}:
            offsets = [(-pad_sx * 0.5, -pad_sy * 0.5), (pad_sx * 0.5, -pad_sy * 0.5), (0.0, pad_sy * 0.5)]
            if footprint.startswith("qfn"):
                offsets = [
                    (-pad_sx * 0.45, -pad_sy * 0.45),
                    (pad_sx * 0.45, -pad_sy * 0.45),
                    (-pad_sx * 0.45, pad_sy * 0.45),
                    (pad_sx * 0.45, pad_sy * 0.45),
                ]
            for i, (ox, oy) in enumerate(offsets, start=1):
                pads.append(
                    self._box(
                        f"pad_{i}",
                        (ox, oy, pad_t * 0.5),
                        (pad_w, pad_h, pad_t),
                        "#bd7a33",
                        "#d79c5f",
                        movable=True,
                    )
                )
        else:
            pads.append(
                self._box(
                    "pad_1",
                    (-pad_sx * 0.5, 0.0, pad_t * 0.5),
                    (pad_w, pad_h, pad_t),
                    "#bd7a33",
                    "#d79c5f",
                    movable=True,
                )
            )
            pads.append(
                self._box(
                    "pad_2",
                    (pad_sx * 0.5, 0.0, pad_t * 0.5),
                    (pad_w, pad_h, pad_t),
                    "#bd7a33",
                    "#d79c5f",
                    movable=True,
                )
            )

        return [substrate, *pads, body]

    @staticmethod
    def _box_faces(center: Vec3, size: Vec3) -> List[Face]:
        cx, cy, cz = center
        sx, sy, sz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5

        p000 = (cx - sx, cy - sy, cz - sz)
        p001 = (cx - sx, cy - sy, cz + sz)
        p010 = (cx - sx, cy + sy, cz - sz)
        p011 = (cx - sx, cy + sy, cz + sz)
        p100 = (cx + sx, cy - sy, cz - sz)
        p101 = (cx + sx, cy - sy, cz + sz)
        p110 = (cx + sx, cy + sy, cz - sz)
        p111 = (cx + sx, cy + sy, cz + sz)

        return [
            (p001, p101, p111, p011),
            (p000, p010, p110, p100),
            (p000, p001, p011, p010),
            (p100, p110, p111, p101),
            (p010, p011, p111, p110),
            (p000, p100, p101, p001),
        ]

    @classmethod
    def _box(cls, name: str, center: Vec3, size: Vec3, color: str, outline: str, movable: bool) -> BoxMesh:
        return BoxMesh(
            name=name,
            center=center,
            size=size,
            movable=movable,
            faces=cls._box_faces(center, size),
            color=color,
            outline=outline,
        )

    @staticmethod
    def _cross(a: Vec3, b: Vec3) -> Vec3:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @staticmethod
    def _normalize(v: Vec3) -> Vec3:
        n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if n < 1e-8:
            return (0.0, 0.0, 0.0)
        return (v[0] / n, v[1] / n, v[2] / n)

    def _camera_basis(self) -> Tuple[Vec3, Vec3, Vec3]:
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        forward = self._normalize((sy * cp, cy * cp, -sp))
        world_up = (0.0, 0.0, 1.0)
        right = self._normalize(self._cross(world_up, forward))
        if abs(right[0]) + abs(right[1]) + abs(right[2]) < 1e-6:
            right = self._normalize(self._cross((0.0, 1.0, 0.0), forward))
        up = self._normalize(self._cross(forward, right))
        return forward, right, up

    def _to_camera(self, p: Vec3) -> Tuple[float, float, float]:
        forward, right, up = self._camera_basis()
        cam_pos = (
            -forward[0] * self.distance,
            -forward[1] * self.distance,
            -forward[2] * self.distance,
        )

        vx, vy, vz = p[0] - cam_pos[0], p[1] - cam_pos[1], p[2] - cam_pos[2]
        x_cam = vx * right[0] + vy * right[1] + vz * right[2]
        y_cam = vx * up[0] + vy * up[1] + vz * up[2]
        z_cam = vx * forward[0] + vy * forward[1] + vz * forward[2]
        return x_cam, y_cam, z_cam

    def _project_cam(self, x_cam: float, y_cam: float, z_cam: float, width: int, height: int) -> Tuple[float, float]:
        z_cam = max(1.0, z_cam)
        focal = min(width, height) * 0.95
        sx = width * 0.5 + self.pan_x + (x_cam * focal / z_cam)
        sy = height * 0.5 + self.pan_y - (y_cam * focal / z_cam)
        return sx, sy

    def redraw(self) -> None:
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        if not self.meshes:
            self.canvas.create_text(
                width * 0.5,
                height * 0.5,
                fill="#cfd8dc",
                text="No profile loaded",
                font=("TkDefaultFont", 11),
            )
            return

        draw_faces: List[Tuple[float, List[Tuple[float, float]], str, str]] = []
        edge_segments: List[Tuple[Tuple[float, float], Tuple[float, float], str]] = []
        self._hit_regions = []
        for mesh_index, mesh in enumerate(self.meshes):
            vis = self._visibility_vars.get(mesh.name)
            if vis is not None and not vis.get():
                continue
            for face in mesh.faces:
                cam_pts = [self._to_camera(v) for v in face]
                if max(pt[2] for pt in cam_pts) <= 1.0:
                    continue
                # Cull backfaces for stable solid filling; keep edges in overlay pass.
                a = cam_pts[0]
                b = cam_pts[1]
                c = cam_pts[2]
                ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
                ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
                n = self._cross(ab, ac)
                if n[2] >= 0.0:
                    continue
                poly = [self._project_cam(pt[0], pt[1], pt[2], width, height) for pt in cam_pts]
                depth = sum(pt[2] for pt in cam_pts) / 4.0 + mesh_index * 1e-3
                draw_faces.append((depth, poly, mesh.color, mesh.name))

                # Collect face edges; render them in a dedicated pass to avoid outline flicker.
                for i in range(4):
                    a = cam_pts[i]
                    b = cam_pts[(i + 1) % 4]
                    if a[2] <= 1.0 or b[2] <= 1.0:
                        continue
                    pa = self._project_cam(a[0], a[1], a[2], width, height)
                    pb = self._project_cam(b[0], b[1], b[2], width, height)
                    edge_segments.append((pa, pb, mesh.outline))

        draw_faces.sort(key=lambda item: item[0], reverse=True)
        for _depth, poly, color, name in draw_faces:
            flat: List[float] = []
            for x, y in poly:
                flat.extend([x, y])
            self.canvas.create_polygon(flat, fill=color, outline="", width=0)
            self._hit_regions.append((name, poly))

        # Edge overlay pass
        for (x1, y1), (x2, y2), edge_color in edge_segments:
            self.canvas.create_line(x1, y1, x2, y2, fill=edge_color, width=1)

        # Highlight selected mesh with stronger edge color.
        if self._selected_name:
            selected = self._mesh_by_name(self._selected_name)
            if selected is not None:
                for face in selected.faces:
                    cam_pts = [self._to_camera(v) for v in face]
                    for i in range(4):
                        a = cam_pts[i]
                        b = cam_pts[(i + 1) % 4]
                        if a[2] <= 1.0 or b[2] <= 1.0:
                            continue
                        p1 = self._project_cam(a[0], a[1], a[2], width, height)
                        p2 = self._project_cam(b[0], b[1], b[2], width, height)
                        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#ffffff", width=2)

        self.canvas.create_text(10, 10, anchor="nw", fill="#8fa3ad", text="LMB orbit | RMB pan | Wheel/Slider zoom")
        if self.move_objects_mode.get():
            self.canvas.create_text(
                10,
                28,
                anchor="nw",
                fill="#b0bec5",
                text="Move mode: click pad and drag to update geometry",
            )
        self._draw_selection_gizmo(width, height)

    def _draw_selection_gizmo(self, width: int, height: int) -> None:
        if not self._selected_name:
            return
        mesh = self._mesh_by_name(self._selected_name)
        if mesh is None or not mesh.movable:
            return
        x_cam, y_cam, z_cam = self._to_camera(mesh.center)
        if z_cam <= 1.0:
            return
        cx, cy = self._project_cam(x_cam, y_cam, z_cam, width, height)
        L = 32.0
        self.canvas.create_line(cx, cy, cx + L, cy, fill="#ff6b6b", width=2, arrow=tk.LAST)
        self.canvas.create_line(cx, cy, cx, cy - L, fill="#57d68d", width=2, arrow=tk.LAST)
        self.canvas.create_text(cx + L + 10, cy, text="X", fill="#ff6b6b", anchor="w")
        self.canvas.create_text(cx, cy - L - 8, text="Y", fill="#57d68d", anchor="s")

    def _mesh_by_name(self, name: str) -> BoxMesh | None:
        for mesh in self.meshes:
            if mesh.name == name:
                return mesh
        return None

    def _pick_object(self, x: float, y: float) -> str | None:
        for name, poly in reversed(self._hit_regions):
            if self._point_in_poly(x, y, poly):
                return name
        return None

    @staticmethod
    def _point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
        inside = False
        n = len(poly)
        if n < 3:
            return False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def _emit_geometry_updates_from_scene(self) -> None:
        if self._on_geometry_changed is None or self._scene_scale <= 1e-8:
            return

        updates: GeometryUpdates = {}
        scale = self._scene_scale

        def m(name: str) -> BoxMesh | None:
            return self._mesh_by_name(name)

        if self._scene_footprint == "chip_2pad":
            p1, p2 = m("pad_1"), m("pad_2")
            if p1 and p2:
                spacing = abs(p2.center[0] - p1.center[0]) / scale
                updates[("geometry_ranges", "pad_spacing")] = [spacing, spacing]
        elif self._scene_footprint == "sot23":
            p1, p2, p3 = m("pad_1"), m("pad_2"), m("pad_3")
            if p1 and p2 and p3:
                left_x = (p1.center[0] + p2.center[0]) * 0.5
                spacing = abs(p3.center[0] - left_x) / scale
                spacing_y = abs(p2.center[1] - p1.center[1]) / scale
                updates[("geometry_ranges", "pad_spacing")] = [spacing, spacing]
                updates[("geometry_ranges", "pad_spacing_y")] = [spacing_y, spacing_y]
        elif self._scene_footprint.startswith("qfn"):
            p1, p2, p3, p4 = m("pad_1"), m("pad_2"), m("pad_3"), m("pad_4")
            if p1 and p2 and p3 and p4:
                left_x = (p1.center[0] + p3.center[0]) * 0.5
                right_x = (p2.center[0] + p4.center[0]) * 0.5
                top_y = (p3.center[1] + p4.center[1]) * 0.5
                bot_y = (p1.center[1] + p2.center[1]) * 0.5
                spacing = abs(right_x - left_x) / (0.9 * scale)
                spacing_y = abs(top_y - bot_y) / (0.9 * scale)
                updates[("geometry_ranges", "pad_spacing")] = [spacing, spacing]
                updates[("geometry_ranges", "pad_spacing_y")] = [spacing_y, spacing_y]

        if updates:
            self._on_geometry_changed(updates)

    def _emit_settings_changed(self) -> None:
        if self._on_settings_changed is not None:
            self._on_settings_changed()

    def get_user_settings(self) -> Dict[str, Any]:
        return {
            "auto_fit_on_profile_change": bool(self.auto_fit_on_profile_change.get()),
            "move_objects_mode": bool(self.move_objects_mode.get()),
            "invert_orbit_x": bool(self.invert_orbit_x.get()),
            "invert_orbit_y": bool(self.invert_orbit_y.get()),
            "invert_pan_x": bool(self.invert_pan_x.get()),
            "invert_pan_y": bool(self.invert_pan_y.get()),
            "invert_zoom": bool(self.invert_zoom.get()),
            "camera": {
                "yaw": float(self.yaw),
                "pitch": float(self.pitch),
                "distance": float(self.distance),
                "pan_x": float(self.pan_x),
                "pan_y": float(self.pan_y),
            },
            "component_visibility": {k: bool(v.get()) for k, v in self._visibility_vars.items()},
        }

    def apply_user_settings(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        self.auto_fit_on_profile_change.set(bool(data.get("auto_fit_on_profile_change", True)))
        self.move_objects_mode.set(bool(data.get("move_objects_mode", False)))
        self.invert_orbit_x.set(bool(data.get("invert_orbit_x", True)))
        self.invert_orbit_y.set(bool(data.get("invert_orbit_y", True)))
        self.invert_pan_x.set(bool(data.get("invert_pan_x", True)))
        self.invert_pan_y.set(bool(data.get("invert_pan_y", True)))
        self.invert_zoom.set(bool(data.get("invert_zoom", True)))

        cam = data.get("camera") or {}
        if isinstance(cam, dict):
            try:
                self.yaw = float(cam.get("yaw", self.yaw))
                self.pitch = float(cam.get("pitch", self.pitch))
                self.distance = max(80.0, min(1600.0, float(cam.get("distance", self.distance))))
                self.pan_x = float(cam.get("pan_x", self.pan_x))
                self.pan_y = float(cam.get("pan_y", self.pan_y))
                self._camera_initialized = True
            except Exception:
                pass

        vis = data.get("component_visibility") or {}
        if isinstance(vis, dict):
            for name, flag in vis.items():
                if name in self._visibility_vars:
                    self._visibility_vars[name].set(bool(flag))

        self._sync_zoom_slider()

    def apply_theme(self, *, dark: bool) -> None:
        self._dark_mode = dark
        self.canvas.configure(bg="#0f1116" if dark else "#e9eef3")
        self.redraw()
