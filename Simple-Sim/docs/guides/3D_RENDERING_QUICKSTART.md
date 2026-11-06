# 3D Rendering Quickstart Guide

How the Blender backend (`render.backend: blender_3d`) builds its images and what to change where.

## Pipeline

```
profile + run config
  -> generator samples nominal geometry + defect per sample (same as 2D)
  -> jobs.jsonl                                   (scripts/generate.py, previews, gallery, profile editor)
  -> render_blender_batch()                       (simple_sim/generators/blender3d/runner.py)
       1. bake per job (blender3d/bake.py):
          photoreal 2D board -> PNG texture (mask, traces, vias, silkscreen, pad finish)
          + layout: pad rectangles, body pose, part family, thermal pad
       2. Blender: simple_sim/blender/render_batch.py builds the 3D scene and renders
       3. camera model (same as 2D: lens blur, uneven light, highlight roll-off, sensor noise)
  -> caller applies the usual image filters (augment)
```

Both backends therefore share one look and one pad layout (`draw._compute_pad_positions`). Blender
only models what needs real 3D: solder fillets, part bodies, leads, lighting.

## What the scene contains

1 Blender unit = 1 mm. Layout values come from the bake in camera pixels and are scaled with
`roi.mm_per_px`.

| Element | Details |
|---|---|
| Board | Plane (2x the ROI, so rotations never show its edge) with the baked texture; low specular because the baked colours already include the mask sheen |
| Solder | One height-field mesh per pad, metallic. Concave fillet rising to the terminal (chips, QFN) or the lead foot (SOT/SOIC); reflowed paste dome when nothing touches the pad; window-pane islands on a free thermal pad |
| Part | Body box (height from `component.nominal_dims_mm.height`, capped at the real package's height/width ratio). Resistor: metal end caps + printed value code. Capacitor (`component.type` contains `cap`): tan ceramic + end caps. SOT/SOIC/QFN: mold body with fine speckle, pin-1 dot, laser marking; SOT/SOIC get gull-wing leads to every pad |
| Defects | Same semantics as 2D: shift/rotation, `MISSING`, `TOMBSTONE` (body pivots about the bottom edge of its anchored end; from above you see the steep top face and the upright end cap, `tolerances.tombstone_tilt_deg`), `SOLDER_BRIDGE` (baked into the board), `SOLDER_BALL` (solder spheres), `EXCESS_SOLDER` (convex, higher fillet meshes); ball positions and affected pads come from the same layout function as in 2D. `CORNER_LIFT` renders like a flat part |
| Light | AOI rig around the camera axis, invisible to the camera: low-angle ring light (emissive torus; highlights on slopes and edges) plus a diffuse top light (emissive disc; makes flat metal such as solder tops and upright end caps bright), dim ambient |
| Camera | Orthographic, framed exactly to the ROI, top-down by default |
| Orientation | `augment.rotation_deg` (cardinal 0/90/180/270 + jitter) rotates the whole scene, like the 2D image rotation |

## Profile settings (`render_3d`)

```yaml
render_3d:
  camera:
    location_mm: [0.0, 0.0, 9.0]   # straight down; an x/y offset tilts the view (foreshortens pads)
    look_at_mm: [0.0, 0.0, 0.0]
  lighting:
    ring_light: {}                 # optional: strength (6), height_mm (2x ROI), radius_mm (35 deg ring)
    top_light: {}                  # optional: strength (0.9)
  materials:
    body:   {base_color: [0.02, 0.02, 0.02], roughness: 0.6, metallic: 0.0}   # linear RGB
    solder: {base_color: [0.70, 0.70, 0.72], roughness: 0.18, metallic: 1.0}  # also used for leads
```

The default ring strength is calibrated so the board renders at its baked (photo-calibrated) colours.
Board colours, traces, silkscreen and pad finish are changed in the 2D generator
(`simple_sim/generators/opencv2d/photoreal.py`), or replaced with real photos via `render.texture_dir`
(see `PHOTOREAL_RENDERING.md`).

`geometry_ranges` have the same meaning as in 2D profiles (pixels at `roi.mm_per_px`;
`component_length` along x). For SOIC-16, `pad_spacing` is the distance between the pad columns and
`pad_spacing_y` the span from the first to the last pad centre.

## Run config (`render.blender`)

Requires Blender 4.0 or newer (Principled BSDF v2 input names); tested with 5.2.

```yaml
render:
  backend: blender_3d
  blender:
    executable: "blender"
    engine: "CYCLES"     # CYCLES (deterministic per seed, denoised) or BLENDER_EEVEE (needs a GPU context)
    samples: 64
    device: "CPU"        # or GPU
```

## Debugging

```bash
./run_render_debug.sh --profiles <profile_id> --non-interactive --samples 32
```

Output: `outputs/debug_previews/index.html`. The baked board textures live in
`<output>/_board_textures/` only while Blender runs.

Common issues:

- **Board too dark/bright**: scale `ring_light.strength` and `top_light.strength` together (the defaults render the board at its baked colours).
- **Solder/end caps too dark or too flat**: shift the ratio towards `top_light` (brighter flat metal) or `ring_light` (more edge contrast).
- **Pads look squashed**: the camera is tilted; use `location_mm: [0, 0, z]`.
- **`KeyError: 'pad_spacing_y'`**: SOT-23, QFN and SOIC profiles need `pad_spacing_y` in `geometry_ranges`.
- **Render failures**: check `render.blender.executable`; the Blender log is printed to the console.

## Where to change what

| What | Where |
|---|---|
| Pad layout (both backends) | `simple_sim/generators/opencv2d/draw.py` -> `_compute_pad_positions()` |
| Board look (both backends) | `simple_sim/generators/opencv2d/photoreal.py` |
| Bake / camera model hook | `simple_sim/generators/blender3d/bake.py` |
| Fillets, bodies, leads, markings, light, camera | `simple_sim/blender/render_batch.py` |
