# Photoreal 2D Rendering

The OpenCV 2D backend has two render styles, selected with `render.style` in the run config:

| Style | Description |
|---|---|
| `photoreal` (default) | Procedural textures + shading calibrated against photos of real assembled boards |
| `flat` | Legacy flat-colour rectangles (`substrate_color`, `copper_color`, `component_color`) |

Existing run configs do not need changes: a missing `render.style` means `photoreal`.
Set `style: flat` to reproduce the old look (e.g. for ablations or extending a flat dataset).

## What `photoreal` renders

Implemented in `simple_sim/generators/opencv2d/photoreal.py`. Everything is rendered at 2x and
downsampled (anti-aliasing); all randomness comes from the sample's seeded `rng`, so images stay
deterministic per seed.

| Layer | Details |
|---|---|
| Substrate | Solder-mask green sampled from a palette measured on real boards, resin mottling, fibre weave, sensor grain |
| Copper under mask | 1-3 autorouted traces with 45° jogs, traces leaving pads, 0-3 vias with annular ring and dark hole |
| Silkscreen | White footprint outline (bars for passives, box + pin-1 dot for ICs) and a reference designator (`R123`, `C42`, `Q7`, `U12`) |
| Pads | Mask-defined opening with dark rim, ENIG (gold) or HASL (silver) finish visible at the pad edge |
| Solder | Height map with slight lumpiness, shaded as a rough mirror: flat areas reflect the overhead light, slopes reflect the dark board, sharp highlight towards the light. Concave fillet rising to the terminal when a body touches the pad; flat reflowed paste when it does not (MISSING / lifted end) |
| Chip body | Resistor: dark body, metallic end caps, white value code (`103`, `4R7`, ...). Capacitor (`component.type` contains `cap`): tan ceramic, nickel caps, no marking |
| IC body | SOT-23 / SOIC-16 / QFN: speckled mold compound, side walls, gloss, pin-1 dimple, laser marking; gull-wing leads for SOT/SOIC; QFN gets real pad rows (`qfn_<N>` -> N/4 per side) plus the exposed thermal pad |
| Shadows | Contact shadow around every body, long cast shadow for tombstones, lit facet for corner lifts |
| Camera | Applied after downsampling so the image is not vector-sharp: lens blur, uneven illumination, soft highlight roll-off, signal-dependent and slightly correlated sensor noise |

Defect semantics are unchanged: `MISSING`, `MISALIGNED` (shift/rotation), `TOMBSTONE` (foreshortened
body anchored on one end), `SOLDER_BRIDGE`, `CORNER_LIFT` are driven by the same `defect_params`
and `tolerances` as before. Realism-group and custom image filters are applied afterwards exactly as
for the flat style.

`render.component_type` (filled from the profile's `component.type` by the pipeline) selects the
resistor/capacitor look for `chip_2pad`.

## Calibration source

Colours and proportions were taken from macro photos of real boards (dark olive solder mask, traces
shimmering through the mask, white silkscreen, bright concave fillets, black 0603 resistors with
white codes, tan MLCCs, black SOT/SOIC/QFN with grey markings).

## Roadmap towards "indistinguishable from the camera"

1. **Done – procedural photoreal 2D** (this document).
2. **Blender PBR upgrade** (`simple_sim/blender/render_batch.py`): texture the substrate with the same
   trace/via/silkscreen generator baked to an image, add a solder-fillet mesh instead of flat pads,
   marking decals, HDRI/ring-light rig matching the AOI box.
3. **Real-photo texture bank**: optional `render.texture_dir` with ROI-sized crops of empty board
   areas from the production camera, used as substrate background (domain randomisation + real
   backgrounds is the cheapest way to close the remaining gap).
4. **Sim-to-real check**: train on synthetic, measure on a small labelled set of real ROIs per
   package; track feature-distance (e.g. kNN on ResNet embeddings) between synthetic and real images
   per release so realism changes are measurable, not eyeballed.
