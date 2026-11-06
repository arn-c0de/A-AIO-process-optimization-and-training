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
downsampled (anti-aliasing; `render.supersampling: 1-4`, default 2, 4 = "Ultra": cleaner edges and
markings at about 4-5x the render time; also used for the baked board of the Blender backend); all randomness comes from the sample's seeded `rng`, so images stay
deterministic per seed.

| Layer | Details |
|---|---|
| Substrate | Solder-mask colour sampled from a palette measured on real part crops of the production boards (olive/bottle green, hue ~135-150°, HSV saturation ~90-110), resin mottling, fibre weave, sensor grain |
| Mask gloss | In `render.mask_gloss` of the samples (default 0.5, 0 = off): broad whitish-green glare of a light source (phone flash, ring light) on the glossy mask, broken up by orange-peel texture. Bare board only, parts and joints are drawn on top |
| Copper under mask | `render.board.routing: legacy` (default): copper pours (large planes with 45° chamfers, clearance around the part), 1-3 autorouted traces with 45° jogs, traces leaving pads, typically 2-16 vias (mostly tented, with a highlight on the mask dome); trace widths and via sizes scale with the part's pads. `routing: escape`: the same look from a **board model** in which every pad belongs to a net and the copper is routed around parts and other nets, see [Board model](#board-model-renderboard). Brightness ratio mask-over-copper / mask-over-laminate measured on the photos |
| Residue | In `render.rework_artifacts` of the samples (default 0.15, 0 = clean): amber flux residue with a dried rim, smeared yellow contamination (colour and width measured on a board photo), thin patch wires (white or enamel) with shadow. Always outside the inspected part, so the label is unaffected: the model learns to ignore it |
| Neighbour parts | Chip resistors/capacitors around the inspected part (pads, fillets, bodies), sized from its pads, possibly cut by the ROI border, never in its keep-out. 70 %: a row/column array of identical parts on a placement grid aligned with the inspected part (like real decoupling caps / resistor rows); otherwise 0-6 mixed parts at 0/90° |
| Silkscreen | In 55 % of samples: white footprint outline (bars for passives, box + pin-1 dot for ICs) and a reference designator (`R123`, `C42`, `Q7`, `U12`), placed around the *nominal* footprint so it never moves with a defect |
| Pads | Mask-defined opening with dark rim, ENIG (gold) or HASL (silver) finish visible at the pad edge. Large (thermal) pads without a part show window-pane stencil islands |
| Solder | `render.solder: legacy` (default; see [Solder joint model](#solder-joint-model-rendersolder) for `meniscus`): height map with slight lumpiness, shaded as a rough mirror: flat areas reflect the overhead light, slopes reflect a light grey-green environment, highlight towards the light; frosted (lead-free) with a generous diffuse share, so joints read as bright grey blobs as in real photos. Concave fillet rising to the terminal when a body touches the pad; flat reflowed paste when it does not (MISSING / lifted end) |
| Chip body | Resistor: dark body, metallic end caps, white value code (`103`, `4R7`, ...). Capacitor (`component.type` contains `cap`): tan ceramic, nickel caps, no marking. Printed codes have blotchy ink |
| IC body | SOT-23 / SOIC-16 / QFN: speckled mold compound, side walls, gloss, pin-1 dimple, grainy laser marking (two Hershey fonts, lines slightly offset); gull-wing leads for SOT/SOIC (shoulder bending out of the mold + flat tin-plated foot, laid out from the nominal footprint so they move rigidly with the body; SOT-23 body long axis along the pin-1/2 column); QFN gets real pad rows (`qfn_<N>` -> N/4 per side) plus the exposed thermal pad |
| Shadows | Contact shadow around every body, cast shadow of the standing part for tombstones, lit facet for corner lifts |
| Camera | Applied after downsampling so the image is not vector-sharp: lens blur, uneven illumination, soft highlight roll-off, signal-dependent and slightly correlated sensor noise |

Defect semantics are unchanged: `MISSING`, `MISALIGNED` (shift/rotation), `TOMBSTONE` (part pivoting about
the bottom edge of its anchored end: from above the steep, foreshortened top face plus the upright
metal end cap, with a shadow projected along the light), `SOLDER_BRIDGE`, `CORNER_LIFT`,
`SOLDER_BALL` (1-4 shiny solder balls on the mask next to the joints, never on a pad or under the
part) and `EXCESS_SOLDER` (high convex joints on 1-4 pads, over the end cap of chips, over the lead
foot of SOT/SOIC, bulging past the package edge on QFN; the part itself is placed like OK) are driven by the same `defect_params`
and `tolerances` as before. Realism-group and custom image filters are applied afterwards exactly as
for the flat style.

`render.component_type` (filled from the profile's `component.type` by the pipeline) selects the
resistor/capacitor look for `chip_2pad`.

## Board model (`render.board`)

The legacy copper generator draws traces and vias at random: traces run through the pads of the
inspected part and under its neighbours, nothing is connected to anything. `render.board` replaces
it with a small board model (`simple_sim/board/`, in mm, shared by the 2D render and the board
baked for Blender):

```yaml
render:
  board:
    routing: escape          # legacy (default: random copper, no board model) | none (no copper) | escape
    design_rules: consumer_dense   # consumer_dense | industrial | hdi (track widths, clearances, via sizes)
    pour_probability: 0.6    # share of boards with a ground pour
    drc: off                 # off | warn (log violations) | strict (re-draw the routing until clean, else raise)
```

With `routing: escape` the pipeline is footprints -> placement -> netlist -> pours -> routing ->
stitching vias -> DRC -> raster:

- **Footprints** carry pin numbers, roles (`signal`, `gnd`, `power`, `thermal`, `nc`) and the
  fan-out direction of every pad; the pad geometry is the same as before (regression-tested), so
  labels, defects and the Blender scene are unchanged.
- **Netlist** per family: chips on both ends (capacitors mostly with one pad on ground), SOT-23
  on all three pins, ICs on ~85-90 % of their pins with a few NC, QFN thermal pad on ground.
  Neighbour chips share nets with the inspected part (decoupling, pull-ups, series parts).
- **Fan-out router**: a straight stub out of the pad edge (necked down at fine-pitch parts), then
  a 45° route to the window border or a via; thermal pads get a via grid; ground pads inside a pour
  are connected by the pour with thermal-relief spokes. 1-3 through-nets cross the window, some
  as a parallel bus. Every candidate is checked on an occupancy grid (half the clearance per cell)
  and dropped on conflict, so nets never touch and nothing crosses a part's keep-out.
- **Pours** keep `zone_clearance` to foreign copper, have thermal reliefs on their own pads and
  stitching vias on a jittered grid.
- **Design rules** are drawn per board from the preset's ranges (variation stays), in mm at the
  profile scale. The profiles do not share one physical scale (`roi.mm_per_px` is 0.01 for most,
  0.03 for QFN-32, physical for the newer ones), so the renderer scales the rules with the drawn
  pad size: `photoreal.board_rule_scale` = smallest pad in mm / 0.31 mm (the 0603 pad the presets
  are tuned against), x2 for QFN and x1.5 for SOIC / SOT-6 (fine-pitch ICs: their real tracks are
  about as wide as their pads), clipped to [0.3, 6].
- **DRC** (`simple_sim/board/drc.py`): clearance, dangling track ends, copper on thermal/NC pads,
  foreign copper in the inspected part's keep-out, track widths. Used by the tests
  (`tests/test_board.py`) and optionally per sample.

The board's random streams are keyed (`default_rng([board_seed, stage])`), so a change in the
router never moves the neighbours or the netlist. Switching from `legacy` to `escape` changes
every image of a seed once (the shared stream of solder and camera shifts too). `legacy` only
means "no board model": it is not pixel-identical to v1.0.3, since the realism changes of this
release changed the copper, vias, colours and camera too. Datasets record the renderer version
(`generator.renderer_version` in `dataset_manifest.json`, `simple_sim/generators/version.py`) and
`generate --extend` refuses to mix versions (`--allow-renderer-mismatch` overrides);
`tests/test_render_golden.py` pins the 2D output so that any change is deliberate (bump the
version, regenerate the hashes). Timing is the same as the legacy generator.

## Solder joint model (`render.solder`)

`render.solder: legacy` (default) keeps the joints described in the table above, bit-identical to the
earlier output. `render.solder: meniscus` (both backends; 3D: the joints of the baked neighbour parts)
replaces them with a model calibrated against the public SolDef_AI macro crops (Calabrese et al.,
CC BY 4.0), where the legacy joints read as matte grey slabs with a ruler-straight outline:

| Aspect | `meniscus` |
|---|---|
| Outline | Contact line with rounded corners and a slightly wavy edge; on some pads a rim of bare ENIG/HASL finish stays visible |
| Profile | Concave meniscus `(1 - s)^p` (p 1.7-2.6) from the terminal (s = 0) to the contact line (s = 1), mean slope capped at ~25° over the visible toe, small rounded lip at the contact line. `EXCESS_SOLDER`: convex, ~1.5x higher. Reflowed paste (no part): low domes, one per stencil window |
| Surface | Low-frequency lumps plus fine grain (lead-free solder is frosted and wrinkled), so highlights break up into streaks |
| Shading | Environment mirror: the reflected view ray picks the bright overhead area, the light (full ring light or 1-3 lamps at 25-60° elevation, saturated white streaks) or, on steep slopes, the mask colour and the black part body (dark green/grey band next to the terminal). 15-32 % frosted diffuse share |
| End caps | Flat, matte tin top instead of the tube-like bright ridge |

Lighting and surface parameters (`solder.SolderLook`) are drawn once per image, so all joints of an
image share them. All meniscus randomness comes from a **side stream** of the sample generator
(`solder.side_stream`, which does not advance it): with the same seed, everything except the joints,
end caps, solder balls and bulges is pixel-identical to the legacy render, which makes A/B
comparisons (arena, `tools/sim_to_real.py`) direct. Code: `simple_sim/generators/opencv2d/solder.py`.

```yaml
render:
  backend: opencv_2d
  solder: meniscus
```

`generate --extend` does not compare `render.solder` yet: do not extend a legacy-solder dataset with
`meniscus` (or the other way round) unless you want both looks mixed.

## Real board backgrounds (`render.texture_dir`)

Optional. Point `render.texture_dir` at a folder of photos of **empty board areas** taken with the
production camera (PNG/JPG/BMP/TIFF). Each sample then uses a random crop of a random photo
(random 90° rotation and flip) as substrate instead of the procedural solder mask, traces and vias.
Pads, solder, silkscreen, parts and the camera model are drawn on top as usual.

```yaml
render:
  backend: opencv_2d
  texture_dir: data/board_textures
```

- Photos at least as large as the ROI are cropped at 1:1 camera scale; smaller ones are stretched.
- Files are picked in sorted order by the sample's seed, so datasets stay reproducible as long as the
  folder content does not change.
- The run fails at config validation if the folder does not exist or contains no images.

## Calibration source

Colours and proportions were taken from macro photos of real boards (dark olive solder mask, traces
shimmering through the mask, white silkscreen, bright concave fillets, black 0603 resistors with
white codes, tan MLCCs, black SOT/SOIC/QFN with grey markings). Mask colours, the copper-under-mask
brightness ratio, copper pours, tented vias and the density of neighbour parts come from 7 photos
(JPEG and raw) of the production boards. Mask, MLCC and solder colours were re-measured on the real
per-part crops of the DeepVision xpa boards (top: phone photo, bottom: flatbed-like capture): median
bare mask BGR (64, 89, 58) / (68, 86, 50), MLCC ~(120, 152, 170), joints ~(187, 196, 196). The earlier
palette (hue 150-170°, saturation ~150) plus the default ±15° hue jitter rendered a lime/cyan mask
that no real board shows; the default hue jitter is now ±5°, saturation 0.85-1.1.

## Camera model (`render.camera`) and calibration (`tools/calibrate_camera.py`)

Every image (2D and 3D) passes through a camera model: viewing geometry, lens blur, uneven
illumination, highlight roll-off, read + shot noise, optionally phone-style sharpening and JPEG.
Two presets, every value can be overridden (ranges are sampled per image):

```yaml
render:
  camera:
    preset: aoi                # aoi (default): fixed top-down AOI camera | phone: handheld phone photos
    blur_sigma: [0.55, 0.95]   # lens blur, Gaussian sigma in camera pixels
    read_noise: [1.0, 4.0]     # read noise variance (8-bit levels^2)
    shot_noise: 0.04           # shot noise variance per intensity level
    tilt_deg: [0, 0]           # oblique view (0-60)
    zoom: [1, 1]               # >= 1: farther away, the part appears this much smaller (more board around it)
    defocus: [0, 0]            # extra blur towards the near/far edge of a tilted view (depth of field)
    sharpen: [0, 0]            # phone ISP sharpening (unsharp mask amount)
    jpeg_quality: [100, 100]   # < 95 compresses
    capture_scale: [1, 1]      # sensor px per ROI px; < 1: captured at low resolution and upscaled to the ROI
    chroma_shift: [0, 0]       # lateral chromatic aberration (R/B shift at the border, sensor px)
```

`preset: phone` sets `tilt_deg: [5, 40]`, `zoom: [1, 1.5]`, `blur_sigma: [0.5, 1.2]`,
`defocus: [0.5, 2.5]`, `read_noise: [4, 16]`, `shot_noise: 0.08`, `sharpen: [0.3, 0.9]`,
`jpeg_quality: [70, 92]`, `capture_scale: [0.2, 0.7]`, `chroma_shift: [0, 0.8]`. Real per-part crops
from phone photos are small (a 0603 crop is ~50 px, ~10 px/mm) and get upscaled to the model input;
`capture_scale` reproduces that: blur, noise, sharpening and JPEG happen at the low sensor resolution,
then the image is resized to the ROI (soft, blocky, sharpening halos); `configs/run_0001_phone.yaml` is `run_0001` with this preset. 2D renders a
larger board area for zoomed-out/oblique views (more neighbour parts) and projects it; the Blender
backend tilts its camera and widens its view accordingly. The part always stays centred, so labels
are unchanged.

To fit them to the production camera, put a few hundred real ROIs (same `roi.mm_per_px`, similar
parts) in a folder, render a synthetic set with the run config and run:

```bash
python tools/calibrate_camera.py --real data/real_rois --synthetic data/run_0001/images --config configs/run_0001.yaml
```

It compares edge steepness (relative to contrast) and sensor noise (flat areas) and prints the
`render.camera` block to use. Re-render and repeat until it reports an extra blur near 0 and a noise
ratio near 1 (usually one or two rounds). Overview photos taken with a phone are not suitable: they
have a different scale, optics and lighting than the AOI camera.

## Where the domain filters are applied (`render.post_filters`)

The run config's `domains.<name>` ranges (`blur_sigma`, `noise_stddev`, `lighting_brightness`) and the
`augment` rotation predate the camera model. With the default `render.post_filters: legacy` they are
still applied to the *finished* camera image: the sensor noise is blurred, then a second noise is
added, and the rotation (`warpAffine` of the final image) interpolates the noise and smears JPEG
blocks. `render.post_filters: camera` (2D photoreal only) moves them into the camera model:

```yaml
render:
  post_filters: camera   # legacy (default, bit-identical) | camera
```

- the domain `blur_sigma` is combined with the lens `blur_sigma` (`sqrt(a^2 + b^2)`, sensor pixels),
- the domain `noise_stddev` squared is added to the read-noise variance (the shot noise stays),
- the rotation (cardinal quarter turn + jitter) is a camera *roll* before the camera model: quarter
  turns are exact (`np.rot90`), the jitter is part of the view projection, the board is rendered on
  a correspondingly larger canvas so the corners stay filled.

Brightness/contrast, colour and the extended image filters (`IMAGE_FILTERS`) are applied as before.
Datasets rendered with `camera` look different for the same seed (the larger canvas changes the
board around the part), so do not `--extend` a legacy dataset with it.

### Several domains, held-out camera (`domains`, `splits.*_domains`)

Samples are spread evenly over all `domains` (`index % n`, so every class has the same share per
domain). Each domain has its own filter ranges and may override the camera:

```yaml
domains:
  domain_A: {lighting_brightness: [0.9, 1.1], blur_sigma: [0, 0.6], noise_stddev: [0, 6]}
  domain_B:
    lighting_brightness: [0.8, 1.2]
    blur_sigma: [0, 0.4]
    noise_stddev: [0, 4]
    camera: {preset: phone}      # keys as render.camera, override the render section's
splits:
  train_domains: [domain_A]
  val_domains: [domain_B]        # held out: val and test see only the phone camera
  test_domains: [domain_B]
```

A domain listed only under `val_domains` / `test_domains` is split between val and test with the
fractions renormalised over those two, so the `eval` report measures generalisation to a camera the
model never trained on instead of interpolation inside one domain. A domain listed everywhere is
split as before. `configs/run_0001_xdomain.yaml` combines this with `post_filters: camera`,
`defects.placement_sampling: continuous` and the training options (`train.augment`, `scheduler`,
`label_smoothing`), see `README.md`.

## Measuring the sim-to-real gap (`tools/sim_to_real.py`)

Compares a folder of synthetic ROIs with a folder of real camera ROIs (both searched recursively,
no labels needed) using ResNet18 embeddings (ImageNet weights, or a trained checkpoint via
`--weights`):

```bash
python tools/sim_to_real.py --synthetic data/run_0001/images --real data/real_rois --json gap.json
```

| Metric | Meaning |
|---|---|
| `one_nn_accuracy` | 1-NN two-sample test on equally many real/synthetic images. 0.5 = indistinguishable, 1.0 = fully separable |
| `nn_distance_ratio` | Mean distance real -> nearest synthetic divided by real -> nearest other real. ~1 = synthetic covers the real images as well as they cover each other |

Sanity check on renders: photoreal vs. photoreal (other seeds) gives 0.52 / 1.01, flat vs. photoreal
1.00 / 9.8. Use the same real folder and `--seed` between releases so numbers are comparable;
a few hundred real ROIs per package are enough.

## Roadmap towards "indistinguishable from the camera"

1. **Done – procedural photoreal 2D** (this document).
2. **Done – Blender upgrade**: the 3D backend bakes this generator's board as texture, models solder
   fillets as meshes, adds markings, leads and an AOI ring light (see `3D_RENDERING_QUICKSTART.md`).
3. **Done – real-photo texture bank**: `render.texture_dir` (see above).
4. **Done – sim-to-real check**: `tools/sim_to_real.py` (see below). Still open: a small labelled
   set of real ROIs per package to measure classifier accuracy on real data.
