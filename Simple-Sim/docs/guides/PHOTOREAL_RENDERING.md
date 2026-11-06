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
| Substrate | Solder-mask colour sampled from a palette measured on photos of the production boards (teal-green), resin mottling, fibre weave, sensor grain |
| Copper under mask | Copper pours (large planes with 45° chamfers, clearance around the part), 1-3 autorouted traces with 45° jogs, traces leaving pads, 0-12 vias (mostly tented). Brightness ratio mask-over-copper / mask-over-laminate measured on the photos |
| Residue | In `render.rework_artifacts` of the samples (default 0.3, 0 = clean): amber flux residue with a dried rim, smeared yellow contamination (colour and width measured on a board photo), thin patch wires (white or enamel) with shadow. Always outside the inspected part, so the label is unaffected: the model learns to ignore it |
| Neighbour parts | 0-6 chip resistors/capacitors around the inspected part (pads, fillets, bodies), sized from its pads, at 0/90°, possibly cut by the ROI border, never in its keep-out |
| Silkscreen | In 55 % of samples: white footprint outline (bars for passives, box + pin-1 dot for ICs) and a reference designator (`R123`, `C42`, `Q7`, `U12`), placed around the *nominal* footprint so it never moves with a defect |
| Pads | Mask-defined opening with dark rim, ENIG (gold) or HASL (silver) finish visible at the pad edge. Large (thermal) pads without a part show window-pane stencil islands |
| Solder | Height map with slight lumpiness, shaded as a rough mirror: flat areas reflect the overhead light, slopes reflect the dark board, sharp highlight towards the light. Concave fillet rising to the terminal when a body touches the pad; flat reflowed paste when it does not (MISSING / lifted end) |
| Chip body | Resistor: dark body, metallic end caps, white value code (`103`, `4R7`, ...). Capacitor (`component.type` contains `cap`): tan ceramic, nickel caps, no marking. Printed codes have blotchy ink |
| IC body | SOT-23 / SOIC-16 / QFN: speckled mold compound, side walls, gloss, pin-1 dimple, grainy laser marking (two Hershey fonts, lines slightly offset); gull-wing leads for SOT/SOIC; QFN gets real pad rows (`qfn_<N>` -> N/4 per side) plus the exposed thermal pad |
| Shadows | Contact shadow around every body, cast shadow of the standing part for tombstones, lit facet for corner lifts |
| Camera | Applied after downsampling so the image is not vector-sharp: lens blur, uneven illumination, soft highlight roll-off, signal-dependent and slightly correlated sensor noise |

Defect semantics are unchanged: `MISSING`, `MISALIGNED` (shift/rotation), `TOMBSTONE` (part pivoting about
the bottom edge of its anchored end: from above the steep, foreshortened top face plus the upright
metal end cap, with a shadow projected along the light), `SOLDER_BRIDGE`, `CORNER_LIFT` are driven by the same `defect_params`
and `tolerances` as before. Realism-group and custom image filters are applied afterwards exactly as
for the flat style.

`render.component_type` (filled from the profile's `component.type` by the pipeline) selects the
resistor/capacitor look for `chip_2pad`.

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
(JPEG and raw) of the production boards: median of bare mask pixels (hue 150-170°, i.e. teal-green
although it reads as green) and of pour vs. laminate areas.

## Camera model (`render.camera`) and calibration (`tools/calibrate_camera.py`)

Every image (2D and 3D) passes through a camera model: lens blur, uneven illumination, highlight
roll-off, read + shot noise. Its parameters can be set per run config:

```yaml
render:
  camera:
    blur_sigma: [0.55, 0.95]   # lens blur, Gaussian sigma in camera pixels (sampled per image)
    read_noise: [1.0, 4.0]     # read noise variance (8-bit levels^2, sampled per image)
    shot_noise: 0.04           # shot noise variance per intensity level
```

To fit them to the production camera, put a few hundred real ROIs (same `roi.mm_per_px`, similar
parts) in a folder, render a synthetic set with the run config and run:

```bash
python tools/calibrate_camera.py --real data/real_rois --synthetic data/run_0001/images --config configs/run_0001.yaml
```

It compares edge steepness (relative to contrast) and sensor noise (flat areas) and prints the
`render.camera` block to use. Re-render and repeat until it reports an extra blur near 0 and a noise
ratio near 1 (usually one or two rounds). Overview photos taken with a phone are not suitable: they
have a different scale, optics and lighting than the AOI camera.

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
