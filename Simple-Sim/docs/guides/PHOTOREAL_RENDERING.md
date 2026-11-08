# Photoreal 2D Rendering

The OpenCV 2D backend has two render styles, selected with `render.style` in the run config:

| Style | Description |
|---|---|
| `photoreal` (default) | Procedural textures + shading calibrated against photos of real assembled boards |
| `flat` | Legacy flat-colour rectangles (`substrate_color`, `copper_color`, `component_color`) |

Existing run configs do not need changes: a missing `render.style` means `photoreal`.
Set `style: flat` to reproduce the old look (e.g. for ablations or extending a flat dataset).

## What `photoreal` renders

Implemented in the package `simple_sim/generators/opencv2d/photoreal/` (one module per layer, see its
`__init__`; `scene.py` composes them, an immutable `RenderContext` carries the supersampling factor and
the solder look of one render). Everything is rendered at 2x and
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
                             # warn / strict also store the result per sample (render_meta.board_drc in meta.jsonl)
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
  (`tests/test_board.py`) and optionally per sample: with `drc: warn | strict` every row of
  `meta.jsonl` gets `render_meta.board_drc`, e.g.
  `{"ok": false, "violations": 2, "kinds": {"clearance": 2}, "routing": "escape", "design_rules": "consumer_dense", "drc": "warn"}`
  (counts per violation kind, no messages; `strict` rows are always `ok`). The renderer returns
  only the image; the result is collected on the side (`simple_sim.board.collect_drc`,
  `opencv2d.render_record_with_meta`) without touching any random stream. Not recorded for
  `legacy` / `none` copper, `drc: off` or the Blender backend. The GUI (Profile Preview "Quality"
  box, Pipeline "Source" section) can override `routing`, `drc` and `design_rules` of the run
  config, and `render.look`, without editing the YAML.

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
comparisons (arena, `tools/sim_to_real.py`) direct. Code: `simple_sim/generators/opencv2d/photoreal/meniscus.py`.

```yaml
render:
  backend: opencv_2d
  solder: meniscus
```

`generate --extend` refuses to mix joint models: it compares the resolved `render.solder` (and the
other look options, see `generation/render_options.py`) with the dataset's `config.yaml`;
`--allow-renderer-mismatch` turns that into a warning.

## Target look (`render.look`)

`render.look` picks the domain the renderer imitates. `legacy` (default) is the palette above,
bit-identical to earlier output. `quins` imitates the **QUINS AOI flatbed scans of the DeepVision xpa
production boards** (`DeepVision-Web/data/store/datasets/xpa-top/quins/crops`, 23.7 px/mm). It was
tuned by measuring ~20 000 good crops (median colour of each part form, centre patch of the part)
and comparing whole-image statistics with `tools/look_stats.py`:

| Material / effect | QUINS scans | `legacy` | `quins` |
|---|---|---|---|
| Solder mask (BGR, per-image median) | (62..72, 80..94, 46..55) | similar palette, stronger weave/mottling | measured palette, weave 1.2 %, mottling 2 % |
| Copper under the mask | faint | x1.29-1.40 | x1.12-1.20 |
| MLCC ceramic | (86, 108, 121) | (120, 152, 170) | measured |
| Moulded IC/SOT, resistor, ferrite, power inductor, LED housing | (64, 68, 65), ~(44, 47, 44), dark, (99, 105, 104), white | all (34, 34, 34) | measured (only when the profile keeps the dark default colour) |
| Chip terminals | matte tin, nearly white (pixels > 200: median BGR (209, 228, 214), p90 253), brightest where the plating rolls over the cap edges | tube-like ridge, (170, 175, 177) | flat (198, 214, 204), bright edges |
| Joints | two line lights along the scan direction: the fillet slopes facing +-x blaze (vertical bright bands on every terminal, whatever the part's orientation), flat solder mirrors the dark camera (grey dome tops on empty pads inside a bright rolled rim), green-cast metal | legacy fillets | `meniscus` with two side lights along image x, rotated with the camera roll (`meniscus.sample_scanner_look(rng, roll_deg)`), metal tint (0.94, 1, 0.95), high domes on empty pads |
| Contact shadows, silkscreen | hardly any shadow (diffuse light), almost no print next to passives | shadow 0.35-0.4, 55 % printed | `Look.shadow` 0.3x, `Look.silkscreen_skip` 85 % |
| Mask glare, rework artifacts | none / rare | 50 % / 25 % | 0 % / 3 % |
| Camera | 23.7 px/mm, upscaled, ~0.9 levels noise, blue/orange fringes along the scan axis | AOI preset | preset `quins` (`channel_shift` 1.6-3.2 px along x) |

Whole images (p10/p50/p90 over 600 real crops vs. 42 renders of 7 run configs, `tools/look_stats.py`):

| Metric | QUINS | `legacy` | `quins` |
|---|---|---|---|
| mask hue (OpenCV) | 71/74/75 | 61/69.5/75 | 70/71/74 |
| p1 grey | 44/51/62 | 17/34/60 | 28/50/62 |
| p99 grey | 170/199/229 | 142/166/186 | 141/193/219 |
| white share | 0.3/1.2/3.8 % | 0.02/0.5/1.3 % | 0.1/1.8/3.1 % |

**Second round (2026-11-07), tuned on side-by-side montages of real and synthetic 0402 crops (3x body
length, upscaled to 256 px; `tools/look_montage.py --real real_eval/ --data DATASET --out m.png --stats`,
one row of real crops per class folder above the synthetic row of the same class) and on the statistics
that separated them (the `--stats` table).** The first `quins` joints were
far too dark: the scans show blazing fillets and bright empty pads, the renders matte grey ones
(an earlier tuning had dimmed them on purpose because a classifier confused bright joints with
excess solder). Per-image means over 300 crops each (`tools/look_stats.py` plus LAB spread):

| Metric (256 px crops) | real good 0402 MLCC | `quins` before | `quins` now | real empty 0402 pad | `quins` MISSING now |
|---|---|---|---|---|---|
| L std (LAB) | 30.4 | 21.9 | 29.3 | 36.8 | 30.3 |
| b std (yellow-blue, fringes + tan body) | 8.4 | 4.6 | 6.8 | 8.7 | 7.9 |
| share of pixels > 200 | 1.0 % | 0.0 % | 1.5 % | 3.7 % | 1.8 % |
| share of pixels > 150 | 6.7 % | 2.7 % | 6.9 % | 10.1 % | 7.4 % |
| p90 / p99 grey | 139 / 201 | 123 / 164 | 135 / 213 | 150 / 229 | 137 / 216 |
| median BGR of pixels > 200 | (209, 228, 214) | (206, 212, 201) | (199, 233, 216) | (211, 222, 213) | (193, 234, 212) |

What changed in `quins` (legacy untouched, 31 golden variant cases updated, `RENDERER_VERSION` unchanged):
the scanner joint light (`meniscus.sample_scanner_look`) is two line lights along image x instead of
a ring (the real crops show vertical bright bands on every terminal and the left/right rims of every
empty pad, in all part orientations), wide in azimuth (half width 50-65 deg) but a narrow elevation
band (22-34 +- 11-17 deg: bright lines on the slopes, not blocks) and strong (gain 440-540), the
flat-solder brightness is low (120-155: a mirror of the dark camera), the metal has the measured green
cast (`SolderLook.tint`), empty pads get high domes (`paste_scale` 1.0-1.3), the terminal tops are
grey mirrors with blazing rolled edges on both sides (the "bracket" look of every real terminal;
`parts.draw_body`, `flat_caps`: 0.7 + 0.7 x edges of (200, 216, 206)), contact shadows are faint
(`Look.shadow`), silkscreen is rare (`Look.silkscreen_skip`), and the RGB misregistration is stronger and
fixed to the scan axis (`camera.channel_axis: x`). The light direction is fixed in the finished image:
`RenderContext.for_render(config, rng, roll_deg)` rotates the azimuths by the camera roll
(`post_filters: camera`) or the final image rotation (`post_filters: legacy`,
`render_photoreal_base(..., light_roll_deg=)`), tested in `tests/test_look.py`. Still open: real empty
pads are brighter than the rendered ones (table), the glossy MLCC body shows a light stripe along its
middle in the scans, and the scans have a little more yellow-blue spread.

The camera preset `quins` adds three keys to `render.camera` (all off by default, without a random
draw, so other presets render as before):

- `sensor_px_per_mm: [min, max]`: sensor resolution. Replaces `capture_scale` by
  `sensor_px_per_mm x roi.mm_per_px x zoom` (at most 1), so the scan resolution holds for every profile
  scale and field of view.
- `channel_shift: [min, max]`: line-scanner RGB misregistration in sensor pixels. Blue and red are
  shifted apart along one axis, which gives the colour fringes at bright edges.
- `channel_axis: random | x | y`: that axis. `random` (default) draws one of the four axis directions
  per image; a line scanner has a fixed scan direction, so the `quins` preset uses `x`. Like the scanner
  lights, a fixed axis is kept in the finished image: with `post_filters: legacy` it is pre-rotated
  against the augmentation rotation (`camera_model(..., axis_roll_deg=)`).
- `crop_part_lengths: [min, max]` (2D): field of view in body lengths, replacing `zoom`. DeepVision cuts
  every QUINS crop at 3x the body length (measured 3.02-3.04 on all chip forms), so the preset uses
  `[3, 3]`; for the 0402 profiles that magnifies the view (zoom 0.73), for 0603 at 100 px/mm it zooms out.

`render.look: quins` sets the defaults `camera.preset: quins`, `mask_gloss: 0`, `rework_artifacts: 0.03`
and `solder: meniscus`, and packs the neighbour parts densely (`Look.packing`: `look.DENSE_PACKING`; the xpa boards are densely
populated); anything set explicitly in `render` wins. Look-specific randomness is drawn from the
meniscus side stream only, so `quins` shares part placement and labels of a seed with `legacy`. Code:
`simple_sim/generators/opencv2d/photoreal/look.py`.

The example config `configs/run_0402_c_quins.yaml` also sets `post_filters: camera` and a calm domain
(blur <= 0.3, noise <= 2, brightness 0.95-1.05). The domain noise of the other run configs (up to
sigma 10, added at ROI resolution after the camera) alone made the images ~20x sharper than the scans
(Laplacian variance 68 vs. 4.7).

```yaml
render:
  backend: opencv_2d
  look: quins
  post_filters: camera
```

A/B against 300 good 0402-MLCC crops (form-02) of the scans, 300 OK renders each, ImageNet ResNet18
embedding (`tools/sim_to_real.py`; NN distance ratio, 1.0 = as close as real to real):

| Render | NN distance ratio | Laplacian variance (real 4.7) |
|---|---|---|
| `run_0402_c.yaml` (legacy) | 3.42 | 101 |
| `look: quins`, run_0402_c domain | 2.70 | 68 |
| `run_0402_c_quins.yaml` (crop 3x, crowd 2, calm domain, camera post filters; before the classifier tuning below) | 2.57 | 4.5 |
| same config, second tuning round (side lights, bright joints, grey terminal tops; 300 vs. 300) | 2.52 | 5.2 |

1-NN accuracy stays ~1.0 (still separable): what is left is content (silkscreen and via placement,
part arrangement) rather than colour. Variants of the joint lighting (low ring light, dark fillet
slopes, more frost) and a darker mask did not move the ratio or made it worse.

### Classifier test on real crops (`tools/eval_real.py`)

`tools/eval_real.py --checkpoint model.pt --real DIR` scores a checkpoint on real crops sorted into one
sub-folder per class (`OK`, `MISSING`, `MISSING@empty_pad`, ...; the part after `@` is a sub-group).
Test set used: 400 good 0402 MLCCs (form-02, QUINS test split), 35 crops labelled "fehlt", 400 empty
0402 pads ("leer", val/test splits) and 5 "versetzt". ResNet18, 2400 synthetic images (400 per class),
6 epochs on CPU, one run each (single runs vary strongly):

| Training data | good -> OK | "fehlt" detected | empty pad detected | false calls | escapes |
|---|---|---|---|---|---|
| `run_0402_c` (legacy) | 3 % | 0 % | 0 % | 97 % | 0 % |
| `run_0402_c_quins` (state of the A/B table above) | 0 % | 0 % | 1 % | 100 % | 0 % |
| + train augment, high pillow-shaped paste on empty pads (`SolderLook.paste_scale`) | 0 % | 57 % | 79 % | 100 % | 0 % |
| + narrow terminals (`Look.cap_len`), dim overhead / bright ring joint light | 0 % | 0 % | 0 % | 100 % | 0 % |
| + dense neighbour rows (`Look.packing`, `look.DENSE_PACKING`) | 0 % | 0 % | 0 % | 100 % | 0 % |
| + 200 real good crops (QUINS train split) | 99.5 % | 9 % | 4 % | 0.5 % | 96 % |
| + 200 real good + 200 real empty pads (train split) | 96.5 % | 94 % | 100 % | 3.5 % | 0.7 % |
| only those 400 real crops, no synthetic images | 96 % | 97 % | 100 % | 4 % | 0.2 % |
| dense-packing quins + 200 + 200 other real crops, mixed with `tools/mix_real.py` | 99.8 % | 97 % | 100 % | 0.3 % | 0.5 % |

("detected" = predicted as any defect; "fehlt" and the tested empty pads were never trained on.)

Second tuning round of the look (side lights, blazing joints, grey terminal tops; see "Target look"),
2400 synthetic images, 8 epochs, train augment + cosine, benchmark from `tools/real_crops.py`
(form-02: 500 good test crops, 10 "fehlalarm", 26 "fehlt", 5 "versetzt", 1 "verdreht", 9 "loetfehler",
4 other defects, 300 empty form-01 pads; form-01: 300 good, 48 "fehlalarm", 9 "fehlt", 11 foreign objects):

| Training data | good -> OK (form-02 / form-01) | "fehlt" detected | empty pad detected | "versetzt"/"loetfehler" detected | false calls | escapes |
|---|---|---|---|---|---|---|
| synthetic only, second-round look | 0 % / 0 % | 100 % | 100 % | 100 % | 100 % | 0 % |
| + 300 real good (form-02 train) + 300 real empty pads (form-01 train), `tools/mix_real.py` | 100 % / 87 % | 85 % | 100 % | 5/5, 7/9 | 0.2 % | 2.9 % |
| those 600 real crops with only 8 synthetic images (real-only control) | 100 % / 82 % | 92 % | 100 % | 3/5, 4/9 | 0 % | 3.8 % |

(form-01 good parts are black 0402 chips the models never saw as good; "versetzt"/"loetfehler" are
detected as MISSING, not as their own class; foreign objects and wrong parts pass as OK in both.)

The synthetic-only model still rejects every real crop (mostly as TOMBSTONE, the rest SOLDER_BALL),
although the image statistics now match (table above, NN distance ratio 2.52). Probing the model with
edited real crops: matching the real crops' LAB statistics or histogram to the renders, blurring,
sharpening or re-sampling them changes nothing; a real part pasted into a synthetic surround is
TOMBSTONE/MISALIGNED, a synthetic part pasted into a real surround is SOLDER_BALL, and a synthetic
OK image with its part painted over stays "OK" in 67 % of the cases. So the classifier keys on
texture details of both the part and its surroundings that the statistics do not capture, not on
colour or sharpness, and it never learned to look at the joint. Closing that with renderer tuning
alone is open-ended; the practical route stays real good parts + real empty pads in the training set.
With them, the synthetic defects do earn their place on the rare classes: the mixed model flags all
5 shifted and 7 of 9 solder-defect crops where the real-only model flags 3 and 4, and it accepts more
of the unseen black form-01 chips (87 % vs. 82 %); on the plentiful classes (good, empty pad, missing)
both are at 92-100 % and the differences are within single-run noise. The 10 "fehlalarm" crops (AOI
false alarms reviewed as good) pass with both.

Findings:

- **Synthetic images alone do not transfer.** Real good parts are rejected (TOMBSTONE, MISALIGNED,
  SOLDER_BALL) by every synthetic-only model. The decision is not about the joint: with the part
  masked out the prediction stays TOMBSTONE, and giving synthetic OK images the LAB colour statistics
  of the scans flips ~45 % of them to TOMBSTONE. The scans have more luminance spread (L std 32 vs. 20
  with dense packing 22) and twice the yellow-blue spread.
- **Real crops in only one class teach the domain, not the class**: with real good crops added, every
  real image is "OK", defects included.
- **Real good crops plus real empty pads work** (both are plentiful in production: 46 000 good and
  2 800 empty 0402 pads in the train split), but in this test the synthetic images add nothing
  measurable for OK / MISSING: the real-only model is as good.
- Where synthetic data still has to carry the load are the classes without real examples
  (MISALIGNED: 5 real crops, TOMBSTONE, solder defects). The real-only model flags all 5 "versetzt"
  crops as MISSING (detected, wrong class); the mixed model flags 4 of 5. Too few to compare.

Recommendation: train on synthetic defects plus real good parts and real empty pads, and collect real
crops of the rarer defects for evaluation before relying on the synthetic ones.

`tools/mix_real.py` does the mixing reproducibly: it takes a folder with one sub-folder per class
(the same layout as `tools/eval_real.py`), resizes the crops to the ROI size and appends them to a
split of a generated dataset (`images/real/`, `meta.jsonl` with `render_backend: real` and
`render_meta.real: true`, `labels.jsonl`, split file, `dataset_manifest.json` `real_crops` history):

```bash
python -m simple_sim generate --config configs/run_0402_c_quins.yaml --out data/run_0402_c_quins
python tools/mix_real.py --data data/run_0402_c_quins --real real_train/ --limit 200   # OK/, MISSING@empty_pad/
python -m simple_sim train --data data/run_0402_c_quins --out models/run_0402_c_quins.pt
python tools/eval_real.py --checkpoint models/run_0402_c_quins.pt --real real_eval/   # crops never mixed in
```

Real rows carry the dataset's median nominal geometry and zero defect offsets (the true values are
unknown), so they must not be used for geometry regression or seed replay.

`generate --extend` compares the resolved look (`render.look` and the camera preset, mask gloss and
rework defaults it implies) with the dataset, so looks are not mixed by accident.

## DeepVision app looks (`dv_top`, `dv_bottom`)

The DeepVision inspection app does not use the QUINS scanner: it photographs the TOP side of a board
with a phone (handheld, flash, ~19-22 px/mm source) and captures the BOTTOM side with a flatbed-like
top-down camera, cuts one crop per part and lets reviewers judge it. `render.look: dv_top` and
`dv_bottom` imitate those crops; `legacy` and `quins` are untouched (`tests/test_render_golden_variants.py`:
620 earlier cases unchanged, 60 added). Both looks exist for the `aoi-judge` Siamese model (see
[Export for DeepVision](#export-for-deepvision-toolsexport_deepvisionpy)).

### The real crops (`tools/dv_crops.py`)

`tools/dv_crops.py` exports the reviewed app crops (`.../products/xpa/aoi/labels/<side>.ndjson`, newest
review per `(board, id)`, retracted reviews and reviews without a colour crop skipped) into the class
folders of `tools/eval_real.py` / `tools/look_montage.py`, with the package of each crop from the
product's `pnp.csv` (nearest placement; TOP centres match within 0.02 mm, BOTTOM within 0.3 mm median,
0.6 mm p90):

```bash
python tools/dv_crops.py --labels .../products/xpa/aoi/labels --side top --out outputs/dv_real/top \
    --pnp .../products/xpa/pnp.csv --limit-ok 500                        # all packages
python tools/dv_crops.py ... --out outputs/dv_real_pkg/top/CAPC1005 --packages CAPC1005 --limit-ok 300
```

Review `fault` -> folder: `fehlt` -> `MISSING`, `versetzt` -> `MISALIGNED`, `verdreht` -> `TOMBSTONE` (the
export contract below maps TOMBSTONE back to `verdreht`; on the xpa boards "verdreht" is a *rotated* part
and the 5 true tombstones are labelled `grabstein` -> `TOMBSTONE@grabstein`), `loetbruecke` ->
`SOLDER_BRIDGE`, everything else `DEFECT@<fault>`. Counts (2026-11-08, `outputs/dv_real/<side>/summary.json`):

| Side | latest reviews | OK available (exported) | fehlt | loetbruecke | verdreht | grabstein | versetzt | other faults | skipped |
|---|---|---|---|---|---|---|---|---|---|
| TOP | 4197 | 3417 (500) | 116 | 68 | 28 | 5 | 6 | 29 | dnp 168, unclear 129, unjudgeable 86, no colour crop 133 |
| BOTTOM | 3238 | 2624 (500) | 28 | 150 | 16 | 0 | 1 | 82 | dnp 127, unclear 29, unjudgeable 55, no colour crop 120 |

Crop geometry (PNG sizes of the exported colour crops): the app cuts **3.04 body lengths** (`fieldMm`
3.037 for a 1.0 mm 0402, 4.81 for a 1.6 mm 0603, 6.03 for the 2.0 mm arrays) and resamples to a fixed
**49 px (TOP) / 50 px (BOTTOM)** for the small parts, i.e. **16.1 px/mm for 0402**, 10.1 px/mm for 0603
and 8.1 px/mm for arrays on TOP; BOTTOM keeps ~16 px/mm for arrays (68 px) and 0603 (76 px) in most runs
but also has 34/38 px variants (older runs). The 1210 MLCC crops are 30 px on TOP (9.2 mm field, 3.2 px/mm)
and 24-156 px on BOTTOM. The judge upscales every crop to 64 px.

### Measurements and the derived looks

Per-image medians of 300 good 0402 crops per side (centre patch of the part, bare mask, pixels > 200;
`tools/look_stats.py`, `tools/look_montage.py --stats`, 256 px):

| Material / effect | TOP (phone) | BOTTOM (flatbed) | QUINS scans | `dv_top` / `dv_bottom` |
|---|---|---|---|---|
| Solder mask, per-image median BGR (p25-p75) | (85, 105, 47) ((69, 91, 24)-(115, 127, 70)), teal glare up to hue 89 | (70, 89, 48) ((64, 84, 37)-(95, 119, 58)) | (67, 87, 49) | k-means centres of the medians as palettes (9 / 6 colours), mottling 6 % / 3 % |
| MLCC body (centre) | (114, 138, 143) | (85, 106, 113) | (86, 108, 121) | `Look.cap_body` (110, 134, 142) / (85, 106, 113) |
| 0402 resistor body (centre, blurred) | (80, 82, 59) | (68, 80, 65) | ~(44, 47, 44) | (48, 52, 42) / (40, 44, 38) before shading |
| Metal (median of pixels > 200, share) | (209, 219, 215), 7.3 % | (207, 220, 221), 2.9 % | (209, 228, 214), 1.0 % | flat tin caps (205, 214, 210) / (196, 206, 204); meniscus joints under a flash lobe (`meniscus.sample_flash_look`: one wide lobe at 50-72 deg, bright overhead) / a diffuse ring (`sample_flatbed_look`) |
| p1 grey (dark gaps, bodies) | 32 | 34 | 51 | `Look.shadow` 1.0, `DV_PACKING` with 50 % resistor neighbours (xpa pnp: 278 R vs 272 C 0402) |
| Laplacian variance (256 px) | 4.8 | 3.8 (CAPC1005) / 4.7 (RESC1005) | 4.7 | camera presets `dv_top` / `dv_bottom` |
| Flat-area noise (`calibrate_camera`) | 0.28 | 0.26 | ~0.9 | read noise 0.5-3 / 0.3-2 |
| Mask glare, silkscreen | flash glare in most crops, little print | hardly any glare | none | `mask_gloss` 0.6 / 0.15, `silkscreen_skip` 0.8 |

The camera presets (`camera.CAMERA_PRESETS`): `sensor_px_per_mm: [15.5, 16.5]` (the app's 16 px/mm crop
is the resolution the judge sees), `crop_part_lengths: [3.0, 3.1]`, JPEG 75-92 / 80-95, lens blur 0.35-0.8 /
0.35-0.75 sensor px, `dv_top` additionally tilt 0-12 deg with depth of field and ISP sharpening (handheld
phone), `dv_bottom` stronger lateral chromatic aberration (`chroma_shift` 0.2-0.9: the red/blue fringes on
every terminal edge) and almost no tilt. The `phone` preset was not a usable starting point: its
blur/JPEG/upscale happen at `capture_scale` 0.2-0.7 of the ROI, not at a fixed 16 px/mm, so its renders are
17x sharper than the app crops (Laplacian variance 80 vs 4.8).

Whole-image statistics (p10/p50/p90 over the images; real: 300 good 0402 crops of one package, synthetic:
160 renders (120 OK + 40 MISSING) of the matching 0402 run config, 256 px):

| Metric | real TOP CAPC1005 | `legacy` (run_0402_c) | `quins` | `phone` preset | **`dv_top`** | real BOTTOM RESC1005 | `legacy` (run_0402_r) | `quins` | **`dv_bottom`** |
|---|---|---|---|---|---|---|---|---|---|
| mask hue (OpenCV) | 72/79/89 | 62/70/74 | 69/72/74 | 62/71/75 | 76/80/86 | 70/75/83 | 62/70/74 | 69/72/74 | 72/74/77 |
| p1 grey | 9/32/55 | 27/56/70 | 40/64/73 | 28/52/69 | 36/64/90 | 8/34/59 | 22/34/52 | 34/44/65 | 28/39/59 |
| p99 grey | 187/221/249 | 163/184/222 | 170/208/231 | 154/172/207 | 198/228/250 | 163/222/254 | 161/184/222 | 168/207/229 | 214/231/250 |
| white share | 1.4/7.3/17 % | 0.4/1.5/3.6 % | 0.2/0.8/2.6 % | 0.1/0.8/2.3 % | 2.3/4.4/8.1 % | 0.3/4.4/16 % | 0.3/1.5/3.6 % | 0.2/0.8/2.4 % | 3.3/5.3/8.5 % |
| Laplacian variance | 3.2/4.8/9.4 | 33/102/234 | 4.0/5.3/6.8 | 16/80/218 | 3.2/4.8/8.3 | 2.7/4.7/8.6 | 37/99/236 | 4.3/5.4/6.8 | 3.2/4.7/7.1 |

Sim-to-real gap (`tools/sim_to_real.py`, ImageNet ResNet18, 300 real good crops vs. the same 160 renders,
128 px; NN distance ratio, 1 = covered; the 1-NN accuracy is 0.99-1.0 for every variant):

| Real crops | `legacy` | `quins` | `phone` preset | **`dv_top`** | **`dv_bottom`** |
|---|---|---|---|---|---|
| TOP CAPC1005 (vs. run_0402_c renders) | 2.40 | 2.49 | 2.45 | **1.90** | - |
| BOTTOM RESC1005 (vs. run_0402_r renders) | 2.33 | 2.35 | - | - | **2.03** |
| BOTTOM CAPC1005 (vs. run_0402_c renders) | 2.13 | 2.16 | 2.22 | 1.86 | - |
| TOP RESC1005 (vs. run_0402_r renders) | - | 2.42 | - | - | 1.91 |

`tools/calibrate_camera.py` on the real crops upscaled to 256 px: noise 0.28 / 0.26 real vs. 0.26 / 0.23
synthetic (variance ratio 1.15 / 1.30, i.e. matched within the tool's resolution); its edge-steepness
measure asks for +2.5 px more blur on TOP and -1.6 px on BOTTOM while the Laplacian variance already
matches (4.8 vs 4.8, 4.7 vs 4.7); the two statistics disagree on these saturated, upscaled crops, so the
Laplacian variance of `look_stats` was used as the sharpness criterion.

Reading: both criteria of the task are met (NN distance ratio and Laplacian variance closer to the app
crops than `quins`), with three tuning rounds (round 1: 2.05 / 2.18, Laplacian 3.8 / 4.5, highlights too hot
(p99 238 / 244) and bottom fringes too strong (LAB b std 7.0 vs 4.9); round 2: 1.88 / 2.03 after dimming
the joint lights and halving `chroma_shift`; round 3: more resistor neighbours, final blur). The cross
table shows what carries the gain: `dv_top` is as close to the BOTTOM crops (1.86) as to the TOP crops, so
the 16 px/mm crop scale, the bright flat joints and the dense packing matter, the side-specific mask
colours and lights much less on this metric. Still open (montage `outputs/dv_look/dv_top_0402c.png`,
`dv_bottom_0402r.png`, `tools/look_montage.py --stats`): the real TOP crops have far more bright area
(21.7 % of pixels > 150 vs. 10.7 %, LAB L std 41 vs 29) because flash-lit fillets and glare merge into
large white blobs, and darker gaps (p1 32 vs 64); the BOTTOM crops match on contrast (L std 36.9 vs 35.9)
and dark end (p1 34 vs 39) but also show more bright area (15.8 % vs 9.3 %).

Configs: `configs/run_0402_c_dv_top.yaml` (0402 MLCC, TOP) and `configs/run_0402_r_dv_bottom.yaml` (0402
resistor, BOTTOM), both `post_filters: camera` with the calm domain of `run_0402_c_quins.yaml`. Looks and
presets validate like the others (`render.look`, `render.camera.preset`); a look's preset, gloss and
rework defaults yield to explicit `render` keys as for `quins`. Tests: `tests/test_look.py` (registration,
every footprint and class, determinism, presets, joint lights, materials, bottom mask colour),
`tests/test_dv_crops.py`, golden variants `look_dv_top` / `look_dv_bottom`.

## Export for DeepVision (`tools/export_deepvision.py`)

The DeepVision `aoi-judge` is a Siamese network: it scores the crop of a part against a good reference crop
of the same part. `tools/export_deepvision.py` therefore renders **pairs per seed** from a run config: a
reference (OK render), an OK probe (the same board under another exposure) and one probe per defect class
(the same board with the inspected part defective). It uses the 2D renderer's new `camera_rng` argument
(`render_roi(..., camera_rng=)`, default `None` = bit-identical to before): the sensor model (lens blur draw,
illumination gradient, noise, sharpening, JPEG quality) draws from that generator instead of the render's
main stream.

What is constant within a pair (code: the renderer draws the view pose, the board, the neighbours and the
joint light from the main stream *before* any defect-dependent draw, see `scene._render_photoreal_base`):
nominal footprint geometry, crop orientation, camera pose / field of view, the board (mask, copper,
silkscreen, residue, neighbour parts), the joint lighting (meniscus side stream) and, for the reference and
the defect probes, the sensor draws (shared `camera_rng`). What varies: the inspected part (placement /
absence / tilt), its joints' micro-relief, solder balls and bulges; the OK probe additionally gets other
sensor draws and brightness / contrast factors (another photo) and a fresh OK placement within tolerance.
`tests/test_export_deepvision.py` checks it: reference and MISSING probe differ by < 1 grey level outside
the part, the OK probe differs everywhere a little.

```bash
python tools/export_deepvision.py --config configs/run_0402_c_dv_top.yaml --out outputs/deepvision-export/run_0402_c_dv_top \
    --pairs 120 --max-size 49 [--faults-per-pair 2 | all] [--px-per-mm 16.1] [--side top] [--seed N] [--workers 6]
```

Output `<out>/manifest.json` (`run, side, profile, look, renderer_version, classes: {name: renders}, pairs,
created, config`, plus `form`, `board`, `class_weights`, `sizes`, `faultMapping`, `faults` (renders per fault),
`constant_per_pair`),
`records.ndjson` (one line per render) and `crops/<id>.png` (grey) / `crops/rgb/<id>.png` (colour):

```json
{"id": "<seed>-ref" | "<seed>-probe-<class>", "pairId": "<seed>", "role": "ref" | "probe",
 "board": "sim-<profile>-<look>-<run>", "side": "top" | "bottom", "label": "ok" | "fault",
 "fault": "fehlt" | "versetzt" | "verdreht" | "grabstein" | "loetbruecke" | "loetfehler" | "fremdkoerper" | null, "form": "CAPC1005",
 "size": 49, "fieldMm": 3.0576, "crop": "crops/<id>.png", "cropColor": "crops/rgb/<id>.png",
 "sim": {"class": "MISSING", "seed": ..., "profile": "chip_0402_capacitor@1", "look": "dv_top",
         "defect_params": {...}, "derived_class": "MISSING", "camera_seed": [seed, 1], "renderer_version": "2"}}
```

Mapping (spelling of the app's reviews; `manifest.json` carries it as `faultMapping`): MISSING -> `fehlt`,
TOMBSTONE -> `grabstein` (the reviewers' word for a tombstone, 5 TOP crops), SOLDER_BRIDGE and EXCESS_SOLDER ->
`loetbruecke`, SOLDER_BALL -> `fremdkoerper`, CORNER_LIFT -> `loetfehler` (a lifted corner is an open joint, a
solder fault in the app's vocabulary), MISALIGNED -> `verdreht` when the rotation dominates (|`rotation_deg`| >
the profile's `ok_rotation_deg` while the shift magnitude stays within `ok_shift_px`), else `versetzt`
(`export_deepvision.fault_of`; the sampled `defect_params` are in every record, so the rule can be re-applied).
`form` is the package code of the profile
(`component.package`, e.g. `CAPC1005`, `RESCAXS50P100X200-8`). The defect classes of a pair are drawn from
the config's `classes` counts as weights (`--faults-per-pair 2` default; `all` renders every class), so an
export has 1 OK probe per 2 defect probes.

Crops: the ROI (256 px at the profile scale, 3.0-3.1 body lengths through the `dv_*` presets) is resampled
(area) to `round(fieldMm x px/mm)` with `--px-per-mm 16.1`, `fieldMm` = the drawn field in mm (`crop_part_lengths`
x body length x `roi.mm_per_px`, so the run config's `mm_per_px` must be the profile's physical scale:
`run_0603_r_dv_bottom.yaml` sets 0.02667 for the `@2` profile, whose 55-65 px body is 1.6 mm). Sizes of the
first export vs. the real app crops: 0402 48-51 px (real 48-50), 0603 BOTTOM 72-85 px (real 76 / 38),
arrays 49 px with `--max-size 49` (real TOP 49 px at a 6.0 mm field, so `run_array_r_dv_top.yaml` crops
6.0-6.2 x the 1.0 mm `component_length` = 3 x the 2.0 mm long side), 1210 49 px with `--max-size 49` (the real
TOP 1210 crops are 30 px for a 9.2 mm field: the app does not keep 16 px/mm for large parts; without the
cap the export gives ~150 px). The judge upscales every crop to 64 px.

First export (2026-11-08, `outputs/deepvision-export/<run>/`, RTX 3080 host, 6 render threads, 9-20 s per run):

| Run | Side | Pairs | Renders (OK / fault) | Classes (renders) | Crop sizes |
|---|---|---|---|---|---|
| `run_0402_c_dv_top` | TOP | 120 | 480 (240 / 240) | SOLDER_BALL 45, EXCESS_SOLDER 56, MISSING 49, MISALIGNED 46, TOMBSTONE 44 | 48-49 px |
| `run_array_r_dv_top` | TOP | 120 | 480 (240 / 240) | MISSING 91, SOLDER_BRIDGE 94, MISALIGNED 30, EXCESS_SOLDER 25 | 49 px |
| `run_1210_c_dv_top` | TOP | 120 | 480 (240 / 240) | SOLDER_BRIDGE 81, MISALIGNED 81, TOMBSTONE 40, EXCESS_SOLDER 38 | 49 px |
| `run_0402_r_dv_bottom_bridges` | BOTTOM | 110 | 440 (220 / 220) | SOLDER_BRIDGE 77, EXCESS_SOLDER 65, MISALIGNED 28, MISSING 25, SOLDER_BALL 25 | 48-51 px |
| `run_0402_r_dv_bottom_missing` | BOTTOM | 110 | 440 (220 / 220) | MISSING 92, MISALIGNED 41, TOMBSTONE 39, SOLDER_BALL 26, EXCESS_SOLDER 22 | 48-51 px |
| `run_0603_r_dv_bottom` | BOTTOM | 110 | 440 (220 / 220) | SOLDER_BRIDGE 68, EXCESS_SOLDER 52, MISSING 35, MISALIGNED 30, SOLDER_BALL 20, TOMBSTONE 15 | 72-85 px |

360 pairs / 1440 renders for TOP, 330 pairs / 1320 renders for BOTTOM (77 MB). Sample pairs:
`outputs/dv_look/export_pairs_sample.png`. The exporter refuses Blender configs (2D only).

## Configs for the judge weaknesses (S4)

One run config per weak spot of the DeepVision judge, class counts weighted towards the problem, `dv_*` look,
`post_filters: camera`, calm domain. 16 example images each in `outputs/dv_look/s4_<run>.png`; render time
of a 256 px image at one worker (wall time of 16 images incl. ~0.5 s start-up): 0402 / 0603 ~30 ms, arrays
~25 ms, 1210 ~60 ms; the 1800-image A/B datasets above rendered in 8-9 s with 8 worker processes.

| Config | Side / look | Classes (count) | Note |
|---|---|---|---|
| `run_0402_r_dv_bottom_bridges.yaml` | BOTTOM `dv_bottom` | OK 300, SOLDER_BRIDGE 300, EXCESS_SOLDER 200, MISSING / MISALIGNED / SOLDER_BALL 60 | 0402 resistor; a chip bridge runs between the two terminals along the body edge |
| `run_0603_r_dv_bottom.yaml` | BOTTOM `dv_bottom` | OK 300, SOLDER_BRIDGE 250, EXCESS_SOLDER 150, MISSING 120, MISALIGNED 60, TOMBSTONE / SOLDER_BALL 40 | `chip_0603_resistor@2` (run_0001 profile) |
| `run_0402_r_dv_bottom_missing.yaml` | BOTTOM `dv_bottom` | OK 300, MISSING 400, MISALIGNED 80, TOMBSTONE 60, SOLDER_BALL / EXCESS_SOLDER 40 | `render.missing_residue: 1.5` (see below) |
| `run_1210_c_dv_top.yaml` | TOP `dv_top` | OK 600, MISALIGNED 150, SOLDER_BRIDGE 150, TOMBSTONE 50, EXCESS_SOLDER 50 | CAPC3225, brown-tan body; OK variants from the per-sample colour / gloss / placement jitter |
| `run_array_r_dv_top.yaml` | TOP `dv_top` | OK 400, MISSING 250, SOLDER_BRIDGE 250, MISALIGNED 60, EXCESS_SOLDER 40 | RESCAXS50P100X200-8, bridge between neighbouring 0.5 mm terminals |

`render.missing_residue` (new, 2D photoreal, default 1.0 = bit-identical to before) scales the reflowed-paste
domes on pads without a part (MISSING, the raised end of a tombstone, a part shifted off its pad): 0 leaves
the bare pad finish (the meniscus draws are still consumed, so the other joints do not change), 1 is the
look's `paste_scale`, up to 4 for heavy residue. It encodes the DeepVision hypothesis that real missing parts
leave solder on the pads instead of clean pads (hypothesis, not measured here: the 28 real BOTTOM "fehlt"
crops do show grey solder domes with bright rims on both pads, see `outputs/dv_look/dv_bottom_0402r.png`).
`generate --extend` treats it as a look option. Validated in `config._validate_missing_residue`; tests in
`tests/test_missing_residue.py`.

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

### Cutting backgrounds from the DeepVision stills (`tools/board_backgrounds.py`)

The DeepVision autopilot keeps a still per inspection run (`test-results/auto/**/<run>/still.jpg`, 1047 runs,
293 distinct photos, phone photos at 17-29 px/mm (median 20 / 22 px/mm TOP / BOTTOM) plus 33 scans) with
`geometry.json` (`boardToSource`: board mm -> still px homography, `side`) and `components.json` (every part
with `centerMm`, `fieldMm`; body polygons on 30 % of the runs). `tools/board_backgrounds.py` slides a window
over the populated board area, drops windows that touch a part (square keep-out from `fieldMm / 3.04`, the
body polygon when present, plus 0.4 mm), rectifies the rest through the homography to the profile scale
(`--mm-per-px`, default 0.01613 = 62 px/mm of the 0402 configs, 5.2 mm window -> 322 px) and keeps only
windows that look like bare mask: >= 90 % green pixels, <= 0.5 % bright (grey > 200), <= 2 % grey/white print,
<= 1 % very dark pixels. One run per photo and side (the one with the most parts), at most 12 windows per
photo, `--dry-run` prints the summary only:

```bash
python tools/board_backgrounds.py --runs .../DeepVision-Web/test-results/auto --out data/board_textures [--dry-run]
```

Result (2026-11-08; `data/board_textures/<side>/summary.json`, the folder is git-ignored):

| Side | photos used | candidate windows | accepted textures | rejected: bright / print / dark / colour |
|---|---|---|---|---|
| TOP | 106 | 888 | 326 | 371 / 74 / 39 / 72 |
| BOTTOM | 65 | 17 717 (sparse side) | 780 | 82 / 23 / 78 / 35 |

Known gap: `geometry.json` has the pose, not the board outline, drill or test-point layout, and the pose
of a run can be off by a few mm (`refine.error: implausible:shift` on some runs); the bounding-box margin
(4 mm inside the part centres) and the colour heuristic stand in for both. The sample montage
`outputs/dv_look/board_textures_sample.png` shows mostly bare mask with vias; a few cuts still carry the
edge of a label or a light-green board region.

**A/B with and without real backgrounds** (`configs/run_0402_c_dv_top_bg.yaml`, `run_0402_r_dv_bottom_bg.yaml`
= the dv configs plus `texture_dir`; 1800 images per dataset (OK 400, MISSING 400, SOLDER_BRIDGE 300,
MISALIGNED / TOMBSTONE 200, SOLDER_BALL / EXCESS_SOLDER 150), ResNet18, 8 epochs, flips / quarter turns /
colour jitter, RTX 3080; scored with `tools/eval_real.py` on the exported app crops of the whole side
(`outputs/dv_real/<side>`: 500 good + all faults) and with `tools/sim_to_real.py` (300 OK renders vs. 300
real good 0402 crops)):

| Training data | real good -> OK | "fehlt" detected (as any defect) | false calls | escapes | NN distance ratio (OK renders) |
|---|---|---|---|---|---|
| `run_0402_c_dv_top` (procedural board) | 0.4 % | 100 % | 99.6 % | 0 % | 1.79 |
| `run_0402_c_dv_top_bg` (real backgrounds) | 0.4 % | 100 % | 99.6 % | 0.4 % | 1.85 |
| `run_0402_r_dv_bottom` (procedural board) | 1.2 % | 100 % | 98.8 % | 0.4 % | 1.96 |
| `run_0402_r_dv_bottom_bg` (real backgrounds) | 0.2 % | 100 % | 99.8 % | 0 % | 2.00 |

Reading: the real backgrounds change nothing measurable. The synthetic-only classifiers reject almost every
real crop (mostly as TOMBSTONE, then SOLDER_BALL / MISALIGNED), exactly as the earlier QUINS experiment found
("Synthetic images alone do not transfer"), and the embedding distance is even slightly larger with the
textures (the 20 px/mm stills are upscaled 3x to the profile scale, so their texture is softer than the
procedural mask before the camera model resamples it). The textures remain available for the mixed
training recipe (synthetic defects + real good parts) and for the judge export, where the board around the
part is not what the model compares.

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
    max_oversampling: 0        # > 0 (2D): cap on supersampling x zoom; zoomed-out views render at a lower
                               # supersampling (never below 1, never raised), e.g. 2 halves the time of a
                               # QFN32 under `crop_part_lengths` (zoom 2: 254 -> 125 ms). 0: off (default)
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

## A/B of the opt-in render options against the real 0402 crops (2026-11-07)

`tools/sim_to_real.py` (ImageNet ResNet18 embeddings, 300 synthetic OK images of
`configs/run_0402_c_quins.yaml` per variant vs. 300 of the 500 real good form-02 test crops, 128 px): the
NN distance ratio says how far the real crops are from their nearest render relative to their nearest
other real crop (1 = covered); the 1-NN accuracy stays at 1.0 for every variant, i.e. real and synthetic
are still perfectly separable by a generic embedding, so only the ratio is informative.

| Variant (relative to `run_0402_c_quins`) | NN distance ratio |
|---|---|
| `look: legacy`, `post_filters: legacy` (the pre-quins render) | 3.61 |
| `look: quins` (the config: `post_filters: camera`, implied meniscus joints) | **2.49** |
| `post_filters: legacy` instead of `camera` | 2.55 |
| `solder: legacy` (quins materials, legacy fillets) | 2.92 |
| `board.routing: escape` | 2.58 |
| `board.routing: escape`, `pour_probability: 1.0` | 2.58 |
| `defects.placement_sampling: continuous` | 2.49 |
| `supersampling: 3` | 2.49 |

Reading: the look (materials, scanner camera) and the meniscus joints carry the whole gain; the camera
post-filter mode helps a little; the board-model copper and the continuous placement sampler do not
change the distance on this metric (the routed traces are hardly visible under the quins mask), and a
third supersampling level is wasted. The remaining gap is not resolved by a generic embedding; the
classifier test above is the better guide. Defaults are unchanged by this measurement.
