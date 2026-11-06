# Changelog

All notable changes to **Simple-Sim** are documented in this file.



## [Unreleased]

### Added
- `render.solder: legacy | meniscus` (default `legacy`, bit-identical, both backends): new solder-joint model `simple_sim/generators/opencv2d/solder.py` calibrated against the public SolDef_AI macro crops. Concave meniscus with a rounded, wavy contact line (sometimes bare finish visible), frosted/wrinkled surface, and environment-mirror shading (overhead area, ring light / lamps, dark reflections of mask and body), so joints get the high contrast of real ones instead of a matte grey slab. Flat matte end caps, and solder balls and bulges use the same shading. Meniscus randomness comes from a side stream of the sample generator, so everything except the joints stays pixel-identical to `legacy` for the same seed. Tests: `tests/test_solder.py`. See `docs/guides/PHOTOREAL_RENDERING.md`.
- Renderer version (`simple_sim.generators.version.RENDERER_VERSION`, now `2`) in `dataset_manifest.json` (`generator.renderer_version`, also per `extend_history` entry). `generate --extend` refuses to append to a dataset from another renderer version (datasets from v1.0.3 and older have none) unless `--allow-renderer-mismatch` is given, so a dataset never silently mixes two looks. A golden-hash test (`tests/test_render_golden.py`) catches unintended changes of the 2D output.
- Board model for the `photoreal` copper (`render.board`, new package `simple_sim/board/`, both backends): footprints with pin numbers, roles and fan-out directions, a netlist per part family, a fan-out router with occupancy checks (stubs out of the pad edge, 45° routes to the window border or a via, via grid in thermal pads, through-nets and buses), ground pours with clearance, thermal reliefs and stitching vias, design-rule presets (`consumer_dense`, `industrial`, `hdi`) and a DRC (`drc: off | warn | strict`). `routing: legacy` (default) means "no board model" (the random copper generator), it is *not* pixel-identical to v1.0.3: the realism changes of this release (copper, vias, palettes, rework, camera) change every image of a seed; `routing: escape` enables the model; `none` renders no copper. The design rules scale with the drawn pad size (`photoreal.board_rule_scale`), not with `roi.mm_per_px`. `roi.mm_per_px` is passed to the renderer (`render.mm_per_px`) by every caller and always wins over a stale `render.mm_per_px` (also for Blender jobs). See `docs/guides/PHOTOREAL_RENDERING.md` and `docs/refactoring/PCB_TRACE_ROUTING_PLAN.md`.
- Pad geometry (`draw._compute_pad_positions`), neighbour placement and the part-family lookup moved to `simple_sim.board` (`footprints.footprint_pads`, `placement.neighbour_layout`, `footprints.family_of`); the drawing code calls them, results are unchanged (regression-tested). Via drawing is one helper for both copper generators. The Blender bake job fields come from one `bake.bake_fields()` instead of three hand-written dicts.

### Changed (photoreal realism, compared against real DeepVision crops)
- Solder-mask palette re-measured on real part crops: olive/bottle green with HSV saturation ~100 instead of a saturated lime/cyan (~150). Default augmentation hue jitter ±15° -> ±5°, saturation 0.8-1.2 -> 0.85-1.1.
- Solder joints, solder balls and bulges shaded brighter (frosted lead-free solder, light environment), fuller fillets; MLCC body colour less orange.
- New `render.mask_gloss` (default 0.5): glossy mask glare with orange-peel texture.
- Traces and vias scale with the part (were hairlines/pin pricks), more vias with a tented-dome highlight.
- Neighbour parts are mostly placed as aligned rows/columns of identical parts on a placement grid instead of a random scatter.
- Camera model: `capture_scale` (low-resolution capture upscaled to the ROI, like real small crops) and `chroma_shift` (lateral chromatic aberration); phone preset uses both, zoom 1-2.5 -> 1-1.5 (part stays at the real crop's share of the image). `run_0001_phone.yaml` drops the extra full-resolution grain/blur.
- `render.rework_artifacts` default 0.3 -> 0.15.
- SOT-23 rendered with the body's long axis along the pin-1/2 column (was a bar lying across the pads), proper gull-wing leads (shoulder + flat foot, solder fillet around the foot) instead of round-capped lines, 2-3 character marking. Leads of SOT/SOIC are rigid with the body (2D and Blender): a misplaced part's feet miss their pads instead of stretching to them.
- New profile `sot23_transistor@2` with real SOT-23 land/body proportions (`run_sot23.yaml` uses it); `@1` unchanged for existing datasets/models.
- IC bodies: no thick bevel frame any more (QFN sawn edges ~vertical, small draft for SOT/SOIC).

### Added (component forms missing compared with the DeepVision xpa / motorboard PnP data)
- Footprints `chip_array_8` (4-element R/C array, 4 pads per side, 0.5 mm pitch) and `sot_6` (SOT-363 / SOT-23-6, 3 gull-wing leads per side); `soic_16` is now one case of the generic two-column layout (`DUAL_ROW_PADS_PER_SIDE`, pixel-identical).
- Families `array_r` (black body, C-shaped terminal wraps, 1-3 character code), `array_c` (light grey-beige ceramic, inner-electrode stripes) and `sot6` (moulded body, short code along the long side).
- Profiles with real proportions, scaled so the footprint fills ~1/3 of the ROI like the real per-part crops: `chip_array_8_resistor@1`, `chip_array_8_capacitor@1`, `sot363_ic@1`, `sot23_6_ic@1`, `chip_0402_resistor@1` (unmarked), `chip_0402_capacitor@1`, `chip_1210_capacitor@1` (brown body); run configs `run_array_r/_c`, `run_sot363`, `run_sot23_6`, `run_0402_r/_c`, `run_1210_c`.
- `render.marking_probability` (profile `render` section): share of parts with a printed/laser code. Light `component_color_bgr` on ceramic parts sets the body colour.
- IC markings run along the long side of the package (SOIC, SOT-23-6).
- Chip families `inductor` (ferrite bead / chip inductor: dark grey ferrite, large caps, unmarked), `power_inductor` (moulded 2520: matte grey block with an embossed coil ring, terminals underneath) and `led` (white housing, yellowish lens with die, bond wire and cathode mark), chosen by `component.type`; footprints `tssop_14` / `tssop_20` (0.65 mm pitch, family `soic`). Profiles `chip_0603_ferrite@1`, `chip_2520_power_inductor@1`, `chip_0603_led@1`, `tssop20_ic@1`, `tssop14_ic@1`; run configs `run_0603_ferrite`, `run_2520_inductor`, `run_0603_led`, `run_tssop20`, `run_tssop14`.
- Board netlist knows every family (`routing: escape` no longer KeyErrors on families without net rules).
- Existing datasets render differently with the same seed; extending an old dataset mixes both looks.

### Performance
- 2D `photoreal` render ~2x faster with pixel-identical output: alpha blends only touch the bounding box of their mask instead of the whole supersampled board.
- Supersampling factor is thread-local instead of a module global behind a lock, so 2D renders run in parallel threads (new `simple_sim.parallel.imap_ordered`, `$SIMPLE_SIM_RENDER_WORKERS`, default `min(8, CPUs)`). Results are identical to a sequential run (tested).
- `generate` / `generate-profile-dataset` (2D) render and write images in parallel and stream them to disk instead of holding the whole dataset in memory. Profile Preview tab renders its 2D images in parallel (Ultra: ~3 -> ~16 images/s on 8 threads).
- Blender backend: board textures are baked in parallel threads before the Blender run, and the camera model / post filters on the renders run in parallel too (3D dataset generation no longer bakes every board sequentially). Profile Preview (3D) bakes the next boards in the background while Blender renders the current image, so only the first bake is on the critical path (new `bake.prepare_job`, `runner.render_prepared_batch`, `generation.blender.run_prepared_blender_jobs`).
- Profile Preview gallery caches decoded images; resizing the window only re-grids the thumbnails instead of re-reading every PNG.

### Fixed
- Filter popup previews: 3D preview was rotated twice (Blender rotates the scene, the popup rotated again) and skipped the board bake/camera settings; 2D preview ignored the run config (camera, textures, supersampling) and used a ROI table with wrong footprint names. Both now use the run config matching the profile and the dataset render path; the four duplicated render functions are one module (`gui/components/filter_popup/preview_render.py`). Results reach Tk via polling instead of Tk calls from the worker thread.
- Randomized image filters in previews (Profile Preview tab, debug previews) are seeded from the preview seed, so the same seed gives the same image.
- 3D previews (`preview/render.py`, profile editor HQ preview) bake the board with the run config's `render` settings (`bake_fields`: board model, camera, gloss, rework, supersampling) like the dataset, instead of the defaults.
- `bake_fields` no longer carries `mm_per_px`: the Blender job's `roi.mm_per_px` is the one scale, a stale `render.mm_per_px` (e.g. from a dataset's `config.yaml`) can no longer override it.
- `render.mask_gloss`, `render.rework_artifacts` and `render.texture_dir` are validated for `blender_3d` too (the bake uses them); `style: flat` warns about the photoreal-only settings it ignores.
- Board model: the inspected part's keep-out follows its body orientation (SOT-23 body is vertical), design rules scale with the pad size (QFN at 0.03 mm/px no longer gets hairline tracks), `drc: strict` raises after the last attempt.
- Optional effects that are off no longer consume random draws (`mask_gloss: 0`, `rework_artifacts: 0`, fixed camera `capture_scale` / `chroma_shift` as in the `aoi` preset).
- Vias are drawn on a local window (same pixels, no full-board masks and blur per via).
- `run_sot23*.yaml` run ids bumped to `run_sot23_v2` / `run_sot23_v2_3d` (new profile versions).
- Board router / DRC (`routing: escape`): clearance is measured with exact geometry (`board/geometry.py`) instead of a coarse raster (false conflicts up to ~2x clearance). GND pads of the inspected part inside a pour are connected by relief spokes (were isolated), a failed fan-out is undone exactly (no holes in keep-outs, no phantom vias), stubs to the window border keep full clearance against foreign copper, tracks no longer run along the window border, thermal vias stay on the pad, design rules shrink to fit fine pin pitches (`fit_to_pitch`), neighbours at ~90° jitter are modelled with the right orientation, unconnected pour islands are removed. New DRC check `unconnected`. 1260 fuzzed boards (all footprints, 3 presets, 60 seeds) are DRC-clean.
- GUI task runner: the dispatcher keeps polling while a callback shows a modal dialog; a task that finished normally is not reported as cancelled; progress updates are coalesced; closing the app cancels tasks and waits up to 3 s (Analysis, Validation and Weights compare stop cooperatively); Profile Preview Stop / close terminates a running Blender process (`should_stop` in `run_blender_jobs` / `run_prepared_blender_jobs`). Dialogs: Return no longer triggers the destructive button of danger confirmations, "Show Details" grows the error dialog, single-choice list picker cannot return a hidden earlier choice. Log views have a working per-view Auto-Scroll toggle; striped tables insert in O(1); Profile Preview mouse wheel scrolls only the gallery and defaults to quality "Standard".

## [1.0.3] - 2026-11-06

### Summary
- New GUI tab **Profile Preview**: renders a few images per defect state of one profile with chosen settings (run config, quality preset Draft/Standard/Ultra, supersampling, style, camera preset, Cycles samples/engine/device, seed, variants, optional GUI image filters) without generating a dataset; click a thumbnail for full size. Output in `outputs/gui/profile_preview/`.
- `photoreal` supersampling configurable via `render.supersampling` (1-4, default 2 = unchanged images); applies to the 2D render and the board baked for Blender.
- Blender backend now really renders on the GPU: `render.blender.device: GPU` enables a Cycles compute backend (OptiX, CUDA, HIP, oneAPI) in the add-on preferences; before, Blender silently fell back to the CPU. Falls back to CPU if no GPU is found; the chosen device is logged. All `*_3d` run configs default to `device: GPU`.
- New default 2D render style `photoreal` (`render.style`), calibrated on photos of real boards: textured solder mask with traces and vias, silkscreen outlines and reference designators, ENIG/HASL pads with height-map shaded solder fillets, resistor value codes and tan capacitors, speckled IC mold bodies with pin-1 marks, laser markings and gull-wing leads, contact/cast shadows. `style: flat` keeps the previous look. See `docs/guides/PHOTOREAL_RENDERING.md`.
- `photoreal` no longer looks like a vector drawing: a camera model (lens blur, uneven illumination, highlight roll-off, sensor noise) is applied after supersampling, solder is shaded as a lumpy metallic mirror instead of milky plastic, silkscreen ink is blotchy.
- `photoreal`: optional `render.texture_dir` uses real photos of empty board areas as substrate; large thermal pads without a part show window-pane stencil islands; IC laser marking is grainy and uses varied fonts/line offsets.
- Blender backend rebuilt around the 2D look: the photoreal board (mask, traces, vias, silkscreen, pad finish) is baked per job and used as texture, Blender gets the 2D pad layout (no own pad logic, QFN/SOT special cases removed), solder fillets are height-field meshes, parts get end caps, gull-wing leads, pin-1 marks and markings, tombstones pivot about their anchored end, lighting is a calibrated AOI ring light, the camera is orthographic top-down framed to the ROI with sRGB output, and the 2D camera model is applied to every render. 3D profiles now share the 2D geometry semantics (SOT-23/QFN ranges from the 2D profiles, SOIC-16 at real scale), drop the unused substrate/copper materials and key/fill lights, and use a top-down camera. Because geometry semantics and look changed, the 3D profiles are now `*_3d@2`; datasets of the `@1` profiles cannot be extended (profile hash check) and should be regenerated.
- Default image filters no longer enable motion blur (11 px on every image) and the random dark shadow blob; neither occurs on a static AOI camera. Both stay available in the filter settings.
- `photoreal` 2D: silkscreen is placed around the nominal footprint instead of the shifted body (removes a label leak for MISALIGNED).
- Tombstones rendered as standing parts in both backends: from above the steep top face (length*cos(tilt), starting behind the anchor) and the upright end face (height*sin(tilt)) are visible; on chips that end face is the bright tin end cap, shaded as metal. 2D shadow is the projection of the standing part along the light. Blender: satin end caps and a diffuse top light next to the ring light, so flat metal (solder tops, upright end caps) is bright as on AOI images; board colours recalibrated.
- Tombstone tilt is sampled from `tolerances.tombstone_tilt_deg` up to fully upright (90 deg) instead of only 10 deg above the threshold; both renderers handle any tilt. `chip_0603_resistor_3d@2` uses a 30 deg threshold (partly lifted ends count as tombstones). Blender end caps no longer share the body's end face (upright caps rendered black).
- New default profile `chip_0603_resistor@2`: TOMBSTONE from 30 deg tilt (partly lifted ends) instead of 75 deg; run configs and GUI default to it (`simple_sim.config.DEFAULT_COMPONENT_PROFILE`). `chip_0603_resistor@1` stays for existing datasets and models.
- Camera model configurable via `render.camera` (lens blur, read/shot noise; 2D and 3D) and new `tools/calibrate_camera.py`, which fits these values to real camera ROIs (self-tested: recovers a known blur/noise setting). Solder-mask palette extended with the brighter greens of the production boards.
- `photoreal` calibrated on photos of the production boards: measured teal-green mask palette and copper-under-mask ratio (replacing estimates), copper pours, more and mostly tented vias, neighbour chip parts around the inspected part (2D and baked into the 3D board), silkscreen only in 55 % of samples.
- Residue as a label-neutral nuisance (`render.rework_artifacts`, default 0.3): flux residue, smeared yellow contamination and patch wires as seen on the production boards, never inside the inspected part's area (tested).
- New defect classes `SOLDER_BALL` and `EXCESS_SOLDER` (2D photoreal and flat, 3D): one layout function decides ball positions and affected pads for both backends. Added to the default run configs (`run_0001`, `run_0500`, all `*_3d`) and to the unreleased `@2` profiles; released `@1` profiles are unchanged.
- Phone photos: `render.camera.preset: phone` (oblique 5-40 deg, zoomed out up to 2.5x with proportionally more neighbour parts, softer with depth-of-field falloff, more noise, ISP sharpening, JPEG); all values overridable (`tilt_deg`, `zoom`, `defocus`, `sharpen`, `jpeg_quality`, ...) and validated. 2D projects a larger board area, Blender tilts its camera. New `configs/run_0001_phone.yaml`. Parts are drawn in local windows (worst-case phone image 98 s -> 1.5 s).
- New `tools/sim_to_real.py`: measures the gap between synthetic and real ROI folders (1-NN two-sample accuracy and NN distance ratio on ResNet18 embeddings).
- QFN footprints now render real pad rows (`qfn_<N>` -> N/4 pads per side) plus the exposed thermal pad.
- Code review fixes across core, scripts, tools and GUI (sharpen filter saturation, rotation-strength scaling of cardinal orientations, ensemble checkpoints in `eval.py`, bundle report location, merged-dataset validation, v2 manifest extend history, Blender light target, GUI settings/dataset-selection bugs, split-file creation on move, and removal of dead or duplicated code).

## [1.0.2] - 2026-02-20

### Summary
- Added a modular profile editor with live 3D/HQ preview and persistence.
- Added shared filter profiles, per-filter randomization, and 90-degree orientation controls.
- Refactored core modules and GUI tabs for better maintainability.
- Improved pipeline stability and responsiveness during training.
- Fixed path persistence, preview reload, filter popup persistence, and key 3D geometry issues.
- Added live filter preview panel to the Image Filters popup: profile dropdown (categorized 2D/3D), one-click render, and inline image display using current filter settings.
- Added dataset customization in GUI: delete single images and move images between datasets from Analyze and Datasets views.
- Added new Datasets tab with dataset catalog management: categories, category rename/delete, archive/unarchive, and filtering.
- Updated Pipeline and Analysis dataset selectors to show categories and hide archived datasets.
- Updated multi-dataset selection popup/tooling to use categorized labels and ignore archived datasets.
- Refactored generators into structured backend packages:
  - `simple_sim.generators.opencv2d` split into `augment`, `geometry`, `draw`, `filters`, and `render`.
  - `simple_sim.generators.blender3d` split into `io` and `runner`.
- Kept backward compatibility with wrapper/facade modules:
  - `simple_sim.generator_2d`, `simple_sim.generator_3d`
  - `simple_sim.generators.opencv_2d`, `simple_sim.generators.blender_3d`
- Centralized filter defaults/normalization in `simple_sim.generators.filter_settings` and reused them in GUI filter popup.
- Replaced hardcoded monitor tab wiring with `TabRegistry` and centralized tab order/metadata.
- Reduced pipeline tab duplication by extracting persisted field mapping and dataset-selection serialization helpers.
- Added typed config parsing baseline in `simple_sim.config_schema` and integrated it into config validation.
- Added regression tests for registry, filter settings, config schema, and wrapper compatibility.
- Added switchable filter modes in the Image Filters popup: `Custom` and `Realism`.
- Introduced grouped realism augmentation with sparse group mixing (`K=0/1/2`), correlated effects, and compatibility constraints.
- Added optional YAML-driven realism profile presets via `configs/realism_profiles.yaml`, with runtime override support through `SIMPLE_SIM_REALISM_PRESETS_PATH`.
- Extended filter normalization and payload handling with realism keys (`filter_mode`, `realism_*`) while preserving backward compatibility.
- Refactored filter popup into a maintainable package:
  - `gui/components/filter_popup/popup.py`
  - `gui/components/filter_popup/ui_custom.py`
  - `gui/components/filter_popup/ui_realism.py`
  - `gui/components/filter_popup/preview_panel.py`
  - `gui/components/filter_popup/controller.py`
  - `gui/components/filter_popup/models.py`
  - `gui/components/filter_popup/constants.py`
- Kept all existing imports and call sites working via backward-compatible package exports (`gui.components.filter_popup`).
- Added shared persistence helpers for filter profile extra keys in `gui/utils/filter_profile_store.py` and reused them across Pipeline tab and debug preview tooling.
- Added/updated tests for filter settings and realism behavior (constraints, sparse mode behavior, YAML preset usage).

### Extension Opportunities
- Profile diff/compare tooling.
- Reusable filter preset packs per profile.
- GPU/CPU auto-tuning presets for training and rendering.

## [1.0.1] - 2026-02-15

### Summary
- Introduced versioned component profiles, deterministic profile hashing, and dataset manifests.
- Added profile-safe pipeline guards for generate, train, eval, and predict workflows.
- Added multi-dataset training and full model-bundle lifecycle support.
- Introduced initial Blender-based 3D synthesis and footprint-aware generation.
- Improved GUI profile handling and fixed class/profile sync and geometry-related issues.
