# Changelog

All notable changes to **Simple-Sim** are documented in this file.



## [Unreleased]

### Summary
- New default 2D render style `photoreal` (`render.style`), calibrated on photos of real boards: textured solder mask with traces and vias, silkscreen outlines and reference designators, ENIG/HASL pads with height-map shaded solder fillets, resistor value codes and tan capacitors, speckled IC mold bodies with pin-1 marks, laser markings and gull-wing leads, contact/cast shadows. `style: flat` keeps the previous look. See `docs/guides/PHOTOREAL_RENDERING.md`.
- `photoreal` no longer looks like a vector drawing: a camera model (lens blur, uneven illumination, highlight roll-off, sensor noise) is applied after supersampling, solder is shaded as a lumpy metallic mirror instead of milky plastic, silkscreen ink is blotchy.
- `photoreal`: optional `render.texture_dir` uses real photos of empty board areas as substrate; large thermal pads without a part show window-pane stencil islands; IC laser marking is grainy and uses varied fonts/line offsets.
- Blender backend rebuilt around the 2D look: the photoreal board (mask, traces, vias, silkscreen, pad finish) is baked per job and used as texture, Blender gets the 2D pad layout (no own pad logic, QFN/SOT special cases removed), solder fillets are height-field meshes, parts get end caps, gull-wing leads, pin-1 marks and markings, tombstones pivot about their anchored end, lighting is a calibrated AOI ring light, the camera is orthographic top-down framed to the ROI with sRGB output, and the 2D camera model is applied to every render. 3D profiles now share the 2D geometry semantics (SOT-23/QFN ranges from the 2D profiles, SOIC-16 at real scale), drop the unused substrate/copper materials and key/fill lights, and use a top-down camera. Because geometry semantics and look changed, the 3D profiles are now `*_3d@2`; datasets of the `@1` profiles cannot be extended (profile hash check) and should be regenerated.
- Default image filters no longer enable motion blur (11 px on every image) and the random dark shadow blob; neither occurs on a static AOI camera. Both stay available in the filter settings.
- `photoreal` 2D: silkscreen is placed around the nominal footprint instead of the shifted body (removes a label leak for MISALIGNED).
- Tombstones rendered as standing parts in both backends: from above the steep top face (length*cos(tilt), starting behind the anchor) and the upright end face (height*sin(tilt)) are visible; on chips that end face is the bright tin end cap, shaded as metal. 2D shadow is the projection of the standing part along the light. Blender: satin end caps and a diffuse top light next to the ring light, so flat metal (solder tops, upright end caps) is bright as on AOI images; board colours recalibrated.
- Tombstone tilt is sampled from `tolerances.tombstone_tilt_deg` up to fully upright (90 deg) instead of only 10 deg above the threshold; both renderers handle any tilt. `chip_0603_resistor_3d@2` uses a 30 deg threshold (partly lifted ends count as tombstones). Blender end caps no longer share the body's end face (upright caps rendered black).
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
