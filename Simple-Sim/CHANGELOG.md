# Changelog

All notable changes to **Simple-Sim** are documented in this file.

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
