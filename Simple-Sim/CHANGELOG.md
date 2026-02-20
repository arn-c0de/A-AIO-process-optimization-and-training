# Changelog

All notable changes to **Simple-Sim** are documented in this file.

## [1.0.2] - 2026-02-20

### Summary
- Added a modular profile editor with live 3D/HQ preview and persistence.
- Added shared filter profiles, per-filter randomization, and 90-degree orientation controls.
- Refactored core modules and GUI tabs for better maintainability.
- Improved pipeline stability and responsiveness during training.
- Fixed path persistence, preview reload, filter popup persistence, and key 3D geometry issues.

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
