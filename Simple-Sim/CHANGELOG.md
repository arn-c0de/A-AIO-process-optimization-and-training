# Changelog

All notable changes to **Simple-Sim** will be documented in this file.

## [Unreleased]

### Added
- New `sot23_transistor@1` component profile for a 3-pad SOT-23 transistor (configs/profiles/sot23_transistor@1.yaml).
- Workflow config `configs/run_sot23.yaml` targeting the new profile (800 samples, ResNet-18 training) to seed datasets/trainings for SOT-23 inspections.
- New `qfn32_ic@1` profile for a 32-pin QFN (configs/profiles/qfn32_ic@1.yaml) plus `configs/run_qfn32.yaml` to generate larger ROI datasets with solder-bridge/corner-lift defects.

## [1.0.1] - 2026-02-13

### Added
- Profile-based multi-component architecture using versioned YAML component profiles in `configs/profiles/`.
- Deterministic profile hashing (SHA256) to detect semantic profile changes and prevent silent data corruption.
- Dataset manifest system (`dataset_manifest.json`) with provenance (profile id/hash/path, generator version, git commit) and extend-history tracking.
- Migration tool to backfill manifests for legacy datasets: `tools/backfill_manifest.py`.

### Changed
- Pipeline guards in CLI scripts to enforce profile compatibility for generate/extend, train/resume, eval and predict.
- Training checkpoints now record component profile metadata and dataset manifest hash.
- Config schema v2 introducing `run.component_profile` (with backward compatibility for v1 configs).
- GUI integration: profile selection dropdown, dataset/model profile display, profile compatibility indicator, and profile info dialogs.

### Notes
- Full implementation report: `PROFILE_SYSTEM_IMPLEMENTATION.md`.
