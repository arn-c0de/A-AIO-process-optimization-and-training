# Changelog

All notable changes to **Simple-Sim** will be documented in this file.

## [Unreleased]

### Added
- Footprint-aware pad rendering in `generator_2d.py`. The renderer now dispatches pad geometry based on the `footprint` field from the component profile:
  - `chip_2pad` (default): two symmetric pads (left/right), backward-compatible with existing 0603 datasets.
  - `sot23`: three pads in a T-arrangement (two smaller pads on the left, one full-size pad on the right).
  - `qfn` (matches any `qfn*` footprint): four pads arranged around the perimeter (left, right, top, bottom) with rotated orientation for top/bottom pads.
- New `pad_spacing_y` geometry parameter for multi-pad footprints. Sampled from `geometry_ranges` when present; absent profiles default to 2-pad behavior without any config changes.
- New `sot23_transistor@1` component profile for a 3-pad SOT-23 transistor.
- New `qfn32_ic@1` profile for a 32-pin QFN IC with 4-pad rendering and corrected geometry ranges.
- Workflow configs `configs/run_sot23.yaml` and `configs/run_qfn32.yaml` for the new profiles.
- "Add new model" button ("+") in the Pipeline tab next to the model dropdown. Allows creating a new model entry by name when no compatible model exists yet (e.g. after switching to a new profile).
- Bundle support in the Weights tab. `.bundle` directories now appear in the checkpoint list with a `[Bundle]` marker and aggregated file size. Bundles can be grouped, favorited, and managed like regular checkpoints.
- Extended bundle metadata (`bundle_details.json`). When merging, per-model details are recorded: source path, accuracy, F1, profile hash, size, and modification date. The "Bundle Info" dialog displays all details.
- Group categories in the Merge tab. The model tree now uses the same group structure as the Weights tab (Favorites, Snapshots, Imports, custom groups, Uncategorized), loaded from the same `favorites.json` and settings store. Expanded/collapsed state is preserved across refreshes.

### Changed
- QFN-32 profile: removed `SOLDER_BRIDGE` and `CORNER_LIFT` defect classes (not distinguishable from `MISALIGNED` in 2D rendering). The profile now uses the standard defect set (OK, MISSING, MISALIGNED, TOMBSTONE).
- QFN-32 profile: corrected `pad_spacing` / `pad_spacing_y` from 300-340 to 520-560 so pads extend beyond the component body and are visible in rendered images.
- `MetaRow` schema validation: nominal geometry now uses superset check instead of exact set match. The five base keys are required, but additional keys (e.g. `pad_spacing_y`) are accepted.
- Weights tab: tree state (expanded groups, selection) is preserved across refresh operations (delete, favorite toggle, etc.).
- Weights tab: `_is_deletable_checkpoint` now allows deleting any file under `outputs/models/`, not just those in `versions/` or `imports/` subdirectories.

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
