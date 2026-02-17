# Profile Editor

Interactive editor for Simple-Sim profiles and run configurations with a 3-column layout:

- Left: preview (`Fast 3D` + `Blender HQ`)
- Center: field editor
- Right: direct YAML editor

The editor uses a single source of truth state, so all panels stay synchronized.

## Start

```bash
./tools/run_profile_editor.sh
```

Or:

```bash
.venv/bin/python tools/profile_editor/app.py --sim-root .
```

## Requirements

- Python + Tkinter (GUI)
- `PyYAML`
- For `Blender HQ`: Blender available via `run.render.blender.executable`
- For HQ image display: `Pillow`

## Supported Files

- Profiles: `configs/profiles/*.yaml`
- Runs: `configs/run_*.yaml`

## Complete Feature Overview

### 1) File Discovery and Auto-Refresh

- Automatically detects profile and run files in config folders.
- Automatically refreshes lists when files are:
  - created
  - deleted
  - modified
- Polling interval: about 1.4s.
- Manual refresh via `Refresh Files`.

### 2) Top Bar

- `Profile` dropdown: load selected profile.
- `Run` dropdown: load selected run configuration.
- Run list is filtered by `run.component_profile` for the current profile.
- `Save Profile`: save profile YAML.
- `Copy Profile`: copy current profile, append a suffix to `profile_id`, save as a new file.
- `Save Run`: save run YAML.
- `Render HQ Now`: trigger immediate HQ Blender render.
- `Refresh Files`: force registry reload.
- Top-right live usage display: `CPU | GPU | RAM`.
- `Night Mode`: global day/night theme toggle for the full editor.

### 3) Left Panel: Preview

#### `Fast 3D` Tab

- Instant live preview based on current profile values.
- Uses simplified geometry reconstruction (substrate, pads, component body).
- Supported footprints:
  - `chip_2pad`
  - `sot23`
  - `qfn_32` (and qfn-like variants)
- Mouse controls:
  - LMB drag: orbit (full 360°)
  - RMB drag: pan
  - Mouse wheel: zoom
- Vertical zoom slider on the left edge.
- `Scene Components` dropdown: per-component visibility toggles.
- `Auto-fit on profile change` checkbox:
  - on: reframe camera on profile switch
  - off: keep camera on profile switch
- `Reset Camera`: reset to default view.
- `Settings` popup:
  - invert orbit X/Y
  - invert pan X/Y
  - invert zoom
- `Move Objects` mode:
  - click pads in `Fast 3D` and drag them
  - geometry values are updated live in profile YAML/form (`geometry_ranges.*`)
- Selected movable objects show a small X/Y gizmo overlay.
- Updates instantly when values change in center/right panels.

#### `Blender HQ` Tab

- Physically rendered preview via existing Blender batch pipeline.
- Runs in a background thread (UI remains responsive).
- Debounced trigger (~900ms) on changes.
- Status states:
  - queued
  - rendering
  - ready
  - error
- Render output path: `outputs/profile_editor/hq_preview/`.

### 4) Center Panel: Field Editor

- Dynamic form automatically generated from YAML leaf values.
- No hardcoded field list.
- Grouped by top-level sections (e.g. `profile`, `component`, `geometry_ranges`, `render_3d`, `run`, `roi`, `render`, ...).
- Commit on `Enter` or focus loss.
- Values are type-parsed via YAML parsing:
  - numbers, booleans, lists, strings

### 5) Right Panel: Direct YAML Editor

- Two tabs:
  - `Profile YAML`
  - `Run YAML`
- `Apply` button per tab.
- Also auto-apply after short typing pause (~350ms).
- YAML parse errors are shown directly in the corresponding tab.
- On syntax error, last valid state remains active.

### 6) Synchronization Rules

- Central state managed by `EditorStateStore`.
- Form change:
  - updates state
  - rewrites YAML text
  - updates fast preview
  - schedules HQ render
- YAML change:
  - parses and validates mapping structure
  - updates form and preview
- Profile-to-run coupling:
  - `run.component_profile` is auto-synced to current `profile.profile_id` on profile changes.

### 7) Save Behavior

- Atomic writes (`.tmp` + replace).
- Dirty state shown in the window title (`*`).
- Write errors shown in the status line.

### 7.1) Cross-Session Settings (Save/Load)

- File: `outputs/profile_editor/settings.json`
- Loaded on startup and auto-saved continuously.
- Stored values:
  - theme mode (`day` / `night`)
  - window geometry
  - last selected profile
  - last selected run
  - Fast 3D settings:
    - auto-fit toggle
    - invert options (orbit/pan/zoom)
    - camera state (yaw/pitch/zoom/pan)
    - component visibility

### 8) Validation and Error Handling

- YAML must be a top-level mapping.
- Parse errors only block the invalid apply step, not the whole app.
- HQ render is skipped if:
  - YAML errors exist
  - profile/run is missing
  - `run.render.backend != blender_3d`
- HQ render failures are shown as status (`error: ...`).

## HQ Render: Technical Details

- Worker hashes current state (`profile + run`) to derive output filenames.
- Each render writes a job JSONL and calls `simple_sim.generator_3d.render_blender_batch(...)`.
- Current preview job renders an `OK` sample (no defect).
- Used run parameters:
  - `roi.width_px`, `roi.height_px`, `roi.mm_per_px`
  - `run.seed`
  - `render.blender.executable`, `render.blender.samples`, `render.blender.device`

## Module Overview

- `tools/profile_editor/app.py`
  - main window, UI wiring, debounce, worker lifecycle, session settings, system stats
- `tools/profile_editor/profile_registry.py`
  - profile/run discovery, change signature
- `tools/profile_editor/sync_controller.py`
  - registry/state coordination (load/save/matching runs)
- `tools/profile_editor/state_store.py`
  - central document state, listeners, leaf updates, YAML apply, dirty flags
- `tools/profile_editor/form_renderer.py`
  - dynamic form rendering from YAML leaves
- `tools/profile_editor/yaml_editor.py`
  - YAML editor with auto-apply and error display
- `tools/profile_editor/preview_3d.py`
  - fast 3D canvas with orbit/pan/zoom, component menu, control settings
- `tools/profile_editor/hq_preview_panel.py`
  - HQ image panel + status
- `tools/profile_editor/blender_live_preview.py`
  - background worker for Blender HQ renders
- `tools/profile_editor/system_monitor.py`
  - CPU/GPU/RAM sampling and formatting
- `tools/profile_editor/yaml_io.py`
  - YAML I/O + atomic save

## Recommended Workflow

1. Select a profile.
2. Select a matching run.
3. Change values in center panel (or directly edit YAML on the right).
4. Check `Fast 3D`.
5. Check `Blender HQ` (auto or `Render HQ Now`).
6. Save profile and run.

## Known Limits

- Fast 3D is intentionally simplified and not a full Blender replacement.
- Field editor is generic for all YAML leaves; no specialized widgets (e.g. color pickers) yet.
- HQ preview currently renders only one preview state (no defect scenario switch in UI).

## Troubleshooting

- Black HQ tab:
  - check Blender path (`run.render.blender.executable`)
  - check `run.render.backend == blender_3d`
  - inspect status line / HQ tab status for `error:`
- Fast 3D shows nothing:
  - verify profile is loaded
  - verify `geometry_ranges` / `component` values are valid
- New profiles do not appear:
  - click `Refresh Files`
  - verify file extension/structure (`*.yaml`, valid `profile.profile_id`)
