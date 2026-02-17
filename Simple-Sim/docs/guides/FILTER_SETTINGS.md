# Filter Settings (Short)

## Where settings are saved
- Shared file: `outputs/gui/settings.json`
- This is used by both:
  - Main GUI pipeline tab (`gui/tabs/pipeline/tab.py`)
  - Debug preview tool (`scripts/render_debug_previews.py`, via `run_render_debug.sh`)

## Main keys in `settings.json`
- `pipeline.filter_profiles_json`
  - JSON string map: profile name -> filter values
- `pipeline.filter_profile_active`
  - active profile name
- `pipeline.filter.cardinal_rotation_90`
- `pipeline.filter.enable_rotation`
- `pipeline.filter.enable_blur`
- `pipeline.filter.enable_grain`
- `pipeline.filter.enable_brightness`
- `pipeline.filter.enable_contrast`
- `pipeline.filter.rotation_strength`
- `pipeline.filter.blur_strength`
- `pipeline.filter.grain_strength`
- `pipeline.filter.brightness_strength`
- `pipeline.filter.contrast_strength`

## Which filters exist
- `90° base rotation` (cardinal: 0/90/180/270)
- `Rotation` (jitter)
- `Blur`
- `Grain` (noise)
- `Brightness`
- `Contrast`

Each filter has:
- enable toggle (`enable_*`)
- strength value (`*_strength`)

## How loading/saving works
- On startup, active profile and filter values are loaded from `settings.json`.
- In popup/profile actions (new/save/rename/delete/select/close), active profile values are persisted.
- Debug preview and main GUI are synchronized through the same file.

## Runtime transfer into generation
- Pipeline run passes filter config through environment variable: `IMAGE_FILTERS` (JSON).
- Generator applies these values in:
  - `scripts/generate.py`
  - `scripts/generate_profile_dataset.py`
  - `simple_sim/generator_2d.py`
- For 3D previews, Blender output is post-processed for blur/grain/brightness/contrast in `scripts/render_debug_previews.py`.
