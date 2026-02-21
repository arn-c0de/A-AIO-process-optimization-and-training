# Filter Settings

## Where settings are saved
- Shared file: `outputs/gui/settings.json`
- Used by:
  - Main GUI pipeline tab (`gui/tabs/pipeline/tab.py`)
  - Debug preview tool (`scripts/render_debug_previews.py`)

## Profile storage keys in `settings.json`
- `pipeline.filter_profiles_json`: JSON map `profile_name -> filter values`
- `pipeline.filter_profile_active`: active profile name

All filter values are stored inside each profile map.

## Modes
- `filter_mode`:
  - `custom`: classic per-filter toggles + strengths
  - `realism`: grouped realism sampling mode
- `realism_enabled`: additional explicit enable switch for realism mode

Realism mode is active only when:
- `filter_mode == "realism"`
- `realism_enabled == true`

## Custom mode keys
Base toggles:
- `cardinal_rotation_90`
- `enable_rotation`
- `enable_blur`
- `enable_grain`
- `enable_brightness`
- `enable_contrast`
- `enable_perspective`
- `enable_motion_blur`
- `enable_saturation`
- `enable_hue_shift`
- `enable_shadow`
- `enable_reflection`
- `enable_vignetting`
- `enable_chromatic_aberration`
- `enable_jpeg_compression`
- `enable_color_temperature`
- `enable_lens_distortion`
- `enable_dust`
- `enable_sharpen`

Main numeric keys:
- `rotation_strength`
- `blur_strength`
- `grain_strength`
- `brightness_strength`
- `contrast_strength`
- `perspective_strength`
- `motion_blur_strength`
- `saturation_factor`
- `hue_shift_deg`
- `shadow_strength`
- `reflection_strength`
- `vignetting_strength`
- `chromatic_strength`
- `jpeg_quality`
- `color_temperature_kelvin`
- `distortion_k1`
- `dust_density`
- `sharpen_strength`

Per-filter randomization keys (for supported numeric keys):
- `<key>_randomize` (bool)
- `<key>_min` (float/int)
- `<key>_max` (float/int)
- Example: `blur_strength_randomize`, `blur_strength_min`, `blur_strength_max`

## Realism mode keys
- `realism_constraints_enabled` (default `true`)
- `realism_profile_id` (default `profile_industrial_cam`)
- `realism_k_prob_0`
- `realism_k_prob_1`
- `realism_k_prob_2`
- `realism_group_G1_prob`
- `realism_group_G2_prob`
- `realism_group_G3_prob`
- `realism_group_G4_prob`
- `realism_group_G5_prob`
- `realism_group_G6_prob`
- `realism_group_G7_prob`
- `realism_group_G8_prob`

Current defaults (`gui/components/filter_popup/constants.py`):
- `K`: 0.60 / 0.35 / 0.05
- `G1..G8`: 0.35, 0.15, 0.20, 0.25, 0.12, 0.18, 0.10, 0.08

## Realism group semantics (runtime)
From `simple_sim/generators/opencv2d/realism.py`:
- `G0` (always active baseline): mild blur/noise/brightness/contrast/rotation drift
- `G1`: focus/optics softness
- `G2`: motion/conveyor blur
- `G3`: geometry/mounting (perspective)
- `G4`: illumination non-uniformity (shadow/vignetting)
- `G5`: specular/glare reflection
- `G6`: color pipeline/white balance
- `G7`: compression/transport artifacts
- `G8`: contamination (dust)

Selection logic:
- Sample `K in {0,1,2}` using `realism_k_prob_*`
- Choose `K` groups from `G1..G8` via group probabilities
- If constraints enabled:
  - `G5 + G7` conflict resolved by removing `G7`
  - At most one heavy group among `{G2, G5, G7}`

## Normalization and ranges
Central normalization is in `simple_sim/generators/filter_settings.py`:
- clamps numeric values to safe ranges
- normalizes `realism_k_prob_0/1/2` to sum to `1.0`
- invalid `filter_mode` falls back to `custom`

## UI grouping in popup
`gui/components/filter_popup/ui_custom.py` groups custom filters into:
- Existing Filters
- High Priority
- Medium Priority (disabled)
- Low Priority (disabled)

The Realism tab is built by `gui/components/filter_popup/ui_realism.py`.

## Runtime usage
- Pipeline passes filter config via `IMAGE_FILTERS` JSON.
- 2D generation uses normalized values and applies:
  - custom augmentation chain, or
  - realism grouped sampling (`apply_realism_groups`) when realism mode is active.
- Debug previews read/write the same keys, so GUI and preview stay synchronized.
