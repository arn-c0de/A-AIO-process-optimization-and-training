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

---

# Recommended Additional Filters

The following filters can significantly improve the quality and robustness of your synthetic PCB dataset:

## 1. **Perspective Transform / Skew**
**Purpose:** Simulates different camera angles and perspectives
- Important for real inspection systems that are not always perfectly perpendicular to the PCB
- Helps the model recognize components from various viewing angles
- **Parameters:**
  - `enable_perspective`: bool
  - `perspective_strength`: float (0.0-2.0) - strength of perspective distortion
  - `perspective_angle_x`: float (-15° to +15°) - tilt around X-axis
  - `perspective_angle_y`: float (-15° to +15°) - tilt around Y-axis

## 2. **Motion Blur**
**Purpose:** Simulates motion blur in high-speed inspection systems
- Realistic for conveyor belt inspection systems
- Differs from Gaussian blur by having a directional component
- **Parameters:**
  - `enable_motion_blur`: bool
  - `motion_blur_strength`: float (0.0-3.0)
  - `motion_blur_angle`: float (0°-360°) - direction of motion

## 3. **Chromatic Aberration**
**Purpose:** Simulates color fringing from optical system lens errors
- Typical for lower-cost camera systems
- Especially visible at high-contrast edges
- **Parameters:**
  - `enable_chromatic_aberration`: bool
  - `chromatic_strength`: float (0.0-2.0) - color channel displacement

## 4. **Vignetting**
**Purpose:** Darkening towards image edges
- Realistic effect in camera optics
- Helps the model handle uneven illumination
- **Parameters:**
  - `enable_vignetting`: bool
  - `vignetting_strength`: float (0.0-2.0) - strength of edge darkening

## 5. **Saturation / Hue Shift**
**Purpose:** Color variation from different lighting and camera systems
- Simulates different white balance settings
- Important for robustness against various light sources (LED, halogen, daylight)
- **Parameters:**
  - `enable_saturation`: bool
  - `saturation_factor`: float (0.5-1.5) - saturation (1.0 = neutral)
  - `enable_hue_shift`: bool
  - `hue_shift_deg`: float (-30° to +30°) - hue shift in HSV space

## 6. **Sharpen**
**Purpose:** Controlled sharpening to simulate different camera settings
- Compensates for natural blur
- Useful for high-resolution inspection systems
- **Parameters:**
  - `enable_sharpen`: bool
  - `sharpen_strength`: float (0.0-2.0)

## 7. **Lens Distortion**
**Purpose:** Simulates barrel or pincushion distortion
- Typical for wide-angle or telephoto lenses
- Important for systems with overview cameras
- **Parameters:**
  - `enable_lens_distortion`: bool
  - `distortion_k1`: float (-0.3 to +0.3) - radial distortion 1st order
  - `distortion_k2`: float (-0.1 to +0.1) - radial distortion 2nd order

## 8. **JPEG Compression Artifacts**
**Purpose:** Simulates compression artifacts in JPEG-stored images
- Realistic for systems that compress for performance reasons
- Makes the model more robust against compression losses
- **Parameters:**
  - `enable_jpeg_compression`: bool
  - `jpeg_quality`: int (50-95) - JPEG quality (lower values = more artifacts)

## 9. **Shadow / Occlusion**
**Purpose:** Simulates shadows from equipment, grippers, or other components
- Very important for real production environments
- Can occlude parts of the component
- **Parameters:**
  - `enable_shadow`: bool
  - `shadow_strength`: float (0.0-0.8) - darkness of shadow
  - `shadow_size`: float (0.1-0.5) - size/extent relative to ROI

## 10. **Reflection / Glare**
**Purpose:** Simulates reflections and overexposure from glossy surfaces
- Common with metallic components or fresh solder
- Can obscure important features
- **Parameters:**
  - `enable_reflection`: bool
  - `reflection_strength`: float (0.0-1.0)
  - `reflection_size`: float (0.05-0.3) - size of reflection

## 11. **Dust / Dirt Particles**
**Purpose:** Simulates dust and contamination on lens or PCB
- Very realistic for production environments
- Can be added as noise/texture overlay
- **Parameters:**
  - `enable_dust`: bool
  - `dust_density`: float (0.0-1.0) - number of particles
  - `dust_size`: float (1-5) - average particle size in pixels

## 12. **Color Temperature Variation**
**Purpose:** Simulates different color temperatures of lighting
- More important than simple hue shift
- Models difference between warm white/cool white LEDs
- **Parameters:**
  - `enable_color_temperature`: bool
  - `color_temperature_kelvin`: int (2500K-7500K) - light color

## Implementation Priority

**High Priority** (greatest impact on model robustness):
1. Perspective Transform
2. Motion Blur
3. Saturation / Hue Shift
4. Shadow / Occlusion
5. Reflection / Glare

**Medium Priority** (useful for specific scenarios):
6. Vignetting
7. Chromatic Aberration
8. JPEG Compression
9. Color Temperature

**Low Priority** (fine-tuning):
10. Lens Distortion
11. Dust Particles
12. Sharpen

## Integration into Existing Architecture

All new filters should follow the same pattern as existing ones:

```python
# In generator_2d.py
def normalize_image_filters(image_filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # Add new filters:
    "enable_perspective": _bool("enable_perspective", False),
    "perspective_strength": _float("perspective_strength", 1.0, 0.0, 2.0),
    # ...
```

```python
# In generator_2d.py
def apply_perspective_transform(img: np.ndarray, strength: float, angle_x: float, angle_y: float) -> np.ndarray:
    """Apply perspective transformation."""
    # Implementation here
    pass
```

```python
# In gui/tabs/pipeline/ui.py
# New Tkinter variables and UI elements for each filter
self.var_filter_enable_perspective: tk.BooleanVar = tk.BooleanVar(value=False)
self.var_filter_perspective_strength: tk.StringVar = tk.StringVar(value="1.0")
```

Each new filter is then applied in the `render_roi()` function in `generator_2d.py`, after the existing filters.
