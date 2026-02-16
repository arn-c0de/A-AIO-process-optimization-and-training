# 3D Rendering Quickstart Guide

This guide explains how to modify and customize the 3D rendering system for Simple-Sim.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Configuration Files](#configuration-files)
3. [Modifying Pad Positions](#modifying-pad-positions)
4. [Modifying Components](#modifying-components)
5. [Materials and Appearance](#materials-and-appearance)
6. [Lighting Setup](#lighting-setup)
7. [Camera Configuration](#camera-configuration)
8. [Debugging and Testing](#debugging-and-testing)

---

## Architecture Overview

The 3D rendering system consists of three main parts:

```
configs/profiles/*.yaml     → Component profiles (geometry, materials, lighting)
configs/run_*.yaml          → Run configurations (ROI, classes, rendering settings)
simple_sim/blender/         → Blender rendering backend
```

**Rendering Flow:**
1. Profile YAML defines component geometry ranges and render settings
2. Run config specifies which profile to use and general settings
3. Generator samples geometry from ranges and creates jobs
4. Blender renders each job using the render_batch.py script

---

## Configuration Files

### Component Profile (`configs/profiles/*.yaml`)

Defines the component-specific geometry, appearance, and rendering setup.

**Key sections:**
- `geometry_ranges`: Physical dimensions (in pixels)
- `render_3d.materials`: Material properties (color, roughness, metallic)
- `render_3d.lighting`: Light positions and intensities
- `render_3d.camera`: Camera position and settings

### Run Configuration (`configs/run_*.yaml`)

Defines the dataset generation parameters.

**Key sections:**
- `run.component_profile`: Which profile to use
- `roi`: Region of interest size and resolution
- `render.blender`: Blender executable and render settings
- `classes`: Number of samples per defect type

### Rotation Strategy (2D + 3D)

Dataset generation now uses:
- **Coarse orientation**: random `0° / 90° / 180° / 270°`
- **Fine jitter**: `augment.rotation_deg_range` (e.g. `[-3, 3]`)

So each sample gets:

```
rotation_deg = cardinal_orientation + jitter
```

In 2D, this is applied as global image rotation.
In 3D, the same `augment.rotation_deg` is applied as a global in-plane scene rotation (pads + component + solder).

---

## Modifying Pad Positions

### Variables Controlling Pad Position

**File:** `configs/profiles/<profile_name>.yaml`

```yaml
geometry_ranges:
  pad_spacing: [250.0, 280.0]      # Distance between opposite pads (left/right)
  pad_spacing_y: [180.0, 210.0]    # Distance between opposite pads (top/bottom)
  pad_width: [45.0, 55.0]          # Pad tangential dimension
  pad_height: [110.0, 130.0]       # Pad radial dimension (length)
```

### How Pad Positioning Works

For QFN components, pads are positioned as follows:

```
Left pad:   x = -pad_spacing / 2
Right pad:  x = +pad_spacing / 2
Top pad:    y = -pad_spacing_y / 2
Bottom pad: y = +pad_spacing_y / 2
```

**To move pads closer to center:** Decrease `pad_spacing` and `pad_spacing_y`
**To move pads farther from center:** Increase `pad_spacing` and `pad_spacing_y`

### Example: Positioning Pads Directly Under Component

For pads to align with component edges:

```
pad_spacing = component_length - pad_height
```

Example calculation:
- component_length: 500px
- pad_height: 120px
- **pad_spacing: 380px** (pads align with component edges)

For pads closer to center (under component):
- **pad_spacing: 250-280px** (pads well inside component footprint)

### Code Implementation

Pad positions are calculated in:
**File:** `simple_sim/blender/render_batch.py`
**Function:** `_pad_positions_mm()`

```python
return [
    # Left/right pads
    (-pad_spacing / 2.0, 0.0, pad_h, pad_w),
    (+pad_spacing / 2.0, 0.0, pad_h, pad_w),
    # Top/bottom pads
    (0.0, -pad_spacing_y / 2.0, pad_w, pad_h),
    (0.0, +pad_spacing_y / 2.0, pad_w, pad_h),
]
```

---

## Modifying Components

### Component Dimensions

**File:** `configs/profiles/<profile_name>.yaml`

```yaml
component:
  nominal_dims_mm:
    length: 5.0      # Nominal length in mm
    width: 4.0       # Nominal width in mm
    height: 0.85     # Height in mm

geometry_ranges:
  component_length: [485.0, 515.0]   # Length variation (in pixels)
  component_width: [400.0, 430.0]    # Width variation (in pixels)
```

### Component Types

Components are created based on footprint type in `render_batch.py`:

```python
if footprint == "chip_2pad":
    # Chip resistor/capacitor with rounded edges
    comp_obj = _mk_chip_resistor(...)

elif footprint == "sot23":
    # SOT-23 transistor package
    comp_obj = _mk_sot23_transistor(...)

elif footprint.startswith("qfn"):
    # QFN IC package
    comp_obj = _mk_qfn_package(...)
```

**To add a new component type:**
1. Add a `_mk_<component>()` function in `render_batch.py`
2. Add a case in the component creation section
3. Add solder joints for the new footprint

---

## Materials and Appearance

### Material Configuration

**File:** `configs/profiles/<profile_name>.yaml`

```yaml
render_3d:
  materials:
    substrate:
      base_color: [0.05, 0.20, 0.05]   # RGB (PCB green)
      roughness: 0.85                   # 0=mirror, 1=matte
      metallic: 0.0                     # 0=dielectric, 1=metal

    copper:
      base_color: [0.80, 0.45, 0.10]   # Copper/gold finish
      roughness: 0.25
      metallic: 1.0

    body:
      base_color: [0.02, 0.02, 0.02]   # Component body (dark)
      roughness: 0.60
      metallic: 0.0

    solder:
      base_color: [0.70, 0.70, 0.72]   # Solder (silver)
      roughness: 0.18
      metallic: 1.0
```

### Material Properties

**base_color**: RGB values [0.0-1.0]
- `[0.05, 0.20, 0.05]` = dark green (PCB)
- `[0.80, 0.45, 0.10]` = copper/gold
- `[0.70, 0.70, 0.72]` = silver (solder)

**roughness**: Surface smoothness [0.0-1.0]
- `0.0` = perfect mirror
- `0.5` = semi-glossy
- `1.0` = completely matte

**metallic**: Material type [0.0-1.0]
- `0.0` = dielectric (plastic, ceramic)
- `1.0` = metal (conductor)

### Code Implementation

Materials are created in `render_batch.py`:

```python
m_solder = _new_material("mat_solder",
    base_color=(0.7, 0.7, 0.72),
    roughness=0.15,
    metallic=1.0)
```

---

## Lighting Setup

### Light Configuration

**File:** `configs/profiles/<profile_name>.yaml`

```yaml
render_3d:
  lighting:
    key_light:
      location_mm: [10.0, -10.0, 24.0]   # Position [x, y, z]
      power_w: 1200.0                     # Light intensity
      size_mm: 28.0                       # Area light size

    fill_light:
      location_mm: [-10.0, 10.0, 20.0]
      power_w: 480.0
      size_mm: 32.0
```

### Lighting Parameters

**location_mm**: Light position in 3D space
- `x`: left (-) to right (+)
- `y`: back (-) to front (+)
- `z`: down (-) to up (+)
- Origin `[0, 0, 0]` is at component center

**power_w**: Light intensity in watts
- Higher values = brighter
- Typical range: 300-2000W for AOI-style lighting

**size_mm**: Physical size of area light
- Larger lights = softer shadows
- Smaller lights = harder, more defined shadows

### Making A Render Brighter (Practical)

If a component (often QFN) renders much darker than others, it is usually because the lights are much farther away.

What to change (in this order):
1. Move lights closer with `location_mm`. Light falloff is roughly proportional to `1 / distance^2`, so doubling the distance can make the scene about 4x darker.
2. If needed, increase `power_w`. In `render_batch.py`, Blender energy is set as `energy = power_w * power_boost` (currently `power_boost = 8.0`), so changes to `power_w` scale brightness linearly.
3. If highlights look too flat, reduce `size_mm` a bit (smaller area lights increase contrast and specular punch).

Example: bring lights closer (same power)

```yaml
render_3d:
  lighting:
    key_light:
      location_mm: [20.0, -20.0, 45.0]
      power_w: 1200.0
      size_mm: 28.0
    fill_light:
      location_mm: [-16.0, 16.0, 40.0]
      power_w: 480.0
      size_mm: 32.0
```

```yaml
render_3d:
  lighting:
    key_light:
      location_mm: [10.0, -10.0, 24.0]
      power_w: 1200.0
      size_mm: 28.0
    fill_light:
      location_mm: [-10.0, 10.0, 20.0]
      power_w: 480.0
      size_mm: 32.0
```

### Two-Light Setup (Standard)

**Key Light**: Main light source, creates primary illumination and shadows
**Fill Light**: Secondary light, reduces shadow intensity, adds detail to dark areas

Typical positioning:
- Key: 45° angle, high intensity
- Fill: Opposite side, lower intensity (30-50% of key)

---

## Camera Configuration

### Camera Settings

**File:** `configs/profiles/<profile_name>.yaml`

```yaml
render_3d:
  camera:
    location_mm: [0.0, -30.0, 40.0]   # Camera position [x, y, z]
    look_at_mm: [0.0, 0.0, 0.0]       # Target point camera looks at
    focal_length_mm: 60.0              # Lens focal length (not used in ortho)
```

### Camera Type

The system uses **orthographic projection** (parallel projection, no perspective distortion) to match AOI inspection systems.

**Orthographic scale** is automatically calculated to fit the ROI:
```python
cam.data.ortho_scale = max(board_w_mm, board_h_mm) * 1.2
```

### Camera Positioning

Standard AOI camera setup:
- Position above and slightly behind component (`y < 0`, `z > 0`)
- Look straight down at component center
- Typical position: `[0, -30, 40]` (slightly angled top-down view)

For pure top-down view:
```yaml
location_mm: [0.0, 0.0, 50.0]    # Directly above
look_at_mm: [0.0, 0.0, 0.0]      # Looking straight down
```

---

## Debugging and Testing

### Debug Rendering Script

Quickly test profile changes:

```bash
./run_render_debug.sh --profiles <profile_name> --non-interactive --samples 32
```

**Output**: `outputs/debug_previews/index.html`

### Debug Output

The render script prints diagnostic information:

```
[QFN] YAML pad_spacing=367.3px, desired=396.5px, FINAL=367.3px
[COMP] Length=5.070mm (pad_spacing=3.673mm, pad_w=0.517mm)
```

**YAML**: Value from profile configuration
**desired**: Calculated optimal value (from code)
**FINAL**: Actual value used in rendering

### Common Issues and Solutions

#### Pads too far from center
**Problem**: `pad_spacing` too large
**Solution**: Decrease `pad_spacing` and `pad_spacing_y` in profile YAML

#### Pads at edge of PCB
**Problem**: Board margin too small
**Solution**: Check `render_batch.py` line ~412 for margin calculation

#### Component not visible
**Problem**: Defect type is `MISSING`
**Solution**: Check job defect type or render `OK` state

#### Colors look wrong
**Problem**: Color management or material settings
**Solution**: Check `view_transform = 'Raw'` in `_configure_cycles()`

#### Rendering too dark/bright
**Problem**: Lighting intensity
**Solution**: First move lights closer/farther via `location_mm`, then tune `power_w`, then adjust `size_mm` for shadow softness/contrast

### Incremental Testing Workflow

1. **Modify profile YAML** (change geometry, materials, etc.)
2. **Run debug render** to generate test images
3. **Check output** in `outputs/debug_previews/index.html`
4. **Iterate** until desired appearance is achieved
5. **Run full dataset generation** with `run_*.yaml` config

---

## Quick Reference: File Locations

| What to Change | File Location |
|----------------|---------------|
| Pad positions | `configs/profiles/<name>.yaml` → `geometry_ranges.pad_spacing` |
| Component size | `configs/profiles/<name>.yaml` → `geometry_ranges.component_length/width` |
| Materials | `configs/profiles/<name>.yaml` → `render_3d.materials` |
| Lighting | `configs/profiles/<name>.yaml` → `render_3d.lighting` |
| Camera | `configs/profiles/<name>.yaml` → `render_3d.camera` |
| ROI size | `configs/run_*.yaml` → `roi.width_px/height_px` |
| Render quality | `configs/run_*.yaml` → `render.blender.samples` |
| Pad creation code | `simple_sim/blender/render_batch.py` → `_pad_positions_mm()` |
| Component creation | `simple_sim/blender/render_batch.py` → Component type functions |
| Material creation | `simple_sim/blender/render_batch.py` → `_new_material()` |
| Lighting code | `simple_sim/blender/render_batch.py` → `add_area()` function |

---

## Advanced: Code Structure

### Main Rendering Pipeline

```
scripts/render_debug_previews.py
  ↓
simple_sim/generator_3d.py → write_jobs_jsonl()
  ↓
simple_sim/blender/render_batch.py → _build_scene_for_job()
  ↓
Blender renders scene → PNG output
```

### Key Functions in render_batch.py

| Function | Purpose |
|----------|---------|
| `_pad_positions_mm()` | Calculate pad positions from geometry |
| `_mk_box()` | Create rectangular objects (pads, basic components) |
| `_mk_chip_resistor()` | Create realistic chip component with rounded edges |
| `_mk_qfn_package()` | Create QFN IC package |
| `_add_solder_joint()` | Create solder fillet at pad/component junction |
| `_new_material()` | Create PBR material with color/roughness/metallic |
| `_configure_cycles()` | Set up Blender render engine and color management |

---

## Tips and Best Practices

### Geometry Ranges
- Use realistic tolerances (±5-10% variation typical)
- Ensure `pad_spacing < component_length` for QFN (pads should be under component)
- Test extreme values (min and max of range) during debugging

### Materials
- Keep `base_color` values realistic (avoid pure black/white)
- PCB green: `[0.05, 0.20, 0.05]` (dark green, not bright)
- Copper finish: `[0.80, 0.45, 0.10]` (warm metallic)
- Match real-world AOI image appearance

### Lighting
- Use two-light setup for balanced illumination
- Key/fill ratio typically 2:1 to 3:1
- Position lights 30-50mm above component
- Area lights (large size) create soft, realistic shadows

### Performance
- Use `--samples 32` for fast debug renders
- Use `--samples 128+` for production datasets
- CPU rendering is slower but more compatible
- GPU rendering (CUDA/OptiX) much faster if available

---

## Example: Creating a New Component Profile

```yaml
profile:
  profile_id: "my_component_3d@1"
  schema_version: 1
  supported_render_backends: ["blender_3d"]

component:
  footprint: "qfn_32"
  nominal_dims_mm:
    length: 5.0
    width: 4.0
    height: 0.85

geometry_ranges:
  pad_width: [45.0, 55.0]
  pad_height: [110.0, 130.0]
  pad_spacing: [250.0, 280.0]        # Pads close to center
  pad_spacing_y: [180.0, 210.0]
  component_length: [485.0, 515.0]
  component_width: [400.0, 430.0]

render_3d:
  camera:
    location_mm: [0.0, -30.0, 40.0]
    look_at_mm: [0.0, 0.0, 0.0]

  lighting:
    key_light:
      location_mm: [20.0, -20.0, 45.0]
      power_w: 1200.0
      size_mm: 28.0
    fill_light:
      location_mm: [-16.0, 16.0, 40.0]
      power_w: 480.0
      size_mm: 32.0

  materials:
    substrate:
      base_color: [0.05, 0.20, 0.05]
      roughness: 0.85
      metallic: 0.0
    copper:
      base_color: [0.80, 0.45, 0.10]
      roughness: 0.25
      metallic: 1.0
```

Test with:
```bash
./run_render_debug.sh --profiles my_component_3d@1 --non-interactive
```

---

## Support and Troubleshooting

For issues or questions:
1. Check debug output for actual values used
2. Verify YAML syntax is valid
3. Test with minimal samples first (`--samples 32`)
4. Review generated images in `outputs/debug_previews/`
5. Check Blender console output for errors

Common error patterns:
- `KeyError`: Missing required field in YAML
- `ValueError`: Invalid range (min > max)
- Render failures: Check Blender executable path in run config
- Missing objects: Verify footprint type matches code implementation
