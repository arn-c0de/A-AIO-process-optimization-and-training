# Simple-Sim Professional GUI

Multi-tab professional monitor for the Simple-Sim pipeline.

## Quick Start

```bash
cd Simple-Sim
./gui/run.sh
```

## Features

### Tab 1: Pipeline Control
- **Pipeline Management**: Start/stop full pipeline (generate → train → eval)
- **Live Monitoring**: Real-time logs and telemetry events
- **Dataset Operations**: Validate, Train, Eval buttons for quick operations
- **Enhanced Image Grid**: 4×3 grid showing latest 12 images
- **Status Indicators**: Phase, epoch, img/s, last processed image

### Tab 2: Analysis
- **Image Browser**: Navigate all samples with class filtering
- **Defect Overlays**: Visual indicators for defect types:
  - ✓ OK: Green checkmark
  - ✗ MISSING: Red X
  - ⚠ MISALIGNED: Yellow arrow showing shift + rotation
  - ⚠ TOMBSTONE: Purple bar with tilt angle
- **Model Predictions**: Compare ground truth vs predictions
- **Batch Analysis**: Generate accuracy report for entire dataset
- **Metadata Display**: View all defect parameters

### Tab 3: Validation
- **Quality Tests**:
  - Schema validation (JSONL format, files, IDs)
  - Image outlier detection (blur, brightness)
  - Duplicate detection (perceptual hashing)
  - Edge case detection (near tolerance boundaries)
- **Flagging System**: Mark problematic samples
- **Export Reports**: JSON validation reports

### Profile Preview
- **Look before you generate**: renders 1-8 variants per defect state of one profile, no dataset needed
- **Quality presets**: Draft (1x supersampling, 16 Cycles samples), Standard (2x, 64), Ultra (4x, 256); values editable (Custom)
- **Render settings**: run config, style (photoreal/flat), camera preset (aoi/phone), Blender engine/device, seed, 90° orientation, optional GUI image filters
- **Gallery**: thumbnail size, click for a full-size view (1-4x zoom); images in `outputs/gui/profile_preview/<profile>/<backend>/`

### Tab 4: Predictions
- **Batch Prediction**: Run `predict.sh` (batch_predict) on a split (train/val/test/all)
- **Metrics + Confusion Matrix**: Accuracy, macro-F1, critical FN rates, CM chart
- **Per-sample Browser**: Inspect correct/wrong predictions with image overlay

## Look & Feel

- **Light / dark theme**: View → Dark Mode (Ctrl+Shift+D); remembered across restarts.
- **UI scale** for HiDPI screens: View → UI Scale, or `SIMPLE_SIM_UI_SCALE=1.5 ./run.sh`.
- **Help everywhere**: hover any control for a tooltip (also shown in the status bar),
  "?" on a card explains the section, F1 opens the quick start, Help → Keyboard Shortcuts.
- **Status bar**: messages, running background tasks with progress, CPU/GPU/RAM.

Shortcuts: F1 help · F5 start pipeline · Shift+F5 stop · Ctrl+R refresh · Ctrl+1…9 pages · Ctrl+Q quit.

## Architecture

```
gui/
├── monitor.py      # shell: window, menu, shortcuts, status bar, theme, AppContext
├── core/           # AppContext, EventBus, TaskRunner (tk-free)
├── theme/          # design tokens (colors, spacing, fonts) + ttk styles
├── widgets/        # Section, FormGrid, PathPicker, ScrolledTree, LogView, StatusBar, ...
├── dialogs.py      # confirm / error / text viewer / list picker
├── help/texts.py   # all tooltip and help texts
├── tabs/<name>/    # tab.py (controller) · ui.py (widgets) · logic.py (tk-free)
├── components/     # filter popup, precise popup, overlays, charts
├── utils/          # settings, processes, dataset/model helpers
└── devtools/       # sandbox root for tests and screenshots
```

Details and "add a tab in 3 steps": [docs/architekture/GUI_ARCHITECTURE.md](../docs/architekture/GUI_ARCHITECTURE.md).
Screenshots of all tabs (light + dark): `python -m simple_sim gui-screenshots --out /tmp/shots --demo`.

## Performance

- **Lazy Loading**: Tabs initialize only when first accessed
- **Image Caching**: LRU cache for 200 thumbnails + 20 full images
- **Background Processing**: Validation and analysis run in threads
- **Memory Efficient**: Typical usage < 100 MB

## Output Files

### Analysis Results
`{dataset_dir}/analysis_results.json`:
```json
{
  "dataset_path": "outputs/sim_data/runs/run_0001",
  "summary": {
    "total": 400,
    "correct": 395,
    "accuracy": 0.9875
  }
}
```

### Flags
`{dataset_dir}/flags.jsonl`:
```json
{"id": "run_0001/domain_A/train/00234", "flag": "OUTLIER", "reason": "Brightness 3.2σ below mean", "timestamp": "2026-02-13T16:30:00"}
```

Flag types: `CORRUPT`, `OUTLIER`, `SUSPECT_LABEL`, `EDGE_CASE`, `DUPLICATE`, `MANUAL`

## Backward Compatibility

- ✅ All pipeline scripts unchanged
- ✅ Telemetry system unchanged
- ✅ Dataset schemas unchanged
- ✅ Model format unchanged

## Troubleshooting

### Tkinter not available
```bash
sudo apt-get install python3-tk
```

### Dependencies missing
```bash
cd Simple-Sim
.venv/bin/pip install -r requirements.txt
```

### Model predictions not showing
Train a model first using the "Train" button in Pipeline Control tab.
