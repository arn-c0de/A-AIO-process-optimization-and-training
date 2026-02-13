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

## Architecture

```
gui/
├── monitor.py                # Main multi-tab application
├── run.sh                    # Launch script
├── state.py                  # Shared state across tabs
├── tabs/
│   ├── pipeline_tab.py      # Pipeline control
│   ├── analysis_tab.py      # Image analysis
│   └── validation_tab.py    # Quality validation
├── components/
│   ├── image_cache.py       # LRU cache for performance
│   ├── overlay_renderer.py  # Defect visualization
│   └── chart_widgets.py     # Matplotlib charts
└── utils/
    ├── model_inference.py   # Model loading
    ├── flag_manager.py      # Sample flagging
    └── validation_suite.py  # Validation tests
```

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
- ✅ Original monitor backed up as `monitor_old.py.backup`

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
