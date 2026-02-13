# Simple-Sim Command Cheatsheet

Quick reference for common operations.

## Setup

```bash
# One-time setup
cd Simple-Sim
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Complete Pipeline

```bash
# Run everything with one command
./run_pipeline.sh
```

## Individual Commands

### Generate Dataset
```bash
python3 scripts/generate.py \
  --config configs/run_0001.yaml \
  --out outputs/sim_data/runs/run_0001
```

### Validate Dataset
```bash
python3 tools/validate_dataset.py \
  --data outputs/sim_data/runs/run_0001
```

### Train Model
```bash
# With GPU (default)
python3 scripts/train.py \
  --data outputs/sim_data/runs/run_0001 \
  --out outputs/models/run_0001.pt

# With CPU only
python3 scripts/train.py \
  --data outputs/sim_data/runs/run_0001 \
  --out outputs/models/run_0001.pt \
  --device cpu
```

### Evaluate Model
```bash
python3 scripts/eval.py \
  --data outputs/sim_data/runs/run_0001 \
  --model outputs/models/run_0001.pt
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_schema.py -v
pytest tests/test_rng.py -v
pytest tests/test_splits.py -v
pytest tests/test_pipeline_e2e.py -v

# Run with coverage
pytest tests/ --cov=simple_sim
```

## Verify Results

```bash
# Check test metrics
cat outputs/models/report_run_0001.json | python3 -m json.tool

# View specific metric
cat outputs/models/report_run_0001.json | grep -A 3 '"accuracy"'

# List generated images
ls -lh outputs/sim_data/runs/run_0001/images/

# Count samples per split
wc -l outputs/sim_data/runs/run_0001/splits/*.txt
```

## Determinism Check

```bash
# Generate twice
python3 scripts/generate.py --config configs/run_0001.yaml --out outputs/test_a
python3 scripts/generate.py --config configs/run_0001.yaml --out outputs/test_b

# Compare labels (should be identical)
diff outputs/test_a/labels.jsonl outputs/test_b/labels.jsonl
echo $?  # Returns 0 if identical
```

## Custom Configuration

```bash
# Copy reference config
cp configs/run_0001.yaml configs/my_experiment.yaml

# Edit parameters
nano configs/my_experiment.yaml

# Generate with custom config
python3 scripts/generate.py \
  --config configs/my_experiment.yaml \
  --out outputs/sim_data/runs/my_experiment
```

## Common Config Tweaks

Edit `configs/run_0001.yaml`:

```yaml
# Change sample counts
classes:
  OK: 200
  MISSING: 200
  MISALIGNED: 200
  TOMBSTONE: 200

# Adjust tolerances
tolerances:
  0603:
    ok_shift_px: 5.0        # More lenient
    ok_rotation_deg: 10.0   # More lenient
    tombstone_tilt_deg: 80.0  # Stricter

# Modify training
train:
  epochs: 20              # More epochs
  batch_size: 64          # Larger batch
  lr: 0.0001              # Lower learning rate
```

## Inspect Dataset

```bash
# View metadata sample
head -n 1 outputs/sim_data/runs/run_0001/meta.jsonl | python3 -m json.tool

# View label sample
head -n 1 outputs/sim_data/runs/run_0001/labels.jsonl | python3 -m json.tool

# Count samples by class
python3 -c "
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, '.')
from simple_sim.schema import read_jsonl, LabelRow

labels = read_jsonl(Path('outputs/sim_data/runs/run_0001/labels.jsonl'), LabelRow)
counts = Counter(row.class_name for row in labels)
for cls, count in sorted(counts.items()):
    print(f'{cls}: {count}')
"
```

## Troubleshooting

```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check installed packages
pip list | grep -E "numpy|opencv|torch|sklearn|yaml|tqdm|pytest"

# Verify imports
python3 -c "import simple_sim; print('✓ Package imports OK')"

# Clean outputs
rm -rf outputs/sim_data/runs/run_0001
rm -rf outputs/models/run_0001.pt

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## Performance Tips

```bash
# Use GPU for training
python3 scripts/train.py ... --device cuda

# Reduce batch size if OOM
# Edit config: train.batch_size: 16

# Generate smaller dataset for testing
# Edit config: classes.OK: 10 (for each class)

# Use fewer workers
# Edit scripts/train.py: num_workers=2
```

## File Locations

| Item | Location |
|------|----------|
| Generated images | `outputs/sim_data/runs/run_0001/images/*.png` |
| Metadata | `outputs/sim_data/runs/run_0001/meta.jsonl` |
| Labels | `outputs/sim_data/runs/run_0001/labels.jsonl` |
| Splits | `outputs/sim_data/runs/run_0001/splits/*.txt` |
| Trained model | `outputs/models/run_0001.pt` |
| Test report | `outputs/models/report_run_0001.json` |
| Config | `configs/run_0001.yaml` |

## Quick Metrics Check

```bash
# View confusion matrix from report
python3 -c "
import json
with open('outputs/models/report_run_0001.json') as f:
    report = json.load(f)
print('Accuracy:', report['metrics']['accuracy'])
print('Macro F1:', report['metrics']['macro_f1'])
print('\nPer-class:')
for cls, metrics in sorted(report['metrics']['per_class'].items()):
    print(f'  {cls}: F1={metrics[\"f1\"]:.3f}, Recall={metrics[\"recall\"]:.3f}')
"
```

## Clean Start

```bash
# Remove all generated data
rm -rf outputs/sim_data/runs/*
rm -rf outputs/models/*

# Deactivate and remove venv
deactivate
rm -rf venv/

# Fresh install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Help

```bash
# Script help
python3 scripts/generate.py --help
python3 scripts/train.py --help
python3 scripts/eval.py --help
python3 tools/validate_dataset.py --help
```

## Success Indicators

After running pipeline, verify:
- [ ] Generation took < 5 min
- [ ] Validation passed all 8 checks
- [ ] Training completed without errors
- [ ] Val accuracy > 70% (shown during training)
- [ ] Test accuracy > 70% (in report.json)
- [ ] No class has 0% recall
- [ ] report_run_0001.json exists

If all checked: **SUCCESS!** ✅
