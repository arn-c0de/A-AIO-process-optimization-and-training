# AGENTS.md - Simple-Sim for AI agents

Synthetic PCB-defect ROI generator + ResNet18 train/eval pipeline. This page is the short,
machine-oriented guide; humans start with [README.md](README.md).

All commands below run from the `Simple-Sim/` directory.

## Setup

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .   # optional: `import simple_sim` from anywhere + `simple-sim` script
# or reuse an existing interpreter that has torch (CPU is fine), torchvision, opencv-python, PyYAML, numpy
```

`pyproject.toml` declares the package (`simple_sim` only; `gui/`, `scripts/`, `tools/` stay
checkout-relative). Core deps cover config, rendering, generation and metrics; extras `[train]`
(torch), `[gui]`, `[dev]`, `[all]`. `simple-sim <cmd>` equals `python -m simple_sim <cmd>`.

Blender is only needed for `render.backend: blender_3d` configs; tkinter only for the GUI
(`./run.sh`) and the profile editor. Neither is needed for the CLI workflow below.

## Entrypoint: `python -m simple_sim`

```bash
python -m simple_sim commands            # human-readable list
python -m simple_sim commands --json     # name, script, description, supports_json, exists
python -m simple_sim status --json       # profiles, run configs, datasets (+manifest), models, reports
python -m simple_sim <command> --help    # options of a command
```

Every other command is registered in `simple_sim.cli.COMMANDS`. Library commands (`generate`,
`train`, `eval`, `predict`, `batch-predict`, `sample-check`, `arena-report`, ...) live in
`simple_sim/commands/<name>.py` and run in-process; `scripts/<name>.py` are shims of the same
`main`, so `python -m simple_sim train --data D --out M` equals `python scripts/train.py --data D --out M`.
The helper tools (`validate`, `create-multi-dataset`, the sim-to-real tools, ...) still run
`tools/*.py` as a subprocess. `commands --json` is read from the command sources (descriptions,
`SUPPORTS_JSON`) and is the authoritative list.

To spawn a command from another program use `simple_sim.cli.argv_for("train", data=D, out=M)`
(`[sys.executable, "-m", "simple_sim", "train", "--data", D, "--out", M]`); read the result with
`--json` and `cli_utils.parse_result` / `parse_result_line`, not by scraping `Report saved to:`.

`status` is tk-free and torch-free; use `--root DIR` to inspect another checkout.

## The `--json` result convention

Commands whose `supports_json` is true accept `--json`. They then print exactly one line,
the **last line of stdout**:

```
SIMPLE_SIM_RESULT {"ok": true, "command": "...", ...}
```

- `ok` (bool) and `command` (str, the `python -m simple_sim` subcommand name, e.g. `batch-predict`) are always present; on failure `error` (str) is set and
  the exit code is non-zero.
- All other keys are command-specific - inspect the result line instead of assuming them.
- Other output (progress, tables) is human-oriented; do not parse it.

Read it back in Python:

```python
import subprocess, sys
from simple_sim.cli_utils import parse_result

proc = subprocess.run([sys.executable, "-m", "simple_sim", "validate", "--data", ds, "--json"],
                      capture_output=True, text=True)
res = parse_result(proc.stdout)   # dict or None
```

or in the shell: `... --json | grep '^SIMPLE_SIM_RESULT ' | tail -1 | cut -d' ' -f2-`.

## Typical workflow

```bash
CFG=configs/run_0001.yaml               # see `status --json` -> run_configs
DS=outputs/sim_data/runs/run_0001
M=outputs/models/run_0001.pt

python -m simple_sim generate      --config "$CFG" --out "$DS"            # add --extend to append (refused on renderer-version mismatch unless --allow-renderer-mismatch)
python -m simple_sim validate      --data "$DS" --json
python -m simple_sim train         --data "$DS" --out "$M" --device cpu --json
python -m simple_sim eval          --data "$DS" --model "$M" --device cpu --json
python -m simple_sim batch-predict --data "$DS" --model "$M" --split test --device cpu --json
python -m simple_sim arena-report  --json
```

For a quick experiment copy a run config and shrink it: small `classes:` counts (e.g. 4 per
class), `train.epochs: 1`, `train.pretrained: false` (avoids downloading weights). Write
scratch outputs outside the repo (e.g. `/tmp/...`); `outputs/` is git-ignored.

Optional quality settings (all default to the old behaviour; `configs/run_0001_xdomain.yaml` has
them all): `defects.placement_sampling: continuous`, `defects.units: mm` (tombstone / bridge /
corner-lift shift ranges in mm via `roi.mm_per_px`), several `domains` with a held-out one in
`splits.val_domains`/`test_domains` (per-domain `camera` override), `render.post_filters: camera`,
`train.augment` / `train.scheduler: cosine` / `train.label_smoothing`. CPU training speed:
`train.cache_images: true` (decoded images stay in RAM), `train.freeze: [stem, layer1]` (frozen
ResNet stages, changes the model), `train --threads N` (torch threads; on hybrid CPUs try the number
of performance cores).

Other commands: `generate-profile-dataset` / `train-profile` / `eval-profile` (component
type classifier), `predict` (single image), `predict-two-stage`, `sample-check`,
`create-multi-dataset` (merge datasets), `backfill-manifest` / `backfill-labels` (legacy
datasets), `render-debug-previews`, `update-sample-gallery`. Sim-to-real helpers (real camera
crops vs. the synthetic look): `real-crops`, `look-stats`, `look-montage`, `sim-to-real`,
`calibrate-camera`, `eval-real`, `mix-real` (each also runs as `python tools/<name>.py`).

## Where outputs land

| What | Path |
|---|---|
| Datasets | `outputs/sim_data/runs/<run>/` (`meta.jsonl`, `labels.jsonl`, `images/`, `splits/{train,val,test}.txt`, `config.yaml`, `dataset_manifest.json`) |
| Dataset snapshots | `outputs/sim_data/versions/<name>/<version>/` |
| Checkpoints | path given to `train --out` (plus `<stem>_last.pt`); multi-profile bundles are `*.bundle/` dirs |
| Eval report | `report_<model_stem>.json` next to the checkpoint |
| Batch-predict reports | `<model_dir>/history/batch_report_<stem>_<split>_<timestamp>.json` (or `--out-dir`) |
| Arena | `outputs/models/arena.json` -> `ARENA_REPORT.md` |
| Telemetry (optional) | JSONL events at `$SIMPLE_SIM_EVENT_LOG` if set |

The exact report path is also in the command's result line.

## Evaluating results

- **Datasets:** `validate --json` returns `checks` (each `{name, status: pass|fail|warn, detail}`)
  plus `failed` / `warnings` name lists. `ok` is false iff any check failed.
- **Models:** report JSON files (`report_*.json`, `batch_report_*.json`) share these keys:
  `timestamp`, `model_path`, `dataset_path`, `split`, `seen_samples`, and `metrics` with
  `accuracy`, `macro_f1`, `per_class` (`{class: {precision, recall, f1, support}}`),
  `confusion_matrix`, `critical_fn_rates` (`{class: fn_rate}`).
  Batch reports add `mode: "batch_predict"`, `model_stem`, `elapsed_s`, `img_per_s`,
  `dataset_class_distribution`, `checkpoint_val_accuracy`, ... (and `profile_accuracy` /
  `profile_metrics` with `--profile-model`).
- **Comparing runs:** `status --json` -> `reports.latest` and `reports.best_accuracy`; in
  Python use `simple_sim.reports` (`report_search_dirs`, `load_reports`, `best_report`,
  `metric_value`).
- Built-in success criteria (printed by `eval`): test accuracy > 70 %, see `simple_sim/metrics.py`.

## Tests

```bash
python -m pytest -q   # tkinter-only tests skip without tkinter / an X display
```

`gui/tabs/*/logic.py` import without tkinter (`tests/test_headless_imports.py` keeps it so);
import tab classes from their own modules (`gui.tabs.<name>.tab`), not from `gui.tabs`.

`tests/test_pipeline_e2e.py` runs a tiny generate -> validate -> train -> eval on CPU.

Renderer changes: `tests/test_render_golden.py` (dataset pipeline) and
`tests/test_render_golden_variants.py` (589 cases: every 2D config, class and `render` option) must
stay green; both skip when the OpenCV/NumPy versions differ from the golden files, so check they ran.
An intended look change bumps `RENDERER_VERSION` and regenerates both (`python -m tests.<name> --update`).

Rendering parallelism: `SIMPLE_SIM_RENDER_WORKERS` (count), `SIMPLE_SIM_RENDER_EXECUTOR=thread|process`
(dataset generation picks processes for large runs by itself).

## Code map

- `simple_sim/` - library (no CLI side effects: no `print`/`sys.exit`, progress and warnings via
  `logging` under the `simple_sim` logger, neutral exceptions). Shared helpers: `io_utils`
  (atomic writes, JSON/JSONL/YAML, `now_iso`/`now_tag`), `classes` (`DefectClass`, `Split`,
  `Backend`, class sets), `coerce` (`as_bool`/`as_float`/`as_int`), `cli_utils` (`--json`
  convention, `configure_cli_logging`, `with_hint`), `reports`, `manifest`, `config`, `schema`.
  Model lifecycle: `training` (`train_model`, `train_profile_model`, `make_checkpoint`),
  `evaluation` (`run_eval`, `run_eval_profile`, `write_report`), `inference` (`Classifier`,
  `predict_image`, `sample_check`), `batch_predict` (`BatchPredictOptions`, `run_batch_predict`),
  `two_stage`, `model_store` (snapshot naming). `simple_sim/commands/` - argparse + console output
  of every library command (`print` is allowed there and in `cli.py` only).
  `simple_sim/board/` - PCB model (footprints, nets, router, pours, DRC, raster) behind
  `render.board` (`routing: legacy | none | escape`); `generators/` - the 2D/3D renderers
  (`generators/opencv2d/photoreal/` one module per layer; render one image from a config with
  `opencv2d.RenderSpec.from_run_config(cfg, profile)` + `render_sample`; a whole dataset from a config
  dict with `generation.dataset.generate_dataset(cfg, out_dir, image_filters=...)`; backend capabilities in
  `generators/backends.py`).
- `scripts/` - shims of `simple_sim/commands/` (kept for `python scripts/<name>.py`); `tools/` - dataset maintenance CLIs, `tools/_common.sh`
  (venv/python detection for the `*.sh` launchers), `tools/profile_editor/` (Tk GUI).
- `gui/` - Tk monitor GUI (`./run.sh`). `configs/` - run configs and `profiles/`.
