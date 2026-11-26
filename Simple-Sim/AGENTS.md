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

Part catalog (`simple_sim.parts`, `python -m simple_sim parts list|lookup|check|add`): datasheet geometry of real
parts (body l / w / **height above the board**, kind, orientation, material, source url + `datasheet|distributor|estimate`),
shared by every tool. Layers: local files (`$SIMPLE_SIM_PARTS_DB`, `os.pathsep`-separated, else `outputs/parts/catalog.json`;
board- and library-specific data lives there and stays off GitHub) over `simple_sim/parts/builtin.json` (published: generic
parts from public datasheets only, e.g. JST series and 2.54 mm headers); match by MPN,
then package name, then regex. `load_board3d` applies it (`board3d["partCatalog"]`); the viewer re-exports its GLB when
the applied heights change. Use `PartCatalog.load().lookup(texts=[value, part], package=pkg)` in generators too.

Real SMD lands (`python -m simple_sim board3d-bare --board3d <dir> --photo <bare photo or run dir> ...`,
`simple_sim/board3d/barepads.py`): without CAD footprints the pads are rule-of-thumb estimates; a phone photo of the
unpopulated board side is registered to board mm (mounting holes -> land ICP -> dense match against
`substrate/scan_rgb.png`) and the land under every estimated pad is measured (marker watershed, half-contrast edge).
Writes `<dir>/bare/lands.json` (board frame `[x, y, w, h, rotDeg]` per pad index, `null` = keep the estimate),
`registration.json`, `bare_rgb.png`, `lands_mask.png`. `load_board3d` puts the lands on the components (`c["lands"]`)
and `footprint_for` places them relative to the current, possibly fitted pose: lands belong to the board, not the
part. A photo of the other board side, or one that registers worse than `--max-residual`, is dropped.

Other commands: `generate-profile-dataset` / `train-profile` / `eval-profile` (component
type classifier), `predict` (single image), `predict-two-stage`, `sample-check`,
`create-multi-dataset` (merge datasets), `backfill-manifest` / `backfill-labels` (legacy
datasets), `render-debug-previews`, `update-sample-gallery`. Sim-to-real helpers (real camera
crops vs. the synthetic look): `real-crops`, `look-stats`, `look-montage`, `sim-to-real`,
`calibrate-camera`, `eval-real`, `mix-real`; DeepVision app helpers: `dv-crops` (reviewed app crops),
`board-backgrounds` (board textures from stills), `export-deepvision` (reference / probe pairs for the
judge). Each also runs as `python tools/<name>.py`.

## Where outputs land

| What | Path |
|---|---|
| Datasets | `outputs/sim_data/runs/<run>/` (`meta.jsonl`, `labels.jsonl`, `images/`, `splits/{train,val,test}.txt`, `config.yaml`, `dataset_manifest.json`) |
| Dataset snapshots | `outputs/sim_data/versions/<name>/<version>/` |
| Checkpoints | path given to `train --out` (plus `<stem>_last.pt`); multi-profile bundles are `*.bundle/` dirs |
| Eval report | `report_<model_stem>.json` next to the checkpoint; for a bundle dir `report_<bundle_stem>_<profile>.json` next to the dir |
| Batch-predict reports | `<model_dir>/history/batch_report_<stem>_<split>_<timestamp>.json` (or `--out-dir`) |
| Arena | `outputs/models/arena.json` -> `ARENA_REPORT.md` |
| Telemetry (optional) | JSONL events at `$SIMPLE_SIM_EVENT_LOG` if set |
| Part catalog (local) | `outputs/parts/catalog.json` or `$SIMPLE_SIM_PARTS_DB` (git-ignored); generic public parts in `simple_sim/parts/builtin.json` |

The exact report path is also in the command's result line.

## Production data stays local (3D-Board-Renderer and everything else)

Only the **tools** go to GitHub (code, `web/board3d/` viewer, vendored three.js, tests with synthetic data). Never
commit, push or paste into code, tests, docs, commit messages or PR texts anything from the real boards:

- board data: scans, phone photos, the photo mosaic, GLB exports, `board3d.json`, renders, defect crops, datasets
  (all under `outputs/`, git-ignored: `outputs/board3d/<product>/<side>/`, `outputs/sim_data/`, ...);
- identifying details: customer, product and PCB names, part numbers, serials / MAC labels, DeepVision run ids,
  store or network paths (`Database/...`, `store/...`, `testing-content/...`), PnP contents, part counts of a
  product;
- the 3D-Board-Renderer plan with its measurements (`BACKLOG-Sim/board3d-plan.md`, git-ignored).

So: no product defaults in code or scripts (`PRODUCT=<id> ./3d-board-viewer.sh`, `--product` is required), examples
use `<product>`, comments say "the first product", per-dataset settings live next to the data (e.g. the mosaic's
excluded runs in `outputs/board3d/<product>/<side>/photos/exclude.txt`). Before a commit check
`git status --untracked-files=all` and `git diff --cached` for such content; earlier local commits are checked too
before a push (`git log origin/<branch>..`).

Forbidden words (company, customer and product names) are blocked mechanically: `python tools/forbidden_words.py
--install-hooks` installs pre-commit / commit-msg / pre-push hooks that refuse any commit or push whose added lines,
file names, messages or author fields contain one of them; `--tree` scans all tracked files. The word list is
`<repo>/.forbidden-words` (git-ignored, one word per line) or `$FORBIDDEN_WORDS_FILE` - never write the words into
tracked files, not even into this guide or a test. Without the list the hooks refuse, so create it on every clone.
`board3d.json` paths into the DeepVision store follow `$SIMPLE_SIM_DV_ROOT` (default: `../../DeepVision-Web` next to
the checkout), so a board3d folder copied to another machine still finds its sources.

## Commits and pull requests

Never add AI attribution: no `Co-Authored-By: Claude ...` trailer, no "Generated with Claude Code" line, no
mention of Claude / an AI assistant as author in commit messages, PR titles or PR bodies. Commits carry the
user's git identity only.

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
