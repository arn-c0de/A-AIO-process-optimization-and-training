# Library Modernization Plan

Goal: make `simple_sim/` a clean, installable, headless library that other programs can
`import` and call (render one image, generate a dataset from an in-memory config, train,
evaluate, predict, validate, merge, inspect reports) without the CLI scripts, the Tk GUI,
a repo checkout or environment variables. Scripts, tools and GUI become thin consumers.

Baseline: branch `1.0.3`, HEAD `1d3ef37`, 2026-11-06. Read-only audit of all 43k lines
(`simple_sim/` 11k, `gui/` 15k, `scripts/` + `tools/` 5k, tests 4k). Companion plans:
[GUI_MODERNIZATION_PLAN.md](GUI_MODERNIZATION_PLAN.md) (GUI internals),
[PCB_TRACE_ROUTING_PLAN.md](PCB_TRACE_ROUTING_PLAN.md) (board model),
[summary.md](summary.md) and [gui_utils_refactor.md](gui_utils_refactor.md) (2026-02 pass).

Hard constraint for everything below: legacy 2D rendering stays **bit-identical**
(`tests/test_render_golden.py`, `routing: legacy`). Any change that alters the order or
count of `rng` draws, integer casts around cv2 calls, or float32/float64 colour constants
is a behaviour change, not a refactor.

## 1. Measured baseline

| Fact | Evidence |
|---|---|
| No `pyproject.toml`, no `requires-python`, no console script | only `requirements.txt`; code already needs 3.10+ (`X \| Y` in dataclass fields, `slots=True`) |
| 28 `sys.path.insert` hacks | every file in `scripts/`, `tools/`, 4 tests, `gui/monitor.py`, `gui/components/filter_popup/preview_panel.py:258` |
| `scripts/` and `tools/` are imported as packages without `__init__.py` | `scripts/train_profile.py:27`, `scripts/eval_profile.py:26`, `gui/utils/validation_suite.py:13`, `tests/test_eval_utils.py:114-152`, `tests/test_profile_editor_tools.py:12-16` |
| `python -m simple_sim <cmd>` spawns a new interpreter per command and sniffs `--json` support by `"add_json_flag" in src` | `simple_sim/cli.py:75,264`; locked in by `tests/test_cli.py:38-42` |
| Typing is modern | 0 uses of `typing.Optional/List/Dict`, 2 `os.path`, `from __future__ import annotations` almost everywhere |
| Logging is not | 112 `print()` in `simple_sim/`, `logging` in 2 modules (`config.py`, `gui/utils/*`) |
| Import cost | `import simple_sim.board` 0.70 s (cv2 via `occupancy/zones/drc/raster`); `simple_sim.generators` pulls cv2 + `subprocess` through star imports; `simple_sim.filter_profiles` pulls cv2; `simple_sim.schema` pulls PyYAML for a dataclass; `simple_sim.metrics` pulls scikit-learn |
| No `gui.tabs.*.logic` module is importable without tkinter | `gui/tabs/__init__.py:3-12` and `gui/tabs/datasets/__init__.py:1` eagerly import every `tab.py`; `gui/components/filter_popup/__init__.py:6` likewise (verified with a tk-blocking import hook) |
| Library/tools import the GUI | `simple_sim/preview/viewer_tk.py:215`, `tools/update_sample_gallery.py:33`, `tools/profile_editor/app.py:22,34,415`, `tools/profile_editor/system_monitor.py:5` |
| Public API | `simple_sim/__init__.py` exports `__version__` only; no `__all__` outside `generators/` and `board/` |

The board model (`simple_sim/board/`) is already the reference for what the rest should
look like: frozen/slots dataclasses, `__all__`, a `Router` Protocol plus `ROUTERS` registry,
no imports from `generators/` or `generation/`.

## 2. Behavioural divergences found on the way

These are not style issues. Each one means two code paths that are supposed to agree do
not, and each disappears when the duplicate is deleted.

| # | Divergence | Where |
|---|---|---|
| X1 | GUI single-image inference inserts `Resize((256,256))`; `predict`, `batch-predict`, `two_stage` and training (`ROIDataset`) do not resize | `gui/utils/model_inference.py:20-25` vs `simple_sim/data_loader.py:22-39` |
| X2 | Filter-popup preview baseline is brightness 1.1 / contrast 1.1; the shared preview baseline is 1.15 / 1.20 | `gui/components/filter_popup/preview_render.py:21-27` vs `simple_sim/preview/render.py:320-329` |
| X3 | GUI edge-case checks hard-code OK shift 3 px, rotation 5°, tombstone 75° instead of the dataset profile's `tolerances` | `gui/utils/validation_suite.py:18-20` |
| X4 | `train_profile` checkpoints omit `component_profile`, so `manifest.check_checkpoint_profile` treats them as legacy | `scripts/train_profile.py:107-118` vs `scripts/train.py:390-407` |
| X5 | Flat render style calls `solder_defect_layout` outside the supersampling context, so it silently uses `_ss() == 2` at camera resolution | `simple_sim/generators/opencv2d/render.py:110-123`, `photoreal.py:515` (must be preserved as-is for pixel identity) |
| X6 | Two snapshot naming schemes write into the same `outputs/models/versions/<stem>/` | `gui/tabs/pipeline/logic.py:388` (`<stem>_<ts>_run<NNN>.pt`) vs `gui/tabs/weights/logic.py:121` (`<stem>_<ts>.pt`) |
| X7 | Predictions tab reads only manifest v1 `component_profile`; v2 manifests give an empty profile id | `gui/tabs/predictions/logic.py:84-98` vs `manifest.resolve_dataset_profile` |
| X8 | `fine_pitch` defined twice with different family sets; a third "chip vs fine pitch" definition in photoreal | `board/footprints.py:190` (`soic, qfn`, unused), `board/routing/escape.py:30-34`, `photoreal.py:1224-1232` |
| X9 | `tools/sim_to_real.py` and `tools/calibrate_camera.py` use `--json PATH` (a file) against the project-wide result-line convention; neither is in `cli.COMMANDS` | `tools/sim_to_real.py:99`, `tools/calibrate_camera.py:108` |
| X10 | Fast-3D preview derives pad layouts itself (qfn as 4 pads) instead of `board.footprints.footprint_pads` | `tools/profile_editor/preview_3d.py:374-523` |
| X11 | GUI arena favourites key by relative path, `simple_sim.arena` by absolute path; GUI report generation spawns system `python3` instead of the venv | `gui/tabs/weights/logic.py:43-110` vs `simple_sim/arena.py:61-80` |
| X12 | 2D renderer constants are re-typed in the Blender script and already drift (thickness 0.55 vs 0.56, paste dome exponent 3 vs 2.5 in `solder.py`) | `photoreal.py:111,124,126,1033` vs `blender/render_batch.py:36-38,258`; `opencv2d/solder.py:148` |

## 3. Findings

IDs: **P** packaging, **I** imports/headless, **L** logic in the wrong layer, **D**
duplication, **R** renderer structure, **A** API/typing, **M** modernization. Effort
S < 0.5 d, M 0.5-2 d, L > 2 d.

### P. Packaging and entry points

- **P1** Add `pyproject.toml`: `requires-python = ">=3.10"`, dependencies from
  `requirements.txt` with extras `[train]` (torch, torchvision, scikit-learn), `[gui]`
  (pillow, matplotlib), `[dev]` (pytest), and `[project.scripts] simple-sim = "simple_sim.cli:main"`.
  `pip install -e .` makes every `sys.path.insert` unnecessary. S.
- **P2** Command registry instead of subprocess dispatch. `simple_sim/commands/<name>.py`
  each exposing `main(argv: list[str] | None = None) -> int`, `DESCRIPTION`,
  `SUPPORTS_JSON`; `cli.COMMANDS: dict[str, CommandSpec]` with lazy `importlib.import_module`.
  `scripts/*.py` and `tools/*.py` become 3-line shims so subprocess-based tests
  (`tests/test_arena_report.py:119`, `tests/test_preview_render.py:67,76`), the GUI call
  sites and ~42 doc mentions keep working. Fixes the inconsistent `main()` signatures
  (argv vs none vs returning int, see `scripts/*.py` and `tools/backfill_manifest.py:164`)
  and the underscore/hyphen command-name mismatch papered over by
  `cli_utils.emit_result` (`cli_utils.py:37`). M. Risk: `tests/test_cli.py:38-42` must change.
- **P3** Shell launchers: `predict.sh` and `sample_predict.sh` are byte-identical except
  the script name; `tools/update_sample_gallery.sh` and `tools/run_profile_editor.sh`
  repeat the pattern. One `exec "$PYTHON" -m simple_sim "$@"` replaces all four.
  `run_pipeline.sh:73-99` embeds a 25-line Python heredoc that recomputes
  `manifest.dataset_stats`; `:141` rebuilds the eval report path by string surgery although
  `eval --json` returns it. `run_render_debug.sh:44-87` probes four modules via heredocs.
  Caveat: the GUI drives `run_pipeline.sh` through env vars (`gui/tabs/pipeline/logic.py:198,24-27`),
  so a Python `pipeline` command must accept the same env. S each.
- **P4** `gui/utils/pipeline_helpers.py:19` and `gui/tabs/predictions/logic.py:101`
  hard-code `./.venv/bin/python`; `gui/tabs/weights/logic.py:94-110` spawns system
  `python3`. Use `sys.executable` (the GUI already runs inside the venv). S.

### I. Import hygiene and headless importability

- **I1** `gui/tabs/__init__.py`, `gui/tabs/datasets/__init__.py`,
  `gui/components/filter_popup/__init__.py`: empty them or use lazy `__getattr__` (pattern
  already in `gui/utils/__init__.py`). Only consumer of `gui.tabs` is `gui/monitor.py:40`
  for `BaseTab`. Makes 8 of 9 `logic.py` modules and `tests/test_profile_preview_logic.py`
  tk-free; update the AGENTS.md test note. S, do first.
- **I2** `simple_sim/generators/__init__.py:3-4` star-imports both backends; delete the
  four compat shims `generator_2d.py`, `generator_3d.py`, `generators/opencv_2d.py`,
  `generators/blender_3d.py` (users: `gui/tabs/profile_preview/logic.py:186`,
  `gui/components/filter_popup/preview_render.py:41-81`, `tools/profile_editor/blender_live_preview.py:13`,
  `tools/update_sample_gallery.py:38-39`, tests). S.
- **I3** Move `generators/filter_settings.py` (pure json/os) to `simple_sim/filter_settings.py`
  so `filter_profiles`, `generation/common` and `gui/utils/pipeline_helpers` stop loading cv2. S.
- **I4** `simple_sim/board/__init__.py:21-29` imports everything eagerly; the data model
  (`model`, `footprints`, `netlist`, `rules`, `geometry`, `placement`, `routing/base`) is
  cv2-free. Lazy submodule attributes (PEP 562) or split `board/pipeline.py`. S.
- **I5** `eval_utils.py:16-24` top-imports torch/tqdm but holds torch-free helpers
  (`MANIFEST_NAME`, `metrics_summary`, `require_manifest`, `dataset_profile`,
  `resolve_bundle_checkpoint`, `checkpoint_eval_config`, `checkpoint_val_accuracy`).
  Move them to `manifest.py` / `model_bundle.py`. S.
- **I6** `schema.py:13` imports `simple_sim.config` (PyYAML) for `VALID_CLASSES`;
  `metrics.py:9-10` needs scikit-learn for three functions that are trivial in numpy.
  New dependency-free `simple_sim/classes.py` (see A3); numpy metrics with exact-value
  tests. S + M.
- **I7** `config.validate_config` lazily imports `simple_sim.board` and renderer modules
  (`config.py:304,314,323,333,343`) to fetch option constants, so validating YAML loads cv2.
  Move all `render.*` option constants and validators into a light
  `generators/options.py` (camera presets, supersampling choices, solder styles, texture
  dir). S.
- **I8** GUI heavy imports at module level: `gui/tabs/pipeline/logic.py:18 import cv2`
  (only for merge resize), `gui/tabs/merge/logic.py:10 import torch`,
  `gui/tabs/weights/logic.py:22` (torch). Lazy-import inside functions. S.
- **I9** `simple_sim/preview/viewer_tk.py` (602 lines, a full Tk app) lives in the library
  and imports `gui.components.filter_popup`. Move to `gui/preview_viewer.py`. S.
- **I10** `simple_sim/filter_profiles.py` reads/writes `outputs/gui/settings.json` with
  GUI key names; `filter_settings.default_filter_values_for_popup` /
  `clean_filter_values_for_popup` encode popup strings; `load_image_filters_env` reads
  `IMAGE_FILTERS` and prints. Keep `normalize_image_filters` + `NEUTRAL_AUGMENT` in the
  library, move the rest to `gui/`, call `load_image_filters_env` at the CLI boundary only. S.

### L. Logic in the wrong layer

Library functions that currently live in scripts, tools or the GUI. Target module names
are proposals; the point is one implementation per concern, importable without Tk, argparse
or a checkout.

| ID | What | Today | Target | Effort |
|---|---|---|---|---|
| L1 | Training loop, checkpoint dict, best-F1 tracking, `_last.pt` naming | `scripts/train.py:41-432` (229-line `train_model`, ~35 prints), `scripts/train_profile.py:38-145` imports from it | `simple_sim/training.py`: `train_model(TrainConfig) -> TrainResult`, `make_checkpoint(...)` (fixes X4) | M |
| L2 | Eval loop and report dict, written three times | `scripts/eval.py:52-143`, `scripts/eval_profile.py:29-86` (structurally identical), `scripts/batch_predict.py:438-464` | `simple_sim/evaluation.py`: `evaluate_model`, `build_report`, `write_report` (the literal `Report saved to:` is parsed by the GUI, keep it in one place) | M |
| L3 | Batch prediction behind `run(args: Namespace)` | `scripts/batch_predict.py:349-505`; `predict_once` re-implements `eval_utils.run_inference` with a hand-rolled EMA; `resolve_model_path` = `eval_utils.resolve_bundle_checkpoint` | `simple_sim/batch_predict.py` with `BatchPredictOptions` dataclass; `eval_utils.oom_retry` | L |
| L4 | Dataset validation checks; `_Checks.add` prints as it records | `tools/validate_dataset.py:29-230`; GUI imports it from `tools.` (`gui/utils/validation_suite.py:13`) | `simple_sim/validation.py`: `run_validation(data_dir, on_check=None) -> ValidationResult`; extra checks (outliers, duplicates, edge cases, taking `tolerances`, fixes X3) in `simple_sim/quality.py` | M |
| L5 | Dataset merge, implemented twice with different features | `tools/create_multi_dataset.py:62-266` (8 `raise SystemExit` inside compute), `gui/tabs/pipeline/logic.py:591-793` (adds ROI harmonisation, hardlinks, class-set check; non-atomic split files) | `simple_sim/dataset_merge.py: merge_datasets(sources, out, *, resize_to_common_roi, link_images) -> MergeResult` raising `UsageError` | M-L |
| L6 | Dataset editing (list/delete/move samples with rollback, recompute stats) | `gui/utils/dataset_ops.py` (tk-free) | `simple_sim/dataset_edit.py`; make `splits.write_splits` atomic and delete `_write_split_files` | M |
| L7 | Single-image prediction | `scripts/predict.py:59-123`, `gui/utils/model_inference.py` (X1), `simple_sim/two_stage.py:73-86` | `simple_sim/inference.py: Classifier` (load once, `predict_bgr`, `predict_path`); decide the Resize question explicitly | S |
| L8 | Checkpoint metadata and mtime cache | `gui/utils/checkpoint_meta.py`; `model_bundle._checkpoint_profile_id` does a full `torch.load` per `.pt` (`model_bundle.py:103-112`) | `simple_sim/checkpoints.py` (or `models.py`); profile id from `bundle.json` sidecar | S |
| L9 | Output layout and discovery (`outputs/models`, runs/versions, profile classifiers) | `gui/utils/paths.py`, `gui/utils/dataset_catalog.py:47-88`, `cli.py:125-128,159-169`, `reports.py:22-33`, `arena.py:39-45`; `"outputs" / "models"` literal 18x in gui, `"configs" / "profiles"` 12x | `simple_sim/layout.py` with `sim_root(explicit=None)` (arg > `SIMPLE_SIM_ROOT` env > walk-up), `models_root`, `runs_root`, `profiles_dir`, `list_checkpoints`, `list_datasets` | M |
| L10 | Profile / run-config discovery, six implementations | `preview/render.py:156-185` (canonical), `gui/tabs/pipeline/logic.py:485-579`, `gui/tabs/profile_preview/logic.py:111-124`, `tools/profile_editor/profile_registry.py` (parses each YAML up to 3x per poll), `tools/update_sample_gallery.py:46-118`, `cli.py:196-197` | `simple_sim/catalog.py` (`list_profiles(backend=None)`, `run_configs_for(profile_id)`, mtime-cached) | M |
| L11 | Dataset info readers, five of them; 16 files open `dataset_manifest.json` by hand | `gui/tabs/datasets/logic.py`, `gui/tabs/weights/logic.py:468-515`, `cli.py:131-156`, `gui/tabs/predictions/logic.py:242-263`, `scripts/batch_predict.py:68-94,132-137` | `simple_sim/dataset_index.py: DatasetIndex(data_dir)` (`meta_by_id`, `label_by_id`, `image_path`, `gt_profile_map`, counts) built on a typed `DatasetManifest` (A2) | S-M |
| L12 | Model store ops (snapshot, import zip, export, activate, delete, rename, duplicate) | `gui/tabs/weights/logic.py:112-272,383-391`, `gui/tabs/pipeline/logic.py:368-414` (X6) | `simple_sim/model_store.py`; one snapshot naming scheme | M |
| L13 | Arena favourites | `gui/tabs/weights/logic.py:43-110` (X11) | `simple_sim.arena.add_models/remove_models`; GUI calls `write_arena_report` via TaskRunner | S |
| L14 | Bundle build and ensemble writer | `gui/tabs/merge/logic.py:95-197` (hard-codes `"simple_sim_ensemble_v1"` instead of `models.ENSEMBLE_FORMAT`) | `model_bundle.build_bundle/build_ensemble`; enables a `merge-models` command | M |
| L15 | Flags / feedback sidecars (`flags.jsonl`, `feedback.jsonl`, `effective_label`) | `gui/utils/flag_manager.py`, `gui/utils/feedback_manager.py` (non-atomic save) | `simple_sim/annotations.py` with `io_utils.atomic_write_text` | S |
| L16 | Defect overlay drawing | `gui/components/overlay_renderer.py`, imported by `tools/update_sample_gallery.py:33` | `simple_sim/overlay.py`, lazy re-export in `gui/components` | S |
| L17 | Sample gallery builder re-implements `preview.render` (profile discovery, ROI regex, seed, 2D render, Blender job building, batching) | `tools/update_sample_gallery.py:46-523` (one 245-line `_run(args)`) | `simple_sim/gallery.py: build_gallery(GalleryOptions)` on top of `preview.render`; keep the gallery's own seed derivation if `SAMPLE_GALLERY` images must stay identical | L |
| L18 | Profile preview renderer with variants/cancellation | `gui/tabs/profile_preview/logic.py` (imports private `_preview_seed`); `tools/profile_editor/blender_live_preview.py:72-170` builds Blender jobs by hand with the legacy facade API | `simple_sim/preview/gallery.py`; make `preview_seed` public; `render_debug_previews` shares it | M |
| L19 | Pipeline planning and command-line building | `gui/utils/pipeline_helpers.py:57-222,360-418` (tk-free; hard-codes script paths that `cli.COMMANDS` owns) | `simple_sim.cli.argv_for("train", ...)` + `simple_sim/pipeline_plan.py` | M |
| L20 | Subprocess streaming with CR flush and tqdm throttling, events.jsonl tailer | `gui/utils/process.py`, `gui/tabs/pipeline/logic.py:233-366` (second implementation) | `simple_sim/proc.py`; `generators/blender3d/runner.py` can share it | M |
| L21 | System stats sampler | `gui/utils/system_stats.py`, wrapped by `tools/profile_editor/system_monitor.py` | `simple_sim/sysinfo.py`; delete the wrapper | S |
| L22 | Preview request resolution with `input()` prompts and 9 `raise SystemExit` in the flow | `scripts/render_debug_previews.py:155-281` | `preview/select.py: resolve_preview_request(...)`; prompts injected as callbacks | S-M |
| L23 | Already-clean functions in the wrong place | `tools/backfill_labels_v2.py:26-83`, `scripts/sample_check.py:31-111`, `tools/backfill_manifest.py:41-161` (returns `"ok\|skipped\|failed"` strings and prints; builds the manifest by hand instead of `write_dataset_manifest`), `tools/sim_to_real.py`, `tools/calibrate_camera.py` (X9) | `simple_sim/labels.py`, `simple_sim/inference.py`, `manifest.backfill_manifest(...) -> BackfillResult`, `simple_sim/realism/` | S each |

### D. Duplication catalogue (library-internal and cross-layer)

Each row is one helper to write and N copies to delete.

| ID | Helper | Copies |
|---|---|---|
| D1 | `resolve_device(spec) -> torch.device` + `default_device_name()` | `scripts/train.py:223,452`, `train_profile.py:44,152`, `eval.py:69,158`, `eval_profile.py:33,93`, `predict.py:87,149`, `batch_predict.py:352,513`, `sample_check.py:74,131`, `gui/utils/model_inference.py:39`, `two_stage.py:51` (`predict_two_stage.py:415` defaults to cpu, inconsistent) |
| D2 | argparse fragments `add_data_arg/add_model_arg/add_out_arg/add_amp_flags/add_profile_mismatch_flag` in `cli_utils` (torch-free) | `--amp/--no-amp` group `train.py:454-457` = `train_profile.py:154-157`; `--allow-profile-mismatch` `train.py:440-447` = `eval.py:150-157`; `--data/--model/--out` in 10 scripts |
| D3 | `rel_to(root, p)` / `abs_from(root, p)` | `io_utils.py:99-104`, `generation/common.py:27-32`, `model_bundle.py:177-180`, `generation/profile_dataset.py:71` (no fallback, raises for a profile outside root), `arena.py:48-58`, `gui/tabs/pipeline/tab.py:584,1314`, `gui/tabs/predictions/tab.py:90,107` |
| D4 | `class_counts(label_rows)` | `manifest.py:67-72`, `generation/common.py:180`, `scripts/batch_predict.py:84`, `gui/utils/dataset_ops.py:227` |
| D5 | Manifest history / generator blocks | `manifest.py:146-153`, `:185-201`, `:390-412` (three hand-built copies in one file) |
| D6 | Manifest v1/v2 (`component_profile` vs `component_profiles`) branching | `manifest.py:207-219,257-281`, `cli.py:142-145`, `generation/dataset.py:69-111`, `scripts/train.py`, `scripts/train_profile.py`, `tools/create_multi_dataset.py`, `gui/utils/pipeline_helpers.py`, `gui/tabs/pipeline/logic.py:421-440`, `gui/tabs/predictions/logic.py:84-98` (X7) |
| D7 | Dataset file-name constants (`dataset_manifest.json`, `meta.jsonl`, `labels.jsonl`, `config.yaml`) | `manifest.py:119,379`, `cli.py:133`, `generation/dataset.py:120-123`, `eval_utils.py:26`, `dataset_store.py:25`, `data_loader.py:85-86`, +20 in gui/scripts/tools |
| D8 | `config.yaml` loader for a dataset dir | `scripts/train.py:133-136`, `tools/create_multi_dataset.py:150-155`, `tools/backfill_manifest.py:76`, `cli.py:112` |
| D9 | Bundle checkpoint listing (`*.pt` minus `_last`) | `model_bundle.py:122-126`, `cli.py:164-168`, `arena.py:114-116` |
| D10 | Bundle checkpoint resolution with the identical error text "Multi-model bundle has no checkpoint for profile" | `eval_utils.py:232-247` (canonical), `scripts/batch_predict.py:109-129`, `gui/tabs/weights/logic.py:281-301`, `gui/tabs/predictions/logic.py:83-131` |
| D11 | Profile YAML parsed twice per load (`load_profile` then `hash_profile` re-reads) | `generation/dataset.py:237-239`, `profile_dataset.py:64-69`, `profile_hash.py:75-76,112-113`; add `hash_profile_dict` |
| D12 | Split-name constant | `generation/common.py:19`, `schema.py:47`, `config.py:124`, `splits.py:56`, `scripts/batch_predict.py:74,98,101`, `tools/validate_dataset.py:145`, gui x5 |
| D13 | Defect-class string sets | `config.py:37`, `defects.py:68,105-115`, `metrics.py:12`, `opencv2d/draw.py:199`, `opencv2d/render.py:17`, `preview/render.py:32`, `tools/update_sample_gallery.py:43`, gui x6 |
| D14 | Numeric/bool coercion `_to_bool/_to_float/_to_int` | `arena.py:83-87`, `filter_profiles.py:35-43`, `generators/filter_settings.py:134-158`, `gui/components/filter_popup/models.py:12-29` |
| D15 | Timestamps (naive local ISO, UTC `Z`, `%Y%m%dT%H%M%SZ`, `%Y-%m-%d %H:%M:%S`) | `manifest.py:147,176,194,391,395`, `model_bundle.py:132,154`, `arena.py:421,601,622`, `io_utils.now_tag:46`, `scripts/eval.py:119`, `eval_profile.py:62`, `batch_predict.py:439`; `datetime.utcnow()` (deprecated 3.12) at `gui/tabs/pipeline/logic.py:531`, `gui/tabs/merge/logic.py:144,191` |
| D16 | SHA-256 with `sha256:` prefix | `manifest.hash_file:22-43`, `profile_hash.hash_profile:54-86`, prefix stripped at `gui/utils/checkpoint_meta.py:29` |
| D17 | `format_bytes` | `io_utils.py:61`, `gui/utils/formatting.py:8` |
| D18 | JSON/JSONL I/O without the io_utils helpers | `manifest.py:124-125,240-241` (`json.load(open(path))`, no encoding), `schema.py:106-125` (own atomic write with `.tmp` suffix), `schema.py:150` |
| D19 | YAML readers | `config.load_config:48-67`, `profile_hash.load_profile:112-113`, `cli.py:112`, `tools/profile_editor/yaml_io.load_yaml` |
| D20 | Filter-setting key specs, three descriptions of the same keys | `generators/filter_settings.py`, `gui/utils/filter_profile_store.py:52-146`, `gui/components/filter_popup/models.py:37-99`; `gui/tabs/pipeline/tab.py:438-472` re-implements `filter_profiles.load/save_shared_filter_profiles` |
| D21 | Image-folder loader (recursive glob + random subset) | `tools/sim_to_real.py:38-50`, `tools/calibrate_camera.py:40-46` |
| D22 | Output scraping instead of `--json` ("Report saved to:", "Predictions saved to:", `newest_file("batch_report_*.json")`) | `gui/tabs/weights/logic.py:326-377`, `gui/tabs/predictions/logic.py:154-183`, `gui/tabs/predictions/tab.py:198-203`; AGENTS.md says not to parse human output |

Renderer-internal duplicates (all pixel-safe if the single rng draw is kept):

| ID | Helper | Copies |
|---|---|---|
| D23 | `_smooth_noise` | `photoreal.py:184-188` == `opencv2d/solder.py:96-99` |
| D24 | heightmap normals + mirror shading | `photoreal.py:159-162` == `solder.py:193-197` |
| D25 | uint8 clip | `photoreal.py:131-132`, inline at `opencv2d/filters.py:22,28,34,163`, `draw.py:63` |
| D26 | rotated-rect SDF / local coords | `photoreal.py:710-720` (used 5x) vs ad-hoc `np.mgrid` at `photoreal.py:241,264,465,536,673`, `solder.py:105,147`, `blender/render_batch.py:147-151` |
| D27 | Legacy copper vs board model: 45° jog polyline and chamfered pour octagon with the same random ranges | `photoreal.py:347-352` vs `board/routing/base.py:63-69`; `photoreal.py:337-342` vs `board/zones.py:23-40` (int cast in `_polyline` must stay) |
| D28 | Family tuples and refdes prefixes | `photoreal.py:119,121,408,974,1224` vs `board/footprints.py:25`; `photoreal.py:112-114` vs `board/placement.py:17-18` |
| D29 | mm/px conversion | `photoreal.py:1268,1326,1337,1342`, `board/footprints.py:169,182`, `board/occupancy.py:45-53`, `board/raster.py:33-35`, `render_batch.py:225,434,445` |
| D30 | Render defaults applied in several layers (`rework_artifacts`, `mask_gloss`, `supersampling`, `component_type`, `footprint`, Blender settings) | `photoreal.py:57,1277,1282,1314,1368`, `bake.py:28,44-45,67-69,80`, `render.py:134`, `draw.py:11-21`, `generation/blender.py:32-40,176`, `preview/render.py:373,419,627-630`, `render_batch.py:47-48` |
| D31 | Colour jitter `rng.uniform(0.9x, 1.0x)` (11 sites) and grain `rng.normal(0, sd, shape)` (5 sites) | `photoreal.py:235,245,441,692-694,815,886,889,954-958,972,988,994,1108,1278` |
| D32 | Preview augment sampling re-implements `augment.py:168-176` and hard-codes the neutral subset | `preview/render.py:295-330`, `gui/components/filter_popup/preview_render.py:21-27` (X2) |
| D33 | Pour raster resolution | `board/zones.py:130` (`clearance / 4.0`) vs `board/drc.py:31` (`POUR_CELLS_PER_CLEARANCE = 4`) |
| D34 | `Vec`/`BBox` aliases; 4-tuple bbox spelled out 14x; vector helpers in `routing/base.py:50-60` imported by `zones.py:18` (layer inversion) | `board/model.py:16`, `board/geometry.py:14`, `routing/base.py:72-101`, `escape.py:37-60` |
| D35 | 2D constants re-typed in the Blender script (X12) | see X12; bake the per-part decisions into the scene dict so `render_batch.py` only builds geometry |

### R. Renderer structure

- **R1** No "render one image from a config" entry point. `render_roi(nominal, defect_params, augment, roi_size, config, tolerances, rng, footprint)` (`opencv2d/render.py:126-157`) needs a config dict hand-merged by `config.merge_profile_render` (`config.py:19-35`); every caller re-assembles it (`generation/common.py:35-77`, `preview/render.py:370-450`, `gui/components/filter_popup/preview_render.py:41-81`, `tools/update_sample_gallery.py:39`). Add a frozen `RenderSpec` with `from_run_config(cfg, profile)` and `render_sample(spec, nominal, defect, augment, rng) -> ndarray`; `render_roi` stays as an adapter. M.
- **R2** Thread-local hidden state (`photoreal.py:35-88`: `_TLS.factor`, `_TLS.solder_look`, read via `_ss()` at 67 sites) plus a module `__getattr__` hack for `SS` kept for tests. Replace with an explicit `RenderContext(ss, solder_look, light)` parameter after the split; pass `ss=2` explicitly in the flat path to preserve X5. L, mechanical.
- **R3** Backend dispatch is string `if/else` in three packages (`generation/dataset.py:233,339,358-359,404-421`, `config.py:278-283`, `preview/render.py:29-31,477,557`, `generation/profile_dataset.py:58-60`). `generators/backends.py` with `Backend(StrEnum)` and a `RenderBackend` Protocol registry, modelled on `board/routing/__init__.py:8`. M.
- **R4** `generation/dataset.py:generate_dataset` (`:204-484`, one 280-line function, 28 prints) takes a config *path*, hard-codes `profiles_dir = root/"configs"/"profiles"` (`:236`), reads `IMAGE_FILTERS` env (`:225`), bakes in tqdm and telemetry, and embeds CLI wording in exceptions (`:124-130`). Split into `plan_dataset(config, profile) -> list[Record]` (pure), `render_records(backend, records, spec, out_root)`, `write_dataset(...)`; `generate_dataset(config: dict, output_dir, *, profiles_dir, image_filters=None, progress=None)` composes them. Keep the seed derivation (`generation/common.py:54`, `dataset.py:369`) byte-identical; note `profile_dataset.py:111-113` continues the sampling rng instead of reseeding (two rng contracts for the same call, document or unify behind a flag). L, golden-guarded.
- **R5** `photoreal.py` (1449 lines) mixes 11 responsibilities. Split by whole-function moves (no change of call order or rng consumption): `context.py`, `palette.py`, `imageops.py`, `substrate.py`, `copper.py` (legacy copper becomes one entry of a `COPPER_SOURCES = {legacy, none, <router>}` table next to the existing `ROUTERS` dispatch, replacing the special-casing at `photoreal._board:1261-1274`), `silkscreen.py`, `joints.py`, `solder_defects.py`, `camera.py` (the only piece the 3D backend needs besides `render_board`; `bake.py:17` then imports `camera`), `parts.py`, `scene.py`, `pipeline.py` with `photoreal.py` kept as a facade. `_render_board` and `_render_photoreal_base` share a prologue (`:1311-1322` vs `:1365-1374,1420-1423`) worth one `_prepare_scene`. M, golden test after every move.
- **R6** `photoreal.py:31` and `render.py:10` import private names from `draw.py` (`_draw_solder_bridge`, `_param_selector`, `_compute_pad_positions`); `draw.py` is the legacy flat renderer but hosts shared geometry (`body_geometry:190-212`, `lifted_corner_geometry:128-159`). Move shared pieces to `opencv2d/geometry.py` as public functions. S.
- **R7** Blender coupling to the checkout: `generators/blender3d/runner.py:90-94` resolves the script via `sim_root`, runs with `cwd=sim_root`; every caller passes `sim_root`. Use a package-relative path (`importlib.resources`), drop `sim_root`, and type the job contract (`bake.py:307`, `render_batch.py:6-19`) with `TypedDict`s in a stdlib-only `blender/job_schema.py` that both sides import. S-M.
- **R8** Realism presets: repo-relative YAML plus `SIMPLE_SIM_REALISM_PRESETS_PATH` env plus a module-global cache with blanket `except Exception: pass` (`opencv2d/realism.py:71-72,122-183`). Presets become a parameter; loader moves to the config/CLI layer. S.
- **R9** `build_window_board` mutates `neighbours` in place (`board/__init__.py:160-179`) and photoreal relies on it (`:1267-1281`). Return the kept list. S.
- **R10** Global state inventory to document for library users: `photoreal.texture_files` lru_cache keyed by path string (`:208`), `parallel._CV2_USERS` reference-counted `cv2.setNumThreads` (`parallel.py:27-29`), `telemetry.emit` reads `SIMPLE_SIM_EVENT_LOG` on every call (`telemetry.py:20`; add `set_sink(callable)`), `manifest._get_git_commit` spawns `git` on every manifest write against the *library's* checkout (`manifest.py:46-64`; make it a cached optional parameter).

### A. API surface and typed records

- **A1** `simple_sim/__init__.py`: lazy (PEP 562) exports for `load_config`, `validate_config`, `MetaRow`, `LabelRow`, `read_jsonl`, `write_jsonl`, `read_dataset_manifest`, `generate_dataset`, `ROIDataset`, `build_resnet18`, `load_checkpoint_model`, `compute_metrics`, `run_inference`, `TwoStageClassifier`, `load_reports`, `best_report`, `write_arena_report`, `render_sample` (torch modules resolved lazily); `__all__` in every module; a `NullHandler` logger. S.
- **A2** Typed records instead of dicts with magic keys: `DatasetManifest` (`from_dict/to_dict`, `profile_ids`; kills D6), `EvalReport` + `write_eval_report` (B14 in scripts), `Checkpoint` + `save_checkpoint`, `LoadedModel(model, class_names, checkpoint)` instead of the 3-tuple from `models.load_checkpoint_model:61-63`, `Report(path, data)` instead of the injected `"_path"` key (`reports.py:64`, consumed at `arena.py:220`, `cli.py:176`), `ModelRanking` instead of 10-key dicts (`arena.py:246-257`). M.
- **A3** `simple_sim/classes.py` with `DefectClass(StrEnum)` and derived sets (`CRITICAL_CLASSES`, `PLACEMENT_CLASSES`, `SOLDER_ONLY_CLASSES`, `NEVER_TOMBSTONE`), plus `Split(StrEnum)`, `Backend`, `RenderStyle`, `SolderStyle`, `Routing`, `DrcMode`. String values unchanged so JSON/YAML stay identical; `dataset_store.py:67` must still dump plain strings. The class set is currently closed and enforced in `MetaRow/LabelRow.__post_init__` (`schema.py:73,100`); consider an `allowed_classes` parameter so another program can define its own classes. S-M.
- **A4** `config.py` (473 lines) vs `config_schema.py` (92): `parse_config_typed` is called at `config.py:80` and its result discarded; it covers only run/roi/train/eval and has no other consumer besides one test. Either make `SimulationConfig` the real typed model (L) or delete `config_schema.py` and move `merge_profile_render` to `generation/common.py` (S). Rename `schema.py` to `rows.py` to end the schema/config_schema confusion.
- **A5** Compute functions that print, call `sys.exit`, or take `argparse.Namespace`: `manifest.check_checkpoint_profile` (`:317-346`, prints and hard-codes `--allow-profile-mismatch`), `metrics.check_success_criteria` (`:102-125`, print-only with an `ascii_only` flag), `models.build_resnet18:24` (bare `except Exception` + print), `data_loader.make_loader_kwargs:55-58`, `eval_utils.print_checkpoint_info/print_distribution`, `eval_utils.require_manifest:218-222` (hint mentions `.venv/bin/python tools/backfill_manifest.py`), `generation/common.finalize_splits` (`mark="✓ "`), `tools/create_multi_dataset.py` (8 `SystemExit`), `scripts/render_debug_previews.py` (9), `tools/sim_to_real.py:41,48,107`, `tools/calibrate_camera.py:43`, `scripts/arena_report.py:36`, `tools/update_sample_gallery.py:288,308,444`. Pattern: return structured results with `warnings: list[str]`, raise `UsageError`, `logging` for progress, `warnings.warn` for user warnings (already used in `model_bundle.py:163`, `data_loader.py:98`); the CLI layer appends the `--flag` hints. M across many files, low test risk (tests assert return values).
- **A6** Smaller API fixes: `splits.generate_splits` takes the whole config for three fractions (`splits.py:14-53`); `manifest.write_dataset_manifest` has 11 positional parameters (`:84-96`); `TwoStageClassifier.__init__(device: str)` vs `str | torch.device` elsewhere; `parse_iso_ts` returns `0.0` on failure while `metric_value` returns `None`; `schema.read_jsonl` special-cases `LabelRow/MetaRow` compat (`:171-176`) and rejects unknown fields (`:166-168`), blocking forward compatibility; `model_bundle.kind: str = "best"` should be `Literal["best","last"]`; `eval_utils.run_inference` hard-wires tqdm and `emit` (`:100,127`), add `progress`/`on_batch`.

### M. Modernization leftovers

- `typing.Mapping/Callable/Iterable/Sequence` instead of `collections.abc` at `board/__init__.py:17`, `board/rules.py:13`, `defects.py:5`, `io_utils.py:13`, `splits.py:9`; missing `from __future__ import annotations` in `data_loader.py`, `models.py`, `schema.py`; untyped samplers `defects.py:138-213`; `reports.py:92 os.path.getmtime`, `telemetry.py:25 os.makedirs`; `open()` without `encoding=` at `schema.py:119,150`, `manifest.py:124,240`.
- No `Enum`, `StrEnum` or `match` anywhere in `simple_sim/`; `if/elif` chains on `defect_type` at `photoreal.py:1383-1425`, `draw.py:214-242` and on `family` at `photoreal.py:832-845,953-1003` are `match` candidates once A3 exists.
- `slots=True` missing on `solder.SolderLook:46`, `preview.RenderSettings:41`, `PreviewOptions:55`, `PreviewRun:506`, `generation.dataset.ExtendState:38`, `schema.MetaRow/LabelRow:19,82`, `config_schema.*`.
- `defects.sample_defect_params` rebuilds its sampler dict per call (`:105-115`); `arena.render_markdown` (`:345-604`, 260 lines) mixes aggregation, chart I/O and markdown; `svg_charts.py:14-19` palette as module globals.
- GUI: `print()` logging in `gui/tabs/analysis/logic.py:43-157`; dozens of bare `except Exception: pass` in `gui/tabs/pipeline/tab.py` (e.g. `:401-409,1431-1440`); one-line compound statements (`tab.py:1190-1196,1596-1600,1875-1878`); `SettingsStore` stringifies non-primitives so tabs store JSON-in-JSON (`settings_store.py:35-40`).
- Dead or stale: `gui/tabs/base_tab.py` (5-line shim, zero importers), `gui/utils/formatting.py`, `tools/profile_editor/system_monitor.py`, `eval_utils.py:250-253` empty "# CLI" header, `footprints.fine_pitch` (X8); `gui/README.md` references a non-existent `monitor_old.py.backup` and documents 4 of 9 tabs; `tools/profile_editor/README.md` says "Night Mode" and "auto-synced" link.
- GUI structure (details in GUI_MODERNIZATION_PLAN.md): `pipeline/tab.py` 2030 lines with 13 responsibilities, ~450 of which leave with L5/L7/L19; dataset combobox populated five ways (`pipeline/tab.py:1182`, `analysis/tab.py:120`, `predictions/tab.py:292`, `validation/tab.py:47`, `weights/tab.py:148`) with two different labelling helpers; settings-persistence boilerplate in four tabs; TaskRunner used by four tabs while five roll their own threads/queues/`after()` polls; `*Logic` classes take `queue.Queue` and `tk.StringVar` dicts (`pipeline/logic.py:31,104-106,334-342`); two dataset-change propagation mechanisms (virtual event + bus, no tab subscribes to the bus); `UiState` and `AppContext` both hold the settings store.

## 4. Phases

Each phase is independently mergeable. Run the full test suite plus
`tests/test_render_golden.py` after every step; the golden test skips when the OpenCV or
NumPy version differs from `tests/golden/render_2d.json`, so confirm it actually ran.

| Phase | Content | Exit criteria |
|---|---|---|
| 0 Unlock (S) **done 2026-11-06** | I1, I2, I3, I8, P1 (pyproject + editable install), D17, dead-code removals | `pip install -e .` works; `python -c "import gui.tabs.pipeline.logic"` succeeds with tkinter blocked; `import simple_sim.filter_profiles` does not load cv2; AGENTS.md test note updated |
| 1 Logging and errors (M) **done 2026-11-06** | A5, A3 (`classes.py`, enums), I6 (`schema` without yaml), D14, D15 (`io_utils.now_iso`), D18, D19 | `grep -c "print(" simple_sim/` only in `cli*.py` and `commands/`; no `SystemExit` outside `commands/`; `metrics` without scikit-learn (exact-value test) |
| 2 Shared helpers (S-M) | D1-D13, D16, D21, I5, I7, L9 (`layout.py`), L10 (`catalog.py`), L11 (`dataset_index.py`), L8, L13, L15, L16, L21, L23 | each listed copy deleted; GUI and tools import from `simple_sim` only; no `tools -> gui` or `simple_sim -> gui` import (`grep -rn "from gui\|import gui" simple_sim tools` empty) |
| 3 Typed records (M) | A2, A4 decision, A6, D5, D6 | `DatasetManifest`, `EvalReport`, `Checkpoint`, `LoadedModel`, `Report` in use by scripts, GUI and tools; `config_schema.py` either real or gone |
| 4 Logic into the library (L) | L1, L2, L3, L4, L5, L6, L7 (decide X1), L12, L14, L17, L18, L19, L20, L22 | `scripts/*.py` and `tools/*.py` contain argparse and printing only; `gui/tabs/*/logic.py` import nothing from `scripts`/`tools`; X1-X7, X9-X11 resolved |
| 5 Command registry (M) | P2, P3, P4, D22 | `cli.py` dispatches by import; `commands --json` reads `SUPPORTS_JSON`; `simple-sim` console script; shell launchers reduced to venv bootstrap; GUI parses `--json` result lines |
| 6 Renderer (M-L) | I7 already done; R1 (`RenderSpec`, `render_sample`), R3 (backend registry), R6, R7, R8, R9, D23-D35, R5 (photoreal split), then R2 (`RenderContext`), then R4 (`generate_dataset` decomposition) | golden md5 unchanged after every commit; `render_sample(RenderSpec.from_run_config(cfg, profile), ...)` renders without the dataset pipeline; `import simple_sim.board.model` does not load cv2; Blender runs without `sim_root` |
| 7 GUI internals | remaining items of GUI_MODERNIZATION_PLAN.md phase 3-4 (pipeline/tab.py split, BaseTab helpers, dataset picker widget, one event path) | tracked there |

### Phase 0 status (2026-11-06)

Done: P1 (`pyproject.toml`, `pip install -e .`, `simple-sim` script, pytest config), I1, I2
(four shims deleted, `generators/__init__.py` empty), I3 (`simple_sim/filter_settings.py`),
I8, `gui/tabs/base_tab.py`, empty `# CLI` header in `eval_utils.py`, `footprints.fine_pitch`,
stale README lines. Guard: `tests/test_headless_imports.py`. Golden test ran and passed.

Corrections to the audit:
- `tools/profile_editor/system_monitor.py` is not dead: `tools/profile_editor/app.py:21,33` imports it. Kept.
- `gui/utils/formatting.py` already delegates to `io_utils.format_bytes` (`decimals=1` for
  the GUI), so D17 is not a duplicate. Kept as the GUI's one formatting entry point.
- All nine `logic.py` modules are tk-free now (not eight). `analysis` and `board_detection`
  still load torch at import (`gui.utils.model_inference`, `simple_sim.two_stage`; analysis also scikit-learn via `metrics`); that is phase 1/4 work.
- `sys.path.insert` hacks stay until phase 5: only `simple_sim` is packaged, and scripts run
  as files still need the checkout on `sys.path` for `gui`/`tools`.

### Phase 1 status (2026-11-06)

Done: A5 (logging instead of `print`, neutral `ProfileMismatchError` / `RendererMismatchError`
plus `cli_utils.with_hint` for the flag advice, `format_*` instead of `print_*`,
`evaluate_success_criteria` + `format_success_criteria`), A3 (`simple_sim/classes.py`), I6
(`schema` takes `VALID_CLASSES` from `classes`; `metrics` numpy-only, bit-identical to
scikit-learn), D14 (`simple_sim/coerce.py`), D15 (`io_utils.now_iso` / `now_tag(micro=False)`,
no `datetime.utcnow()` left), D18 (`io_utils.load_json`, `atomic_write_jsonl`), D19
(`io_utils.load_yaml`, all `safe_load(read_text())` sites). New tests: `test_metrics.py`
(parity), `test_coerce.py`, `test_library_logging.py`. Golden test ran and passed.

Exit criteria: no `print(` in `simple_sim/` except `cli.py`, `cli_utils.py` and the Blender
script `blender/render_batch.py` (runs as `__main__` inside Blender, it is a CLI); no
`SystemExit` outside those plus `__main__.py`. `commands/` does not exist before phase 5.

Deviations from the plan:
- A3: only `DefectClass`, `Split`, `Backend` are enums. `RenderStyle`, `SolderStyle`, `Routing`,
  `DrcMode` stay as the existing option tuples until I7 (`generators/options.py`, phase 2)
  moves them, so they are defined once. The class set stays closed (no `allowed_classes`
  parameter yet); revisit with A2's typed records.
- `require_manifest` still names a command (`python -m simple_sim backfill-manifest`), but
  no venv path or script file.
- D19: `tools/profile_editor/state_store.py` parses YAML text from the editor buffer
  (`yaml.safe_load(text)`), not a file; left as is.

### Phase 6 status (2026-11-07, renderer part)

Done, every 2D image bit-identical (both golden tests; `RENDERER_VERSION` unchanged):
- Guard rail first: `tests/test_render_golden_variants.py` (589 cases: 16 2D configs x 8 classes,
  flat, supersampling 1/3/4, meniscus, post_filters camera, routing none/escape/escape+DRC, phone,
  gloss/rework/marking, texture_dir, all extended filters, realism filters, bake, camera model).
- R5: `photoreal.py` split into the package `opencv2d/photoreal/` by whole-function moves (generated
  from the AST, then reviewed); `opencv2d/solder.py` -> `photoreal/meniscus.py`. Legacy copper is
  one entry of `copper.COPPER_SOURCES` next to `none` and the board-model routers.
- R2: `RenderContext(ss, solder_look)` passed explicitly; thread-local `_TLS`, `_ss()`,
  `photoreal.SS` removed. X5 is explicit (`solder_defect_layout(..., ss=DEFAULT_SUPERSAMPLING)` in
  the flat path).
- R6: `pad_positions`, `param_selector`, `body_geometry`, `lifted_corner_geometry`,
  `draw_solder_bridge` public in `opencv2d/geometry.py`.
- R1: `opencv2d/spec.py` (`RenderSpec.from_run_config`, `render_sample`, `render_record`), used by
  `generate`, `generate-profile-dataset` and `preview.render`.
- D23, D24 (noise, height-map normals shared by legacy and meniscus joints), D26 partly (all
  `np.mgrid` in the photoreal package use `imageops.grid` / `local_coords` windows).
- Speed: window-local drawing of bodies, leads, lead feet, tombstones and bulges (random fields
  still drawn at full size via `imageops.WindowedNoise`), growing shared grid; 60 -> 35 ms per
  image. Dataset rendering in worker processes (`parallel.imap_ordered(processes=True)`,
  `render_executor`), 83 -> 264 images/s on 16 cores.

Second round (2026-11-07, "go"): R3 (`generators/backends.py`, capabilities instead of string
compares in config, generation, preview), R4 (`generate_dataset` = `resolve_run` + `plan_records` +
render/write; config mapping, `profiles_dir`, `image_filters` parameters), R7 (package-relative
Blender script, absolute paths, no `sim_root`; the `TypedDict` job contract is still open), R8
(`realism_presets` parameter, `load_realism_presets`, mtime-keyed cache, no silent `except`), R9
(`WindowBoard`), D25 (`clip8` in filters), D28 (chip families from `board.footprints`), RenderSpec
in the GUI filter popup, profile-preview tab and sample gallery. New: `--extend` compares the
resolved look options (`generation/render_options.py`), closing the open item of the board,
solder, training-options and look work.

Still open: D27 (legacy copper vs board-model jog/pour helpers), D29-D35 (mm/px helper, render
defaults in one place, colour-jitter helper, Blender constants re-typed in `render_batch.py` / X12),
the Blender job `TypedDict`s, `profile_dataset.py` continuing the sampling rng (documented, kept for
pixel identity), the `typing` leftovers outside the renderer.

## 5. Guard rails

- Legacy pixel identity: `tests/test_render_golden.py` after every renderer commit; add a
  second golden for `routing: escape` and for the flat style before touching R2/R5 so
  the `_ss()` quirk (X5) is pinned.
- Tests that import from `scripts.`/`tools.` (`tests/test_eval_utils.py`,
  `tests/test_pipeline_e2e.py`, `tests/test_profile_editor_tools.py`,
  `tests/test_sim_to_real.py`, `tests/test_calibrate_camera.py`): keep the shim re-exports
  until the tests are moved to the library modules in the same PR.
- Tests that pin current shape: `tests/test_cli.py:38-42` (`add_json_flag` sniffing),
  `tests/test_gui_helpers.py:308` (exact `./.venv/bin/python scripts/train.py` argv),
  `tests/test_parallel_render.py:63` and `tests/test_photoreal.py:207` (`photoreal.SS`).
- `SAMPLE_GALLERY.md` images: L17 changes the seed derivation unless the gallery keeps
  `_stable_int_seed`; decide before the move.
- Docs to update alongside: AGENTS.md (test note, command registry), README/QUICKSTART
  (`simple-sim` script), `docs/architekture/*-Overview.md`, `architektur.md` via
  `tools/update_architektur_md.sh`, `gui/README.md` (stale sections).
