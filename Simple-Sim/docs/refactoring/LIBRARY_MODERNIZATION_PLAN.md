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
  **Done 2026-11-07:** both. `board/options.py` (cv2-free: `render.board` options, streams,
  `collect_drc` sink, `ROUTER_NAMES`, `POUR_PROBABILITY`), `board/pipeline.py`
  (`build_window_board`, `WindowBoard`, `min_row_pitch`, `fanout_band_px`, logger still
  `simple_sim.board`); `board/__init__.py` imports options/footprints/geometry/model/netlist/
  placement/rules eagerly and resolves the DRC, routers, pours, pipeline and the submodules via
  PEP 562 `__getattr__`. Every `__all__` name stays importable. Guarded by
  `tests/test_headless_imports.py::test_board_data_model_does_not_load_cv2` (7 modules) and
  `test_board_lazy_exports_load_on_access`.
- **I5** `eval_utils.py:16-24` top-imports torch/tqdm but holds torch-free helpers
  (`MANIFEST_NAME`, `metrics_summary`, `require_manifest`, `dataset_profile`,
  `resolve_bundle_checkpoint`, `checkpoint_eval_config`, `checkpoint_val_accuracy`).
  Move them to `manifest.py` / `model_bundle.py`. S.
  **Done 2026-11-07** differently: torch, `torch.nn`, `DataLoader` and tqdm are imported inside
  the functions that use them (annotations via `TYPE_CHECKING`), so the helpers stay where they
  are and `import simple_sim.eval_utils` is torch-free (`tests/test_headless_imports.py`).
- **I6** `schema.py:13` imports `simple_sim.config` (PyYAML) for `VALID_CLASSES`;
  `metrics.py:9-10` needs scikit-learn for three functions that are trivial in numpy.
  New dependency-free `simple_sim/classes.py` (see A3); numpy metrics with exact-value
  tests. S + M.
- **I7** `config.validate_config` lazily imports `simple_sim.board` and renderer modules
  (`config.py:304,314,323,333,343`) to fetch option constants, so validating YAML loads cv2.
  Move all `render.*` option constants and validators into a light
  `generators/options.py` (camera presets, supersampling choices, solder styles, texture
  dir). S.
  **Done 2026-11-07:** `simple_sim/generators/options.py` (look names, solder styles, supersampling
  choices, post-filter modes, camera defaults/preset names + `validate_camera_config`); `config.py` imports only
  from it, the photoreal definitions stay and `tests/test_generator_options.py` fails on drift;
  `test_validate_config_without_cv2`.
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
| D1 | `resolve_device(spec) -> torch.device` + `default_device_name()` | `scripts/train.py:223,452`, `train_profile.py:44,152`, `eval.py:69,158`, `eval_profile.py:33,93`, `predict.py:87,149`, `batch_predict.py:352,513`, `sample_check.py:74,131`, `gui/utils/model_inference.py:39`, `two_stage.py:51` (`predict_two_stage.py:415` defaults to cpu, inconsistent) **Done 2026-11-07:** `simple_sim/devices.py` (torch imported on call); 7 scripts + `two_stage.py`; `gui/utils/model_inference.py` and `predict_two_stage.py` (cpu default) left |
| D2 | argparse fragments `add_data_arg/add_model_arg/add_out_arg/add_amp_flags/add_profile_mismatch_flag` in `cli_utils` (torch-free) | `--amp/--no-amp` group `train.py:454-457` = `train_profile.py:154-157`; `--allow-profile-mismatch` `train.py:440-447` = `eval.py:150-157`; `--data/--model/--out` in 10 scripts **Done 2026-11-07:** `cli_utils.add_*` (help text passed by the caller) in 11 scripts; `--help` pinned by `tests/test_script_help.py` + `tests/golden/help/` |
| D3 | `rel_to(root, p)` / `abs_from(root, p)` | `io_utils.py:99-104`, `generation/common.py:27-32`, `model_bundle.py:177-180`, `generation/profile_dataset.py:71` (no fallback, raises for a profile outside root), `arena.py:48-58`, `gui/tabs/pipeline/tab.py:584,1314`, `gui/tabs/predictions/tab.py:90,107` |
| D4 | `class_counts(label_rows)` | `manifest.py:67-72`, `generation/common.py:180`, `scripts/batch_predict.py:84`, `gui/utils/dataset_ops.py:227` |
| D5 | Manifest history / generator blocks | `manifest.py:146-153`, `:185-201`, `:390-412` (three hand-built copies in one file) **Done 2026-11-07:** `manifest._history_entry` / `_generator_block` (same keys and order) |
| D6 | Manifest v1/v2 (`component_profile` vs `component_profiles`) branching | `manifest.py:207-219,257-281`, `cli.py:142-145`, `generation/dataset.py:69-111`, `scripts/train.py`, `scripts/train_profile.py`, `tools/create_multi_dataset.py`, `gui/utils/pipeline_helpers.py`, `gui/tabs/pipeline/logic.py:421-440`, `gui/tabs/predictions/logic.py:84-98` (X7) |
| D7 | Dataset file-name constants (`dataset_manifest.json`, `meta.jsonl`, `labels.jsonl`, `config.yaml`) | `manifest.py:119,379`, `cli.py:133`, `generation/dataset.py:120-123`, `eval_utils.py:26`, `dataset_store.py:25`, `data_loader.py:85-86`, +20 in gui/scripts/tools **Done 2026-11-07:** `simple_sim/dataset_files.py` (`DATASET_MANIFEST`, `META_JSONL`, `LABELS_JSONL`, `CONFIG_YAML`, `SPLITS_DIR`) at the path joins in `simple_sim/` (not `generation/`), `scripts/`, `gui/`; message strings and `generation/`, `tools/` left |
| D8 | `config.yaml` loader for a dataset dir | `scripts/train.py:133-136`, `tools/create_multi_dataset.py:150-155`, `tools/backfill_manifest.py:76`, `cli.py:112` |
| D9 | Bundle checkpoint listing (`*.pt` minus `_last`) | `model_bundle.py:122-126`, `cli.py:164-168`, `arena.py:114-116` **Done 2026-11-07:** `model_bundle.list_bundle_checkpoints(include_last=)` |
| D10 | Bundle checkpoint resolution with the identical error text "Multi-model bundle has no checkpoint for profile" | `eval_utils.py:232-247` (canonical), `scripts/batch_predict.py:109-129`, `gui/tabs/weights/logic.py:281-301`, `gui/tabs/predictions/logic.py:83-131` **Done 2026-11-07:** `model_bundle.resolve_bundle_checkpoint(hint=, show_available=)` (re-exported by `eval_utils`), used by `batch_predict` and the weights tab with their exact texts; `predictions/logic.py` (log lines, auto-train) left |
| D11 | Profile YAML parsed twice per load (`load_profile` then `hash_profile` re-reads) | `generation/dataset.py:237-239`, `profile_dataset.py:64-69`, `profile_hash.py:75-76,112-113`; add `hash_profile_dict` **Done 2026-11-07** (helper): `profile_hash.hash_profile_dict`; the two callers in `generation/` still call `hash_profile(path)` |
| D12 | Split-name constant | `generation/common.py:19`, `schema.py:47`, `config.py:124`, `splits.py:56`, `scripts/batch_predict.py:74,98,101`, `tools/validate_dataset.py:145`, gui x5 |
| D13 | Defect-class string sets | `config.py:37`, `defects.py:68,105-115`, `metrics.py:12`, `opencv2d/draw.py:199`, `opencv2d/render.py:17`, `preview/render.py:32`, `tools/update_sample_gallery.py:43`, gui x6 |
| D14 | Numeric/bool coercion `_to_bool/_to_float/_to_int` | `arena.py:83-87`, `filter_profiles.py:35-43`, `generators/filter_settings.py:134-158`, `gui/components/filter_popup/models.py:12-29` |
| D15 | Timestamps (naive local ISO, UTC `Z`, `%Y%m%dT%H%M%SZ`, `%Y-%m-%d %H:%M:%S`) | `manifest.py:147,176,194,391,395`, `model_bundle.py:132,154`, `arena.py:421,601,622`, `io_utils.now_tag:46`, `scripts/eval.py:119`, `eval_profile.py:62`, `batch_predict.py:439`; `datetime.utcnow()` (deprecated 3.12) at `gui/tabs/pipeline/logic.py:531`, `gui/tabs/merge/logic.py:144,191` |
| D16 | SHA-256 with `sha256:` prefix | `manifest.hash_file:22-43`, `profile_hash.hash_profile:54-86`, prefix stripped at `gui/utils/checkpoint_meta.py:29` **Done 2026-11-07:** `io_utils.sha256_tag` / `SHA256_PREFIX` |
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

Third round (2026-11-07, D27-D35 and the job contract; both golden tests unchanged, no rng draw moved):
- D27: `board.geometry.polyline45(p0, p1, dedupe=True)` and `board.geometry.octagon` are the one
  45-degree jog and pour outline; the legacy copper generator (`photoreal/copper.py`, `_polyline` and
  the pours) calls them (`dedupe=False`: `cv2.polylines` blends a zero-length anti-aliased segment, so
  the coinciding jog point stays; the jog of integer pixels is an integral float cast back to int32).
- D33: `POUR_CELLS_PER_CLEARANCE` is defined in `board/zones.py` (`stitch_vias` divides by it);
  `board/drc.py` imports it.
- D34: `Vec` and `BBox` exist once (`board/geometry.py`; `model.py` imports them); `unit`, `perp`,
  `add`, `clip`, `polyline45` moved from `routing/base.py` to `geometry.py` (re-exported by `base`;
  `zones.py` no longer imports from the routing layer); the spelled-out 4-tuple is `BBox` in
  `model.py`, `routing/base.py`, `routing/escape.py`, `board/__init__.py`.
- R7 contract: `simple_sim/blender/job_schema.py` (stdlib only, `simple_sim/blender/__init__.py`
  added): `JobFields`, `BakeOptions`, `BlenderJob`, `PreparedJob`, `BoardScene`, `SceneBody`,
  `CameraPose`; imported by `blender3d/bake.py`, `runner.py`, `generation/blender.py` and by
  `render_batch.py` inside Blender (puts the package root on `sys.path`, imports nothing else).
  `tests/test_blender_bake.py` checks the keys of every writer and reader against it.
- D30: defaults in one place: `job_schema` carries `DEFAULT_BLENDER_EXECUTABLE`,
  `DEFAULT_CYCLES_SAMPLES`, `DEFAULT_DEVICE`, `DEFAULT_ENGINE` (`generation.blender.blender_settings`,
  the runner and the script's argparse), `DEFAULT_COMPONENT_HEIGHT_MM`, `BOARD_SCALE`,
  `DEFAULT_CAMERA_POSE`; `board.footprints.DEFAULT_FOOTPRINT` + `profile_footprint(profile)`
  (`RenderSpec`, `draw.py`, `bake.py`, `preview/render.py`, `tools/update_sample_gallery.py`);
  `bake.profile_component_height_mm(profile)` (`generation/dataset.py`, `preview/render.py`, the
  gallery); `bake.BAKE_CONFIG_DEFAULTS` is the one table behind `bake_fields` and `prepare_job`.
  Left for the owners: `opencv2d/render.py:64` (`footprint="chip_2pad"`), `look.mask_gloss` /
  `rework_probability(config, default)` always called with the same constant (`scene.py:67,72`,
  `generation/render_options.py:42-43`, `bake.py`: make `default` optional in `look.py`),
  `gui/components/filter_popup/preview_render.py:105` and `gui/tabs/profile_preview/logic.py:248`
  (`0.45`, use `profile_component_height_mm`).
- D31: `palette.jittered(colour, rng, lo, hi)` (= `colour * float(rng.uniform(lo, hi))`), used at
  `substrate.py` (mask base) and `rework.py` (flux). Left (owned files): `joints.py:37`,
  `parts.py:119,194,229,273,275,277,307,441`, `scene.py:68`, `meniscus.py:87`. Grain: the home is
  `imageops.WindowedNoise.normal`; the free-standing `rng.normal(0, sd, shape).astype(float32)` at
  `substrate.py:58`, `camera.py:224,226`, `filters.py:23` stay until `imageops` gets a module function.
- D32: `augment.sample_rotation_deg(rng, range, enable_cardinal_rotation_90=)` (the two draws of
  `sample_augment_params`) and `augment.clean_augment(rotation_deg)` (`CLEAN_AUGMENT_KEYS` from
  `NEUTRAL_AUGMENT`); `preview.render.sample_preview_augment` is now one line of both. Left:
  `gui/components/filter_popup/preview_render.py:21-27` (X2, gui).
- D29 partly: `render_batch.py` converts with `_mm` / `_mm_xy` (one place for the y flip); the
  photoreal sites collapsed into `copper.mm_per_px(config)` during R5 and one `job.mm_per_px / ctx.ss`
  (`copper.py:216`). Not unified: `footprints.pads_px` multiplies by the reciprocal and rounds,
  `occupancy.Grid` divides with ceil/round, `part_from_pads` multiplies: different numerics, a shared
  helper would change pixels.
- D35: skipped. The per-part decisions re-typed in the Blender script (`_HEIGHT_RATIO`, cap fraction,
  standoff, lead width, marking vocabularies; X12's 0.56 vs `parts.THICKNESS_RATIO` 0.55, dome
  exponent 3 vs 2.5) would have to be written by `scene.render_board` into the scene dict (owned by
  the look session) and the bake golden hashes that dict; the constants belong in `job_schema` once
  `parts.py` / `meniscus.py` can import them from there.

Still open: the D29/D30/D31/D32 sites listed above, D35, `profile_dataset.py` continuing the
sampling rng (documented, kept for pixel identity), the `typing` leftovers outside the renderer.

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

## 6. Re-audit 2026-11-07 (working tree after phase 6, uncommitted look/perf round)

Read-only pass over the whole tree (`simple_sim/` 15.1k lines, `gui/` 18.9k, `tools/` 4.8k,
`scripts/` 2.4k, `tests/` 6.1k; 610 tests, 607 pass / 3 skip in 178 s; both golden tests pass on
the uncommitted tree, 48 s). Lens: modularity (extension points, layer boundaries) and
scalability (per-call work that is constant per run, whole-dataset memory, UI-thread I/O).

### 6.1 Status of the plan

| Phase | Status | Evidence |
|---|---|---|
| 0, 1 | done | `tests/test_headless_imports.py` guards I1/I8; `print(` only in `cli*.py` + Blender script |
| 2 shared helpers | open except D14/D17/D18/D19/D23-D25 (and D1/D2/D5/D7/D9/D10/D11/D16/I7: done 2026-11-07, see section 3) | D1 device pick at 7 script sites; D5 three history blocks `manifest.py:146,186,399`; D7 108 literal file-name sites; D11 `hash_profile` re-reads YAML; D16 `sha256:` built twice; I5 done 2026-11-07 (torch/tqdm lazy in `eval_utils`, 0.74 -> 0.05 s warm; was 2.8 s cold); I7 `validate_config` still loads cv2 via the photoreal option constants (`config.py:357,376,386,396,407,419`); its board part (`_validate_board`) is cv2-free since I4 (`test_board_options_validate_without_cv2`); I4 done 2026-11-07 (`board/options.py` + `board/pipeline.py`, lazy `board/__init__.py`) |
| 3 typed records | open | `_path` key in 6 places, `load_checkpoint_model` 3-tuple, `parse_config_typed` result discarded (`config.py:79`), 11-positional `write_dataset_manifest` |
| 4 logic into library | open | none of `training/evaluation/batch_predict/validation/dataset_merge/dataset_edit/inference/...` exist; `train_model` 256 lines / 44 prints; `gui/utils/validation_suite.py:13` is the only `gui -> tools` import; `simple_sim/preview/viewer_tk.py:219` the only `simple_sim -> gui` import |
| 5 command registry | open | `cli.py:264 subprocess.call`, `:75 "add_json_flag" in src`; the 7 sim-to-real tools are in `COMMANDS` since 2026-11-07 (N12); 40 `sys.path.insert` (the 6 in those tools now only fire when run as a script, `real_crops.py` has none) |
| 6 renderer | mostly done | R2, R5, R6, R9, X5, D23-D25 done. Partly: R1 (5 callers still hand-merge via `merge_profile_render`: `profile_dataset.py:101`, `preview/render.py:377`, `filter_popup/preview_render.py:58`, `profile_preview/logic.py:190`, `update_sample_gallery.py:148`), R3 (`backends.py` holds capabilities only; dispatch `if backend == Backend.OPENCV_2D` at `dataset.py:456-475`), R4 (`generate_dataset` 208 lines, `generate_profile_dataset` 134 lines not decomposed), R7 (job contract still dicts), R8 (env read inside `realism.py:135`), D26, D28 (`REFDES_PREFIX` twice: `board/placement.py:18-19` and `photoreal/palette.py:64-66`, same mapping; fix: `palette.py` imports it from `board.placement`), D29, X8 (escape side done 2026-11-07: `board.footprints.FINE_PITCH_FAMILIES` / `is_fine_pitch` is the single source, `escape.py` and `pipeline.py` import it; `copper.py:164` `RULE_CHIP_FAMILIES` is a different predicate (rule-scale boost for `sot, soic, qfn, sot6`; arrays are not boosted) and cannot use `is_fine_pitch` bit-identically). Open: D27, D30, D31, D33, D34, D35/X12, R10 |
| 7 GUI | open | `pipeline/tab.py` 2033 lines / 105 methods; threads in 6 tabs; GUI plan phase 4 untouched |
| X1-X12 | only X5 closed | X1 `model_inference.py:22`, X2 `preview_render.py:24`, X3 `validation_suite.py:18`, X4 `train_profile.py:98`, X6, X7 `predictions/logic.py:91`, X9, X10 `preview_3d.py:475`, X11 `weights/logic.py:97` |

### 6.2 Regression in the working tree (fix before commit)

- **G1** `scripts/train_profile.py:85` unpacks 4 values, but the uncommitted freeze change made
  `scripts/train.py:169 build_training` return 5 (`..., frozen`). `train_profile` raises
  `ValueError: too many values to unpack` at start; no test imports it. Fix: unpack `frozen`,
  pass it to `fit_epoch`, add a 1-epoch smoke test. This is the cost of L1 (training loop in a
  script that another script imports).

### 6.3 Findings of the uncommitted renderer round (all pixel-safe)

- **U1** `light_roll_deg` threaded through 4 layers under 4 names (`render.py:76` -> `scene.py:248`
  `roll`/`after` with `None` sentinel -> `context.py:41 roll_deg` -> `camera.py:142 axis_roll_deg`).
  One `RenderContext.post_roll_deg`. S. **Done 2026-11-07:** `RenderContext.post_roll_deg` set by
  `for_render(config, rng, roll_deg, *, post_roll_deg=)`; `_render_photoreal_base` lost its
  `light_roll_deg` parameter and passes `ctx.post_roll_deg` to `camera_model(axis_roll_deg=)`. The
  public `render_photoreal_base(light_roll_deg=)` / `camera_model(axis_roll_deg=)` are unchanged.
- **U2** `context.py:52 if look.name == "quins"` selects the joint sampler by string. Make it a
  `Look.joint_sampler: Callable` field so a third look needs no core edit. S. **Done 2026-11-07:**
  `Look.joint_sampler(stream, roll_deg) -> SolderLook` (legacy: `sample_look`, quins:
  `sample_scanner_look`); `look.py` now imports `meniscus` (no cycle).
- **U3** `Grid.inflated_rect_rows` returns a `None` sentinel; the fallback is duplicated in
  `zones.py:98-103` and `occupancy.py:124-131`. Always return rows. S. **Done 2026-11-07:**
  falls back internally (`_mask_rows(inflate(rect()))`), both call-site fallbacks removed;
  `tests/test_board.py::test_inflated_rect_rows_equal_inflate_of_rect` (300 random grids/boxes/n)
  and `..._fallback_without_kernel_spans`.
- **U4** `drc._end_connected(items=None)` (`drc.py:129-140`) rebuilds `_copper(board)` per call
  when omitted. Make `items` required. S. **Done 2026-11-07** (only caller `check` passes it).
- **U5** `imageops.gradient`, `heightmap_normals`, `contact_shadow`/`_rolled_spans`
  (`imageops.py:59-85,231-269`) re-implement `np.gradient`/`np.linalg.norm`/`np.roll` and claim
  bit identity; only the golden tests pin it. Add equality property tests incl. border cases. S.
  **Done 2026-11-07:** `tests/test_imageops_equivalence.py` (76 cases: shapes 2x2..64x48 and
  2x1000, 1-px maps raise like `np.gradient`, int fallback, float32/float64, shadow masks at all
  corners/edge midpoints/centre for ss 1-4 and both looks, full/empty masks, `_rolled_spans`
  vs `np.roll` for n 1-17 and every shift/span, `blend` bbox vs full frame). All bit-identical.
- **U6** `DENSE_PACKING` (`placement.py:46`, a look constant) lives in the board layer; `crowd`
  overlaps `Packing` (`scene.py:97,160` vs `placement.py:77`). Move to `look.py`, fold `crowd` in. S.
  **Partly done 2026-11-07:** `Look.crowd` (1.0 on every look, a duplicate of `Packing.parts`) is
  removed: pixel-identical since `x * 1.0 == x`. The `crowd` argument of `neighbour_layout`
  (rendered area / ROI area) is geometry, not packing, and stays. **Done 2026-11-07:**
  `DENSE_PACKING` is defined in `photoreal/look.py`; `simple_sim.board` no longer exports it
  (no re-export: the board layer must not depend on the generators), `tests/test_look.py`
  imports it from `look`.
- **U7** `Look` is becoming the render-defaults object (`shadow`, `silkscreen_skip`, `packing`).
  Finish D30 there: numeric `mask_gloss`/`rework` on `LEGACY_LOOK`, drop the `default` args of
  `mask_gloss()/rework_probability()` and the constant re-imports in `bake.py:45-46,69-70`,
  `render_options.py:18`. S. **Done 2026-11-07:** `Look.mask_gloss`/`rework` are `float`
  (legacy defaults `MASK_GLOSS_PROBABILITY = 0.5`, `REWORK_PROBABILITY = 0.15`, now defined in
  `look.py` and re-exported by `substrate`/`rework`/the package); `default` is optional (only used
  for a look whose value is `None`); `scene.py`, `render_options.py`, `bake.py` call
  `mask_gloss(cfg)`/`rework_probability(cfg)`; `BAKE_CONFIG_DEFAULTS` uses `LEGACY_LOOK`'s values.
- **U8** `camera_model` is 83 lines (`camera.py:139`); `camera_settings` returns an untyped dict and
  `CAMERA_DEFAULTS["channel_axis"]` is a string among range lists, so typos surface at render
  time. Frozen `CameraSettings` validated in `config._validate_camera`. M. **Done 2026-11-07
  (settings + validation; `camera_model` not split):** `camera.CameraSettings` is a frozen slots
  dataclass that is also a read-only `Mapping` (`c[key]`, `.get`, `{**c}`, `== dict` keep working;
  values kept as configured so the draws are unchanged); unknown keys / preset / `channel_axis`
  raise on construction. `camera.validate_camera_config` holds the former `config._validate_camera`
  body (+ a construction check), and `config._validate_camera` calls it. `render_options._camera`
  returns `dict(...)` for readable `look_differences`.

### 6.4 New findings: modularity and extension points

- **N1 Part families are a 7-file string chain.** `parts.draw_body` (115 lines, `:272-346`),
  `parts.py:139-153`, `solder_defects.py:88-96`, `silkscreen.py:23-30`, `footprints.family_of:52-71`
  (substring sniffing), `palette.py:58-84` (CHIPS/LEADED/ARRAY/UNMARKED/CAP_LEN/REFDES),
  `render_batch.py:36,227,248-279`, `copper.py:164`. Fix: cv2-free `PartFamily` frozen dataclass
  (kind, cap_len, unmarked, refdes, edge inset, height_ratio, marking rule) in a `FAMILIES`
  table in `board/footprints.py`; `parts.py` dispatches via a drawer registry keyed by `kind`;
  Blender reads the same table from the baked scene (closes half of D35). Pixel-safe. M.
  **Done 2026-11-07** (2D side): `PartFamily` + read-only `FAMILIES` (name -> row; iterates the names as the
  old tuple did), `part_family()`, `BODY_KINDS`; `CHIP_FAMILIES`, `FINE_PITCH_FAMILIES`, `placement.REFDES_PREFIX`,
  palette `CHIPS/LEADED/ARRAY_FAMILIES/UNMARKED/CAP_LEN` derived from it; `parts._BODY_DRAWERS` by kind;
  silkscreen/marking read the table. Blender side open (D35): `render_batch.py` is stdlib-only and the bake
  golden hashes the scene dict; `tests/test_part_families.py` pins `_HEIGHT_RATIO` to `height_ratio`.
- **N2 Render config dict re-resolved per image.** `config` is passed through
  `draw_board/_render_board/_render_photoreal_base` (`scene.py:35,82,135`) and read for
  `texture_dir`, `component_type`, `component_color`, `marking_probability`, `mm_per_px`,
  `camera`, `mask_gloss`, `rework`; `board_options` validates per image (`:51`); `get_look`
  runs 3x per image. Resolve once in `RenderContext.for_render` (typed fields) or a
  `ResolvedRender` cached on `RenderSpec`. Pixel-safe. M. **Done 2026-11-07:** `RenderContext`
  fields `texture_dir`, `component_type`, `component_color`, `marking_probability`, `mm_per_px`,
  `camera`, `mask_gloss`, `rework`, `board` (`BoardOptions`), resolved in `for_render` with one
  `get_look` (`camera_config`/`mask_gloss`/`rework_probability` accept the resolved `look=`).
  `draw_board`/`_render_board`/`_render_photoreal_base` no longer take `config`. No RNG change
  (`for_render` draws nothing from the image stream). `copper.mm_per_px()` is now unused.
- **N3 Scene prologue duplicated.** `_render_board:94-117` and `_render_photoreal_base:154-185`
  both do scaled_geometry, family_of, draw_board, body_geometry, boxPoints,
  solder_defect_layout; the latter has 10 args. Extract `_scene_setup() -> SceneGeometry`
  (frozen) and `_draw_part()`. Pixel-safe by whole-block moves. M.
  **Done 2026-11-07:** `scene.SceneGeometry` / `_scene_setup()` (scaling, family, `draw_board`, body pose + box,
  no rng after `draw_board`) used by both pipelines; `_draw_part()` holds the part block of
  `_render_photoreal_base`. RNG order and both golden tests unchanged.
- **N4 Backend dispatch.** Add a lazily imported `render_records` to `BackendInfo` (or a
  `RenderBackend` Protocol); removes `dataset.py:456-475` and `preview/render.py:34-35`. S.
  **Done 2026-11-07:** `BackendInfo.renderer`/`previewer` (`module:function`, imported on call;
  `RecordRenderer` protocol), adapters in `generation/render_backends.py`; no backend-name
  comparison left in `generate_dataset` / `render_profile_backend`; `render_previews` batches by
  the `needs_blender` capability. `tests/test_generation_backends.py`.
- **N5 `generate_profile_dataset` is a second `generate_dataset`** (`profile_dataset.py:35-168`):
  own profile loop, own sampling loop, hard-coded `profiles_dir`, `load_image_filters_env()`
  inside, ignores the R4 split. Reuse `resolve_run` + `plan_records` with an
  "rng continues into render" flag (keeps `:119-121`). M.
  **Done 2026-11-07:** `resolve_run(profile_id=, domain_cameras=)` per profile, `plan_records(domain=,
  continue_rng=)`, rendering via N4; `profiles_dir=` and mapping input; `$IMAGE_FILTERS` stays the
  default for `image_filters=None` (`scripts/generate_profile_dataset.py` does not pass it yet).
  Byte-identical: `tests/test_profile_dataset_golden.py` (hashes written by the old code).
- **N6 Defect types as raw strings** at 14 renderer sites (`scene.py:99,120,171-212`,
  `solder_defects.py:30,33`) and in `defects.py:85,94,118-127`, `batch_predict.py:518`,
  although `DefectClass(StrEnum)` exists. S. **Done 2026-11-07** for the renderer (`scene.py`,
  `solder_defects.py`, `draw.py`) and `defects.py` (dict keys and comparisons; `classify_defect`
  returns `.value`, i.e. plain `str`, so labels/YAML are unchanged). `batch_predict.py` was owned
  by another session and is left.
- **N7 Registries duplicated:** `LOOKS` tuple vs `_LOOKS` dict (`look.py:41,110`, validated via
  lazy import in `config.py:386`); `BOARD_ROUTERS` vs `ROUTERS` (`board/__init__.py:40`). Derive
  tuples from dicts. `Look`/`Packing` frozen dataclasses hold mutable arrays/dicts
  (`look.py:57-65`): `setflags(write=False)` + `MappingProxyType`. S. **Done 2026-11-07 for the
  looks:** `LOOKS = tuple(_LOOKS)`; `Look.__post_init__` marks every array (also the shared
  `palette` constants and `bodies` values) read-only and wraps `bodies`/`cap_len` in
  `MappingProxyType`; nothing mutated them (all suites pass). `Packing` holds only floats and
  tuples. Note: `Look` is no longer picklable (MappingProxyType); nothing pickles it (`RenderSpec`
  carries the config). `LOOKS` is still not cv2-free (`photoreal/__init__` loads cv2), so
  `gui/utils/pipeline_helpers.py` keeps loading cv2 for it; a cv2-free name list would have to live
  outside the `photoreal` package. `BOARD_ROUTERS` left to the board session.
- **N8 Profile editor duplicates geometry:** `preview_3d.py:374-523 _build_scene` (149 lines) and
  `blender_live_preview.py:150-169` each define `mid()` over `geometry_ranges`; pads hand-placed
  (X10) instead of `footprint_pads`/`geometry.pad_positions`; `_render:72-147` builds the Blender
  job by hand while `generation.blender.blender_job` exists. M.
- **N9 Three subprocess launch conventions in the GUI:** `run_pipeline.sh` env
  (`pipeline/logic.py:185-213`), `bash predict.sh` (`weights/logic.py:306`),
  `.venv/bin/python scripts/train.py` (`predictions/logic.py:101`), `python3 scripts/arena_report.py`
  (`weights/logic.py:97`). One `cli.argv_for(cmd, **opts) -> [sys.executable, "-m", "simple_sim", ...]`
  (goes with P2/P4). S-M.
- **N10 `pipeline/tab.py` god class** (1955-line class, 105 methods; `_build_run_specs` 171,
  `_confirm_pipeline_start` 142, `_start_pipeline` 108 lines). Tk-free `RunSpecBuilder` and
  `FilterProfileController`; `DatasetPicker`/`ModelPicker` widgets replace the five combo
  populators; dataset CRUD (`:1640-1880`) moves to the Datasets tab. L.
- **N11 Polling instead of the dispatcher:** `after(120/150/100)` tick loops in
  `pipeline/tab.py:1082`, `weights/tab.py:146`, `predictions/tab.py:206`, `profile_preview/tab.py:432`,
  `monitor.py:399`, `profile_editor/app.py:465,511`; logic classes take `queue.Queue` and
  `dict[str, StringVar]` although `UiDispatcher.post` (`gui/core/tasks.py:36`) exists. M.
- **N12 Helper tools outside the registry** (D21 grew to 5 copies of the image-folder loader:
  `sim_to_real`, `calibrate_camera`, `look_stats`, `eval_real`, `look_montage`, all importing
  `IMAGE_EXTENSIONS` from `photoreal/substrate.py`); tests load them via
  `spec_from_file_location` or `pythonpath=["."]`; `tools/mix_real.py:114-131` edits the manifest
  by hand. `simple_sim/images.py` (`IMAGE_EXTENSIONS`, `list_images`), `simple_sim/realism/`
  package, register the commands. S.
  **Done 2026-11-07** except the `simple_sim/realism/` package: `simple_sim/images.py`
  (`IMAGE_EXTENSIONS`, `is_image`, `list_images`) used by the four tools (`eval_real.class_folders`
  keeps its signature); the 7 tools are `COMMANDS` entries (`tests/test_cli.py` checks `--help`);
  `mix_real` uses `manifest.record_added_samples`. Open: `photoreal/substrate.py:17` still defines
  its own `IMAGE_EXTENSIONS` (renderer owner: `from simple_sim.images import IMAGE_EXTENSIONS`),
  the tool tests still load via `spec_from_file_location` / `pythonpath`, the tool logic is not
  yet a library package (P2).
- **N13 Packaging gap:** `pyproject.toml` packages only `simple_sim*`; `gui/monitor.py:15`,
  `profile_editor/app.py:15`, `filter_popup/preview_panel.py:258` mutate `sys.path`;
  `simple-sim gui` only works from a checkout. Package `gui` under the `[gui]` extra. M.

### 6.5 New findings: scalability

- **S1 Dataset JSONL parsed up to 5x per run.** `_BaseROIDataset.__init__` (`data_loader.py:156`)
  reads full meta+labels per split instance; `batch_predict --split all` = 3 datasets +
  `dataset_info` + `_gt_profile_map` -> labels 5x, meta 4x (measured 50k rows: ~5.4 s redundant).
  `read_jsonl` recomputes `dataclasses.fields()` per line (`schema.py:150-153`);
  `_get_label_index` does `class_names.index()` per `__getitem__` (`data_loader.py:244`).
  Fix: hoist the field set (S), one `DatasetIndex` (L11) shared by datasets, predictors and GUI (M).
  **Done 2026-11-07** (datasets + `batch_predict`; GUI = S6 still open): `simple_sim/dataset_index.py`
  (per-file cache keyed by path + inode/size/mtime/ctime, 32 files), used by `_BaseROIDataset`,
  `batch_predict.dataset_info/_gt_profile_map/build_dataset` and `sample_check`; field set hoisted;
  `_get_label_index` via dict. Measured on 12k rows (CPU): `--split all` data loading 2.09 -> 0.45 s,
  `read_jsonl(meta)` 473 -> 407 ms, labels 30 -> 21 ms; label lookup unchanged at 6 classes (~77 ns).
  Tests: `tests/test_dataset_index.py`.
- **S2 Per-call env/subprocess/file work in hot paths.** `telemetry.emit` reads the env var,
  `makedirs` and opens the file per batch (`telemetry.py:20-27`, called from
  `eval_utils.py:127`, `train.py:103`); `manifest._get_git_commit` spawns `git rev-parse` on
  every manifest write (`manifest.py:48-61`); `parallel.py:48,61,76`, `realism.py:135`,
  `filter_settings.py:234` read env inside library functions. Fix: `telemetry.set_sink()`,
  cached git commit as optional parameter, env resolution at the CLI boundary only. S.
  **Done 2026-11-07:** `telemetry.set_sink/reset_sink/current_sink`, sink resolved on the first
  event, file kept open and flushed per line (12.2 -> 3.0 us/event); `_get_git_commit` `lru_cache`
  (1.8 ms -> 1 us) + `git_commit=` on both manifest writers; `generate_profile_dataset(image_filters=)`.
  Kept on purpose: `parallel.py` (explicit args already win, env is the fallback) and
  `realism.py:135` (read inside render worker processes, which only inherit the environment; the
  real fix is carrying the presets in `RenderSpec`, R8). Tests: `tests/test_perf_caches.py`.
- **S3 Texture substrate decodes the photo per image** (`substrate.py:29 cv2.imread`); only the
  directory listing is cached. Per-process `lru_cache` of read-only decoded arrays keyed by
  path+mtime. Pixel-safe. S. **Done 2026-11-07:** `substrate.load_texture` /
  `_decoded_texture` (`lru_cache(8)`, key path + mtime_ns + size, read-only arrays; the caller only
  takes views and `cv2.resize` copies). `IMAGE_EXTENSIONS` now comes from `simple_sim.images`.
- **S4 `--extend` holds the whole old dataset in lists** (`dataset.py:186-187,488-489`;
  `records/tmp_meta/tmp_labels/meta_rows` coexist in `common.py:191-207`). Stream rows, keep only
  ids/splits of the old set. M.
  **Done 2026-11-07:** one streamed pass (`validate_jsonl_pair(on_meta=, on_label=)`, `iter_jsonl`);
  kept per old row only `LabelKey(id, class, profile)` (coverage + manifest counts need the class).
  15k rows, 32 px, +90 samples: 1.45 -> 0.74 s, RSS 390 -> 133 MiB, tracemalloc peak 237 -> 8 MiB.
  `build_rows` still holds `tmp_meta` for the new records only (not the old set).
- **S5 Report/arena scans re-read and re-walk.** `reports.load_reports` parses every report per
  call (`reports.py:44-63`, called by `cli.status`, `arena`, `batch_predict.history_rows`);
  `arena.best_rows_by_model` re-parses timestamps inside sort keys (`:176-187`);
  `path_size_bytes` walks a dataset dir per dataset per model per report (`arena.py:189-194,261-263`).
  mtime-keyed caches. S-M.
  **Done 2026-11-07:** `load_reports` caches parsed files per path + inode/size/mtime (fresh
  top-level dicts per call; 400 reports x 300 rows: 64 -> 2 ms warm, 67 ms cold);
  `best_rows_by_model` parses each timestamp once. `path_size_bytes` is not cached across calls (a
  directory mtime does not change when a file in it grows, so an mtime key would go stale; within a
  report each path was already walked once); instead a `scandir` walk, same sum (20k files 144 -> 33 ms).
- **S6 Blocking I/O on the Tk thread.** `pipeline/tab.py:1603-1609` reads whole
  `meta.jsonl`/`labels.jsonl` per dataset selection; `pipeline/ui.py:519-539` decodes 12 images
  + overlays per refresh on the UI thread; `_refresh_models:1290-1298` stats every checkpoint
  synchronously; `weights/logic.py:467-515` reopens manifests per dataset; profile editor
  parses every YAML every 1.4 s (`profile_registry.py:54-70` from `app.py:511`). `DatasetIndex`
  via TaskRunner, thumbnails in a worker, stat-first mtime cache (L10 `catalog.py`). M.
- **S7 Per-image render hot spots (profile, 256² ss 2: legacy 55.7 ms, quins 49.8 ms):** `blend`
  ~10 ms/img (`cv2.boundingRect((alpha>0))` on full-frame alphas, `imageops.py:48`),
  `procedural_substrate` ~9 ms (rng-bound, not skippable), `camera_model` 7 ms,
  `meniscus.shade` 6 ms. Pixel-safe: `blend(..., window=)` so callers that know the box skip
  the boundingRect. S. `RenderSpec` pickles to 370 B, so pool overhead is not an issue.
  **Measured 2026-11-07, not done:** per 256² image (30 renders, ss 2) the `boundingRect` inside
  `blend` costs 0.4 ms (legacy, 37 ms/img) / 0.4 ms (quins, 41 ms/img), ~1 %; the ~10 ms is the
  blend arithmetic itself (largest: `copper.py:35` full-frame copper, 2.2-3.4 ms, whose alpha
  really covers most of the frame). A `window=` argument would save < 1 % and is not worth the
  extra API; revisit only if the copper layer gets a known small box.
- **S8 `model_bundle` does a full `torch.load` per `.pt` to list a bundle** (`model_bundle.py:103-113`;
  plan L8). Sidecar JSON or `torch.load(..., mmap=True)` of the metadata only. S.
  **Done 2026-11-07** with `mmap=True` (no new files in bundles), full-load fallback for legacy
  non-zip checkpoints: 5 ResNet-18 checkpoints 640 -> 164 ms cold, 88 -> 24 ms warm.

### 6.6 New findings: correctness found on the way

- **C1** `sample_check.choose_samples` selects the split by substring `f"/{split}/" in sample_id`
  (`sample_check.py:42`) instead of `splits.read_split`, so it diverges from `ROIDataset` for
  merged or re-split datasets. S. **Done 2026-11-07:** split from `DatasetIndex.split_ids`
  (missing split file -> no samples); regression test with a re-split dataset; identical
  selection when ids name their split.
- **C2** `augment.apply_image_filter_overrides` is a 144-line ladder (`augment.py:13`); table-drive
  it in the existing key order (rng draw order preserved). M.
- **C3** The divergent modules of section 2 have no tests at all (`gui/utils/model_inference.py`,
  `validation_suite.py`, `filter_popup/preview_render.py`, `dataset_store.py`, `two_stage.py`,
  `telemetry.py`, `preview/viewer_tk.py`), so X1-X3 cannot be fixed with a regression net.
  Write the tests against the library targets while doing L4/L7. S each.

### 6.7 Test-suite health

178 s total; `test_render_golden_variants` is one 48 s test (589 cases, no ids);
`test_eval_utils::test_ensemble_and_bundle_eval_predict` and `test_script_cores_end_to_end`
each 20 s (generate + train resnet18); `test_headless_imports` 5 s per torch-loading param.
No `tests/conftest.py`: dataset fixtures hand-rolled in 4 files, a `"roi":` config dict repeated
in 6, `generate_dataset` called for fixtures in 10. Fix: session-scoped `tiny_dataset`/`trained_model`
fixtures, a `slow` marker documented in AGENTS.md, parametrised variants. M.

### 6.8 Recommended order

1. G1 (regression) and U1-U7 before the current round is committed; they are small and the
   golden tests guard them.
2. Phase 2 in the order S1 + L11 `DatasetIndex` (unblocks S6, X7, D6), S2, D1/D2/D7/D9/D10/D16,
   I4 (`board/__init__.py` lazy, PEP 562), I5, I7, N12, S3, S8.
3. N1 + N2 + N3 (renderer extension points), then the open phase-6 leftovers (D27, D30/U7,
   D33, D34, D35 via the baked `FAMILIES` table, R3 via N4, R4 via N5).
4. Phase 4/5 together: L1/L2/L3 into `simple_sim/training.py`, `evaluation.py`, `inference.py`
   (closes X1, X4, G1 for good), then P2 registry + N9 `cli.argv_for` (closes P4, D22, X9, X11).
5. GUI: N11 dispatcher first (mechanical, per tab), then N10 split, then S6.
