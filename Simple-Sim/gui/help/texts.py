"""All user-facing help texts, keyed by ``<tab>.<control>`` (tk-free, i18n-ready).

Style: first sentence says what the control does; optional second sentence gives
an example or a consequence. Keep each text short enough for a tooltip.
``SECTION_HELP`` holds the longer texts shown by a section's "?" button.
"""

from __future__ import annotations

HELP: dict[str, str] = {
    # ---------------------------------------------------------------- common
    "common.refresh": "Reload the list from disk.",
    "common.browse": "Choose a path with a file dialog.",
    "common.dataset": "Dataset to work on (folder under outputs/sim_data).",
    "common.model": "Model checkpoint (.pt) or multi-profile bundle (.bundle).",
    "common.split": "Dataset split to use: train, val, test or all samples.",
    "common.device": "Compute device. 'auto' uses the GPU when one is available, else the CPU.",
    "common.prev": "Previous item (Left arrow).",
    "common.next": "Next item (Right arrow).",
    "common.info": "Show details.",
    "common.copy_log": "Copy the full log to the clipboard.",
    "common.clear_log": "Clear the log view (files on disk are not touched).",
    # -------------------------------------------------------------- pipeline
    "pipeline.start": "Run the full pipeline: generate -> validate -> train -> eval (F5).",
    "pipeline.generate_only": "Generate and validate the dataset only. No training or evaluation.",
    "pipeline.stop": "Stop the running process and all of its child processes (Shift+F5).",
    "pipeline.filters": "Configure image filters applied during generation (rotation, blur, grain, brightness, ...).",
    "pipeline.run_mode": (
        "Single: one run. Multiple: run 'Count' times in a row. Continuous: repeat until Stop. "
        "Precise: exact sample counts per class / per profile."
    ),
    "pipeline.run_count": "Number of runs in 'Multiple' mode.",
    "pipeline.precise": "Set exact sample counts per class and per profile for 'Precise' mode.",
    "pipeline.dataset_mode": (
        "Create New writes a fresh dataset to 'Output'. "
        "Extend Existing appends new samples to the dataset in 'Output'."
    ),
    "pipeline.config": "Run config (YAML) that defines classes, counts, rendering and training.",
    "pipeline.out": "Folder the dataset is written to, e.g. outputs/sim_data/runs/run_0001.",
    "pipeline.model": "Checkpoint the training writes to (and evaluation reads from).",
    "pipeline.model_add": "Create a new, empty model target name (.pt or .bundle).",
    "pipeline.autosnap": "Keep a copy of the model every N runs, so a later run cannot overwrite a good model.",
    "pipeline.snap_every": "Take a model snapshot after every N-th run.",
    "pipeline.snap_keep": "How many snapshots to keep; the oldest are deleted.",
    "pipeline.name": "Base name for the dataset folder and the model file.",
    "pipeline.name_ts": "Append a timestamp (YYYYmmdd_HHMMSS) so names never collide.",
    "pipeline.bundle": "Train into a multi-profile bundle (.bundle) with one checkpoint per component profile.",
    "pipeline.apply_name": "Fill 'Output' and 'Model' from Name (+ timestamp, + bundle).",
    "pipeline.profile": "Component profile to render (e.g. resistor_0603). Picks a matching config automatically.",
    "pipeline.profile_info": "Show the selected profile's geometry and settings.",
    "pipeline.render": "opencv_2d: fast 2D rendering. blender_3d: photoreal 3D rendering (needs Blender, slower).",
    "pipeline.look": (
        "render.look override: legacy = default materials, quins = materials and scanner joints tuned to the "
        "QUINS scans. 'from config' keeps the run config's value. Applied through a temporary config under "
        "outputs/live; extending a dataset with another look is refused like any config change."
    ),
    "pipeline.board_routing": (
        "render.board.routing override: legacy = random copper (no board model), none = no copper, "
        "escape = board model with nets, routed tracks, pours and vias. 'from config' keeps the run config's value."
    ),
    "pipeline.board_drc": (
        "render.board.drc override (board model only): off, warn = log violations and store the result per sample "
        "in meta.jsonl (render_meta.board_drc), strict = redraw the routing until it is clean."
    ),
    "pipeline.board_rules": (
        "render.board.design_rules override (board model only): consumer_dense, industrial or hdi "
        "(track widths, clearances, via sizes)."
    ),
    "pipeline.multi_profile": "Generate several component profiles in one go.",
    "pipeline.multi_mode": (
        "separate: one dataset per profile. "
        "mixed: one combined dataset with all profiles (Generate Only)."
    ),
    "pipeline.multi_pick": "Choose which profiles to generate.",
    "pipeline.dataset_multi": "Select several datasets at once (train one model on all of them).",
    "pipeline.dataset_multi_pick": "Choose the datasets for multi-dataset training.",
    "pipeline.ds_stats": "Dataset size on disk and number of samples. Colour shows whether the size is enough for training.",
    "pipeline.ds_snapshot": "Freeze a copy of the dataset under outputs/sim_data/versions (hard links, fast).",
    "pipeline.ds_rename": "Rename the dataset folder (and matching model files).",
    "pipeline.ds_delete": "Delete the whole dataset folder. Cannot be undone.",
    "pipeline.ds_delete_images": "Delete selected images (and their labels) from the dataset.",
    "pipeline.ds_move_images": "Move selected images to another dataset.",
    "pipeline.ds_create": "Create a new, empty dataset folder.",
    "pipeline.datasets_manage": "Snapshot, rename, create or delete datasets and move/delete images.",
    "pipeline.validate": "Check the selected dataset (schema, files, splits, class balance).",
    "pipeline.train": "Train 'Model' on the selected dataset(s).",
    "pipeline.eval": "Evaluate 'Model' on the test split of the selected dataset.",
    "pipeline.continue_epochs": "Extra epochs when continuing training from an existing checkpoint.",
    "pipeline.continue_out": "Which checkpoint to continue from: 'last' (end of last run) or 'best' (best validation).",
    "pipeline.continue_train": "Resume training of 'Model' for '+Epochs' more epochs.",
    "pipeline.profile_model": "Component-type classifier used to detect the profile of unknown images.",
    "pipeline.profile_model_lock": "Keep this profile model selected (it is not changed automatically).",
    "pipeline.build_preset": "quick: small dataset, few epochs (minutes). full: larger dataset, better accuracy.",
    "pipeline.build_profile_model": "Generate a profile dataset and train the component-type classifier.",
    "pipeline.dataset_profile_info": "Show which component profile(s) the selected dataset was generated with.",
    "pipeline.profile_compat": "Warns when the dataset's profile does not match the model's profile.",
    # ---------------------------------------------------------------- datasets
    "datasets.category": "Filter the list by category.",
    "datasets.set_category": "Assign a category to the selected datasets (for grouping).",
    "datasets.archive": "Move the selected datasets out of the active list (files are kept).",
    "datasets.unarchive": "Bring archived datasets back into the active list.",
    "datasets.select": "Make the selected dataset the active dataset in all tabs.",
    "datasets.refresh": "Rescan outputs/sim_data for datasets and reload categories and archive state.",
    "datasets.details": "Show sample, split and per-profile counts of the selected dataset (or double-click a row).",
    "datasets.categories_menu": "Create, rename or delete categories. Deleting a category moves its datasets to the default category.",
    "datasets.show_archived": "Also list archived datasets (shown greyed out). Turn off to hide them.",
    # ---------------------------------------------------------------- analysis
    "analysis.class_filter": "Show only samples of this class.",
    "analysis.recompute": "Re-run the model on all samples, taking your feedback (relabels) into account.",
    "analysis.batch": "Run the model on every sample and store accuracy per class.",
    "analysis.delete": "Delete the selected samples (image files and labels) from the dataset. Asks first; cannot be undone.",
    "analysis.move": "Move the selected samples to another dataset with the same component profile.",
    "analysis.search": "Filter the list by sample ID or class while you type.",
    "analysis.search_clear": "Clear the search text and show all samples that match the other filters.",
    "analysis.run_filter": "Show only samples of this generation run.",
    "analysis.domain_filter": "Show only samples of this domain (rendering variant).",
    "analysis.split_filter": "Show only samples of this split (train, val or test).",
    "analysis.sort": "Order of samples within each split: by file date or by ID.",
    "analysis.profile_filter": "Show only samples of this component profile. Picking one turns the multi-profile filter off.",
    "analysis.profile_multi": "Filter by several component profiles at once; overrides the single Profile filter.",
    "analysis.feedback_up": "Mark the model's view of this sample as correct (stored in the dataset's feedback log).",
    "analysis.feedback_down": "Mark this sample as wrongly labelled and choose the correct class; the new label is used by Recompute.",
    "analysis.feedback_history": "Show all feedback entries of this dataset; double-click one to jump to the sample.",
    "analysis.feedback_recent": "Your latest feedback entries; click one to open that sample.",
    "analysis.feedback_class": "The class this sample really shows; it replaces the label for feedback-aware analysis.",
    "analysis.feedback_note": "Optional free-text note stored with the feedback entry.",
    # -------------------------------------------------------------- validation
    "validation.run": "Run all quality checks: schema, image outliers (blur/brightness), duplicates, edge cases.",
    "validation.export": "Save the validation results as a JSON report.",
    "validation.dataset": "Dataset to validate; choosing one here makes it the active dataset in all tabs.",
    "validation.refresh": "Rescan outputs/sim_data for datasets.",
    "validation.check_schema": "Check that meta.jsonl, labels.jsonl and the splits are consistent and well-formed.",
    "validation.check_outliers": "Flag images whose blur or brightness is far from the dataset average.",
    "validation.check_duplicates": "Flag near-identical image pairs found by perceptual hashing (slow on large datasets).",
    "validation.check_edge_cases": "Flag samples whose shift, rotation or tilt lies close to a class threshold.",
    "validation.remove_flag": "Remove the selected flag from flags.jsonl of this dataset.",
    "validation.clear_flags": "Remove every flag of this dataset (asks first). Images and labels are not changed.",
    # ------------------------------------------------------------- predictions
    "predictions.run": "Predict every sample of the chosen split and compute accuracy, macro-F1 and the confusion matrix.",
    "predictions.profile_model": "Optional component-type classifier; with a bundle it routes each image to the right checkpoint.",
    "predictions.max_samples": "Limit the number of samples for a quick check; leave empty to predict all samples.",
    "predictions.stop": "Stop the running batch prediction; results written so far are kept.",
    "predictions.dataset_multi": "Predict on several datasets in one run; results appear per dataset in a table.",
    "predictions.dataset_multi_pick": "Choose the datasets used when Multi is on.",
    "predictions.save_preds": "Write one prediction per sample to a JSONL file; needed for the sample list and image viewer.",
    "predictions.auto_train_bundle": "If a bundle has no checkpoint for a profile, train one before predicting (takes longer).",
    "predictions.show_filter": "Filter the sample list: all, wrong or correct class predictions, or wrong profile predictions.",
    "predictions.ds_results": "Metrics per dataset of the last multi-dataset run; select a row to load its report and samples.",
    # ----------------------------------------------------------------- weights
    "weights.evaluate": "Evaluate the selected checkpoint on the current dataset (predict.sh) and add the report to the history.",
    "weights.compare": "Evaluate Model A, then Model B on every selected dataset and log a side-by-side metric comparison.",
    "weights.export": "Save a copy of the selected checkpoint (.pt) or bundle (as .zip) to a location of your choice.",
    "weights.import": "Copy a checkpoint (.pt) or zipped bundle into outputs/models so it appears in the list.",
    "weights.backup": "Store a versioned copy of the checkpoint under outputs/models/versions.",
    "weights.refresh": "Rescan outputs/models and rebuild the checkpoint list (groups, favorites, Arena marks).",
    "weights.snapshot": "Copy the active model to outputs/models/versions/<name>/ with a timestamp, so later training cannot overwrite it.",
    "weights.activate": "Copy the selected checkpoint over the active model path; the current active model is overwritten.",
    "weights.profile_info": "Show the component profile, classes and training metadata stored in the selected checkpoint.",
    "weights.delete": "Delete the selected checkpoints from disk; favorites and Arena-tracked models are protected.",
    "weights.status": "State of the last evaluation or comparison started from this tab.",
    "weights.dataset": "Dataset used for Run Selected; change it in the Pipeline tab.",
    "weights.active_model": "Checkpoint that Snapshot copies from and Activate writes to (defaults to outputs/models/<dataset>.pt).",
    "weights.tree": "Checkpoints grouped by Favorites, Snapshots, Imports and your own groups; right-click for actions, drag onto a group to move.",
    "weights.max_samples": "Evaluate at most this many samples for a quick check; leave empty to use the whole split.",
    "weights.save_preds": "Also write per-sample predictions (one row per image) next to the report, not only the summary metrics.",
    "weights.stop": "Stop the running evaluation or comparison; remaining datasets of a comparison are skipped.",
    "weights.compare_a": "First model of the comparison (run first).",
    "weights.compare_b": "Second model of the comparison (run after A).",
    "weights.compare_datasets": "Datasets to compare on; select several to get a weighted summary. Nothing selected = current dataset.",
    "weights.compare_refresh": "Reload the list of datasets under outputs/sim_data.",
    "weights.compare_use_current": "Select only the dataset that is active in the Pipeline tab.",
    "weights.compare_select_all": "Select every dataset in the list (a comparison then runs on all of them).",
    "weights.reports_refresh": "Reload report files from disk and apply the Scope / Split / Sort filters.",
    "weights.report_scope": "Which reports to list: those of the selected model, of the current dataset, or all.",
    "weights.report_split": "Only list reports of this split ('any' shows all).",
    "weights.report_sort": "Order of the list: newest first, or best accuracy / macro F1 first.",
    "weights.report_summary": "Print a short summary of the selected report into the log below (double-click a row does the same).",
    # ------------------------------------------------------------------- merge
    "merge.bundle": "Copy the selected checkpoints into a new <name>.bundle folder; at prediction time each image is routed to its profile's checkpoint.",
    "merge.ensemble": "Pack the selected checkpoints into one <name>_ensemble.pt file; all of them must use the same class names.",
    "merge.refresh": "Rescan outputs/models for checkpoints and bundles and rebuild both lists.",
    "merge.tree": "Checkpoints grouped by component profile; double-click a model to select it for its profile.",
    "merge.select": "Use the highlighted model as the weight for its profile (replaces an earlier choice for that profile).",
    "merge.deselect": "Remove the weight chosen for the highlighted profile from the merge selection.",
    "merge.name": "Base name of the new bundle or ensemble file under outputs/models.",
    "merge.timestamp": "Append the current date and time to the name so an existing bundle is never overwritten.",
    "merge.bundle_info": "Show the profiles, source checkpoints and metrics stored in the selected bundle.",
    "merge.load_bundle": "Replace the current merge selection with the checkpoints of the selected bundle.",
    "merge.delete_bundle": "Delete the selected bundle folder and all checkpoints inside it from disk.",
    # --------------------------------------------------------- board detection
    "board_detection.profile_model": "Profile classifier checkpoint (.pt) that recognises which component type the image shows.",
    "board_detection.bundle": "Multi-profile defect bundle (.bundle folder); the checkpoint for the detected profile classifies the defect.",
    "board_detection.image": "Image to classify, e.g. a ROI from outputs/sim_data or a real photo.",
    "board_detection.device": "Compute device for both models; cuda needs a GPU-enabled torch build.",
    "board_detection.confidence": "Profile confidence threshold (0..1); results below it get a review flag.",
    "board_detection.predict": "Load both models (cached between runs) and classify the image: profile first, then defect.",
}

SECTION_HELP: dict[str, str] = {
    "pipeline.source": (
        "Where the data comes from.\n\n"
        "Profile selects the component (e.g. a 0603 resistor). Config is the run YAML with class "
        "counts and render settings; it is picked automatically when you change the profile. "
        "Render chooses fast 2D (OpenCV) or photoreal 3D (Blender)."
    ),
    "pipeline.output": (
        "Where results go.\n\n"
        "Create New writes a fresh dataset, Extend Existing appends to it. Use Name + Apply to "
        "fill the dataset and model paths consistently; Timestamp avoids overwriting old runs."
    ),
    "pipeline.run": (
        "How often to run.\n\n"
        "Single runs once. Multiple runs Count times. Continuous repeats until you press Stop. "
        "Precise lets you set exact sample counts per class and profile."
    ),
    "pipeline.model": (
        "The model that is trained and evaluated.\n\n"
        "Bundle trains one checkpoint per component profile. Auto-snapshot keeps copies of the "
        "model every N runs so a bad run cannot destroy a good model."
    ),
    "pipeline.datasets": (
        "Work with existing datasets.\n\n"
        "Pick a dataset, then Validate, Train or Eval it. Enable Multi to train one model on several "
        "datasets. The Manage menu holds snapshot, rename, delete and image operations."
    ),
    "pipeline.profile_model": (
        "Component-type classifier.\n\n"
        "It recognises which component profile an image shows and is used for two-stage prediction "
        "with bundles. Build one with the quick preset to get started."
    ),
    # ---------------------------------------------------------------- analysis
    "analysis.images": (
        "Browse the samples of the selected dataset.\n\n"
        "Samples are grouped by run, domain and split. Combine the filters to narrow the list; "
        "Left/Right on a sample (or in the viewer) steps to the previous/next image. "
        "Right-click a sample to delete or move it."
    ),
    "analysis.feedback": (
        "Correct the dataset labels by hand.\n\n"
        "Thumbs up confirms a sample, thumbs down stores the correct class. Feedback is kept in a log "
        "next to the dataset; Recompute With Feedback uses the corrected labels."
    ),
    # ------------------------------------------------------------- predictions
    "predictions.source": (
        "What to predict with.\n\n"
        "Choose the dataset (or several with Multi) and the model checkpoint or bundle. The optional "
        "profile model detects the component type first and routes each image to the right checkpoint."
    ),
    "predictions.results": (
        "Per-sample results of the last run.\n\n"
        "Wrong predictions are shown in red. Select a row to see the image with its overlay; "
        "Left/Right steps through the list. With Multi, pick a dataset in the table above first."
    ),
    # ---- weights
    "weights.checkpoints": (
        "All checkpoints and bundles under outputs/models.\n\n"
        "Favorites and Arena-tracked models (marked [A]) cannot be deleted here. Right-click for "
        "favorites, Arena, compare, rename, duplicate and group actions; drag a model onto a group "
        "header to move it."
    ),
    "weights.evaluate_section": (
        "Evaluate the selected checkpoint.\n\n"
        "Runs predict.sh on the current dataset and split. Reports are stored under "
        "<dataset>/predictions/weights_tab and appear in History / Reports."
    ),
    "weights.compare_section": (
        "Compare two models.\n\n"
        "Model A and Model B are evaluated one after the other on each selected dataset; datasets whose "
        "classes do not match a model are skipped. The comparison is written to the log."
    ),
    "weights.reports_section": (
        "Evaluation reports found on disk.\n\n"
        "Filter by scope and split, sort by date or metric. Right-click a report to delete its file."
    ),
    # ---- merge
    "merge.weights": (
        "Checkpoints sorted by component profile.\n\n"
        "Pick one weight per profile (double-click). The chosen weight is marked with * and shown in green."
    ),
    "merge.selection": (
        "The weights that will be merged, one per profile.\n\n"
        "Load an existing bundle to start from its checkpoints."
    ),
    "merge.merge": (
        "Create the merged model.\n\n"
        "Bundle: a .bundle folder with one checkpoint per profile (recommended for two-stage prediction). "
        "Ensemble: a single .pt file that contains all selected checkpoints."
    ),
    "merge.bundles": (
        "Bundles under outputs/models.\n\n"
        "Double-click for details. Delete removes the bundle folder from disk."
    ),
    # ---- datasets / validation / board detection
    "datasets.list": (
        "All datasets under outputs/sim_data (runs and snapshots).\n\n"
        "Categories and the archive flag only affect how other tabs group and list datasets; "
        "no files are moved or deleted. Select several rows to change them together; right-click "
        "for a menu, double-click for details."
    ),
    "validation.checks": (
        "Quality checks for the selected dataset.\n\n"
        "Schema validation checks the files. Image Outliers, Duplicate Detection and Edge Cases "
        "add flags to the dataset's flags.jsonl, listed under Flagged Samples."
    ),
    "validation.flags": (
        "Samples flagged by the last validation runs.\n\n"
        "Flags are stored in flags.jsonl in the dataset folder and stay until you remove them. "
        "They do not change images or labels."
    ),
    "board_detection.models": (
        "Two-stage prediction.\n\n"
        "The profile classifier first decides which component type the image shows; the matching "
        "checkpoint of the defect bundle then classifies the defect. Train both in the Pipeline tab."
    ),
    "board_detection.input": (
        "Image and run options.\n\n"
        "When the profile confidence is below Min Confidence, the result is marked for manual review."
    ),
    "board_detection.results": (
        "Probabilities of both stages, highest first.\n\n"
        "A review flag means the component type was uncertain, so the defect result may come from the wrong checkpoint."
    ),
}

QUICK_START = (
    "Quick start\n\n"
    "1. Pipeline -> Source: choose a component profile.\n"
    "2. Output: enter a Name and press Apply.\n"
    "3. Press Start Pipeline (F5). The log shows generate -> validate -> train -> eval.\n"
    "4. Predictions: run the model on the test split and look at the confusion matrix.\n"
    "5. Analysis: browse wrong predictions and give feedback.\n\n"
    "Tip: hover over any control to see what it does; the status bar shows the same hint."
)

SHORTCUTS: list[tuple[str, str]] = [
    ("F1", "Help / quick start"),
    ("F5", "Start pipeline (Pipeline tab)"),
    ("Shift+F5", "Stop the running pipeline process"),
    ("Esc", "Close dialog"),
    ("Ctrl+R", "Refresh the current tab"),
    ("Ctrl+1 ... Ctrl+9", "Switch page (order of the sidebar entries)"),
    ("Ctrl+Q", "Quit"),
    ("Ctrl+Shift+D", "Toggle dark mode"),
    ("Left / Right", "Previous / next image (Analysis, Predictions)"),
]


def help_text(key: str | None) -> str:
    return HELP.get(key or "", "")


def section_help(key: str | None) -> str:
    return SECTION_HELP.get(key or "", "")
