# Model Merge: Bundles vs Ensembles

This document explains the two supported "merge" outputs in Simple-Sim and how they behave during
training, prediction, and evaluation.

## Background: Profiles and Checkpoints

Simple-Sim is profile-based. Each dataset has a `dataset_manifest.json` that declares:

- `component_profile.profile_id` (example: `chip_0603_resistor@1`, `sot23_transistor@1`, `qfn32_ic@1`)
- `component_profile.profile_hash` (guard against accidental mixing)

Training writes checkpoints that include the profile metadata in the checkpoint under:

- `checkpoint["component_profile"]["profile_id"]`
- `checkpoint["component_profile"]["profile_hash"]`

The standard (single) checkpoint is a single `.pt` file.

## Key Concept: "Single .pt" vs "Bundle" vs "General Model"

It is easy to mix up these terms:

- A **single `.pt`** is just one trained model. By default it does not "scan" or "select" a profile.
  You can run it on any image, but if the image is out-of-distribution (OOD) (for example a component
  type it never saw during training), the prediction can be unreliable or overly confident.

- A **bundle** is a directory that contains multiple `.pt` files (one per profile). A bundle can be
  *better* than a single profile-specific `.pt` when the component profile is known, because you can
  dispatch to the correct specialist model. A bundle is not automatically better for unknown images
  unless you also have a component-type classifier (or another resolver) in front of it.

- A **general model** (one model that handles many component types well) is not created by "merging"
  checkpoints. It requires training on mixed data (multiple profiles) and usually needs an explicit
  UNKNOWN/OOD strategy if you expect truly unseen component types in production.

## Option A: Multi-Profile Bundle Directory (`*.bundle/`)

A bundle is a directory containing one checkpoint per profile. Convention:

- `outputs/models/<name>.bundle/<profile_id>.pt`
- `outputs/models/<name>.bundle/<profile_id>_last.pt` (optional)
- `bundle.json` and `bundle_details.json` (metadata)

### How prediction/evaluation works with a bundle

When you pass a bundle directory as `--model`, scripts resolve the dataset profile and load the
matching checkpoint inside the bundle:

1. Read `dataset_manifest.json` from the dataset directory.
2. Resolve the expected checkpoint path: `<bundle>/<dataset_profile_id>.pt`
3. Load that checkpoint and run prediction/eval.

If the bundle does not contain `<dataset_profile_id>.pt`, prediction/eval fails because there is
no weights file to load for that dataset profile.

### Bundles and "unknown images"

Bundles do not magically solve unknown component types. A bundle still needs a way to decide which
profile checkpoint to use:

- **Dataset scoring**: the profile comes from `dataset_manifest.json`, so dispatch is deterministic.
- **Free images (no dataset/manifest)**: you must add a component-type resolver (for example a
  profile classifier + optional OOD/UNKNOWN detection), then dispatch to the corresponding bundle
  checkpoint.

### How training works with a bundle

If you train with `--out <name>.bundle`, training writes into the bundle using the dataset profile:

```bash
cd Simple-Sim
./.venv/bin/python scripts/train.py \
  --data outputs/sim_data/runs/Transistor-v1 \
  --out outputs/models/my_multi.bundle
```

That creates/updates:

- `outputs/models/my_multi.bundle/sot23_transistor@1.pt`

### When bundles are the right choice

- Production-style packaging where you want "one model selection" but still keep weights isolated
  per component profile.
- You want strict prevention of cross-profile weight mixing.

## Option B: Single-File Ensemble Checkpoint (`*_ensemble.pt`)

An ensemble is a single `.pt` file containing multiple sub-model checkpoints. At inference time,
Simple-Sim averages the sub-model logits.

This output is created by the Merge tab button:

- `Merge -> Single .pt (Ensemble)`

Example output:

- `outputs/models/<name>_ensemble.pt`

### How prediction/evaluation works with an ensemble

When a script loads a checkpoint with:

- `checkpoint["format"] == "simple_sim_ensemble_v1"`

it builds an ensemble model and averages logits across all sub-models. No dataset profile dispatch
is required (it is one `.pt`).

Important constraints:

- All sub-models must have the same `class_names` list (same defect classes).
- This does not "learn" new profiles; it only aggregates the behavior of models you included.

### When ensembles are the right choice

- Quick cross-profile scoring where you do not care about strict per-profile dispatch.
- Rapid experimentation for "how does the system behave if I combine multiple experts".

### When ensembles are NOT the right choice

- True "unknown component type" generalization requirements in production.
- Memory/performance constrained deployments (multiple sub-models are heavier than one).

## GUI Notes

### Predictions Tab: "Auto-train missing bundle ckpt"

If you select a bundle for prediction and it is missing the dataset's `<profile_id>.pt`, the GUI can
optionally train the dataset into the bundle automatically (this adds the missing checkpoint).

This option is only meaningful for bundle directories. It is not used for single `.pt` models.

## Practical Workflow

1. Generate/train separate datasets per profile (recommended).
2. Decide packaging:
   - Bundle for production-like dispatch by profile.
   - Ensemble for experimentation.
3. Use Merge tab to create the chosen artifact.
4. Run `predict.sh` / `scripts/batch_predict.py` against any dataset you want to score.
