
## 1) Principle for a simple simulator

Do **not** start with “free-form diffusion prompt → random PCB images”. That produces visually plausible nonsense with unreliable geometry and labels.

For a simple simulation that actually tests the concept, you need:

* **Deterministic geometry** (pads, component bodies, traces)
* **Deterministic defect injection** (missing, offset, tombstone, bridge)
* **Automatic labels** (ground truth)
* **Reproducible generation** (seed + config)
* **Train/eval loop** with metrics

This can be done entirely in **Python + OpenCV** for a first concept test. 3D rendering is optional and can be added later.

---

## 2) Minimal “Simple-Sim” architecture

### Modules

1. **Generator**

   * produces synthetic ROI images (or full board images)
2. **Defect injector**

   * applies parametric defects with known labels
3. **Dataset store**

   * writes images + metadata + labels (JSONL)
4. **Trainer**

   * trains a classifier on generated data (OK vs defect classes)
5. **Tester**

   * evaluates on a fixed test set and prints accuracy/precision/recall/F1 + confusion matrix
6. **Config + seed control**

   * ensures reproducibility for generation + training

### Why ROI-first

Training on full boards is unnecessary complexity. For AOI, ROI-based learning is standard and scales.

---

## 3) What defects to simulate first (simple but meaningful)

Start with four classes:

* `OK`
* `MISSING` (no component body)
* `MISALIGNED` (component shifted relative to pads)
* `TOMBSTONE` (component rotated upright / vertical posture proxy)
* (Optional later) `BRIDGE`

For MVP, you do not need physically correct solder. You need **distinct, learnable visual patterns** and an architecture that can later be fed real ROIs.

---

## 4) Simple image model (2D) that works for concept testing

### Rendering concept

Render an ROI with:

* background substrate
* copper pads (two rectangles)
* solder paste/fillet proxy (blurred highlights)
* component body (rectangle with subtle gradient)
* optional silkscreen marking

Then apply camera-like effects:

* blur
* noise
* brightness/contrast variation
* slight rotation jitter

This gives enough variability for a classifier to learn.

### Tombstone proxy (for MVP)

Simulate tombstoning by:

* rotating component body to near-vertical orientation (e.g., 80–90 degrees)
* changing shadow/highlight pattern on one side
* reducing contact area on one pad

It will not be physically perfect, but it tests the full pipeline.

---

## 5) Dataset “data bank” format

Use a strict folder layout and JSONL metadata.

```
sim_data/
  configs/
    run_0001.yaml
  splits/
    train.txt
    val.txt
    test.txt
  images/
    000000.png
    000001.png
    ...
  meta.jsonl
  labels.jsonl
```

### Example `meta.jsonl` row

* `id`
* `seed`
* `defect`
* defect parameters
* augmentation parameters

This is what makes the dataset reproducible and auditable.

---

## 6) Reproducibility: config + seed policy

Use a single YAML config per run:

* global seed
* number of samples per class
* image size
* pad geometry ranges
* defect parameter ranges
* augmentation ranges
* train hyperparameters
* split definitions

Rule:

* **Generation uses seed + sample_id** (deterministic)
* **Training uses fixed seed** (deterministic)

You can then recreate any dataset and any model exactly.

---

## 7) Training + testing (Python)

### Recommended software (free)

* `python>=3.10`
* `numpy`
* `opencv-python`
* `torch`, `torchvision`
* `scikit-learn`
* `pyyaml`
* (optional) `tqdm`, `matplotlib`

### Baseline model for MVP

* A small CNN (e.g., ResNet18 from torchvision) on ROI images
* Loss: cross-entropy
* Metrics:

  * accuracy
  * per-class precision/recall/F1
  * confusion matrix
  * “false negative rate” for critical defects (tombstone/missing)

### Tester behavior

* Always evaluate on a fixed `test` split that never changes
* Print current metrics after each training run
* Save metrics report as JSON + optionally HTML

---

## 8) Alternating good/bad samples (as you requested)

Do it deterministically, not randomly.

Two valid approaches:

### A) Balanced generation

Generate exactly `N_ok`, `N_missing`, `N_misaligned`, `N_tombstone`.

### B) Interleaved sequence

Generate sample order like:
`OK, TOMBSTONE, OK, MISSING, OK, MISALIGNED, ...`
This is cosmetic for training but useful if you build a live “viewer”.

In both cases, labels are stored in `labels.jsonl`.

---

## 9) Optional: “free online PCB design + 3D”

If you insist on a quick 3D route, keep it controlled:

* Design a tiny board in a free CAD (e.g., KiCad is free, offline, reliable).
* Export 3D (STEP) and render in Blender (free) with fixed camera and lighting.
* Script Blender renders to generate multiple images.
* Inject defects by programmatically moving/removing components before render.

This is heavier than the 2D OpenCV renderer but can later approximate real reflections.

For a concept MVP, the 2D approach is faster and sufficient.

---

## 10) Minimal deliverables for your MVP

You should produce exactly these items:

1. `generate.py`

   * creates `sim_data/` with images + jsonl + splits
2. `train.py`

   * trains model and saves `models/run_0001.pt`
3. `eval.py`

   * prints and saves metrics for `test` split
4. `config.yaml`

   * all settings for reproducibility
5. `report.json`

   * metrics + confusion matrix + run metadata

Once this exists, replacing synthetic ROIs with real ROIs is straightforward.

---

## 11) What this MVP proves (and what it does not)

### It proves

* Your pipeline architecture works end-to-end
* ROI extraction/training/testing infrastructure is correct
* Thresholding and reporting works
* Active-learning style dataset growth is feasible

### It does not prove

* Final production accuracy on real solder appearance
  That requires real data fine-tuning.

---

## 12) Continue plan: two-track roadmap (2D now, 3D next)

Do not block the MVP on CAD/3D. Run two tracks in parallel:

1. **Track A (fast): 2D ROI simulator**
   * Purpose: validate dataset format, labels, training loop, metrics, and defect taxonomy
   * Output: `sim_data/` with reproducible `images/ + jsonl`
2. **Track B (realism): CAD/3D render simulator**
   * Purpose: reduce domain gap (lighting/reflections/occlusions) and prepare for real AOI conditions
   * Output: same dataset format as Track A, but rendered images and optionally pixel-level labels

Rule: **Both tracks write the same schema** so you can mix datasets and compare models fairly.

---

## 13) Local CAD/3D generation: recommended tooling (offline)

### Recommended stack (pragmatic)

* **KiCad** (PCB + footprints + component placement, free/offline)
* **Blender** (rendering, defect injection by scripting, free/offline)
* **Python** (run orchestration + metadata + post-processing)

Why this combination:

* KiCad gives deterministic, editable placement and a clean way to define “nominal” geometry.
* Blender (Cycles) gives physically-based rendering (specular highlights, shadows) which matters for AOI.
* Python glues everything together and keeps reproducibility (seed/config).

### Alternative stack (if you already use mechanical CAD)

* **FreeCAD** (assembly) + export STEP + render in Blender.

---

## 14) CAD/3D pipeline (KiCad → STEP → Blender → dataset)

### Step 1: Define a minimal reference board in KiCad

Keep it tiny and controlled:

* 1–3 footprints (e.g., `0603`, `0402`, `SOT-23`)
* a few pads with solder mask opening
* optional silkscreen line/text

Export:

* board assembly as **STEP** (for Blender import)
* optionally also store the KiCad project as the “source of truth”

### Step 2: Import into Blender and set up an AOI-like “rig”

Scene constraints:

* fixed camera pose (top-down or slight tilt)
* fixed focal length and sensor size (match your AOI camera roughly)
* controlled lights (e.g., ring light approximation)
* materials:
  * solder mask (rough dielectric)
  * copper pads (metallic)
  * component body (plastic/ceramic)
  * solder (metallic, low roughness)

Render pass recommendations:

* RGB (main)
* optional: depth, normals, object-id mask (for perfect labels)

### Step 3: Defect injection in Blender (deterministic transforms)

Defects are applied by modifying object transforms or adding defect geometry before render.
All parameters must be written into metadata (seed + per-sample params).

Minimal defect implementations:

* `MISSING`: hide/remove the component object
* `MISALIGNED`: translate in X/Y, rotate around Z within a param range
* `TOMBSTONE`: rotate around the pad edge axis (X or Y) to ~80–90 degrees and lift one side in Z
* `BRIDGE` (later): add a small mesh “bridge” between pads (simple cylinder/metaball, or geometry-nodes)

### Step 4: Post-processing (camera + sensor effects)

Even for 3D renders, add image-space effects to better mimic real AOI:

* gaussian blur (focus variation)
* shot noise + read noise
* slight lens distortion
* exposure/white-balance jitter

### Step 5: Dataset write-out (same as MVP)

Write exactly the same `sim_data/` layout:

* `images/*.png` (or `.jpg`)
* `meta.jsonl` (seed, nominal geometry id, defect params, render params)
* `labels.jsonl` (class label; optionally also bbox/mask paths)

---

## 15) Label strategy (start simple, keep upgrade path)

### MVP label level (classification)

One label per ROI:

* `OK`, `MISSING`, `MISALIGNED`, `TOMBSTONE` (and later `BRIDGE`)

### Upgrade path (detection/segmentation)

If you want “where is the defect?” later:

* bboxes from object-id pass (cheap)
* segmentation masks (perfect, from object-id/material-id passes)

Keep this optional to avoid scope creep.

---

## 16) How to generate “training + comparison” datasets locally

### Goal

You want **two dataset families** that are comparable (sim-only):

1. `2d_synthetic` (fast, high volume)
2. `3d_synthetic` (slower, higher realism)

They must share:

* same classes and naming
* same ROI size / resolution policy
* same label file format

Then you can run controlled experiments:

* train on `2d_synthetic`, test on `3d_synthetic`
* train on mixed sets, evaluate domain generalization

---

## 17) Fault/defect “generator” policy (for later realism)

Once the pipeline is stable, improve defects by adding constraints and more types:

* `ROTATED` (wrong orientation)
* `SHIFT_XY` (already included as misaligned)
* `LIFTED` (one side lifted)
* `SOLDER_INSUFFICIENT` / `EXCESS` (material proxy)
* `SOLDER_BALL` (small spheres near pads)
* `FOREIGN_OBJECT` (random small debris)

Important: do not add more defect classes until the current ones are separable and stable in evaluation.

---

## 18) Concrete milestones (so this doesn’t sprawl)

### M0 (1–2 days)

* 2D generator produces `sim_data/` + baseline classifier + eval metrics

### M1 (2–4 days)

* KiCad reference board + Blender scene “rig”
* scripted render of nominal + `MISSING/MISALIGNED/TOMBSTONE`
* dataset written with the same schema as M0

### M2 (1–2 weeks)

* better materials and light model, more camera artifacts
* `BRIDGE` + 1–2 additional defect types
* optional object-id masks for pixel-perfect labels

### M3 (ongoing)

* mix `2d_synthetic` + `3d_synthetic` (same schema) and track sim-to-sim domain gap
* keep frozen `test` and frozen challenge sets for honest evaluation

---

## 19) Constraint: no real hardware, so evaluation must be sim-only

For Simple-Sim (current scope): **no camera, no AOI machine, no real images**.
That means you must be disciplined about evaluation, otherwise you only measure "how well the model fits your own generator".

Solution: treat simulation as multiple *domains* and always evaluate on a different domain than training.

---

## 20) Sim-only evaluation strategy (train vs. unseen sim domains)

### Define domains explicitly

Create a small set of render/generator "domains" (each a config preset):

* `domain_A`: baseline lighting/materials/noise
* `domain_B`: changed lighting angles + stronger speculars
* `domain_C`: different camera tilt + blur/noise distribution
* `domain_D`: different pad/component tolerances + textures

Rule:

* Train on `A+B` (or only `A` for a hard test)
* Validate on `C`
* Test on `D` (frozen, never touched)

This gives a **sim-to-sim generalization** score that is meaningful even without real hardware.

### Frozen "challenge sets"

In addition to the normal test set, generate fixed stress sets:

* borderline defects (misalignment close to the OK tolerance threshold)
* rare combinations (rotation + shift, mild tombstone angles)
* occlusion/contamination (foreign object near pads)
* extreme lighting (glare, low exposure)

Measure performance specifically on these sets. This is where AOI-like failure modes appear first.

---

## 21) Target: measurable accuracy and calibrated decisions (not only accuracy)

For AOI-style decisions you need more than top-1 accuracy.
Even in sim-only mode you can demand:

* per-class precision/recall/F1 (especially for critical defects)
* confusion matrix
* **false negative rate** for `MISSING` and `TOMBSTONE`
* calibration proxy: reliability diagram or ECE (optional, later)

Define acceptance thresholds early, e.g.:

* `MISSING` recall >= 0.995 on `test_domain_D`
* `TOMBSTONE` recall >= 0.98 on `challenge_borderline`
* overall macro-F1 >= 0.95 on `test_domain_D`

These numbers are placeholders until you pick defect parameter ranges.

---

## 22) Realism ladder (increase fidelity without changing the schema)

Keep the data format stable and increase realism in steps:

1. **2D parametric** (OpenCV) for speed and pipeline validation
2. **3D render, simple materials** (Blender) for lighting/shadows/speculars
3. **3D render, improved materials** (solder mask + copper + solder + component)
4. **3D render, sensor model** (noise, distortion, motion blur)
5. **3D render, nuisance factors** (dust, scratches, flux residue proxies)

Each step must improve sim-to-sim generalization on the frozen test/challenge sets.
If it does not, do not add complexity.

---

## 23) Defect parameters: define "tolerances" as ground truth

Because there is no hardware, you must encode what "OK" means.
Define numeric tolerances (per footprint type) and label based on them:

* `OK` if shift <= `ok_shift_px` and rotation <= `ok_rot_deg` and lift <= `ok_lift_mm`
* `MISALIGNED` if shift > threshold (and component present)
* `TOMBSTONE` if tilt angle > threshold OR contact area ratio < threshold
* `MISSING` if component absent

This keeps labels consistent across generators and enables "borderline" challenge sets.

---

## 24) Dataset versioning (so tests are honest)

Treat generated datasets like releases:

* `sim_data/runs/run_0001/` ... `run_000N/`
* each run has:
  * `config.yaml`
  * `meta.jsonl`, `labels.jsonl`
  * `splits/` (frozen lists)
  * `README.md` (what changed vs previous run)

Rule: once a `test` split is declared frozen for a domain, you never regenerate it.

---

## 25) What "success" looks like before any real hardware exists

You can claim progress if:

* the generator is deterministic (seeded) and auditable
* models improve **on unseen sim domains** and challenge sets, not only in-domain
* defect thresholds and tolerances are explicit (not implicit in images)
* you can reproduce any sample and its label from `meta.jsonl`

At that point you have a credible training/eval pipeline that is ready to accept real ROIs later,
but it already provides value as a controlled research and regression-test environment.

---

## 26) Architecture “contracts” (so everything plugs together)

The architecture is only "correct" if the interfaces are explicit and stable.
Define these contracts up front:

* **Contract A: Dataset schema** (JSONL fields, types, required/optional)
* **Contract B: Config schema** (YAML keys, defaults, validation)
* **Contract C: Generator API** (inputs/outputs, determinism)
* **Contract D: Renderer API** (2D OpenCV vs 3D Blender share the same output contract)
* **Contract E: Trainer/Eval I/O** (reads dataset by schema only, no generator coupling)

---

## 27) Folder layout (source vs assets vs outputs)

Use a predictable repo layout so code, assets, and generated data are not mixed:

```
Simple-Sim/
  README.md
  configs/
    run_0001.yaml
  simple_sim/
    __init__.py
    schema.py
    config.py
    rng.py
    dataset_store.py
    generator_2d.py
    renderer_3d_blender.py
    defects.py
    splits.py
    metrics.py
  tools/
    validate_dataset.py
    preview_samples.py
  assets/
    kicad/
      board_minimal/
        (kicad project files)
    blender/
      scene_template.blend
      materials.blend
      scripts/
        render.py
  scripts/
    generate.py
    train.py
    eval.py
  outputs/
    sim_data/
      runs/
        run_0001/
          images/
          meta.jsonl
          labels.jsonl
          splits/
          config.yaml
          report.json
          README.md
    models/
      run_0001.pt
```

Rule: anything in `outputs/` can be deleted and recreated from config + seed.

---

## 28) Dataset schema (make it strict)

Keep JSONL strict and versioned. Add a `schema_version` field and fail fast if it mismatches.

### `meta.jsonl` (one row per sample)

Required fields:

* `schema_version` (int)
* `id` (string, unique within run)
* `run_id` (string)
* `domain` (string, e.g. `domain_A`)
* `split` (string: `train|val|test|challenge_<name>`)
* `seed` (int)
* `image_path` (string, relative path)
* `render_backend` (string: `opencv_2d|blender_3d`)
* `footprint` (string, e.g. `0603`)
* `nominal` (object: canonical geometry/pose before defect)
* `defect` (object: defect type + parameters)
* `augment` (object: post effects parameters)

Recommended fields:

* `board_id` (string) and `refdes` (string) if using CAD placements
* `roi` (object: width/height, pixel_size or mm_per_px)
* `hash` (object: sha256 of image file, optional but useful)

### `labels.jsonl` (one row per sample)

Required fields:

* `schema_version` (int)
* `id` (string, matches `meta.jsonl`)
* `class` (string: `OK|MISSING|MISALIGNED|TOMBSTONE|BRIDGE`)

Optional fields (future):

* `bboxes` (list)
* `masks` (paths)

Rule: labels must be derived from **defect parameters + tolerance policy**, not from pixels.

---

## 29) Config schema (YAML) and validation rules

The config must be machine-validated before generation/training starts.
Minimum required sections:

* `run`:
  * `run_id`
  * `seed`
  * `schema_version`
* `roi`:
  * `width_px`, `height_px`
* `classes`:
  * list of class names and per-class sample counts
* `tolerances`:
  * per-footprint thresholds (shift/rotation/lift/contact)
* `domains`:
  * map of `domain_A..D` to generator/render params
* `splits`:
  * which domains go to train/val/test and what is frozen
* `render`:
  * backend selection and backend-specific params
* `augment`:
  * ranges for noise/blur/exposure etc
* `train` and `eval`:
  * model, epochs, batch size, metrics

Validation rules:

* every class name in generator must exist in `classes`
* every domain referenced by splits must exist in `domains`
* no overlap between frozen test IDs and training IDs

---

## 30) Determinism policy (generation and rendering)

Determinism must be true at the dataset level:

* `sample_seed = f(run_seed, domain, split, sample_index)` (stable, collision-resistant)
* `id` must be deterministic (e.g. `run_id/domain/split/index`)
* every stochastic choice uses only `sample_seed`

Rendering determinism notes:

* 2D OpenCV path is straightforward deterministic.
* 3D Blender path must pin:
  * Blender major/minor version
  * render engine settings (samples, denoise on/off)
  * scene template `.blend`

Goal is reproducible datasets on the same machine and Blender version. Cross-machine bit-identical renders are not required for MVP, but the *labels and metadata* must be identical.

---

## 31) CAD asset integration (without coupling training to KiCad)

CAD is an input to generation only. Training/Eval never reads KiCad files.

Two acceptable CAD integration levels:

* **Level 1 (placements only)**:
  * parse KiCad PCB for component placement (refdes, x/y, rotation, footprint)
  * render per-component ROIs using parametric pad/body geometry
* **Level 2 (geometry + render)**:
  * use KiCad export (STEP/VRML) to produce 3D board/component geometry
  * import into Blender via a documented conversion step

Important: STEP is not always a smooth path into Blender.
If STEP import is unreliable in your environment, standardize on one conversion path and document it, or use VRML export if it is stable.

---

## 32) Required tooling scripts (so the workflow is one command)

Minimum scripts and their contracts:

* `scripts/generate.py --config configs/run_0001.yaml --out outputs/sim_data/runs/run_0001`
  * writes dataset + splits + config copy
* `scripts/train.py --data outputs/sim_data/runs/run_0001 --out outputs/models/run_0001.pt`
  * reads only JSONL + images
* `scripts/eval.py --data outputs/sim_data/runs/run_0001 --model outputs/models/run_0001.pt`
  * produces `report.json` and prints metrics
* `tools/validate_dataset.py --data outputs/sim_data/runs/run_0001`
  * checks schema, missing files, id consistency, class distribution, split overlap

Non-negotiable: `validate_dataset.py` must run before training.

---

## 33) Hardware/compute budget (first-order calculations)

You need estimates for:

* dataset generation time (2D vs 3D)
* training time
* RAM / GPU VRAM
* storage growth over time (runs, frozen tests, models, reports)

Because this is sim-only and you control the settings, the best approach is:

1. define target dataset sizes and fidelity (ROI size, png/jpg, 3D samples)
2. run 3 short local benchmarks to measure throughput
3. scale with simple formulas (below)

### Storage sizing

Definitions:

* `N` = number of samples
* `S_img` = average image size on disk (bytes)
* `S_meta` = average metadata+label size on disk per sample (bytes, usually small)

Approx:

* `S_total ~= N * (S_img + S_meta) + overhead`

Rules of thumb to plan with (conservative):

* `S_meta` usually < 2 KB/sample if JSONL is kept compact.
* `S_img` depends heavily on format:
  * PNG: variable (texture/noise can inflate)
  * JPEG: typically smaller but lossy (often acceptable for AOI-like training)

Practical policy:

* keep `outputs/sim_data/runs/run_xxxx/` immutable once "released"
* periodically archive old runs you no longer train on

### RAM sizing

Training typically does NOT require loading the whole dataset into RAM.
Plan RAM for:

* OS + Python + dataloader workers
* caching (optional)

Baseline:

* 16 GB is workable for small ROIs + light augmentation
* 32 GB recommended if using many dataloader workers and heavy augmentations

### GPU VRAM sizing (training)

VRAM is driven by:

* model size
* ROI resolution
* batch size
* mixed precision on/off

For an MVP classifier (ResNet18-like) on 224-256 px ROIs:

* 8 GB VRAM is usually enough for reasonable batch sizes (especially with mixed precision)
* 12-16 GB gives more headroom (bigger batch, bigger ROI, heavier augment)

### GPU/CPU sizing (rendering)

* 2D OpenCV generator: CPU-bound, scales with CPU cores.
* Blender/Cycles rendering: typically GPU-bound (much faster with a CUDA/OptiX-capable GPU), but depends on samples/denoise/resolution.

---

## 34) Runtime formulas (scale to different run durations)

### 2D generation runtime

Measure once:

* `R_2d = samples_per_second_2d`

Then:

* `time_seconds ~= N / R_2d`

### 3D rendering runtime (Blender)

Measure once with your exact scene template:

* `R_3d = renders_per_second_3d` (or renders/hour)

Then:

* `time_seconds ~= N / R_3d`

Important: 3D runtime changes drastically with:

* samples per pixel
* denoiser
* bounces
* output resolution
* enabled render passes (ID masks, depth, normals)

### Training runtime

Definitions:

* `N_train` = number of training samples
* `B` = batch size
* `E` = epochs
* `T_iter` = measured seconds per iteration (one optimizer step)

Steps:

* `steps_per_epoch = ceil(N_train / B)`
* `time_train ~= E * steps_per_epoch * T_iter`

Measure `T_iter` once on your machine and then you can plan "overnight" vs "weekend" runs reliably.

---

## 35) Benchmarks to make the estimates real (required)

Add these quick benchmarks early and store results in each run folder:

* `tools/benchmark_2d.py`
  * generates e.g. 2000 samples, reports `samples/sec`, CPU usage, peak RAM
* `tools/benchmark_blender.py`
  * renders e.g. 200 samples with the template scene, reports `renders/sec`, peak VRAM, wall time
* `tools/benchmark_train.py`
  * trains for e.g. 200 iterations, reports `sec/iter`, `images/sec`, peak VRAM

Write results to:

* `outputs/sim_data/runs/run_xxxx/bench.json`

Then "how much hardware do we need?" becomes:

* pick a target (e.g. N=200k samples)
* choose a time budget (e.g. 8h / 24h / 72h)
* compute required throughput:
  * `required_samples_per_sec = N / time_budget_sec`
* compare with your measured `R_2d`, `R_3d`, `T_iter` and decide if you need:
  * fewer samples, smaller ROIs, cheaper passes, fewer epochs
  * or more GPU/CPU (or parallel rendering)

---

## 36) Parallel monitoring: experiment tracking + "health dashboard"

You need a way to watch generation/training/eval live and to compare runs later.
Do this offline/local-first so it works on your PC without accounts.

Two layers:

1. **Training metrics live** (per step/epoch): loss, accuracy, learning rate, throughput, GPU VRAM, etc.
2. **Run artifacts + eval summaries** (per run): domain breakdown, challenge-set scores, confusion matrices, worst examples.

---

## 37) Logging contracts (so the dashboard is simple)

Add these files per run:

* `train_metrics.jsonl`
  * append-only, one row every `log_every_n_steps`
* `eval_metrics.json`
  * overwritten each eval, contains latest snapshot (val/challenge only during training)
* `final_report.json`
  * written once at the end, includes frozen test results and domain/challenge breakdown
* `system.json`
  * hardware summary + environment (GPU name, driver/cuda if available, CPU model, RAM)
* `events.log`
  * human-readable log (start, config hash, checkpoints saved, errors)

Minimal schema for `train_metrics.jsonl` rows:

* `ts` (unix seconds)
* `step` (int), `epoch` (int)
* `split` (string: `train`)
* `loss` (float)
* `acc` (float, optional early)
* `lr` (float)
* `images_per_sec` (float)
* `gpu_vram_mb` (int, optional)
* `cpu_ram_mb` (int, optional)

Minimal schema for eval snapshots:

* `when`: `during_train|final`
* `splits`: map of split->metrics (macro-F1, per-class recall, confusion matrix)
* `domains`: map of domain->metrics
* `challenges`: map of challenge_set->metrics

Rule to prevent test leakage:

* during training you may evaluate on `val` and `challenge_val`
* the frozen `test` domain/challenge sets are evaluated **only once** in `final_report.json`

---

## 38) Dashboard features (MVP)

The dashboard must answer these questions fast:

* Is training healthy? (loss decreasing, no NaNs, throughput stable, GPU not OOM)
* Did the latest run improve on unseen domains?
* Which classes are failing (false negatives) and under which domain/challenge?
* What changed between runs (config diff + benchmark diff)?

MVP pages:

* **Runs list**: run_id, start time, dataset sizes, backbone, final macro-F1 on val, link to reports
* **Live training**: loss/acc curves from `train_metrics.jsonl`, throughput, VRAM/RAM
* **Eval breakdown**: per-domain and per-challenge metrics, per-class recall, confusion matrix
* **Error inspector**: show top-K worst samples (highest confidence wrong predictions), grouped by class/domain

Artifacts to support "Error inspector":

* store a small `predictions.jsonl` for val/challenge (id, true, pred, prob, domain)
* store `worst_k/` image copies or store only IDs and load images by path

---

## 39) Implementation choice (offline, minimal friction)

Recommended approach:

* **TensorBoard** for live scalar plots (loss, lr, throughput)
* **Streamlit** for the run browser, domain/challenge tables, confusion matrices, and error inspector

Why:

* both are local-first and easy to run in parallel with training
* TensorBoard integrates naturally with PyTorch
* Streamlit is fast for a "health dashboard" that reads JSONL/JSON artifacts

Alternative (if you prefer a single tool):

* only Streamlit (plot from JSONL directly)
* or MLflow (local tracking server + UI), but it adds more moving parts

---

## 40) Required CLI behavior to enable parallel monitoring

Training process must:

* write `train_metrics.jsonl` continuously (flush every write)
* write a checkpoint every N minutes/epochs
* write `eval_metrics.json` after each val eval
* never touch frozen test until training ends

Dashboard process must:

* be read-only (no modifying outputs)
* gracefully handle partially-written JSONL (tailing)

---

## 41) Full system architecture (pipelines + contracts + subfolder chains)

This section is the "single source of truth" for how everything fits together.
If something is not specified here, it is not allowed to be implicit in code.

### Components (logical)

* **Asset pipeline**: CAD/Blender templates and versioning
* **Dataset pipeline**: generate -> validate -> freeze splits -> release
* **Training pipeline**: train -> checkpoint -> val/challenge eval -> finalize
* **Evaluation pipeline**: frozen test eval -> report -> error artifacts
* **Monitoring pipeline**: live metrics + dashboard reading outputs

### Dataflow (high level)

```
configs/ + assets/  ->  generate  -> outputs/sim_data/runs/<run_id>  -> validate -> train -> eval -> dashboard
                                      |                               |
                                      +-> frozen test/challenge sets  +-> outputs/models/<run_id>.*
```

Non-negotiable principle:

* Training/Eval code consumes only **dataset artifacts** (images + JSONL + splits).
* Generator/Renderer code is never imported by Trainer/Eval (prevents coupling and leakage).

---

## 42) Subfolder chains (what goes where, always)

### Source (committed)

* `Simple-Sim/simple_sim/`: library code (schema, rng, splits, metrics, IO)
* `Simple-Sim/scripts/`: CLIs that call the library
* `Simple-Sim/tools/`: utilities (validate, preview, benchmark)
* `Simple-Sim/configs/`: run configs (YAML)
* `Simple-Sim/assets/`: CAD + Blender templates (inputs)

### Outputs (generated, not hand-edited)

* `Simple-Sim/outputs/sim_data/runs/<run_id>/`: datasets (released runs)
* `Simple-Sim/outputs/models/<run_id>/`: model checkpoints + final model
* `Simple-Sim/outputs/dash/`: optional cached dashboard indexes (read-only cache)

Rule:

* everything under `outputs/` must be reproducible from committed inputs + run config + seed
* no code reads or writes outside `Simple-Sim/` (self-contained)

---

## 43) Run identity, registries, and naming

### IDs

* `run_id`: `run_0001` style, unique
* `sample_id`: deterministic string, e.g. `<run_id>/<domain>/<split>/<index>`
* `model_id`: equals `run_id` unless doing sweeps (then add suffix)

### Registry files (append-only)

Add a lightweight registry for browsing runs without scanning the filesystem:

* `Simple-Sim/outputs/registry/runs.jsonl`
  * one line per completed dataset release
* `Simple-Sim/outputs/registry/models.jsonl`
  * one line per completed training run (links to dataset run_id)

Each registry row stores:

* `run_id`
* `created_ts`
* `git_commit` (optional but recommended)
* key config hash (sha256 of `config.yaml`)
* paths to artifacts
* headline metrics (macro-F1 val, critical recalls, etc.)

Registry rules:

* only written at the end of a successful pipeline stage
* dashboard reads registry first, falls back to scanning if missing

---

## 44) Orchestration layer (the pipeline runner)

You need one "front door" so the workflow is consistent.

### Required commands (minimum)

* `generate`: create dataset artifacts
* `validate`: validate schema + splits + distributions
* `train`: train model on a released dataset
* `eval`: evaluate model on frozen test (final only)
* `dash`: run dashboard UI (read-only)

### Execution style

Keep it boring and scriptable:

* `python -m simple_sim ...` (library entrypoints) OR `scripts/*.py`
* optionally a `Makefile` or `justfile` as a thin wrapper

Hard rule:

* every command must be resumable and idempotent for its outputs
  * if outputs exist and match config hash, do nothing
  * if outputs exist but mismatch config hash, fail fast

---

## 45) Pipeline stages (inputs, outputs, gates)

Define stages as a DAG. Each stage has explicit inputs/outputs and a "gate" (validation) before the next stage runs.

### Stage S0: Assets snapshot

Inputs:

* `assets/` + asset version tags

Outputs:

* `outputs/sim_data/runs/<run_id>/assets_manifest.json`
  * list of asset files + hashes + versions

Gate:

* asset manifest is complete; required templates exist

### Stage S1: Dataset generation

Inputs:

* `configs/<run_id>.yaml`
* assets snapshot (optional for 2D)

Outputs:

* `outputs/sim_data/runs/<run_id>/images/*`
* `outputs/sim_data/runs/<run_id>/meta.jsonl`
* `outputs/sim_data/runs/<run_id>/labels.jsonl`
* `outputs/sim_data/runs/<run_id>/config.yaml` (exact copy)

Gate:

* all files exist; counts match; IDs unique; JSONL parseable

### Stage S2: Split freeze (domains + challenges)

Inputs:

* dataset artifacts from S1
* split policy in config

Outputs:

* `outputs/sim_data/runs/<run_id>/splits/*.txt`
  * train/val/test + `challenge_*.txt`

Gate:

* no overlap between splits (especially train vs frozen test)
* class distribution and domain distribution within expected bounds

### Stage S3: Dataset validation (strict)

Inputs:

* S1 + S2 outputs

Outputs:

* `outputs/sim_data/runs/<run_id>/validate.json`
  * pass/fail + stats

Gate:

* must pass before any training can start

### Stage S4: Training (with monitored eval)

Inputs:

* released dataset run folder
* train config (can be in dataset config or separate training override)

Outputs:

* `outputs/models/<model_id>/checkpoints/*.pt`
* `outputs/models/<model_id>/final.pt`
* `outputs/models/<model_id>/train_metrics.jsonl`
* `outputs/models/<model_id>/eval_metrics.json` (val + challenge only)
* `outputs/models/<model_id>/config.yaml` (effective config)
* `outputs/models/<model_id>/system.json`

Gate:

* training completed without NaNs/OOM; metrics present; checkpoint exists

Hard rule:

* the trainer must refuse to run if `validate.json` is missing or failed

### Stage S5: Final evaluation (frozen test only)

Inputs:

* `final.pt`
* frozen test split IDs (domain_D + frozen challenges)

Outputs:

* `outputs/models/<model_id>/final_report.json`
* `outputs/models/<model_id>/predictions_test.jsonl` (optional)
* `outputs/models/<model_id>/worst_k/*` (optional)

Gate:

* report written once; if rerun, must produce identical metrics for same model+dataset

### Stage S6: Registry update

Inputs:

* dataset run folder + model folder

Outputs:

* `outputs/registry/runs.jsonl`
* `outputs/registry/models.jsonl`

Gate:

* only update registry after S5 success (ensures dashboard sees completed runs)

---

## 46) Parallelism and scaling (single PC first)

### Generation parallelism

* 2D: multiprocessing workers per CPU core (cap to avoid RAM pressure)
* 3D: render queue; optionally multiple Blender workers if VRAM allows, otherwise single GPU worker

### Training parallelism

* single GPU training process
* dataloader workers tuned to CPU and storage (avoid oversubscription)

### "Parallel monitoring"

* dashboard runs as a separate process that reads artifacts
* no IPC needed beyond files

---

## 47) Quality gates (fail fast)

Add explicit gates; do not rely on "it looks ok".

Dataset gates:

* schema validation
* split overlap check
* class/domain balance check
* determinism check: regenerate a small subset and verify IDs/labels/metadata identical

Training gates:

* NaN/Inf detection
* OOM detection with actionable message (reduce batch/ROI)
* critical defect FN rate tracked and alerted during val

Eval gates:

* ensure frozen test evaluated only after training complete
* ensure test split IDs are identical to the frozen file (no regeneration)

---

## 48) Testing strategy (so the plan survives refactors)

Minimum tests (unit/integration):

* `schema`: JSONL read/write roundtrip; required keys enforced
* `rng`: seed policy stable; sample_id stable
* `splits`: no-overlap invariants; reproducible split generation
* `labels`: tolerance-based labeling correct across edge cases
* `generator_2d`: produces image + metadata for fixed seed deterministically
* `pipeline`: tiny end-to-end run (e.g. 200 samples) that generates->validates->trains 1 epoch->evals val

This can run on CPU only for CI; GPU tests are optional.

---

## 49) Operational playbooks (how you actually work day to day)

Daily workflow:

1. pick `run_id` + config change
2. run `generate` -> `validate` (must pass)
3. run `train` (watch dashboard)
4. run `eval` (writes final report)
5. compare to previous runs in dashboard; only then iterate

Rules to avoid self-deception:

* frozen test never changes
* any config change that affects generation creates a new `run_id`
* any training hyperparam change creates a new `model_id` (even if dataset same)

---

## 50) Sweeps/Hyperparam search (planned from day 1)

If you want steady progress, you need systematic sweeps:

* optimizer/lr schedule
* augmentation strength
* ROI resolution
* model backbone (small/medium)
* class weighting / focal loss (optional)

Key requirement:

* sweeps must not break reproducibility
* sweeps must not touch frozen test until promotion/finalization

---

## 51) Config layering + hashing (non-negotiable)

To make sweeps safe, define explicit config layers and hashing rules.

### Layers

1. **Base dataset config** (generation): `configs/<run_id>.yaml`
2. **Base training config** (defaults): e.g. `configs/train_base.yaml`
3. **Sweep spec** (grid/random): `configs/sweeps/<sweep_id>.yaml`
4. **Trial override** (one trial params): generated deterministically per trial

### Effective config

Each trial writes an `effective_config.yaml` that is the fully merged config:

* dataset reference (run_id + dataset hash)
* training params
* seed policy
* evaluation policy (val/challenge only during training)

### Hashing

Compute and store:

* `dataset_hash`: sha256 of dataset `config.yaml` + split files + schema_version
* `trial_hash`: sha256 of `effective_config.yaml`

Rules:

* if `trial_hash` exists, the trial is considered identical and must not be re-run unless explicitly forced
* every output folder is keyed by `trial_hash` to prevent collisions

---

## 52) Sweep IDs, trial IDs, and folder layout

### IDs

* `sweep_id`: e.g. `sweep_0001`
* `trial_id`: deterministic, e.g. `trial_<trial_hash_prefix>`
* `model_id`: `run_id/<sweep_id>/<trial_id>` (or a flat sanitized string)

### Outputs

```
outputs/
  sweeps/
    <sweep_id>/
      sweep.yaml
      index.jsonl
      leaderboard.json
      trials/
        <trial_id>/
          effective_config.yaml
          checkpoints/
          final.pt
          train_metrics.jsonl
          eval_metrics.json   (val/challenge only)
          system.json
          status.json         (running|failed|done + exit code)
```

### `index.jsonl`

Append-only, one row per trial:

* `trial_id`, `trial_hash`, `params` (flattened)
* `status`, `start_ts`, `end_ts`
* headline metrics (macro-F1 val, critical recalls on challenge_val)
* pointers to artifact paths

---

## 53) Sweep pipeline stages (S4 becomes S4a/S4b)

Extend the pipeline DAG with sweep-specific stages.

### Stage S4a: Trial generation (planning)

Inputs:

* `sweep.yaml`
* base training config + dataset run_id

Outputs:

* list of trials with deterministic `trial_id` + `effective_config.yaml`

Gate:

* no trial may request evaluation on frozen `test`

### Stage S4b: Trial execution (training + val/challenge)

Inputs:

* one `effective_config.yaml`
* released dataset

Outputs:

* trial artifacts (metrics, checkpoints, final.pt)

Gate:

* trial writes valid `status.json` and at least one eval snapshot on val/challenge

### Stage S4c: Sweep aggregation (leaderboard)

Inputs:

* all trial `eval_metrics.json`

Outputs:

* `leaderboard.json` with ranking
* `best_trial.json` (pointer)

Gate:

* ranking uses only val/challenge scores (not frozen test)

---

## 54) Promotion policy (choose winners without cheating)

Define promotion as an explicit step with a policy file.

### Promotion inputs

* `leaderboard.json`
* `promotion_policy.yaml`:
  * primary metric (e.g. macro-F1 on val)
  * constraints (e.g. `MISSING` recall >= X on challenge_borderline_val)
  * tie-breakers (e.g. throughput, model size)

### Promotion outputs

* `outputs/models/promoted/<model_id>/` (copy or symlink of best trial)
* `promotion.json` (why chosen, metrics snapshot, config hash)

Rule:

* only promoted models are eligible for frozen test evaluation (S5)

---

## 55) Resource scheduling on a single PC (RTX 3080)

Assume:

* 1 GPU for training
* optional GPU for Blender renders (do not render and train concurrently unless measured stable)

Scheduler policy (simple and robust):

* sweeps run **sequentially on GPU** by default
* use CPU for data loading/augmentation with tuned workers
* optionally allow 2 concurrent trials only if VRAM headroom exists and training throughput does not collapse

Add a config knob:

* `max_concurrent_trials`: default 1

---

## 56) Early stopping + pruning (save time)

Sweeps become practical if you prune bad trials early.

Add rules:

* warmup period (e.g. 1 epoch) before judging
* stop trial if:
  * loss diverges or NaNs
  * val macro-F1 is far below current best by a margin after K epochs
  * critical class recall is below minimum threshold and not improving

Record pruning decisions in `status.json` so the leaderboard is explainable.

---

## 57) Frozen test evaluation in sweep mode (only at the end)

Process:

1. run sweep (S4a/b/c)
2. promote top-N models (e.g. 3) by policy
3. run S5 frozen test eval only for promoted models
4. update registries with promoted models and their frozen test scores

Rule:

* no trial may ever access frozen test IDs before promotion

---

## 58) What is "best" for results: staged sweep strategy (Grid + Random + Bayesian)

"Best results" is almost never a single sweep type. The best practical strategy is staged:

1. **Grid (small)**: prove the pipeline is stable and the metrics move in the expected direction.
2. **Random (medium/large)**: explore widely; it finds good regions fast with low complexity.
3. **Bayesian/Optuna (refine)**: exploit the best region efficiently to squeeze out the last gains.

This gives you:

* fast detection of bugs/leakage (grid)
* strong baseline improvements (random)
* best final numbers per compute budget (bayesian)

---

## 59) Unified `sweep.yaml` schema (supports all sweep types)

A single schema that can express grid, random, and bayesian avoids rewriting orchestration.

### Required keys

* `sweep_id`
* `dataset`:
  * `run_id`
  * `dataset_hash` (or "auto" if computed)
* `sampler`:
  * `type`: `grid|random|bayes`
  * `seed`
* `budget`:
  * `max_trials`
  * `max_epochs` (or `max_steps`)
  * `max_wall_time_minutes` (optional)
* `objective`:
  * `primary_metric`: e.g. `val/macro_f1`
  * `direction`: `maximize|minimize`
  * `constraints`: list (e.g. `challenge_borderline_val/MISSING_recall >= 0.995`)
  * `tie_breakers`: list (e.g. smaller model, higher throughput)
* `search_space`:
  * parameter definitions (see below)
* `pruner` (optional but recommended):
  * `enabled`
  * `type`: `median|successive_halving`
  * `warmup_epochs`
  * `min_trials_before_prune`
* `promotion`:
  * `top_n`: e.g. 3
  * `policy_yaml`: path or inline policy

### Search space parameter types

Each param has:

* `name`
* `type`: `choice|uniform|loguniform|int_uniform|int_loguniform|fixed`
* bounds or values

Examples:

* `lr`: `loguniform [1e-5, 5e-3]`
* `weight_decay`: `loguniform [1e-6, 1e-2]`
* `batch_size`: `choice [32, 64, 128]` (trainer auto-reduces on OOM unless forbidden)
* `roi_size`: `choice [224, 256, 320]`
* `aug_strength`: `uniform [0.0, 1.0]`
* `model`: `choice [resnet18, resnet34, efficientnet_b0]`

Rules:

* parameter sampling must be deterministic given `sampler.seed` + `trial_index`
* all sampled params must be written to `effective_config.yaml`

---

## 60) How each sampler maps to trial generation

### Grid

* Expand cartesian product of a small set of discrete params.
* Use for sanity checks and debugging only (avoid explosion).

### Random

* Sample from distributions for `max_trials`.
* Default for most work: best exploration/complexity ratio.

### Bayes (Optuna-like)

* Use past trial results to propose next params.
* Requires a persistent study state:
  * `outputs/sweeps/<sweep_id>/study.db` (sqlite) OR a JSON state file

Policy:

* only enable bayes after you have at least ~20-50 random trials (otherwise it overfits noise)
* bayes stage reuses the same sweep_id with a new `phase` (or a new sweep_id linked to previous)

---

## 61) Recommended default sweep schedule for your single PC (RTX 3080)

Start here and adjust based on your benchmarks:

1. **Grid sanity** (8-16 trials, 3-5 epochs)
   * params: `lr` (2-3 values), `batch_size` (2 values), `aug_strength` (2 values)
2. **Random explore** (50-150 trials, 5-15 epochs with pruning)
   * wide ranges for `lr`, `weight_decay`, augmentation
   * include 1-2 backbones max (keep it comparable)
3. **Bayes refine** (30-80 trials, 10-30 epochs)
   * narrow ranges around best random region
   * tighten constraints on critical recalls

Promotion:

* promote `top_n=3` from the final leaderboard by policy
* evaluate frozen test only for promoted models

---

## 62) Bayesian sweeps with SQLite (Optuna study backend)

Bayesian refinement will use an Optuna-style study with a local SQLite DB so runs are resumable and auditable.

### Study DB location

* `outputs/sweeps/<sweep_id>/study.sqlite3`

### Study naming

* `study_name = "<sweep_id>__<dataset_hash_prefix>"`

This prevents accidentally mixing trials across datasets.

### Resume rules

* If `study.sqlite3` exists and `dataset_hash` matches: resume and continue proposing trials.
* If it exists but `dataset_hash` differs: fail fast (must create a new `sweep_id`).

### Concurrency rules (single PC)

* Default `max_concurrent_trials = 1` for bayes sweeps.
* SQLite supports safe multi-process access when configured, but for stability keep it single-writer unless you explicitly enable concurrency after testing.

### What gets stored

In the Optuna trial user attributes store:

* `run_id`, `dataset_hash`, `trial_hash`, `trial_id`
* effective config hash
* key metrics (val + challenge only)
* status and error reason if failed/pruned

### Required dependencies

* `optuna` (Python)
* SQLite is built-in on most systems; Optuna will use it via SQLAlchemy.
