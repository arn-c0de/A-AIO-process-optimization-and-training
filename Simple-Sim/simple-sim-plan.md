
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

You want **three dataset families** that are comparable:

1. `2d_synthetic` (fast, high volume)
2. `3d_synthetic` (slower, higher realism)
3. `real_rois` (small but high value)

They must share:

* same classes and naming
* same ROI size / resolution policy
* same label file format

Then you can run controlled experiments:

* train on `2d_synthetic`, test on `3d_synthetic`
* train on `3d_synthetic`, test on `real_rois`
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

* mix synthetic + real ROIs, fine-tune and track domain gap
* keep a frozen `test` set of real ROIs for honest evaluation


