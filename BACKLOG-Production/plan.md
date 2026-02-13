
---

## 1) Objective and constraints

**Objective:** Automated optical inspection (AOI) of fully assembled and soldered PCBs to detect defects (initial focus: **tombstoning**, **missing component**, **misalignment**, optional: **solder bridges**, **insufficient/excess solder**).

**Constraints (given by you):**

* All PCBs are designed in-house, CAD/placement data is available.
* Optical scanning happens in a closed box with fixed position and stable conditions.
* Many designs (“patterns”), each potentially requiring separate datasets.
* ~12,000 boards/year, dataset can be carried forward year to year.

**Implication:** Use CAD-driven ROI extraction + per-design (or per-package) models. This will scale.

---

## 2) Recommended system architecture

### 2.1 High-level pipeline (production)

1. **Acquire image(s)** from scanner (optionally multiple lighting states).
2. **Board registration** using fiducials (compute transform CAD → image pixels).
3. **ROI extraction** for each component/footprint using CAD/XY placement.
4. **Inference** per ROI:

   * **Anomaly detection** (per design) for general “something is wrong”
   * **Defect classifier** (global or per package family) for known defects (tombstone, missing, misaligned)
   * Optional **segmentation** for solder shape/bridging if required
5. **Decision logic** (thresholds + routing):

   * High confidence NOK → reject / rework
   * Uncertain → manual review queue (also becomes training data)
6. **Logging and dataset capture**:

   * Store ROI image + metadata + model outputs + final disposition.

### 2.2 Model strategy (what you should actually do)

Given many designs and likely limited labeled NOK examples:

* **Per design:** Anomaly Detection trained mostly on **OK ROIs**
  Purpose: detect rare/unseen defects without needing many NOK labels.

* **Global / per package family:** Supervised classifier for frequent classes
  Purpose: reliably label tombstone/missing/misalignment.

This avoids maintaining one large brittle model per PCB design and reduces labeling cost.

---

## 3) Data design (non-negotiable if you want reliability)

### 3.1 Metadata schema per ROI (store as JSON Lines)

For each extracted ROI:

* `design_id`
* `board_id` (serial / scan id)
* `capture_version` (camera + lighting settings version)
* `refdes` (R12, C35, U7, …)
* `package` (0402, 0603, SOT-23, QFN-32, …)
* `cad_x`, `cad_y`, `cad_rotation`
* `roi_bbox_px` (x, y, w, h)
* `image_file`
* `label` (optional; OK / tombstone / missing / misaligned / bridge / …)
* `inference` (scores, anomaly score, classifier probabilities)

### 3.2 Folder layout (recommended)

```
data/
  designs/
    <design_id>/
      capture_v<k>/
        boards/
          <board_id>/
            full_image.png
            rois/
              <refdes>_<package>.png
            meta.jsonl
            labels.jsonl   (if labeled)
models/
  anomaly/
    <design_id>/v1/...
  classifier/
    global/v1/...
reports/
  eval/<date>_vX.html
```

### 3.3 Golden test set rule

Maintain a **frozen test set** per design (or per major family). Never train on it. Every new model release is evaluated against it.

---

## 4) Implementation stack (Python-first)

### 4.1 Core libraries (Python)

* **Image I/O and processing:** `opencv-python`, `numpy`
* **Data handling:** `pandas`, `pyarrow` (optional), `orjson` (fast JSON)
* **Deep learning:** `pytorch`, `torchvision`
* **Training framework (recommended):** `pytorch-lightning` (optional but helpful)
* **Model export/runtime:** `onnx`, `onnxruntime` (for stable deployment)
* **Experiment tracking:** `mlflow` (recommended) or `wandb` (if allowed)
* **Dataset versioning:** `dvc` (optional but useful) or at minimum Git + structured storage
* **API service:** `fastapi` (for inference service)
* **Queue / async jobs (optional):** `celery` + Redis or just a simple worker loop
* **Labeling tool:** `CVAT` (excellent), or `Label Studio` (simpler)

### 4.2 Non-Python components (practical)

* **Database:** PostgreSQL (recommended) or SQLite for small-scale
* **Storage:** file system / NAS with consistent naming; object storage if available
* **UI for review:** lightweight web UI or integrate with existing MES if you have it

---

## 5) Detailed module breakdown (what you build)

### 5.1 Acquisition module

* Input: scan trigger or file drop
* Output: full image(s) + scan metadata
* Requirements:

  * stable naming, timestamps, design selection logic
  * capture_version tagging

### 5.2 Registration module (fiducials)

* Input: full image + fiducial CAD coordinates
* Output: transform matrix `T` mapping CAD → pixel coordinates
* Implementation options:

  * fiducial detection via template matching + subpixel refinement
  * or circle/marker detection if fiducials are standardized

Deliverable: registration quality metrics (error in pixels). If error > threshold → fail scan.

### 5.3 ROI extractor

* Input: transform `T`, placements (x,y,rot), footprint dimensions
* Output: ROI images + metadata record
* Key details:

  * rotate ROIs to canonical orientation using `cad_rotation`
  * normalize ROI size per package family

### 5.4 Inference engine

Runs per ROI:

* Anomaly model (per design)
* Classifier model (global or package family)

Outputs:

* anomaly score
* class probabilities
* combined decision + confidence

### 5.5 Decision policy

* Hard reject: missing/tombstone above threshold
* Review queue: uncertain region (probabilities in defined band)
* Accept: clear OK

This policy must be explicit and versioned. Otherwise you cannot audit changes.

### 5.6 Logging + dataset builder

Every inspected board contributes to:

* operational logs
* training dataset (for later labeling)
* error analysis reports

---

## 6) Model choices (sane defaults)

### 6.1 Anomaly detection (per design)

Good options:

* **PaDiM** / **PatchCore**-style feature-based anomaly detection
* **Autoencoder** (simpler, usually weaker)
* **Normalizing flows** (more complex)

Practical recommendation: PatchCore-like approach is often strong for AOI ROI anomalies and needs mostly OK data.

### 6.2 Supervised classifier (known defects)

* Backbone: ResNet/EfficientNet (small variant)
* Output classes: OK / tombstone / missing / misaligned (extend later)
* Train either:

  * globally across designs (if visual appearance is consistent)
  * or per package family (often improves generalization)

### 6.3 Optional segmentation (if you must judge solder shape)

* U-Net style segmentation for:

  * solder fillet region
  * bridges
  * lifted leads (for IC packages)

This is more labeling work, so do it only if required.

---

## 7) Release process (minimum viable governance)

To prevent “random model swaps”:

1. **Model versioning**: `v1, v2, ...`
2. **Evaluation report** includes:

   * confusion matrix per defect type
   * false negatives (must be emphasized)
   * false positives (cost metric)
   * performance by package family and by design
3. **Deployment gate**: deploy only if test set metrics meet thresholds.

---

## 8) Rollout plan (phased, realistic)

### Phase 0: Foundations (1–2 weeks of engineering)

* Implement registration + ROI extraction
* Build storage layout + metadata schema
* Establish capture_version discipline

### Phase 1: Pilot dataset (2–4 weeks production data)

* Choose 1–2 designs
* Collect ROIs, label only:

  * tombstone, missing, obvious misalignment
* Create frozen test set

### Phase 2: Models v1

* Anomaly model per design trained on OK
* Classifier trained on labeled ROIs
* Define decision thresholds and review queue

### Phase 3: Production + Active Learning

* Deploy inference
* Send uncertain cases to review
* Feed reviewed cases back into training monthly/quarterly

### Phase 4: Scale to more designs

* Automate per-design anomaly training
* Expand classifier coverage by package family
* Add segmentation only if necessary

---

## 9) Deliverables checklist (what “done” means)

* [ ] Registration accuracy report (per scan)
* [ ] ROI extractor validated vs CAD placements
* [ ] Dataset schema + versioning in place
* [ ] Labeling tool operational and label guidelines written
* [ ] v1 anomaly model per pilot design
* [ ] v1 defect classifier for tombstone/missing/misalignment
* [ ] Decision thresholds defined and audited
* [ ] Production inference service + logging
* [ ] Monthly retraining pipeline documented

---

## 10) Minimal Python project structure (recommended)

```
aoi/
  acquisition/
  registration/
  roi/
  models/
    anomaly/
    classifier/
    segmentation/
  inference/
  decision/
  data/
  evaluation/
  api/
  configs/
  scripts/
```

Use config files (YAML) for:

* design definitions
* fiducial definitions
* ROI sizes per package
* thresholds per defect

---

