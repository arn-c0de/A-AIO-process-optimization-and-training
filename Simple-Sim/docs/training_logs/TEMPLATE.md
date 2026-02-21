# Training Report Template

## Training Log Start (Fill This First)

**Run Name:** `____`
**Start Date:** `____`
**Owner:** `____`
**Objective:** Start a new training run with a freshly generated database.

### Dataset Plan
- **Sampling strategy per profile:**
  - `200` clear image samples
  - `100` realism mode image samples
- **Profiles included:** `____`
- **Total samples formula:** `profiles_count * (200 + 100)`
- **Expected total samples:** `____`

### Data Generation Notes
- **Filter mode for clear set:** `custom/clean` (no realism)
- **Filter mode for realism set:** `realism` (`filter_mode=realism`, `realism_enabled=true`)
- **Generation command(s):** `____`
- **Output dataset path:** `____`
- **Train/Val/Test split plan:** `____`

### Initial Checks Before Training
- [ ] Class balance verified
- [ ] Broken/missing images checked
- [ ] Label integrity validated
- [ ] Duplicate check completed
- [ ] Dataset manifest/version recorded

---

**Report ID:** ____
**Date:** ____
**Model:** ____
**Status:** ☐ Completed ☐ In Progress ☐ Failed

---

## Metrics
| Metric | Value |
|--------|-------|
| Accuracy | |
| Loss | |
| Precision | |
| Recall | |
| F1-Score | |

---

## Config
- **Architecture:**
- **Dataset:**
- **Epochs:**
- **Batch Size:**
- **Learning Rate:**

---

## Results
- **Key Finding:**
- **Improvement vs Baseline:**

---

## Next Steps
1.
2.
3.

---

*Prepared by: ____*
