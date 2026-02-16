# Training Report

**Report ID:** TR-2026-02-16-90ROT-SOIC16
**Date:** 2026-02-16
**Model:** Full pipeline restart (same datasets)
**Status:** ☑ In Progress

---

## Metrics
| Metric | Value |
|--------|-------|
| Accuracy | TBD |
| Loss | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |

---

## Config
- **Architecture:** Same as previous baseline run
- **Dataset:** Same datasets as before, regenerated with random 90° base rotations per image
- **Epochs:** TBD
- **Batch Size:** TBD
- **Learning Rate:** TBD

---

## Results
- **Key Finding:** Training is restarted from scratch with identical dataset setup, now including 0/90/180/270 orientation randomization for stronger 3D-side invariance.
- **Improvement vs Baseline:** TBD after first full run.

---

## Next Steps
1. Run full training end-to-end with 90° rotation enabled.
2. Compare results directly against the previous baseline.
3. Validate and document the new SOIC16 profile behavior.

---

*Prepared by: Arn + Codex*
