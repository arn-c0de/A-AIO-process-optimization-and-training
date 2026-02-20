# Model Arena Research Log – Fine-Tuned Default Filters vs QFN-Focused Fine-Tune (February 20, 2026)

**Research Type:** Comparative training log (new arena cycle)  
**Status:** Completed (analysis), follow-up run required  
**Base References:**  
`Simple-Sim/ARENA_REPORT.md`  
`Simple-Sim/docs/arena-reports-historie/2026-02-90degree-ARENA_REPORT.md`  
`Simple-Sim/docs/arena-reports-historie/2026-02-ARENA_REPORT.md`

## Research Goal

Compare two new fine-tuned checkpoints trained under the new default filter regime and document what changed versus the previous arena baseline.

## Run Definitions

| Run ID | Checkpoint | Training Focus | Epoch | Val Acc | Notes |
|--------|------------|----------------|------:|--------:|-------|
| Run-A | `FINE-TUNED-RandomCrawler-1X.pt` | New default filters, all samples, 100 samples per category/profile | 10 | 0.9452 | Broad fine-tune |
| Run-B | `QFN-FINE-TUNED-RandomCrawler-1X.pt` | QFN-focused fine-tune (QFN-3D + QFN32-2D at 200 samples; others 100) | 10 | 0.9315 | Targeted specialization |

## Overall Arena Comparison

| Model | Avg Accuracy | Avg F1 | Rank |
|-------|-------------:|-------:|-----:|
| `RandomCrawler-5X.pt` (historical baseline, 2026-02-16) | 0.7170 | 0.6744 | 1 (historical) |
| `RandomCrawler-1X_MultiTrained.pt` (historical baseline, 2026-02-16) | 0.7154 | 0.6616 | 2 (historical) |
| `FINE-TUNED-RandomCrawler-1X.pt` (new) | 0.7164 | 0.6851 | 3 (current report) |
| `QFN-FINE-TUNED-RandomCrawler-1X.pt` (new) | 0.7005 | 0.6689 | 4 (current report) |

Key deltas:
- `FINE-TUNED-RandomCrawler-1X.pt` vs historical `RandomCrawler-1X_MultiTrained.pt`: **+0.0010** avg accuracy, **+0.0235** avg F1.
- `QFN-FINE-TUNED-RandomCrawler-1X.pt` vs `FINE-TUNED-RandomCrawler-1X.pt`: **-0.0159** avg accuracy, **-0.0162** avg F1.

## Per-Dataset Delta (Run-B minus Run-A)

| Dataset | Run-A Acc | Run-B Acc | Delta |
|---------|----------:|----------:|------:|
| CLEAN-All-90degree-profile-database | 0.8604 | 0.8031 | -0.0573 |
| CLEAN-QFN-3D | 0.2075 | 0.3187 | +0.1112 |
| CLEAN-QFN32-2D | 0.3150 | 0.2925 | -0.0225 |
| CLEAN-Resistor-2D | 0.9225 | 0.8875 | -0.0350 |
| CLEAN-Resistor-3D | 0.6875 | 0.6925 | +0.0050 |
| CLEAN-Transistor-2D | 0.9413 | 0.8962 | -0.0451 |
| CLEAN-Transistor-3D | 0.8363 | 0.7913 | -0.0450 |
| Filter-Default-Mixed-ALL_20260220_131657 | 0.9604 | 0.9218 | -0.0386 |

## What Changed vs. Previous Reports

- Compared to the **February 16, 2026 90-degree baseline report**, the new broad fine-tune (`FINE-TUNED-RandomCrawler-1X.pt`) is roughly on the same average-accuracy level but with better average F1.
- The **QFN-focused fine-tune** improved `CLEAN-QFN-3D`, but this did not translate into better global performance.
- Compared to the older **fixed-rotation era report** (`2026-02-ARENA_REPORT.md`), absolute score patterns are not directly comparable due to changed data regime and stronger realism/randomization constraints.

## Critical Finding (Requested Follow-up)

- The QFN-focused fine-tune **worsened QFN32-related outcome** (`CLEAN-QFN32-2D`: 0.3150 -> 0.2925).
- A plausible cause is a **dataset image-format mismatch** (wrong/incorrect format characteristics for that dataset split).
- Even with this possible explanation, the measured result is still a regression and must be treated as a real issue until proven otherwise.

## Next Run Actions

1. Verify QFN32-2D image format pipeline end-to-end (resolution, channel mode, normalization, color space, augmentation parity).
2. Re-run QFN-focused fine-tune after format validation with identical seed/config for strict A/B comparison.
3. Add dedicated QFN32 diagnostic metrics (class-wise FN/F1 and confusion slices) to detect early regression.
4. Keep broad default-filter fine-tune as current safer reference until QFN32 issue is resolved.

## Interim Conclusion

`FINE-TUNED-RandomCrawler-1X.pt` is currently the stronger general checkpoint.  
`QFN-FINE-TUNED-RandomCrawler-1X.pt` shows targeted gain on QFN-3D but introduces broader regressions, including QFN32 degradation, and requires a validation-focused follow-up run.
