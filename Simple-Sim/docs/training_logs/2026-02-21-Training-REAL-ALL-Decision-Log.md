# Training Log: REAL-ALL Decision Note

**Report ID:** TL-2026-02-21-REAL-ALL-Decision
**Date:** 2026-02-21
**Model:** Real-ALL-200clear-100filter-RandomCrawler-v2_20260221_142203.pt
**Status:** Completed

---

## Dataset Build Plan Used
- Per profile:
  - 200 clear samples
  - 100 realism-mode samples
- Intent: improve cross-dataset robustness instead of maximizing single-dataset peak only.

---

## Result Summary
- REAL-ALL is currently **not Top-3** in arena average ranking.
- Current overall rank in arena table: **#6** (Avg Accuracy `0.6605`, Avg F1 `0.6400`).
- However, REAL-ALL has the **best worst-dataset score** among listed models:
  - REAL-ALL worst dataset: `CLEAN-IC-16-3D (0.4637)`
  - Top-1 worst dataset: `0.3187`
  - Top-2 worst dataset: `0.2900`
  - Top-3 worst dataset: `0.4900`

Interpretation:
- Peak leaderboard position is lower.
- Performance floor is higher.
- This indicates stronger generalization robustness across difficult datasets.

## Additional Dataset Observations
- `CLEAN-QFN32-2D`:
  - Top-1: `CLEAN-QFN32-2D.pt` (`1.0000` / `1.0000`)
  - Top-3: `Real-ALL-200clear-100filter-RandomCrawler-v2_20260221_142203.pt` (`0.6375` / `0.5885`)
  - Note: REAL-ALL reaches Top-3 on this dataset, behind the specialized main model trained for this target dataset and `Real-ALL-200clear-100filter-RandomCrawler-v2.pt`.

- `CLEAN-QFN-3D`:
  - REAL-ALL: Rank `#5` (`0.4900` / `0.4646`)
  - `QFN-FINE-TUNED-RandomCrawler-1X.pt`: Rank `#7` (`0.3187` / `0.2760`)
  - Note: REAL-ALL outperforms the QFN fine-tuned model on this 3D QFN benchmark.

---

## Decision
- REAL-ALL is selected as the **new general model** for robustness-focused usage.
- Follow-up tuning should target improving mean rank while preserving the current robustness floor.

---

## Next Steps
1. Start next training cycle from REAL-ALL baseline.
2. Compare with same evaluation protocol and report both mean score and worst-dataset floor.
3. Keep this robustness criterion as a primary selection metric.

---

*Prepared by: ____*
