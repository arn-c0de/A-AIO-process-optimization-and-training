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
- Current overall rank in arena table: **#5** (Avg Accuracy `0.6605`, Avg F1 `0.6400`).
- However, REAL-ALL has the **best worst-dataset score** among listed models:
  - REAL-ALL worst dataset: `CLEAN-IC-16-3D (0.4637)`
  - Top-1 worst dataset: `0.3187`
  - Top-2 worst dataset: `0.2900`
  - Top-3 worst dataset: `0.2500`

Interpretation:
- Peak leaderboard position is lower.
- Performance floor is higher.
- This indicates stronger generalization robustness across difficult datasets.

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
