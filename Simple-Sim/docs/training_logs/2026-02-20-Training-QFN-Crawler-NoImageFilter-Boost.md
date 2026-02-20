# Training Log – QFN Crawler Fine-Tune Boost (No Image Filter) – 2026-02-20

**Report ID:** TRAINING-LOG-2026-02-20-QFN-NO-FILTER-BOOST  
**Date:** 2026-02-20  
**Status:** Completed

## Comparison Basis

Compared files:
- `Simple-Sim/ARENA_REPORT.md`
- `Simple-Sim/docs/training_logs/2026-02-20-Training-Comparison-FineTuned-vs-QFN-FineTuned.md`

## Core Difference Between Reports

- The older comparison log still showed `QFN-FINE-TUNED-RandomCrawler-1X.pt` behind `FINE-TUNED-RandomCrawler-1X.pt` (Avg Accuracy 0.7005 vs 0.7164).
- In the current `ARENA_REPORT.md`, `QFN-FINE-TUNED-RandomCrawler-1X.pt` is now **Rank 1** with **Avg Accuracy 0.7693** and **Avg F1 0.7426**.

## Main Training Change

During crawler fine-tuning, additional clean QFN-focused data was added:
- `QFN32-2D`: +50 samples
- `QFN-3D`: +50 samples
- without image filter (clean/no-filter data path)

This change stabilized the QFN data basis and moved the checkpoint significantly forward.

## Result

- `QFN-FINE-TUNED-RandomCrawler-1X.pt` is currently the leading model in the arena ranking.
- The previous gap versus the general fine-tune was turned into a clear lead.

## Interpretation

The additional no-filter QFN samples (2D/3D) were the key lever.  
The earlier assumption of a pure QFN32 regression is no longer supported by the latest results; the updated data composition clearly improved generalization.

## Next Steps

1. Document this change as the new training baseline (`QFN +50/+50 no-filter` as standard variant).
2. Optionally repeat an A/B run with identical seed configuration to confirm lead robustness.
3. Continue separate monitoring of QFN32-2D and QFN-3D to detect drift early.
