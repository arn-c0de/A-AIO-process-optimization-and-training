# Training Log – QFN Crawler Fine-Tune Boost (No Image Filter) – 2026-02-20

**Report ID:** TRAINING-LOG-2026-02-20-QFN-NO-FILTER-BOOST  
**Date:** 2026-02-20  
**Status:** Completed

## Vergleichsgrundlage

Verglichen wurden:
- `Simple-Sim/ARENA_REPORT.md`
- `Simple-Sim/docs/training_logs/2026-02-20-Training-Comparison-FineTuned-vs-QFN-FineTuned.md`

## Kerndifferenz zwischen den Reports

- Der ältere Vergleichslog zeigt `QFN-FINE-TUNED-RandomCrawler-1X.pt` noch hinter `FINE-TUNED-RandomCrawler-1X.pt` (Avg Accuracy 0.7005 vs 0.7164).
- Im aktuellen `ARENA_REPORT.md` ist `QFN-FINE-TUNED-RandomCrawler-1X.pt` jetzt auf **Platz 1** mit **Avg Accuracy 0.7693** und **Avg F1 0.7426**.

## Hauptänderung im Training

Beim Crawler-Fine-Tuning wurden QFN-spezifisch zusätzliche, saubere Daten ergänzt:
- `QFN32-2D`: +50 Samples
- `QFN-3D`: +50 Samples
- ohne Image-Filter (clean/no-filter Datenpfad)

Diese Änderung hat die Datenbasis im QFN-Bereich gezielt stabilisiert und den Checkpoint deutlich nach vorne gebracht.

## Ergebnis

- `QFN-FINE-TUNED-RandomCrawler-1X.pt` ist aktuell das führende Modell im Arena-Ranking.
- Der vorherige Rückstand gegenüber dem allgemeinen Fine-Tune wurde in einen klaren Vorsprung gedreht.

## Interpretation

Die zusätzlichen no-filter QFN-Samples (2D/3D) waren der entscheidende Hebel.  
Die frühere Annahme einer reinen QFN32-Regression ist mit dem neuen Stand nicht mehr haltbar; die aktualisierte Datenzusammensetzung hat die Generalisierung sichtbar verbessert.

## Nächste Schritte

1. Änderung als neue Trainings-Baseline dokumentieren (`QFN +50/+50 no-filter` als Standardvariante).
2. Optionalen A/B-Run mit identischer Seed-Konfiguration wiederholen, um die Robustheit des Vorsprungs zu bestätigen.
3. QFN32-2D und QFN-3D weiterhin separat monitoren, um frühzeitig Drift zu erkennen.
