# Model Arena Report
> Last updated: 2026-02-15 12:18:07

## Charts

### Top Avg Accuracy

![Top Avg Accuracy](ARENA_REPORT_assets/top_avg_accuracy.svg)

### Top Avg F1

![Top Avg F1](ARENA_REPORT_assets/top_avg_f1.svg)

### Model Storage Breakdown

![Model Storage Breakdown](ARENA_REPORT_assets/model_size_pie.svg)

## Overall Ranking

| Rank | Model | Type | Avg Accuracy | Avg F1 | Datasets Tested | Best Dataset | Worst Dataset | Last Run |
|---:|---|---|---:|---:|---:|---|---|---|
| 1 | DeepMind-small-v1_20260214_210452.bundle | Bundle | 0.9962 | 0.9962 | 3 | QFN-v1 (1.0000) | Transistor-v1 (0.9938) | 2026-02-14 21:08:59 |
| 2 | Transistor-v1.pt | Single | 0.7229 | 0.6821 | 3 | Transistor-v1 (0.9938) | QFN-v1 (0.4975) | 2026-02-14 21:07:52 |
| 3 | QFN-v1.pt | Single | 0.6925 | 0.6257 | 3 | QFN-v1 (1.0000) | Transistor-v1 (0.5000) | 2026-02-14 21:08:23 |
| 4 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | Single | 0.5203 | 0.4367 | 4 | Resistor-3d-v1 (0.9350) | QFN-v1 (0.2500) | 2026-02-15 12:17:39 |
| 5 | Resistor-3d-v1.pt | Single | 0.4992 | 0.4257 | 3 | Resistor-3d-v1 (0.9775) | Resistor-v1 (0.2575) | 2026-02-15 12:00:41 |
| 6 | Resistor-v1.pt | Single | 0.4450 | 0.3406 | 4 | Resistor-v1 (0.9950) | Resistor-3d-v1 (0.2500) | 2026-02-15 11:50:15 |

## Per-Dataset Breakdown

### Dataset: QFN-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | DeepMind-small-v1_20260214_210452.bundle | 1.0000 | 1.0000 | all | 800 | 2026-02-14 21:08:44 |
| 2 | QFN-v1.pt | 1.0000 | 1.0000 | all | 800 | 2026-02-14 21:08:11 |
| 3 | Transistor-v1.pt | 0.4975 | 0.3775 | all | 800 | 2026-02-14 21:07:37 |
| 4 | Resistor-v1.pt | 0.2525 | 0.1052 | all | 800 | 2026-02-14 21:07:17 |
| 5 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2500 | 0.1200 | all | 800 | 2026-02-15 12:17:19 |

### Dataset: Resistor-3d-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | Resistor-3d-v1.pt | 0.9775 | 0.9776 | all | 400 | 2026-02-15 11:50:02 |
| 2 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.9350 | 0.9347 | all | 400 | 2026-02-15 12:15:26 |
| 3 | Resistor-v1.pt | 0.2500 | 0.1000 | all | 400 | 2026-02-15 11:50:15 |

### Dataset: Resistor-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | DeepMind-small-v1_20260214_210452.bundle | 0.9950 | 0.9950 | all | 400 | 2026-02-14 21:08:51 |
| 2 | Resistor-v1.pt | 0.9950 | 0.9950 | all | 400 | 2026-02-14 21:06:33 |
| 3 | Transistor-v1.pt | 0.6775 | 0.6750 | all | 400 | 2026-02-14 21:07:44 |
| 4 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.6450 | 0.5734 | all | 400 | 2026-02-15 12:15:16 |
| 5 | QFN-v1.pt | 0.5775 | 0.5128 | all | 400 | 2026-02-14 21:08:17 |
| 6 | Resistor-3d-v1.pt | 0.2575 | 0.1416 | all | 400 | 2026-02-15 11:49:34 |

### Dataset: Transistor-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | DeepMind-small-v1_20260214_210452.bundle | 0.9938 | 0.9937 | all | 800 | 2026-02-14 21:08:59 |
| 2 | Transistor-v1.pt | 0.9938 | 0.9937 | all | 800 | 2026-02-14 21:07:52 |
| 3 | QFN-v1.pt | 0.5000 | 0.3644 | all | 800 | 2026-02-14 21:08:23 |
| 4 | Resistor-v1.pt | 0.2825 | 0.1624 | all | 800 | 2026-02-14 21:06:52 |
| 5 | Resistor-3d-v1.pt | 0.2625 | 0.1581 | all | 800 | 2026-02-15 12:00:41 |
| 6 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2512 | 0.1187 | all | 800 | 2026-02-15 12:17:39 |

## Bundle Details

### DeepMind-small-v1_20260214_210452.bundle
- **Profiles:** chip_0603_resistor@1, qfn32_ic@1, sot23_transistor@1
- **Checkpoints:** 3
- **Created:** 2026-02-14T20:04:52Z

## Per-Model Detail Cards

### DeepMind-small-v1_20260214_210452.bundle (Bundle)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 12 | 1.0000 | 64.6 img/s |
| Resistor-v1 | all | 0.9950 | 0.9950 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 8 | 0.9833 | 450.1 img/s |
| Transistor-v1 | all | 0.9938 | 0.9937 | MISALIGNED:0.0250 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 6 | 0.9917 | 472.4 img/s |

### Transistor-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-v1 | all | 0.4975 | 0.3775 | MISALIGNED:0.0600 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 6 | 0.9917 | 61.2 img/s |
| Resistor-v1 | all | 0.6775 | 0.6750 | MISALIGNED:0.2900 MISSING:0.6100 TOMBSTONE:0.0000 | 400 | 6 | 0.9917 | 449.9 img/s |
| Transistor-v1 | all | 0.9938 | 0.9937 | MISALIGNED:0.0250 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 6 | 0.9917 | 479.4 img/s |

### QFN-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 12 | 1.0000 | 64.3 img/s |
| Resistor-v1 | all | 0.5775 | 0.5128 | MISALIGNED:0.3500 MISSING:0.0000 TOMBSTONE:0.3400 | 400 | 12 | 1.0000 | 453.0 img/s |
| Transistor-v1 | all | 0.5000 | 0.3644 | MISALIGNED:0.9150 MISSING:0.0500 TOMBSTONE:1.0000 | 800 | 12 | 1.0000 | 471.8 img/s |

### Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-v1 | all | 0.2500 | 0.1200 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | - | - | 52.6 img/s |
| Resistor-3d-v1 | all | 0.9350 | 0.9347 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.2100 | 400 | - | - | 445.3 img/s |
| Resistor-v1 | all | 0.6450 | 0.5734 | MISALIGNED:0.0200 MISSING:0.0300 TOMBSTONE:1.0000 | 400 | - | - | 427.6 img/s |
| Transistor-v1 | all | 0.2512 | 0.1187 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | - | - | 442.5 img/s |

### Resistor-3d-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Resistor-3d-v1 | all | 0.9775 | 0.9776 | MISALIGNED:0.0800 MISSING:0.0000 TOMBSTONE:0.0100 | 400 | 9 | 0.9667 | 563.1 img/s |
| Resistor-v1 | all | 0.2575 | 0.1416 | MISALIGNED:0.0300 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 9 | 0.9667 | 441.1 img/s |
| Transistor-v1 | all | 0.2625 | 0.1581 | MISALIGNED:0.1300 MISSING:1.0000 TOMBSTONE:0.9950 | 800 | 9 | 0.9667 | 477.0 img/s |

### Resistor-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-v1 | all | 0.2525 | 0.1052 | MISALIGNED:0.0000 MISSING:0.9900 TOMBSTONE:1.0000 | 800 | 8 | 0.9833 | 60.4 img/s |
| Resistor-3d-v1 | all | 0.2500 | 0.1000 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 8 | 0.9833 | 558.8 img/s |
| Resistor-v1 | all | 0.9950 | 0.9950 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 8 | 0.9833 | 382.3 img/s |
| Transistor-v1 | all | 0.2825 | 0.1624 | MISALIGNED:0.0000 MISSING:0.9850 TOMBSTONE:0.8900 | 800 | 8 | 0.9833 | 485.9 img/s |

## History

- 2026-02-15: Report generated (6 models, 4 datasets)
