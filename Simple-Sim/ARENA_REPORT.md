# Model Arena Report
> Last updated: 2026-02-15 13:49:53

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
| 2 | Random-DataCrawler-v1_MultiTrained.pt | Single | 0.8713 | 0.8512 | 5 | Transistor-3D-v1 (0.9962) | QFN-v1 (0.3887) | 2026-02-15 13:46:14 |
| 3 | QFN-v1.pt | Single | 0.6925 | 0.6257 | 3 | QFN-v1 (1.0000) | Transistor-v1 (0.5000) | 2026-02-14 21:08:23 |
| 4 | Transistor-v1.pt | Single | 0.6295 | 0.5700 | 5 | Transistor-v1 (0.9938) | QFN-3D-v1 (0.3825) | 2026-02-15 13:49:28 |
| 5 | Seqent-DatasetCrawler-v1_MultiTrained.pt | Single | 0.4596 | 0.3624 | 6 | QFN-v1 (1.0000) | Resistor-3d-v1 (0.2500) | 2026-02-15 13:46:57 |
| 6 | Resistor-v1.pt | Single | 0.4445 | 0.3353 | 5 | Resistor-v1 (0.9950) | Resistor-3d-v1 (0.2500) | 2026-02-15 13:49:43 |
| 7 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | Single | 0.4342 | 0.3376 | 6 | Resistor-3d-v1 (0.9350) | QFN-v1 (0.2500) | 2026-02-15 13:47:27 |
| 8 | Resistor-3d-v1.pt | Single | 0.4237 | 0.3306 | 5 | Resistor-3d-v1 (0.9775) | Transistor-3D-v1 (0.2500) | 2026-02-15 13:47:52 |

## Per-Dataset Breakdown

### Dataset: QFN-3D-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | Transistor-v1.pt | 0.3825 | 0.2682 | all | 800 | 2026-02-15 13:49:20 |
| 2 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.3800 | 0.2492 | all | 800 | 2026-02-15 13:46:57 |
| 3 | Resistor-3d-v1.pt | 0.3713 | 0.2507 | all | 800 | 2026-02-15 13:47:52 |
| 4 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2650 | 0.1439 | all | 800 | 2026-02-15 13:47:18 |

### Dataset: QFN-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 1.0000 | 1.0000 | all | 800 | 2026-02-15 12:58:27 |
| 2 | DeepMind-small-v1_20260214_210452.bundle | 1.0000 | 1.0000 | all | 800 | 2026-02-14 21:08:44 |
| 3 | QFN-v1.pt | 1.0000 | 1.0000 | all | 800 | 2026-02-14 21:08:11 |
| 4 | Transistor-v1.pt | 0.4975 | 0.3775 | all | 800 | 2026-02-14 21:07:37 |
| 5 | Random-DataCrawler-v1_MultiTrained.pt | 0.3887 | 0.2887 | all | 800 | 2026-02-15 13:08:12 |
| 6 | Resistor-v1.pt | 0.2525 | 0.1052 | all | 800 | 2026-02-15 12:55:29 |
| 7 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2500 | 0.1200 | all | 800 | 2026-02-15 12:17:19 |

### Dataset: Resistor-3d-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | Random-DataCrawler-v1_MultiTrained.pt | 0.9900 | 0.9900 | all | 400 | 2026-02-15 13:46:00 |
| 2 | Resistor-3d-v1.pt | 0.9775 | 0.9776 | all | 400 | 2026-02-15 11:50:02 |
| 3 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.9350 | 0.9347 | all | 400 | 2026-02-15 12:15:26 |
| 4 | Resistor-v1.pt | 0.2500 | 0.1000 | all | 400 | 2026-02-15 13:49:43 |
| 5 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.2500 | 0.1000 | all | 400 | 2026-02-15 12:58:39 |

### Dataset: Resistor-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | Resistor-v1.pt | 0.9950 | 0.9950 | all | 400 | 2026-02-15 12:35:37 |
| 2 | DeepMind-small-v1_20260214_210452.bundle | 0.9950 | 0.9950 | all | 400 | 2026-02-14 21:08:51 |
| 3 | Random-DataCrawler-v1_MultiTrained.pt | 0.9850 | 0.9850 | all | 400 | 2026-02-15 13:08:30 |
| 4 | Transistor-v1.pt | 0.6775 | 0.6750 | all | 400 | 2026-02-14 21:07:44 |
| 5 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.6450 | 0.5734 | all | 400 | 2026-02-15 12:35:33 |
| 6 | QFN-v1.pt | 0.5775 | 0.5128 | all | 400 | 2026-02-14 21:08:17 |
| 7 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.3900 | 0.3269 | all | 400 | 2026-02-15 12:58:48 |
| 8 | Resistor-3d-v1.pt | 0.2575 | 0.1416 | all | 400 | 2026-02-15 12:35:00 |

### Dataset: Transistor-3D-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | Random-DataCrawler-v1_MultiTrained.pt | 0.9962 | 0.9962 | all | 800 | 2026-02-15 13:46:14 |
| 2 | Transistor-v1.pt | 0.5962 | 0.5354 | all | 800 | 2026-02-15 13:49:28 |
| 3 | Resistor-v1.pt | 0.4425 | 0.3140 | all | 800 | 2026-02-15 13:49:37 |
| 4 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2587 | 0.1347 | all | 800 | 2026-02-15 13:47:27 |
| 5 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.2512 | 0.1026 | all | 800 | 2026-02-15 13:46:31 |
| 6 | Resistor-3d-v1.pt | 0.2500 | 0.1250 | all | 800 | 2026-02-15 13:47:35 |

### Dataset: Transistor-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | Random-DataCrawler-v1_MultiTrained.pt | 0.9962 | 0.9962 | all | 800 | 2026-02-15 13:08:39 |
| 2 | DeepMind-small-v1_20260214_210452.bundle | 0.9938 | 0.9937 | all | 800 | 2026-02-14 21:08:59 |
| 3 | Transistor-v1.pt | 0.9938 | 0.9937 | all | 800 | 2026-02-14 21:07:52 |
| 4 | QFN-v1.pt | 0.5000 | 0.3644 | all | 800 | 2026-02-14 21:08:23 |
| 5 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.4863 | 0.3955 | all | 800 | 2026-02-15 12:58:56 |
| 6 | Resistor-v1.pt | 0.2825 | 0.1624 | all | 800 | 2026-02-14 21:06:52 |
| 7 | Resistor-3d-v1.pt | 0.2625 | 0.1581 | all | 800 | 2026-02-15 12:34:31 |
| 8 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2512 | 0.1187 | all | 800 | 2026-02-15 12:34:27 |

## Bundle Details

### DeepMind-small-v1_20260214_210452.bundle
- **Profiles:** chip_0603_resistor@1, qfn32_ic@1, qfn32_ic_3d@1, sot23_transistor@1
- **Checkpoints:** 4
- **Created:** 2026-02-14T20:04:52Z

## Per-Model Detail Cards

### DeepMind-small-v1_20260214_210452.bundle (Bundle)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 12 | 1.0000 | 64.6 img/s |
| Resistor-v1 | all | 0.9950 | 0.9950 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 8 | 0.9833 | 450.1 img/s |
| Transistor-v1 | all | 0.9938 | 0.9937 | MISALIGNED:0.0250 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 6 | 0.9917 | 472.4 img/s |

### Random-DataCrawler-v1_MultiTrained.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-v1 | all | 0.3887 | 0.2887 | MISALIGNED:0.4500 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 14 | 0.9833 | 68.8 img/s |
| Resistor-3d-v1 | all | 0.9900 | 0.9900 | MISALIGNED:0.0400 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 9 | 0.9769 | 531.6 img/s |
| Resistor-v1 | all | 0.9850 | 0.9850 | MISALIGNED:0.0600 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 14 | 0.9833 | 491.3 img/s |
| Transistor-3D-v1 | all | 0.9962 | 0.9962 | MISALIGNED:0.0150 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 9 | 0.9769 | 578.7 img/s |
| Transistor-v1 | all | 0.9962 | 0.9962 | MISALIGNED:0.0150 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 14 | 0.9833 | 516.3 img/s |

### QFN-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 12 | 1.0000 | 64.3 img/s |
| Resistor-v1 | all | 0.5775 | 0.5128 | MISALIGNED:0.3500 MISSING:0.0000 TOMBSTONE:0.3400 | 400 | 12 | 1.0000 | 453.0 img/s |
| Transistor-v1 | all | 0.5000 | 0.3644 | MISALIGNED:0.9150 MISSING:0.0500 TOMBSTONE:1.0000 | 800 | 12 | 1.0000 | 471.8 img/s |

### Transistor-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.3825 | 0.2682 | MISALIGNED:0.0950 MISSING:0.3750 TOMBSTONE:1.0000 | 800 | 6 | 0.9917 | 76.3 img/s |
| QFN-v1 | all | 0.4975 | 0.3775 | MISALIGNED:0.0600 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 6 | 0.9917 | 61.2 img/s |
| Resistor-v1 | all | 0.6775 | 0.6750 | MISALIGNED:0.2900 MISSING:0.6100 TOMBSTONE:0.0000 | 400 | 6 | 0.9917 | 449.9 img/s |
| Transistor-3D-v1 | all | 0.5962 | 0.5354 | MISALIGNED:0.5650 MISSING:0.0500 TOMBSTONE:0.0000 | 800 | 6 | 0.9917 | 589.1 img/s |
| Transistor-v1 | all | 0.9938 | 0.9937 | MISALIGNED:0.0250 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 6 | 0.9917 | 479.4 img/s |

### Seqent-DatasetCrawler-v1_MultiTrained.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.3800 | 0.2492 | MISALIGNED:0.4450 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 9 | 1.0000 | 74.4 img/s |
| QFN-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 9 | 1.0000 | 56.5 img/s |
| Resistor-3d-v1 | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 9 | 1.0000 | 552.1 img/s |
| Resistor-v1 | all | 0.3900 | 0.3269 | MISALIGNED:0.8700 MISSING:0.0000 TOMBSTONE:0.8700 | 400 | 9 | 1.0000 | 476.1 img/s |
| Transistor-3D-v1 | all | 0.2512 | 0.1026 | MISALIGNED:0.9950 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 9 | 1.0000 | 605.7 img/s |
| Transistor-v1 | all | 0.4863 | 0.3955 | MISALIGNED:0.7600 MISSING:0.1200 TOMBSTONE:1.0000 | 800 | 9 | 1.0000 | 525.7 img/s |

### Resistor-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-v1 | all | 0.2525 | 0.1052 | MISALIGNED:0.0000 MISSING:0.9900 TOMBSTONE:1.0000 | 800 | 8 | 0.9833 | 59.7 img/s |
| Resistor-3d-v1 | all | 0.2500 | 0.1000 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 8 | 0.9833 | 519.1 img/s |
| Resistor-v1 | all | 0.9950 | 0.9950 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 8 | 0.9833 | 478.7 img/s |
| Transistor-3D-v1 | all | 0.4425 | 0.3140 | MISALIGNED:0.0600 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 8 | 0.9833 | 598.0 img/s |
| Transistor-v1 | all | 0.2825 | 0.1624 | MISALIGNED:0.0000 MISSING:0.9850 TOMBSTONE:0.8900 | 800 | 8 | 0.9833 | 485.9 img/s |

### Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.2650 | 0.1439 | MISALIGNED:0.0150 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | - | - | 63.5 img/s |
| QFN-v1 | all | 0.2500 | 0.1200 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | - | - | 52.6 img/s |
| Resistor-3d-v1 | all | 0.9350 | 0.9347 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.2100 | 400 | - | - | 445.3 img/s |
| Resistor-v1 | all | 0.6450 | 0.5734 | MISALIGNED:0.0200 MISSING:0.0300 TOMBSTONE:1.0000 | 400 | - | - | 420.0 img/s |
| Transistor-3D-v1 | all | 0.2587 | 0.1347 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | - | - | 505.3 img/s |
| Transistor-v1 | all | 0.2512 | 0.1187 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | - | - | 443.5 img/s |

### Resistor-3d-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.3713 | 0.2507 | MISALIGNED:0.5150 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 9 | 0.9667 | 74.3 img/s |
| Resistor-3d-v1 | all | 0.9775 | 0.9776 | MISALIGNED:0.0800 MISSING:0.0000 TOMBSTONE:0.0100 | 400 | 9 | 0.9667 | 563.1 img/s |
| Resistor-v1 | all | 0.2575 | 0.1416 | MISALIGNED:0.0300 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 9 | 0.9667 | 481.5 img/s |
| Transistor-3D-v1 | all | 0.2500 | 0.1250 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 9 | 0.9667 | 599.5 img/s |
| Transistor-v1 | all | 0.2625 | 0.1581 | MISALIGNED:0.1300 MISSING:1.0000 TOMBSTONE:0.9950 | 800 | 9 | 0.9667 | 517.1 img/s |

## History

- 2026-02-15: Report generated (8 models, 6 datasets)
