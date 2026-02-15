# Model Arena Report
> Last updated: 2026-02-15 21:51:16

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
| 1 | DeepMind-small-v2_20260215_212646.bundle | Bundle | 0.9948 | 0.9948 | 7 | REFFERENCE_DATA_Transistor (1.0000) | Resistor-3d-v1 (0.9775) | 2026-02-15 21:48:43 |
| 2 | Random-DataCrawler-v2_MultiTrained.pt | Single | 0.7991 | 0.7629 | 7 | REFFERENCE_DATA_Transistor (1.0000) | QFN-3D-v1 (0.2500) | 2026-02-15 21:30:44 |
| 3 | Transistor-v1.pt | Single | 0.5748 | 0.5012 | 7 | Transistor-v1 (0.9938) | Resistor-3d-v1 (0.2525) | 2026-02-15 21:37:56 |
| 4 | Transistor-3D-v1.pt | Single | 0.4964 | 0.4262 | 7 | REFFERENCE_DATA_Transistor (1.0000) | Resistor-v1 (0.0975) | 2026-02-15 21:33:27 |
| 5 | QFN-v1.pt | Single | 0.4336 | 0.3562 | 7 | QFN-v1 (1.0000) | REFFERENCE_DATA_Transistor (0.1650) | 2026-02-15 21:37:05 |
| 6 | Seqent-DatasetCrawler-v1_MultiTrained.pt | Single | 0.4296 | 0.3249 | 7 | QFN-v1 (1.0000) | Resistor-3d-v1 (0.2500) | 2026-02-15 21:34:20 |
| 7 | Resistor-v1.pt | Single | 0.4136 | 0.2964 | 7 | Resistor-v1 (0.9950) | Resistor-3d-v1 (0.2500) | 2026-02-15 21:38:46 |
| 8 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | Single | 0.4095 | 0.3090 | 7 | Resistor-3d-v1 (0.9350) | QFN-v1 (0.2500) | 2026-02-15 21:35:19 |
| 9 | Resistor-3d-v1.pt | Single | 0.3755 | 0.2756 | 7 | Resistor-3d-v1 (0.9775) | REFFERENCE_DATA_Transistor (0.2500) | 2026-02-15 21:36:11 |
| 10 | QFN-3D-v1.pt | Single | 0.3736 | 0.2653 | 7 | QFN-3D-v1 (0.9975) | Resistor-3d-v1 (0.2500) | 2026-02-15 21:32:11 |

## Per-Dataset Breakdown

### Dataset: QFN-3D-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | DeepMind-small-v2_20260215_212646.bundle | 0.9975 | 0.9975 | all | 800 | 2026-02-15 21:44:06 |
| 2 | QFN-3D-v1.pt | 0.9975 | 0.9975 | all | 800 | 2026-02-15 21:31:38 |
| 3 | Transistor-3D-v1.pt | 0.4888 | 0.3648 | all | 800 | 2026-02-15 21:32:52 |
| 4 | QFN-v1.pt | 0.4225 | 0.3077 | all | 800 | 2026-02-15 21:36:32 |
| 5 | Transistor-v1.pt | 0.3925 | 0.2747 | all | 800 | 2026-02-15 21:37:23 |
| 6 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.3800 | 0.2492 | all | 800 | 2026-02-15 21:33:47 |
| 7 | Resistor-3d-v1.pt | 0.3750 | 0.2539 | all | 800 | 2026-02-15 21:35:38 |
| 8 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2712 | 0.1498 | all | 800 | 2026-02-15 21:34:42 |
| 9 | Resistor-v1.pt | 0.2500 | 0.1000 | all | 800 | 2026-02-15 21:38:13 |
| 10 | Random-DataCrawler-v2_MultiTrained.pt | 0.2500 | 0.1000 | all | 800 | 2026-02-15 21:30:10 |

### Dataset: QFN-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | DeepMind-small-v2_20260215_212646.bundle | 1.0000 | 1.0000 | all | 800 | 2026-02-15 21:45:40 |
| 2 | QFN-v1.pt | 1.0000 | 1.0000 | all | 800 | 2026-02-15 21:36:46 |
| 3 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 1.0000 | 1.0000 | all | 800 | 2026-02-15 21:34:02 |
| 4 | Transistor-v1.pt | 0.4975 | 0.3775 | all | 800 | 2026-02-15 21:37:37 |
| 5 | Random-DataCrawler-v2_MultiTrained.pt | 0.3775 | 0.2739 | all | 800 | 2026-02-15 21:30:25 |
| 6 | Resistor-3d-v1.pt | 0.2562 | 0.1483 | all | 800 | 2026-02-15 21:35:53 |
| 7 | Transistor-3D-v1.pt | 0.2562 | 0.1139 | all | 800 | 2026-02-15 21:33:07 |
| 8 | QFN-3D-v1.pt | 0.2525 | 0.1055 | all | 800 | 2026-02-15 21:31:52 |
| 9 | Resistor-v1.pt | 0.2525 | 0.1052 | all | 800 | 2026-02-15 21:38:27 |
| 10 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2500 | 0.1200 | all | 800 | 2026-02-15 21:34:59 |

### Dataset: REFFERENCE_DATA_Transistor

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | DeepMind-small-v2_20260215_212646.bundle | 1.0000 | 1.0000 | all | 800 | 2026-02-15 21:44:23 |
| 2 | Transistor-3D-v1.pt | 1.0000 | 1.0000 | all | 800 | 2026-02-15 21:33:12 |
| 3 | Random-DataCrawler-v2_MultiTrained.pt | 1.0000 | 1.0000 | all | 800 | 2026-02-15 21:30:29 |
| 4 | Transistor-v1.pt | 0.6050 | 0.5410 | all | 800 | 2026-02-15 21:37:41 |
| 5 | Resistor-v1.pt | 0.4225 | 0.2982 | all | 800 | 2026-02-15 21:38:31 |
| 6 | QFN-3D-v1.pt | 0.3075 | 0.2264 | all | 800 | 2026-02-15 21:31:56 |
| 7 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2550 | 0.1317 | all | 800 | 2026-02-15 21:35:03 |
| 8 | Resistor-3d-v1.pt | 0.2500 | 0.1250 | all | 800 | 2026-02-15 21:35:57 |
| 9 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.2500 | 0.1000 | all | 800 | 2026-02-15 21:34:06 |
| 10 | QFN-v1.pt | 0.1650 | 0.1117 | all | 800 | 2026-02-15 21:36:50 |

### Dataset: Resistor-3d-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | Random-DataCrawler-v2_MultiTrained.pt | 0.9900 | 0.9900 | all | 400 | 2026-02-15 21:30:32 |
| 2 | DeepMind-small-v2_20260215_212646.bundle | 0.9775 | 0.9776 | all | 400 | 2026-02-15 21:44:27 |
| 3 | Resistor-3d-v1.pt | 0.9775 | 0.9776 | all | 400 | 2026-02-15 21:36:00 |
| 4 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.9350 | 0.9347 | all | 400 | 2026-02-15 21:35:07 |
| 5 | Transistor-3D-v1.pt | 0.3825 | 0.2851 | all | 400 | 2026-02-15 21:33:15 |
| 6 | Transistor-v1.pt | 0.2525 | 0.1055 | all | 400 | 2026-02-15 21:37:45 |
| 7 | QFN-3D-v1.pt | 0.2500 | 0.1016 | all | 400 | 2026-02-15 21:32:00 |
| 8 | Resistor-v1.pt | 0.2500 | 0.1000 | all | 400 | 2026-02-15 21:38:35 |
| 9 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.2500 | 0.1000 | all | 400 | 2026-02-15 21:34:09 |
| 10 | QFN-v1.pt | 0.2050 | 0.0854 | all | 400 | 2026-02-15 21:36:54 |

### Dataset: Resistor-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | DeepMind-small-v2_20260215_212646.bundle | 0.9950 | 0.9950 | all | 400 | 2026-02-15 21:48:43 |
| 2 | Resistor-v1.pt | 0.9950 | 0.9950 | all | 400 | 2026-02-15 21:38:38 |
| 3 | Random-DataCrawler-v2_MultiTrained.pt | 0.9825 | 0.9825 | all | 400 | 2026-02-15 21:30:36 |
| 4 | Transistor-v1.pt | 0.6775 | 0.6750 | all | 400 | 2026-02-15 21:37:48 |
| 5 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.6450 | 0.5734 | all | 400 | 2026-02-15 21:35:10 |
| 6 | QFN-v1.pt | 0.5775 | 0.5128 | all | 400 | 2026-02-15 21:36:57 |
| 7 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.3900 | 0.3269 | all | 400 | 2026-02-15 21:34:12 |
| 8 | Resistor-3d-v1.pt | 0.2575 | 0.1416 | all | 400 | 2026-02-15 21:36:03 |
| 9 | QFN-3D-v1.pt | 0.2500 | 0.1000 | all | 400 | 2026-02-15 21:32:03 |
| 10 | Transistor-3D-v1.pt | 0.0975 | 0.1195 | all | 400 | 2026-02-15 21:33:18 |

### Dataset: Transistor-3D-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | DeepMind-small-v2_20260215_212646.bundle | 1.0000 | 1.0000 | all | 800 | 2026-02-15 21:44:34 |
| 2 | Transistor-3D-v1.pt | 1.0000 | 1.0000 | all | 800 | 2026-02-15 21:33:22 |
| 3 | Random-DataCrawler-v2_MultiTrained.pt | 1.0000 | 1.0000 | all | 800 | 2026-02-15 21:30:40 |
| 4 | Transistor-v1.pt | 0.6050 | 0.5410 | all | 800 | 2026-02-15 21:37:52 |
| 5 | Resistor-v1.pt | 0.4425 | 0.3140 | all | 800 | 2026-02-15 21:38:42 |
| 6 | QFN-3D-v1.pt | 0.3075 | 0.2264 | all | 800 | 2026-02-15 21:32:07 |
| 7 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2587 | 0.1347 | all | 800 | 2026-02-15 21:35:14 |
| 8 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.2512 | 0.1026 | all | 800 | 2026-02-15 21:34:16 |
| 9 | Resistor-3d-v1.pt | 0.2500 | 0.1250 | all | 800 | 2026-02-15 21:36:07 |
| 10 | QFN-v1.pt | 0.1650 | 0.1117 | all | 800 | 2026-02-15 21:37:01 |

### Dataset: Transistor-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | Random-DataCrawler-v2_MultiTrained.pt | 0.9938 | 0.9937 | all | 800 | 2026-02-15 21:30:44 |
| 2 | DeepMind-small-v2_20260215_212646.bundle | 0.9938 | 0.9937 | all | 800 | 2026-02-15 21:44:38 |
| 3 | Transistor-v1.pt | 0.9938 | 0.9937 | all | 800 | 2026-02-15 21:37:56 |
| 4 | QFN-v1.pt | 0.5000 | 0.3644 | all | 800 | 2026-02-15 21:37:05 |
| 5 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.4863 | 0.3955 | all | 800 | 2026-02-15 21:34:20 |
| 6 | Resistor-v1.pt | 0.2825 | 0.1624 | all | 800 | 2026-02-15 21:38:46 |
| 7 | Resistor-3d-v1.pt | 0.2625 | 0.1581 | all | 800 | 2026-02-15 21:36:11 |
| 8 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2512 | 0.1187 | all | 800 | 2026-02-15 21:35:19 |
| 9 | Transistor-3D-v1.pt | 0.2500 | 0.1000 | all | 800 | 2026-02-15 21:33:27 |
| 10 | QFN-3D-v1.pt | 0.2500 | 0.1000 | all | 800 | 2026-02-15 21:32:11 |

## Bundle Details

### DeepMind-small-v2_20260215_212646.bundle
- **Profiles:** chip_0603_resistor@1, chip_0603_resistor_3d@1, qfn32_ic@1, qfn32_ic_3d@1, sot23_transistor@1, sot23_transistor_3d@1
- **Checkpoints:** 6
- **Created:** 2026-02-15T20:26:46Z

## Per-Model Detail Cards

### DeepMind-small-v2_20260215_212646.bundle (Bundle)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.9975 | 0.9975 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 7 | 1.0000 | 75.6 img/s |
| QFN-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 12 | 1.0000 | 67.6 img/s |
| REFFERENCE_DATA_Transistor | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 11 | 1.0000 | 553.1 img/s |
| Resistor-3d-v1 | all | 0.9775 | 0.9776 | MISALIGNED:0.0800 MISSING:0.0000 TOMBSTONE:0.0100 | 400 | 9 | 0.9667 | 546.5 img/s |
| Resistor-v1 | all | 0.9950 | 0.9950 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 8 | 0.9833 | 483.9 img/s |
| Transistor-3D-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 11 | 1.0000 | 557.2 img/s |
| Transistor-v1 | all | 0.9938 | 0.9937 | MISALIGNED:0.0250 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 6 | 0.9917 | 515.5 img/s |

### Random-DataCrawler-v2_MultiTrained.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 13 | 0.9905 | 61.9 img/s |
| QFN-v1 | all | 0.3775 | 0.2739 | MISALIGNED:0.4900 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 13 | 0.9905 | 65.7 img/s |
| REFFERENCE_DATA_Transistor | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 13 | 0.9905 | 520.7 img/s |
| Resistor-3d-v1 | all | 0.9900 | 0.9900 | MISALIGNED:0.0400 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 13 | 0.9905 | 517.9 img/s |
| Resistor-v1 | all | 0.9825 | 0.9825 | MISALIGNED:0.0500 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 13 | 0.9905 | 446.9 img/s |
| Transistor-3D-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 13 | 0.9905 | 478.9 img/s |
| Transistor-v1 | all | 0.9938 | 0.9937 | MISALIGNED:0.0150 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 13 | 0.9905 | 502.5 img/s |

### Transistor-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.3925 | 0.2747 | MISALIGNED:0.0500 MISSING:0.3800 TOMBSTONE:1.0000 | 800 | 6 | 0.9917 | 77.4 img/s |
| QFN-v1 | all | 0.4975 | 0.3775 | MISALIGNED:0.0600 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 6 | 0.9917 | 66.9 img/s |
| REFFERENCE_DATA_Transistor | all | 0.6050 | 0.5410 | MISALIGNED:0.5550 MISSING:0.0250 TOMBSTONE:0.0000 | 800 | 6 | 0.9917 | 575.6 img/s |
| Resistor-3d-v1 | all | 0.2525 | 0.1055 | MISALIGNED:1.0000 MISSING:0.0000 TOMBSTONE:0.9900 | 400 | 6 | 0.9917 | 548.5 img/s |
| Resistor-v1 | all | 0.6775 | 0.6750 | MISALIGNED:0.2900 MISSING:0.6100 TOMBSTONE:0.0000 | 400 | 6 | 0.9917 | 473.6 img/s |
| Transistor-3D-v1 | all | 0.6050 | 0.5410 | MISALIGNED:0.5550 MISSING:0.0250 TOMBSTONE:0.0000 | 800 | 6 | 0.9917 | 584.1 img/s |
| Transistor-v1 | all | 0.9938 | 0.9937 | MISALIGNED:0.0250 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 6 | 0.9917 | 515.7 img/s |

### Transistor-3D-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.4888 | 0.3648 | MISALIGNED:0.0450 MISSING:1.0000 TOMBSTONE:0.0000 | 800 | 11 | 1.0000 | 67.0 img/s |
| QFN-v1 | all | 0.2562 | 0.1139 | MISALIGNED:1.0000 MISSING:0.9750 TOMBSTONE:0.0000 | 800 | 11 | 1.0000 | 63.1 img/s |
| REFFERENCE_DATA_Transistor | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 11 | 1.0000 | 536.9 img/s |
| Resistor-3d-v1 | all | 0.3825 | 0.2851 | MISALIGNED:0.0000 MISSING:0.4700 TOMBSTONE:1.0000 | 400 | 11 | 1.0000 | 549.6 img/s |
| Resistor-v1 | all | 0.0975 | 0.1195 | MISALIGNED:0.9900 MISSING:0.7100 TOMBSTONE:0.9100 | 400 | 11 | 1.0000 | 469.6 img/s |
| Transistor-3D-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 11 | 1.0000 | 540.1 img/s |
| Transistor-v1 | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:0.0000 | 800 | 11 | 1.0000 | 499.4 img/s |

### QFN-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.4225 | 0.3077 | MISALIGNED:0.2450 MISSING:0.9350 TOMBSTONE:1.0000 | 800 | 12 | 1.0000 | 74.1 img/s |
| QFN-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 12 | 1.0000 | 68.2 img/s |
| REFFERENCE_DATA_Transistor | all | 0.1650 | 0.1117 | MISALIGNED:1.0000 MISSING:0.6450 TOMBSTONE:1.0000 | 800 | 12 | 1.0000 | 578.2 img/s |
| Resistor-3d-v1 | all | 0.2050 | 0.0854 | MISALIGNED:1.0000 MISSING:0.1800 TOMBSTONE:1.0000 | 400 | 12 | 1.0000 | 511.3 img/s |
| Resistor-v1 | all | 0.5775 | 0.5128 | MISALIGNED:0.3500 MISSING:0.0000 TOMBSTONE:0.3400 | 400 | 12 | 1.0000 | 480.9 img/s |
| Transistor-3D-v1 | all | 0.1650 | 0.1117 | MISALIGNED:1.0000 MISSING:0.6450 TOMBSTONE:1.0000 | 800 | 12 | 1.0000 | 583.2 img/s |
| Transistor-v1 | all | 0.5000 | 0.3644 | MISALIGNED:0.9150 MISSING:0.0500 TOMBSTONE:1.0000 | 800 | 12 | 1.0000 | 502.8 img/s |

### Seqent-DatasetCrawler-v1_MultiTrained.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.3800 | 0.2492 | MISALIGNED:0.4450 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 9 | 1.0000 | 67.9 img/s |
| QFN-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 9 | 1.0000 | 67.1 img/s |
| REFFERENCE_DATA_Transistor | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 9 | 1.0000 | 580.8 img/s |
| Resistor-3d-v1 | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 9 | 1.0000 | 539.8 img/s |
| Resistor-v1 | all | 0.3900 | 0.3269 | MISALIGNED:0.8700 MISSING:0.0000 TOMBSTONE:0.8700 | 400 | 9 | 1.0000 | 478.0 img/s |
| Transistor-3D-v1 | all | 0.2512 | 0.1026 | MISALIGNED:0.9950 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 9 | 1.0000 | 599.3 img/s |
| Transistor-v1 | all | 0.4863 | 0.3955 | MISALIGNED:0.7600 MISSING:0.1200 TOMBSTONE:1.0000 | 800 | 9 | 1.0000 | 503.9 img/s |

### Resistor-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.2500 | 0.1000 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 8 | 0.9833 | 75.3 img/s |
| QFN-v1 | all | 0.2525 | 0.1052 | MISALIGNED:0.0000 MISSING:0.9900 TOMBSTONE:1.0000 | 800 | 8 | 0.9833 | 68.3 img/s |
| REFFERENCE_DATA_Transistor | all | 0.4225 | 0.2982 | MISALIGNED:0.1050 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 8 | 0.9833 | 579.8 img/s |
| Resistor-3d-v1 | all | 0.2500 | 0.1000 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 8 | 0.9833 | 541.8 img/s |
| Resistor-v1 | all | 0.9950 | 0.9950 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 8 | 0.9833 | 478.1 img/s |
| Transistor-3D-v1 | all | 0.4425 | 0.3140 | MISALIGNED:0.0600 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 8 | 0.9833 | 598.0 img/s |
| Transistor-v1 | all | 0.2825 | 0.1624 | MISALIGNED:0.0000 MISSING:0.9850 TOMBSTONE:0.8900 | 800 | 8 | 0.9833 | 512.1 img/s |

### Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.2712 | 0.1498 | MISALIGNED:0.0100 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | - | - | 62.5 img/s |
| QFN-v1 | all | 0.2500 | 0.1200 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | - | - | 55.9 img/s |
| REFFERENCE_DATA_Transistor | all | 0.2550 | 0.1317 | MISALIGNED:0.0000 MISSING:0.9950 TOMBSTONE:1.0000 | 800 | - | - | 497.5 img/s |
| Resistor-3d-v1 | all | 0.9350 | 0.9347 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.2100 | 400 | - | - | 446.1 img/s |
| Resistor-v1 | all | 0.6450 | 0.5734 | MISALIGNED:0.0200 MISSING:0.0300 TOMBSTONE:1.0000 | 400 | - | - | 408.9 img/s |
| Transistor-3D-v1 | all | 0.2587 | 0.1347 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | - | - | 505.3 img/s |
| Transistor-v1 | all | 0.2512 | 0.1187 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | - | - | 438.6 img/s |

### Resistor-3d-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.3750 | 0.2539 | MISALIGNED:0.5000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 9 | 0.9667 | 75.9 img/s |
| QFN-v1 | all | 0.2562 | 0.1483 | MISALIGNED:0.0950 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 9 | 0.9667 | 67.3 img/s |
| REFFERENCE_DATA_Transistor | all | 0.2500 | 0.1250 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 9 | 0.9667 | 582.3 img/s |
| Resistor-3d-v1 | all | 0.9775 | 0.9776 | MISALIGNED:0.0800 MISSING:0.0000 TOMBSTONE:0.0100 | 400 | 9 | 0.9667 | 525.2 img/s |
| Resistor-v1 | all | 0.2575 | 0.1416 | MISALIGNED:0.0300 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 9 | 0.9667 | 468.0 img/s |
| Transistor-3D-v1 | all | 0.2500 | 0.1250 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 9 | 0.9667 | 582.8 img/s |
| Transistor-v1 | all | 0.2625 | 0.1581 | MISALIGNED:0.1300 MISSING:1.0000 TOMBSTONE:0.9950 | 800 | 9 | 0.9667 | 497.3 img/s |

### QFN-3D-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.9975 | 0.9975 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 7 | 1.0000 | 72.0 img/s |
| QFN-v1 | all | 0.2525 | 0.1055 | MISALIGNED:1.0000 MISSING:0.9900 TOMBSTONE:0.0000 | 800 | 7 | 1.0000 | 66.1 img/s |
| REFFERENCE_DATA_Transistor | all | 0.3075 | 0.2264 | MISALIGNED:0.0050 MISSING:1.0000 TOMBSTONE:0.7650 | 800 | 7 | 1.0000 | 573.8 img/s |
| Resistor-3d-v1 | all | 0.2500 | 0.1016 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 7 | 1.0000 | 508.7 img/s |
| Resistor-v1 | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:0.0000 | 400 | 7 | 1.0000 | 473.6 img/s |
| Transistor-3D-v1 | all | 0.3075 | 0.2264 | MISALIGNED:0.0050 MISSING:1.0000 TOMBSTONE:0.7650 | 800 | 7 | 1.0000 | 557.3 img/s |
| Transistor-v1 | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:0.0000 | 800 | 7 | 1.0000 | 503.6 img/s |

## History

- 2026-02-15: Report generated (10 models, 7 datasets)
