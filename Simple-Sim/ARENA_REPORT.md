# Model Arena Report
> Last updated: 2026-02-16 17:27:58

## Quick Navigation

- [Charts](#charts)
- [Overall Ranking](#overall-ranking)
- [Per-Dataset Breakdown](#per-dataset-breakdown)
  - Datasets:
    - [FullMergedDataset-allProfiles](#dataset-fullmergeddataset-allprofiles)
    - [QFN-3D-v1](#dataset-qfn-3d-v1)
    - [QFN-v1](#dataset-qfn-v1)
    - [REFFERENCE_DATA_Transistor](#dataset-refference-data-transistor)
    - [Resistor-3d-v1](#dataset-resistor-3d-v1)
    - [Resistor-v1](#dataset-resistor-v1)
    - [Transistor-3D-v1](#dataset-transistor-3d-v1)
    - [Transistor-v1](#dataset-transistor-v1)
- [Bundle Details](#bundle-details)
- [Per-Model Detail Cards](#per-model-detail-cards)
  - Models:
    - [DeepMindSmall-v2_20260216_171226.bundle](#model-deepmindsmall-v2-20260216-171226-bundle)
    - [RandomDataCrawler-v3.pt](#model-randomdatacrawler-v3-pt)
    - [Transistor-v1.pt](#model-transistor-v1-pt)
    - [Transistor-3D-v1.pt](#model-transistor-3d-v1-pt)
    - [Seqent-DatasetCrawler-v1_MultiTrained.pt](#model-seqent-datasetcrawler-v1-multitrained-pt)
    - [QFN-v1.pt](#model-qfn-v1-pt)
    - [Resistor-v1.pt](#model-resistor-v1-pt)
    - [Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt](#model-resistor-ensemble-multi-v1-20260215-121430-ensemble-pt)
    - [Resistor-3d-v1.pt](#model-resistor-3d-v1-pt)
    - [QFN-3D-v1.pt](#model-qfn-3d-v1-pt)
- [History](#history)

<a id="charts"></a>
## Charts

### Top Avg Accuracy

![Top Avg Accuracy](ARENA_REPORT_assets/top_avg_accuracy.svg)

### Top Avg F1

![Top Avg F1](ARENA_REPORT_assets/top_avg_f1.svg)

### Model Storage Breakdown

![Model Storage Breakdown](ARENA_REPORT_assets/model_size_pie.svg)

### Dataset Storage Breakdown

![Dataset Storage Breakdown](ARENA_REPORT_assets/dataset_size_pie.svg)

<a id="overall-ranking"></a>
## Overall Ranking

| Rank | Model | Type | Avg Accuracy | Avg F1 | Datasets Tested | Best Dataset | Worst Dataset | Last Run |
|---:|---|---|---:|---:|---:|---|---|---|
| 1 | DeepMindSmall-v2_20260216_171226.bundle | Bundle | 0.9950 | 0.9950 | 7 | REFFERENCE_DATA_Transistor (1.0000) | Resistor-3d-v1 (0.9775) | 2026-02-16 17:13:38 |
| 2 | RandomDataCrawler-v3.pt | Single | 0.8434 | 0.8100 | 8 | REFFERENCE_DATA_Transistor (1.0000) | QFN-3D-v1 (0.2500) | 2026-02-16 16:49:44 |
| 3 | Transistor-v1.pt | Single | 0.5825 | 0.5159 | 8 | Transistor-v1 (0.9938) | Resistor-3d-v1 (0.2525) | 2026-02-16 16:54:03 |
| 4 | Transistor-3D-v1.pt | Single | 0.5039 | 0.4415 | 8 | REFFERENCE_DATA_Transistor (1.0000) | Resistor-v1 (0.0975) | 2026-02-16 16:51:53 |
| 5 | Seqent-DatasetCrawler-v1_MultiTrained.pt | Single | 0.4247 | 0.3268 | 8 | QFN-v1 (1.0000) | Resistor-3d-v1 (0.2500) | 2026-02-16 16:52:24 |
| 6 | QFN-v1.pt | Single | 0.4222 | 0.3504 | 8 | QFN-v1 (1.0000) | REFFERENCE_DATA_Transistor (0.1650) | 2026-02-16 16:53:45 |
| 7 | Resistor-v1.pt | Single | 0.4076 | 0.2975 | 8 | Resistor-v1 (0.9950) | Resistor-3d-v1 (0.2500) | 2026-02-16 16:54:27 |
| 8 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | Single | 0.4040 | 0.3083 | 8 | Resistor-3d-v1 (0.9350) | QFN-v1 (0.2500) | 2026-02-16 16:52:48 |
| 9 | Resistor-3d-v1.pt | Single | 0.3755 | 0.2820 | 8 | Resistor-3d-v1 (0.9775) | REFFERENCE_DATA_Transistor (0.2500) | 2026-02-16 16:53:11 |
| 10 | QFN-3D-v1.pt | Single | 0.3573 | 0.2528 | 8 | QFN-3D-v1 (0.9975) | FullMergedDataset-allProfiles (0.2432) | 2026-02-16 16:50:27 |

<a id="per-dataset-breakdown"></a>
## Per-Dataset Breakdown

<a id="dataset-fullmergeddataset-allprofiles"></a>
### Dataset: FullMergedDataset-allProfiles
- Size on disk: 418.31 MB
- Total samples: 6800

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | RandomDataCrawler-v3.pt | 0.9956 | 0.9956 | all | 6800 | 418.31 MB | 2026-02-16 16:48:56 |
| 2 | Transistor-v1.pt | 0.6366 | 0.6190 | all | 6800 | 418.31 MB | 2026-02-16 16:54:03 |
| 3 | Transistor-3D-v1.pt | 0.5559 | 0.5483 | all | 6800 | 418.31 MB | 2026-02-16 16:51:53 |
| 4 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.3897 | 0.3405 | all | 6800 | 418.31 MB | 2026-02-16 16:52:24 |
| 5 | Resistor-3d-v1.pt | 0.3754 | 0.3270 | all | 6800 | 418.31 MB | 2026-02-16 16:53:11 |
| 6 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.3660 | 0.3037 | all | 6800 | 418.31 MB | 2026-02-16 16:52:48 |
| 7 | Resistor-v1.pt | 0.3659 | 0.3053 | all | 6800 | 418.31 MB | 2026-02-16 16:54:27 |
| 8 | QFN-v1.pt | 0.3428 | 0.3096 | all | 6800 | 418.31 MB | 2026-02-16 16:53:45 |
| 9 | QFN-3D-v1.pt | 0.2432 | 0.1648 | all | 6800 | 418.31 MB | 2026-02-16 16:50:27 |

<a id="dataset-qfn-3d-v1"></a>
### Dataset: QFN-3D-v1
- Size on disk: 616.25 MB
- Total samples: 800

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | DeepMindSmall-v2_20260216_171226.bundle | 0.9988 | 0.9987 | all | 800 | 616.25 MB | 2026-02-16 17:13:06 |
| 2 | QFN-3D-v1.pt | 0.9975 | 0.9975 | all | 800 | 616.25 MB | 2026-02-15 21:31:38 |
| 3 | Transistor-3D-v1.pt | 0.4888 | 0.3648 | all | 800 | 616.25 MB | 2026-02-15 21:32:52 |
| 4 | QFN-v1.pt | 0.4225 | 0.3077 | all | 800 | 616.25 MB | 2026-02-15 21:36:32 |
| 5 | Transistor-v1.pt | 0.3925 | 0.2747 | all | 800 | 616.25 MB | 2026-02-15 21:37:23 |
| 6 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.3800 | 0.2492 | all | 800 | 616.25 MB | 2026-02-15 21:33:47 |
| 7 | Resistor-3d-v1.pt | 0.3750 | 0.2539 | all | 800 | 616.25 MB | 2026-02-15 21:35:38 |
| 8 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2712 | 0.1498 | all | 800 | 616.25 MB | 2026-02-15 21:34:42 |
| 9 | RandomDataCrawler-v3.pt | 0.2500 | 0.1000 | all | 800 | 616.25 MB | 2026-02-16 16:49:11 |
| 10 | Resistor-v1.pt | 0.2500 | 0.1000 | all | 800 | 616.25 MB | 2026-02-15 21:38:13 |

<a id="dataset-qfn-v1"></a>
### Dataset: QFN-v1
- Size on disk: 664.60 MB
- Total samples: 800

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | DeepMindSmall-v2_20260216_171226.bundle | 1.0000 | 1.0000 | all | 800 | 664.60 MB | 2026-02-16 17:13:20 |
| 2 | QFN-v1.pt | 1.0000 | 1.0000 | all | 800 | 664.60 MB | 2026-02-15 21:36:46 |
| 3 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 1.0000 | 1.0000 | all | 800 | 664.60 MB | 2026-02-15 21:34:02 |
| 4 | RandomDataCrawler-v3.pt | 0.5400 | 0.4235 | all | 800 | 664.60 MB | 2026-02-16 16:49:26 |
| 5 | Transistor-v1.pt | 0.4975 | 0.3775 | all | 800 | 664.60 MB | 2026-02-15 21:37:37 |
| 6 | Resistor-3d-v1.pt | 0.2562 | 0.1483 | all | 800 | 664.60 MB | 2026-02-15 21:35:53 |
| 7 | Transistor-3D-v1.pt | 0.2562 | 0.1139 | all | 800 | 664.60 MB | 2026-02-15 21:33:07 |
| 8 | QFN-3D-v1.pt | 0.2525 | 0.1055 | all | 800 | 664.60 MB | 2026-02-15 21:31:52 |
| 9 | Resistor-v1.pt | 0.2525 | 0.1052 | all | 800 | 664.60 MB | 2026-02-15 21:38:27 |
| 10 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2500 | 0.1200 | all | 800 | 664.60 MB | 2026-02-15 21:34:59 |

<a id="dataset-refference-data-transistor"></a>
### Dataset: REFFERENCE_DATA_Transistor
- Size on disk: 83.80 MB
- Total samples: 800

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | DeepMindSmall-v2_20260216_171226.bundle | 1.0000 | 1.0000 | all | 800 | 83.80 MB | 2026-02-16 17:13:24 |
| 2 | RandomDataCrawler-v3.pt | 1.0000 | 1.0000 | all | 800 | 83.80 MB | 2026-02-16 16:49:30 |
| 3 | Transistor-3D-v1.pt | 1.0000 | 1.0000 | all | 800 | 83.80 MB | 2026-02-15 21:33:12 |
| 4 | Transistor-v1.pt | 0.6050 | 0.5410 | all | 800 | 83.80 MB | 2026-02-15 21:37:41 |
| 5 | Resistor-v1.pt | 0.4225 | 0.2982 | all | 800 | 83.80 MB | 2026-02-15 21:38:31 |
| 6 | QFN-3D-v1.pt | 0.3075 | 0.2264 | all | 800 | 83.80 MB | 2026-02-15 21:31:56 |
| 7 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2550 | 0.1317 | all | 800 | 83.80 MB | 2026-02-15 21:35:03 |
| 8 | Resistor-3d-v1.pt | 0.2500 | 0.1250 | all | 800 | 83.80 MB | 2026-02-15 21:35:57 |
| 9 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.2500 | 0.1000 | all | 800 | 83.80 MB | 2026-02-15 21:34:06 |
| 10 | QFN-v1.pt | 0.1650 | 0.1117 | all | 800 | 83.80 MB | 2026-02-15 21:36:50 |

<a id="dataset-resistor-3d-v1"></a>
### Dataset: Resistor-3d-v1
- Size on disk: 41.10 MB
- Total samples: 400

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | RandomDataCrawler-v3.pt | 0.9900 | 0.9900 | all | 400 | 41.10 MB | 2026-02-16 16:49:33 |
| 2 | DeepMindSmall-v2_20260216_171226.bundle | 0.9775 | 0.9776 | all | 400 | 41.10 MB | 2026-02-16 17:13:27 |
| 3 | Resistor-3d-v1.pt | 0.9775 | 0.9776 | all | 400 | 41.10 MB | 2026-02-15 21:36:00 |
| 4 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.9350 | 0.9347 | all | 400 | 41.10 MB | 2026-02-15 21:35:07 |
| 5 | Transistor-3D-v1.pt | 0.3825 | 0.2851 | all | 400 | 41.10 MB | 2026-02-15 21:33:15 |
| 6 | Transistor-v1.pt | 0.2525 | 0.1055 | all | 400 | 41.10 MB | 2026-02-15 21:37:45 |
| 7 | QFN-3D-v1.pt | 0.2500 | 0.1016 | all | 400 | 41.10 MB | 2026-02-15 21:32:00 |
| 8 | Resistor-v1.pt | 0.2500 | 0.1000 | all | 400 | 41.10 MB | 2026-02-15 21:38:35 |
| 9 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.2500 | 0.1000 | all | 400 | 41.10 MB | 2026-02-15 21:34:09 |
| 10 | QFN-v1.pt | 0.2050 | 0.0854 | all | 400 | 41.10 MB | 2026-02-15 21:36:54 |

<a id="dataset-resistor-v1"></a>
### Dataset: Resistor-v1
- Size on disk: 43.84 MB
- Total samples: 400

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | DeepMindSmall-v2_20260216_171226.bundle | 0.9950 | 0.9950 | all | 400 | 43.84 MB | 2026-02-16 17:13:30 |
| 2 | Resistor-v1.pt | 0.9950 | 0.9950 | all | 400 | 43.84 MB | 2026-02-15 21:38:38 |
| 3 | RandomDataCrawler-v3.pt | 0.9825 | 0.9824 | all | 400 | 43.84 MB | 2026-02-16 16:49:36 |
| 4 | Transistor-v1.pt | 0.6775 | 0.6750 | all | 400 | 43.84 MB | 2026-02-15 21:37:48 |
| 5 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.6450 | 0.5734 | all | 400 | 43.84 MB | 2026-02-15 21:35:10 |
| 6 | QFN-v1.pt | 0.5775 | 0.5128 | all | 400 | 43.84 MB | 2026-02-15 21:36:57 |
| 7 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.3900 | 0.3269 | all | 400 | 43.84 MB | 2026-02-15 21:34:12 |
| 8 | Resistor-3d-v1.pt | 0.2575 | 0.1416 | all | 400 | 43.84 MB | 2026-02-15 21:36:03 |
| 9 | QFN-3D-v1.pt | 0.2500 | 0.1000 | all | 400 | 43.84 MB | 2026-02-15 21:32:03 |
| 10 | Transistor-3D-v1.pt | 0.0975 | 0.1195 | all | 400 | 43.84 MB | 2026-02-15 21:33:18 |

<a id="dataset-transistor-3d-v1"></a>
### Dataset: Transistor-3D-v1
- Size on disk: 83.44 MB
- Total samples: 800

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | DeepMindSmall-v2_20260216_171226.bundle | 1.0000 | 1.0000 | all | 800 | 83.44 MB | 2026-02-16 17:13:34 |
| 2 | RandomDataCrawler-v3.pt | 1.0000 | 1.0000 | all | 800 | 83.44 MB | 2026-02-16 16:49:40 |
| 3 | Transistor-3D-v1.pt | 1.0000 | 1.0000 | all | 800 | 83.44 MB | 2026-02-15 21:33:22 |
| 4 | Transistor-v1.pt | 0.6050 | 0.5410 | all | 800 | 83.44 MB | 2026-02-15 21:37:52 |
| 5 | Resistor-v1.pt | 0.4425 | 0.3140 | all | 800 | 83.44 MB | 2026-02-15 21:38:42 |
| 6 | QFN-3D-v1.pt | 0.3075 | 0.2264 | all | 800 | 83.44 MB | 2026-02-15 21:32:07 |
| 7 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2587 | 0.1347 | all | 800 | 83.44 MB | 2026-02-15 21:35:14 |
| 8 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.2512 | 0.1026 | all | 800 | 83.44 MB | 2026-02-15 21:34:16 |
| 9 | Resistor-3d-v1.pt | 0.2500 | 0.1250 | all | 800 | 83.44 MB | 2026-02-15 21:36:07 |
| 10 | QFN-v1.pt | 0.1650 | 0.1117 | all | 800 | 83.44 MB | 2026-02-15 21:37:01 |

<a id="dataset-transistor-v1"></a>
### Dataset: Transistor-v1
- Size on disk: 177.10 MB
- Total samples: 800

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | DeepMindSmall-v2_20260216_171226.bundle | 0.9938 | 0.9937 | all | 800 | 177.10 MB | 2026-02-16 17:13:38 |
| 2 | Transistor-v1.pt | 0.9938 | 0.9937 | all | 800 | 177.10 MB | 2026-02-15 21:37:56 |
| 3 | RandomDataCrawler-v3.pt | 0.9888 | 0.9887 | all | 800 | 177.10 MB | 2026-02-16 16:49:44 |
| 4 | QFN-v1.pt | 0.5000 | 0.3644 | all | 800 | 177.10 MB | 2026-02-15 21:37:05 |
| 5 | Seqent-DatasetCrawler-v1_MultiTrained.pt | 0.4863 | 0.3955 | all | 800 | 177.10 MB | 2026-02-15 21:34:20 |
| 6 | Resistor-v1.pt | 0.2825 | 0.1624 | all | 800 | 177.10 MB | 2026-02-15 21:38:46 |
| 7 | Resistor-3d-v1.pt | 0.2625 | 0.1581 | all | 800 | 177.10 MB | 2026-02-15 21:36:11 |
| 8 | Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt | 0.2512 | 0.1187 | all | 800 | 177.10 MB | 2026-02-15 21:35:19 |
| 9 | Transistor-3D-v1.pt | 0.2500 | 0.1000 | all | 800 | 177.10 MB | 2026-02-15 21:33:27 |
| 10 | QFN-3D-v1.pt | 0.2500 | 0.1000 | all | 800 | 177.10 MB | 2026-02-15 21:32:11 |

<a id="bundle-details"></a>
## Bundle Details

### DeepMindSmall-v2_20260216_171226.bundle
- **Profiles:** chip_0603_resistor@1, chip_0603_resistor_3d@1, qfn32_ic@1, qfn32_ic_3d@1, sot23_transistor@1, sot23_transistor_3d@1
- **Checkpoints:** 6
- **Created:** 2026-02-16T16:12:26Z

<a id="per-model-detail-cards"></a>
## Per-Model Detail Cards

<a id="model-deepmindsmall-v2-20260216-171226-bundle"></a>
### DeepMindSmall-v2_20260216_171226.bundle (Bundle)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| QFN-3D-v1 | all | 0.9988 | 0.9987 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 616.25 MB | 4 | 1.0000 | 75.5 img/s |
| QFN-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 664.60 MB | 12 | 1.0000 | 70.2 img/s |
| REFFERENCE_DATA_Transistor | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 83.80 MB | 11 | 1.0000 | 607.4 img/s |
| Resistor-3d-v1 | all | 0.9775 | 0.9776 | MISALIGNED:0.0800 MISSING:0.0000 TOMBSTONE:0.0100 | 400 | 41.10 MB | 9 | 0.9667 | 561.7 img/s |
| Resistor-v1 | all | 0.9950 | 0.9950 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 43.84 MB | 8 | 0.9833 | 475.9 img/s |
| Transistor-3D-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 83.44 MB | 11 | 1.0000 | 600.6 img/s |
| Transistor-v1 | all | 0.9938 | 0.9937 | MISALIGNED:0.0250 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 177.10 MB | 6 | 0.9917 | 527.6 img/s |

<a id="model-randomdatacrawler-v3-pt"></a>
### RandomDataCrawler-v3.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| FullMergedDataset-allProfiles | all | 0.9956 | 0.9956 | MISALIGNED:0.0129 MISSING:0.0000 TOMBSTONE:0.0000 | 6800 | 418.31 MB | 14 | 0.9853 | 656.1 img/s |
| QFN-3D-v1 | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 616.25 MB | 14 | 0.9853 | 63.5 img/s |
| QFN-v1 | all | 0.5400 | 0.4235 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.9200 | 800 | 664.60 MB | 14 | 0.9853 | 67.9 img/s |
| REFFERENCE_DATA_Transistor | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 83.80 MB | 14 | 0.9853 | 493.7 img/s |
| Resistor-3d-v1 | all | 0.9900 | 0.9900 | MISALIGNED:0.0400 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 41.10 MB | 14 | 0.9853 | 550.4 img/s |
| Resistor-v1 | all | 0.9825 | 0.9824 | MISALIGNED:0.0700 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 43.84 MB | 14 | 0.9853 | 430.3 img/s |
| Transistor-3D-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 83.44 MB | 14 | 0.9853 | 589.9 img/s |
| Transistor-v1 | all | 0.9888 | 0.9887 | MISALIGNED:0.0150 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 177.10 MB | 14 | 0.9853 | 488.4 img/s |

<a id="model-transistor-v1-pt"></a>
### Transistor-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| FullMergedDataset-allProfiles | all | 0.6366 | 0.6190 | MISALIGNED:0.2800 MISSING:0.1976 TOMBSTONE:0.2741 | 6800 | 418.31 MB | 6 | 0.9917 | 585.7 img/s |
| QFN-3D-v1 | all | 0.3925 | 0.2747 | MISALIGNED:0.0500 MISSING:0.3800 TOMBSTONE:1.0000 | 800 | 616.25 MB | 6 | 0.9917 | 77.4 img/s |
| QFN-v1 | all | 0.4975 | 0.3775 | MISALIGNED:0.0600 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 664.60 MB | 6 | 0.9917 | 66.9 img/s |
| REFFERENCE_DATA_Transistor | all | 0.6050 | 0.5410 | MISALIGNED:0.5550 MISSING:0.0250 TOMBSTONE:0.0000 | 800 | 83.80 MB | 6 | 0.9917 | 575.6 img/s |
| Resistor-3d-v1 | all | 0.2525 | 0.1055 | MISALIGNED:1.0000 MISSING:0.0000 TOMBSTONE:0.9900 | 400 | 41.10 MB | 6 | 0.9917 | 548.5 img/s |
| Resistor-v1 | all | 0.6775 | 0.6750 | MISALIGNED:0.2900 MISSING:0.6100 TOMBSTONE:0.0000 | 400 | 43.84 MB | 6 | 0.9917 | 473.6 img/s |
| Transistor-3D-v1 | all | 0.6050 | 0.5410 | MISALIGNED:0.5550 MISSING:0.0250 TOMBSTONE:0.0000 | 800 | 83.44 MB | 6 | 0.9917 | 584.1 img/s |
| Transistor-v1 | all | 0.9938 | 0.9937 | MISALIGNED:0.0250 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 177.10 MB | 6 | 0.9917 | 515.7 img/s |

<a id="model-transistor-3d-v1-pt"></a>
### Transistor-3D-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| FullMergedDataset-allProfiles | all | 0.5559 | 0.5483 | MISALIGNED:0.4112 MISSING:0.4294 TOMBSTONE:0.1712 | 6800 | 418.31 MB | 11 | 1.0000 | 661.5 img/s |
| QFN-3D-v1 | all | 0.4888 | 0.3648 | MISALIGNED:0.0450 MISSING:1.0000 TOMBSTONE:0.0000 | 800 | 616.25 MB | 11 | 1.0000 | 67.0 img/s |
| QFN-v1 | all | 0.2562 | 0.1139 | MISALIGNED:1.0000 MISSING:0.9750 TOMBSTONE:0.0000 | 800 | 664.60 MB | 11 | 1.0000 | 63.1 img/s |
| REFFERENCE_DATA_Transistor | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 83.80 MB | 11 | 1.0000 | 536.9 img/s |
| Resistor-3d-v1 | all | 0.3825 | 0.2851 | MISALIGNED:0.0000 MISSING:0.4700 TOMBSTONE:1.0000 | 400 | 41.10 MB | 11 | 1.0000 | 549.6 img/s |
| Resistor-v1 | all | 0.0975 | 0.1195 | MISALIGNED:0.9900 MISSING:0.7100 TOMBSTONE:0.9100 | 400 | 43.84 MB | 11 | 1.0000 | 469.6 img/s |
| Transistor-3D-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 83.44 MB | 11 | 1.0000 | 540.1 img/s |
| Transistor-v1 | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:0.0000 | 800 | 177.10 MB | 11 | 1.0000 | 499.4 img/s |

<a id="model-seqent-datasetcrawler-v1-multitrained-pt"></a>
### Seqent-DatasetCrawler-v1_MultiTrained.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| FullMergedDataset-allProfiles | all | 0.3897 | 0.3405 | MISALIGNED:0.8588 MISSING:0.6153 TOMBSTONE:0.8894 | 6800 | 418.31 MB | 9 | 1.0000 | 652.9 img/s |
| QFN-3D-v1 | all | 0.3800 | 0.2492 | MISALIGNED:0.4450 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 616.25 MB | 9 | 1.0000 | 67.9 img/s |
| QFN-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 664.60 MB | 9 | 1.0000 | 67.1 img/s |
| REFFERENCE_DATA_Transistor | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 83.80 MB | 9 | 1.0000 | 580.8 img/s |
| Resistor-3d-v1 | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 41.10 MB | 9 | 1.0000 | 539.8 img/s |
| Resistor-v1 | all | 0.3900 | 0.3269 | MISALIGNED:0.8700 MISSING:0.0000 TOMBSTONE:0.8700 | 400 | 43.84 MB | 9 | 1.0000 | 478.0 img/s |
| Transistor-3D-v1 | all | 0.2512 | 0.1026 | MISALIGNED:0.9950 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 83.44 MB | 9 | 1.0000 | 599.3 img/s |
| Transistor-v1 | all | 0.4863 | 0.3955 | MISALIGNED:0.7600 MISSING:0.1200 TOMBSTONE:1.0000 | 800 | 177.10 MB | 9 | 1.0000 | 503.9 img/s |

<a id="model-qfn-v1-pt"></a>
### QFN-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| FullMergedDataset-allProfiles | all | 0.3428 | 0.3096 | MISALIGNED:0.8765 MISSING:0.3606 TOMBSTONE:0.8441 | 6800 | 418.31 MB | 12 | 1.0000 | 661.1 img/s |
| QFN-3D-v1 | all | 0.4225 | 0.3077 | MISALIGNED:0.2450 MISSING:0.9350 TOMBSTONE:1.0000 | 800 | 616.25 MB | 12 | 1.0000 | 74.1 img/s |
| QFN-v1 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 664.60 MB | 12 | 1.0000 | 68.2 img/s |
| REFFERENCE_DATA_Transistor | all | 0.1650 | 0.1117 | MISALIGNED:1.0000 MISSING:0.6450 TOMBSTONE:1.0000 | 800 | 83.80 MB | 12 | 1.0000 | 578.2 img/s |
| Resistor-3d-v1 | all | 0.2050 | 0.0854 | MISALIGNED:1.0000 MISSING:0.1800 TOMBSTONE:1.0000 | 400 | 41.10 MB | 12 | 1.0000 | 511.3 img/s |
| Resistor-v1 | all | 0.5775 | 0.5128 | MISALIGNED:0.3500 MISSING:0.0000 TOMBSTONE:0.3400 | 400 | 43.84 MB | 12 | 1.0000 | 480.9 img/s |
| Transistor-3D-v1 | all | 0.1650 | 0.1117 | MISALIGNED:1.0000 MISSING:0.6450 TOMBSTONE:1.0000 | 800 | 83.44 MB | 12 | 1.0000 | 583.2 img/s |
| Transistor-v1 | all | 0.5000 | 0.3644 | MISALIGNED:0.9150 MISSING:0.0500 TOMBSTONE:1.0000 | 800 | 177.10 MB | 12 | 1.0000 | 502.8 img/s |

<a id="model-resistor-v1-pt"></a>
### Resistor-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| FullMergedDataset-allProfiles | all | 0.3659 | 0.3053 | MISALIGNED:0.0324 MISSING:0.8712 TOMBSTONE:0.9159 | 6800 | 418.31 MB | 8 | 0.9833 | 588.9 img/s |
| QFN-3D-v1 | all | 0.2500 | 0.1000 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 616.25 MB | 8 | 0.9833 | 75.3 img/s |
| QFN-v1 | all | 0.2525 | 0.1052 | MISALIGNED:0.0000 MISSING:0.9900 TOMBSTONE:1.0000 | 800 | 664.60 MB | 8 | 0.9833 | 68.3 img/s |
| REFFERENCE_DATA_Transistor | all | 0.4225 | 0.2982 | MISALIGNED:0.1050 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 83.80 MB | 8 | 0.9833 | 579.8 img/s |
| Resistor-3d-v1 | all | 0.2500 | 0.1000 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 41.10 MB | 8 | 0.9833 | 541.8 img/s |
| Resistor-v1 | all | 0.9950 | 0.9950 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 43.84 MB | 8 | 0.9833 | 478.1 img/s |
| Transistor-3D-v1 | all | 0.4425 | 0.3140 | MISALIGNED:0.0600 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 83.44 MB | 8 | 0.9833 | 598.0 img/s |
| Transistor-v1 | all | 0.2825 | 0.1624 | MISALIGNED:0.0000 MISSING:0.9850 TOMBSTONE:0.8900 | 800 | 177.10 MB | 8 | 0.9833 | 512.1 img/s |

<a id="model-resistor-ensemble-multi-v1-20260215-121430-ensemble-pt"></a>
### Resistor-ensemble-multi-v1_20260215_121430_ensemble.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| FullMergedDataset-allProfiles | all | 0.3660 | 0.3037 | MISALIGNED:0.0035 MISSING:0.7694 TOMBSTONE:0.9165 | 6800 | 418.31 MB | - | - | 466.1 img/s |
| QFN-3D-v1 | all | 0.2712 | 0.1498 | MISALIGNED:0.0100 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 616.25 MB | - | - | 62.5 img/s |
| QFN-v1 | all | 0.2500 | 0.1200 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 664.60 MB | - | - | 55.9 img/s |
| REFFERENCE_DATA_Transistor | all | 0.2550 | 0.1317 | MISALIGNED:0.0000 MISSING:0.9950 TOMBSTONE:1.0000 | 800 | 83.80 MB | - | - | 497.5 img/s |
| Resistor-3d-v1 | all | 0.9350 | 0.9347 | MISALIGNED:0.0200 MISSING:0.0000 TOMBSTONE:0.2100 | 400 | 41.10 MB | - | - | 446.1 img/s |
| Resistor-v1 | all | 0.6450 | 0.5734 | MISALIGNED:0.0200 MISSING:0.0300 TOMBSTONE:1.0000 | 400 | 43.84 MB | - | - | 408.9 img/s |
| Transistor-3D-v1 | all | 0.2587 | 0.1347 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 83.44 MB | - | - | 505.3 img/s |
| Transistor-v1 | all | 0.2512 | 0.1187 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 177.10 MB | - | - | 438.6 img/s |

<a id="model-resistor-3d-v1-pt"></a>
### Resistor-3d-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| FullMergedDataset-allProfiles | all | 0.3754 | 0.3270 | MISALIGNED:0.0388 MISSING:0.7429 TOMBSTONE:0.8865 | 6800 | 418.31 MB | 9 | 0.9667 | 577.5 img/s |
| QFN-3D-v1 | all | 0.3750 | 0.2539 | MISALIGNED:0.5000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 616.25 MB | 9 | 0.9667 | 75.9 img/s |
| QFN-v1 | all | 0.2562 | 0.1483 | MISALIGNED:0.0950 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 664.60 MB | 9 | 0.9667 | 67.3 img/s |
| REFFERENCE_DATA_Transistor | all | 0.2500 | 0.1250 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 83.80 MB | 9 | 0.9667 | 582.3 img/s |
| Resistor-3d-v1 | all | 0.9775 | 0.9776 | MISALIGNED:0.0800 MISSING:0.0000 TOMBSTONE:0.0100 | 400 | 41.10 MB | 9 | 0.9667 | 525.2 img/s |
| Resistor-v1 | all | 0.2575 | 0.1416 | MISALIGNED:0.0300 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 43.84 MB | 9 | 0.9667 | 468.0 img/s |
| Transistor-3D-v1 | all | 0.2500 | 0.1250 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 83.44 MB | 9 | 0.9667 | 582.8 img/s |
| Transistor-v1 | all | 0.2625 | 0.1581 | MISALIGNED:0.1300 MISSING:1.0000 TOMBSTONE:0.9950 | 800 | 177.10 MB | 9 | 0.9667 | 497.3 img/s |

<a id="model-qfn-3d-v1-pt"></a>
### QFN-3D-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| FullMergedDataset-allProfiles | all | 0.2432 | 0.1648 | MISALIGNED:0.4047 MISSING:0.9912 TOMBSTONE:0.6329 | 6800 | 418.31 MB | 4 | 1.0000 | 645.1 img/s |
| QFN-3D-v1 | all | 0.9975 | 0.9975 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 616.25 MB | 7 | 1.0000 | 72.0 img/s |
| QFN-v1 | all | 0.2525 | 0.1055 | MISALIGNED:1.0000 MISSING:0.9900 TOMBSTONE:0.0000 | 800 | 664.60 MB | 7 | 1.0000 | 66.1 img/s |
| REFFERENCE_DATA_Transistor | all | 0.3075 | 0.2264 | MISALIGNED:0.0050 MISSING:1.0000 TOMBSTONE:0.7650 | 800 | 83.80 MB | 7 | 1.0000 | 573.8 img/s |
| Resistor-3d-v1 | all | 0.2500 | 0.1016 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 41.10 MB | 7 | 1.0000 | 508.7 img/s |
| Resistor-v1 | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:0.0000 | 400 | 43.84 MB | 7 | 1.0000 | 473.6 img/s |
| Transistor-3D-v1 | all | 0.3075 | 0.2264 | MISALIGNED:0.0050 MISSING:1.0000 TOMBSTONE:0.7650 | 800 | 83.44 MB | 7 | 1.0000 | 557.3 img/s |
| Transistor-v1 | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:0.0000 | 800 | 177.10 MB | 7 | 1.0000 | 503.6 img/s |

<a id="history"></a>
## History

- 2026-02-16: Report generated (10 models, 8 datasets)
