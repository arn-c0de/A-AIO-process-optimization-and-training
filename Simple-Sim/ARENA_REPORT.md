# Model Arena Report
> Last updated: 2026-02-20 11:04:25

## Quick Navigation

- [Charts](#charts)
- [Overall Ranking](#overall-ranking)
- [Per-Dataset Breakdown](#per-dataset-breakdown)
- [Bundle Details](#bundle-details)
- [Per-Model Detail Cards](#per-model-detail-cards)
- [History](#history)

<details>
<summary>Datasets</summary>

- [IC-16-3D](#dataset-ic-16-3d)
- [QFN-3D](#dataset-qfn-3d)
- [QFN32-2D](#dataset-qfn32-2d)
- [Resistor-2D](#dataset-resistor-2d)
- [Resistor-3D](#dataset-resistor-3d)
- [Transistor-2D](#dataset-transistor-2d)
- [Transistor-3D](#dataset-transistor-3d)

</details>

<details>
<summary>Models</summary>

- [RandomCrawler-5X.pt](#model-randomcrawler-5x-pt)
- [RandomCrawler-1X_MultiTrained.pt](#model-randomcrawler-1x-multitrained-pt)
- [Transistor-2D.pt](#model-transistor-2d-pt)
- [IC-16-3D.pt](#model-ic-16-3d-pt)
- [Transistor-3D.pt](#model-transistor-3d-pt)
- [Resistor-2D.pt](#model-resistor-2d-pt)
- [QFN32-2D.pt](#model-qfn32-2d-pt)
- [Resistor-3D.pt](#model-resistor-3d-pt)
- [QFN-3D.pt](#model-qfn-3d-pt)

</details>


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
| 1 | RandomCrawler-5X.pt | Single | 0.7170 | 0.6744 | 7 | Transistor-2D (0.9925) | IC-16-3D (0.2900) | 2026-02-16 22:56:29 |
| 2 | RandomCrawler-1X_MultiTrained.pt | Single | 0.7154 | 0.6616 | 7 | Resistor-2D (0.9975) | QFN-3D (0.2500) | 2026-02-20 11:03:43 |
| 3 | Transistor-2D.pt | Single | 0.5891 | 0.5090 | 7 | Transistor-2D (0.9850) | Resistor-3D (0.2500) | 2026-02-16 22:35:03 |
| 4 | IC-16-3D.pt | Single | 0.4888 | 0.4095 | 7 | IC-16-3D (0.9875) | Transistor-2D (0.2712) | 2026-02-16 22:41:03 |
| 5 | Transistor-3D.pt | Single | 0.4561 | 0.3526 | 7 | Transistor-3D (0.9838) | Resistor-3D (0.2500) | 2026-02-16 22:33:53 |
| 6 | Resistor-2D.pt | Single | 0.4330 | 0.3398 | 7 | Resistor-2D (0.9825) | Resistor-3D (0.2500) | 2026-02-16 22:37:14 |
| 7 | QFN32-2D.pt | Single | 0.4280 | 0.3156 | 7 | QFN32-2D (1.0000) | Resistor-3D (0.2500) | 2026-02-16 22:38:47 |
| 8 | Resistor-3D.pt | Single | 0.4030 | 0.3123 | 7 | Resistor-3D (0.9800) | QFN-3D (0.2087) | 2026-02-16 22:36:08 |
| 9 | QFN-3D.pt | Single | 0.3968 | 0.3182 | 7 | QFN-3D (0.9875) | Resistor-2D (0.2350) | 2026-02-16 22:39:57 |

<a id="per-dataset-breakdown"></a>
## Per-Dataset Breakdown

<a id="dataset-ic-16-3d"></a>
### Dataset: IC-16-3D
- Size on disk: 1.04 GB
- Total samples: 800

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | IC-16-3D.pt | 0.9875 | 0.9875 | all | 800 | 1.04 GB | 2026-02-16 22:40:21 |
| 2 | Transistor-3D.pt | 0.5112 | 0.4004 | all | 800 | 1.04 GB | 2026-02-16 22:33:11 |
| 3 | RandomCrawler-1X_MultiTrained.pt | 0.4975 | 0.3734 | all | 800 | 1.04 GB | 2026-02-16 22:31:46 |
| 4 | QFN-3D.pt | 0.4938 | 0.3725 | all | 800 | 1.04 GB | 2026-02-16 22:39:12 |
| 5 | Transistor-2D.pt | 0.4462 | 0.3385 | all | 800 | 1.04 GB | 2026-02-16 22:34:19 |
| 6 | QFN32-2D.pt | 0.3013 | 0.1883 | all | 800 | 1.04 GB | 2026-02-16 22:38:03 |
| 7 | RandomCrawler-5X.pt | 0.2900 | 0.1775 | all | 800 | 1.04 GB | 2026-02-16 22:55:45 |
| 8 | Resistor-2D.pt | 0.2500 | 0.1000 | all | 800 | 1.04 GB | 2026-02-16 22:36:31 |
| 9 | Resistor-3D.pt | 0.2375 | 0.1386 | all | 800 | 1.04 GB | 2026-02-16 22:35:25 |

<a id="dataset-qfn-3d"></a>
### Dataset: QFN-3D
- Size on disk: 614.92 MB
- Total samples: 800

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | QFN-3D.pt | 0.9875 | 0.9875 | all | 800 | 614.92 MB | 2026-02-16 22:39:28 |
| 2 | IC-16-3D.pt | 0.7250 | 0.6648 | all | 800 | 614.92 MB | 2026-02-16 22:40:34 |
| 3 | Transistor-3D.pt | 0.6500 | 0.5784 | all | 800 | 614.92 MB | 2026-02-16 22:33:25 |
| 4 | RandomCrawler-5X.pt | 0.4400 | 0.3345 | all | 800 | 614.92 MB | 2026-02-16 22:56:00 |
| 5 | Transistor-2D.pt | 0.3925 | 0.2671 | all | 800 | 614.92 MB | 2026-02-16 22:34:33 |
| 6 | RandomCrawler-1X_MultiTrained.pt | 0.2500 | 0.1000 | all | 800 | 614.92 MB | 2026-02-20 11:03:43 |
| 7 | QFN32-2D.pt | 0.2500 | 0.1000 | all | 800 | 614.92 MB | 2026-02-16 22:38:17 |
| 8 | Resistor-2D.pt | 0.2500 | 0.1000 | all | 800 | 614.92 MB | 2026-02-16 22:36:45 |
| 9 | Resistor-3D.pt | 0.2087 | 0.1557 | all | 800 | 614.92 MB | 2026-02-16 22:35:39 |

<a id="dataset-qfn32-2d"></a>
### Dataset: QFN32-2D
- Size on disk: 1.29 GB
- Total samples: 800

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | QFN32-2D.pt | 1.0000 | 1.0000 | all | 800 | 1.29 GB | 2026-02-16 22:38:32 |
| 2 | Transistor-2D.pt | 0.5487 | 0.4744 | all | 800 | 1.29 GB | 2026-02-16 22:34:48 |
| 3 | Resistor-2D.pt | 0.4150 | 0.3548 | all | 800 | 1.29 GB | 2026-02-16 22:36:59 |
| 4 | Resistor-3D.pt | 0.3800 | 0.2710 | all | 800 | 1.29 GB | 2026-02-16 22:35:53 |
| 5 | RandomCrawler-5X.pt | 0.3438 | 0.2640 | all | 800 | 1.29 GB | 2026-02-16 22:56:14 |
| 6 | RandomCrawler-1X_MultiTrained.pt | 0.3075 | 0.2054 | all | 800 | 1.29 GB | 2026-02-20 11:01:27 |
| 7 | IC-16-3D.pt | 0.2800 | 0.1605 | all | 800 | 1.29 GB | 2026-02-16 22:40:49 |
| 8 | QFN-3D.pt | 0.2650 | 0.1392 | all | 800 | 1.29 GB | 2026-02-16 22:39:42 |
| 9 | Transistor-3D.pt | 0.2525 | 0.1186 | all | 800 | 1.29 GB | 2026-02-16 22:33:39 |

<a id="dataset-resistor-2d"></a>
### Dataset: Resistor-2D
- Size on disk: 114.79 MB
- Total samples: 400

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | RandomCrawler-1X_MultiTrained.pt | 0.9975 | 0.9975 | all | 400 | 114.79 MB | 2026-02-20 11:01:30 |
| 2 | RandomCrawler-5X.pt | 0.9900 | 0.9900 | all | 400 | 114.79 MB | 2026-02-16 22:56:18 |
| 3 | Resistor-2D.pt | 0.9825 | 0.9825 | all | 400 | 114.79 MB | 2026-02-16 22:37:03 |
| 4 | Transistor-2D.pt | 0.9650 | 0.9650 | all | 400 | 114.79 MB | 2026-02-16 22:34:52 |
| 5 | QFN32-2D.pt | 0.4775 | 0.3218 | all | 400 | 114.79 MB | 2026-02-16 22:38:35 |
| 6 | IC-16-3D.pt | 0.2950 | 0.2102 | all | 400 | 114.79 MB | 2026-02-16 22:40:52 |
| 7 | Transistor-3D.pt | 0.2800 | 0.1566 | all | 400 | 114.79 MB | 2026-02-16 22:33:42 |
| 8 | Resistor-3D.pt | 0.2575 | 0.1350 | all | 400 | 114.79 MB | 2026-02-16 22:35:56 |
| 9 | QFN-3D.pt | 0.2350 | 0.2379 | all | 400 | 114.79 MB | 2026-02-16 22:39:45 |

<a id="dataset-resistor-3d"></a>
### Dataset: Resistor-3D
- Size on disk: 56.05 MB
- Total samples: 400

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | Resistor-3D.pt | 0.9800 | 0.9800 | all | 400 | 56.05 MB | 2026-02-16 22:36:00 |
| 2 | RandomCrawler-1X_MultiTrained.pt | 0.9775 | 0.9775 | all | 400 | 56.05 MB | 2026-02-20 11:01:34 |
| 3 | RandomCrawler-5X.pt | 0.9725 | 0.9724 | all | 400 | 56.05 MB | 2026-02-16 22:56:21 |
| 4 | IC-16-3D.pt | 0.2850 | 0.1575 | all | 400 | 56.05 MB | 2026-02-16 22:40:55 |
| 5 | QFN-3D.pt | 0.2600 | 0.1221 | all | 400 | 56.05 MB | 2026-02-16 22:39:49 |
| 6 | QFN32-2D.pt | 0.2500 | 0.1139 | all | 400 | 56.05 MB | 2026-02-16 22:38:39 |
| 7 | Transistor-3D.pt | 0.2500 | 0.1012 | all | 400 | 56.05 MB | 2026-02-16 22:33:45 |
| 8 | Resistor-2D.pt | 0.2500 | 0.1000 | all | 400 | 56.05 MB | 2026-02-16 22:37:06 |
| 9 | Transistor-2D.pt | 0.2500 | 0.1000 | all | 400 | 56.05 MB | 2026-02-16 22:34:55 |

<a id="dataset-transistor-2d"></a>
### Dataset: Transistor-2D
- Size on disk: 245.13 MB
- Total samples: 800

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | RandomCrawler-5X.pt | 0.9925 | 0.9925 | all | 800 | 245.13 MB | 2026-02-16 22:56:25 |
| 2 | RandomCrawler-1X_MultiTrained.pt | 0.9900 | 0.9900 | all | 800 | 245.13 MB | 2026-02-20 11:01:38 |
| 3 | Transistor-2D.pt | 0.9850 | 0.9850 | all | 800 | 245.13 MB | 2026-02-16 22:34:59 |
| 4 | Resistor-2D.pt | 0.6338 | 0.6416 | all | 800 | 245.13 MB | 2026-02-16 22:37:10 |
| 5 | QFN32-2D.pt | 0.4550 | 0.3084 | all | 800 | 245.13 MB | 2026-02-16 22:38:43 |
| 6 | IC-16-3D.pt | 0.2712 | 0.1725 | all | 800 | 245.13 MB | 2026-02-16 22:40:59 |
| 7 | Transistor-3D.pt | 0.2650 | 0.1295 | all | 800 | 245.13 MB | 2026-02-16 22:33:49 |
| 8 | Resistor-3D.pt | 0.2575 | 0.1263 | all | 800 | 245.13 MB | 2026-02-16 22:36:04 |
| 9 | QFN-3D.pt | 0.2437 | 0.1892 | all | 800 | 245.13 MB | 2026-02-16 22:39:53 |

<a id="dataset-transistor-3d"></a>
### Dataset: Transistor-3D
- Size on disk: 79.92 MB
- Total samples: 800

| Rank | Model | Accuracy | F1 | Split | Samples | Dataset Size | Last Run |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | RandomCrawler-5X.pt | 0.9900 | 0.9900 | all | 800 | 79.92 MB | 2026-02-16 22:56:29 |
| 2 | RandomCrawler-1X_MultiTrained.pt | 0.9875 | 0.9875 | all | 800 | 79.92 MB | 2026-02-20 11:01:43 |
| 3 | Transistor-3D.pt | 0.9838 | 0.9837 | all | 800 | 79.92 MB | 2026-02-16 22:33:53 |
| 4 | IC-16-3D.pt | 0.5775 | 0.5134 | all | 800 | 79.92 MB | 2026-02-16 22:41:03 |
| 5 | Transistor-2D.pt | 0.5363 | 0.4327 | all | 800 | 79.92 MB | 2026-02-16 22:35:03 |
| 6 | Resistor-3D.pt | 0.5000 | 0.3798 | all | 800 | 79.92 MB | 2026-02-16 22:36:08 |
| 7 | QFN-3D.pt | 0.2925 | 0.1791 | all | 800 | 79.92 MB | 2026-02-16 22:39:57 |
| 8 | QFN32-2D.pt | 0.2625 | 0.1769 | all | 800 | 79.92 MB | 2026-02-16 22:38:47 |
| 9 | Resistor-2D.pt | 0.2500 | 0.1000 | all | 800 | 79.92 MB | 2026-02-16 22:37:14 |

<a id="bundle-details"></a>
## Bundle Details

- (No bundle models tracked)

<a id="per-model-detail-cards"></a>
## Per-Model Detail Cards

<a id="model-randomcrawler-5x-pt"></a>
### RandomCrawler-5X.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| IC-16-3D | all | 0.2900 | 0.1775 | MISALIGNED:0.0000 MISSING:0.8400 TOMBSTONE:1.0000 | 800 | 1.04 GB | 13 | 0.9804 | 51.6 img/s |
| QFN-3D | all | 0.4400 | 0.3345 | MISALIGNED:0.0000 MISSING:0.2400 TOMBSTONE:1.0000 | 800 | 614.92 MB | 13 | 0.9804 | 66.9 img/s |
| QFN32-2D | all | 0.3438 | 0.2640 | MISALIGNED:0.0550 MISSING:0.6100 TOMBSTONE:1.0000 | 800 | 1.29 GB | 13 | 0.9804 | 66.0 img/s |
| Resistor-2D | all | 0.9900 | 0.9900 | MISALIGNED:0.0300 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 114.79 MB | 13 | 0.9804 | 473.1 img/s |
| Resistor-3D | all | 0.9725 | 0.9724 | MISALIGNED:0.1100 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 56.05 MB | 13 | 0.9804 | 553.0 img/s |
| Transistor-2D | all | 0.9925 | 0.9925 | MISALIGNED:0.0250 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 245.13 MB | 13 | 0.9804 | 528.4 img/s |
| Transistor-3D | all | 0.9900 | 0.9900 | MISALIGNED:0.0100 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 79.92 MB | 13 | 0.9804 | 600.0 img/s |

<a id="model-randomcrawler-1x-multitrained-pt"></a>
### RandomCrawler-1X_MultiTrained.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| IC-16-3D | all | 0.4975 | 0.3734 | MISALIGNED:0.0000 MISSING:0.0100 TOMBSTONE:1.0000 | 800 | 1.04 GB | 12 | 0.9750 | 52.6 img/s |
| QFN-3D | all | 0.2500 | 0.1000 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 614.92 MB | 12 | 0.9750 | 77.4 img/s |
| QFN32-2D | all | 0.3075 | 0.2054 | MISALIGNED:0.0000 MISSING:0.8250 TOMBSTONE:0.9450 | 800 | 1.29 GB | 12 | 0.9750 | 62.6 img/s |
| Resistor-2D | all | 0.9975 | 0.9975 | MISALIGNED:0.0100 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 114.79 MB | 12 | 0.9750 | 440.3 img/s |
| Resistor-3D | all | 0.9775 | 0.9775 | MISALIGNED:0.0700 MISSING:0.0000 TOMBSTONE:0.0200 | 400 | 56.05 MB | 12 | 0.9750 | 457.7 img/s |
| Transistor-2D | all | 0.9900 | 0.9900 | MISALIGNED:0.0350 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 245.13 MB | 12 | 0.9750 | 432.2 img/s |
| Transistor-3D | all | 0.9875 | 0.9875 | MISALIGNED:0.0400 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 79.92 MB | 12 | 0.9750 | 461.9 img/s |

<a id="model-transistor-2d-pt"></a>
### Transistor-2D.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| IC-16-3D | all | 0.4462 | 0.3385 | MISALIGNED:0.0000 MISSING:0.2150 TOMBSTONE:1.0000 | 800 | 1.04 GB | 12 | 0.9667 | 52.4 img/s |
| QFN-3D | all | 0.3925 | 0.2671 | MISALIGNED:0.4300 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 614.92 MB | 12 | 0.9667 | 70.2 img/s |
| QFN32-2D | all | 0.5487 | 0.4744 | MISALIGNED:0.1000 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 1.29 GB | 12 | 0.9667 | 62.6 img/s |
| Resistor-2D | all | 0.9650 | 0.9650 | MISALIGNED:0.0700 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 114.79 MB | 12 | 0.9667 | 476.6 img/s |
| Resistor-3D | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:0.0000 TOMBSTONE:1.0000 | 400 | 56.05 MB | 12 | 0.9667 | 546.7 img/s |
| Transistor-2D | all | 0.9850 | 0.9850 | MISALIGNED:0.0150 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 245.13 MB | 12 | 0.9667 | 533.3 img/s |
| Transistor-3D | all | 0.5363 | 0.4327 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.8550 | 800 | 79.92 MB | 12 | 0.9667 | 603.1 img/s |

<a id="model-ic-16-3d-pt"></a>
### IC-16-3D.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| IC-16-3D | all | 0.9875 | 0.9875 | MISALIGNED:0.0300 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 1.04 GB | 2 | 1.0000 | 47.0 img/s |
| QFN-3D | all | 0.7250 | 0.6648 | MISALIGNED:0.1600 MISSING:0.0000 TOMBSTONE:0.8650 | 800 | 614.92 MB | 2 | 1.0000 | 74.4 img/s |
| QFN32-2D | all | 0.2800 | 0.1605 | MISALIGNED:0.0150 MISSING:0.9650 TOMBSTONE:0.9000 | 800 | 1.29 GB | 2 | 1.0000 | 68.0 img/s |
| Resistor-2D | all | 0.2950 | 0.2102 | MISALIGNED:0.2700 MISSING:0.9500 TOMBSTONE:0.6000 | 400 | 114.79 MB | 2 | 1.0000 | 482.2 img/s |
| Resistor-3D | all | 0.2850 | 0.1575 | MISALIGNED:1.0000 MISSING:0.0000 TOMBSTONE:0.8600 | 400 | 56.05 MB | 2 | 1.0000 | 548.6 img/s |
| Transistor-2D | all | 0.2712 | 0.1725 | MISALIGNED:0.2150 MISSING:1.0000 TOMBSTONE:0.7000 | 800 | 245.13 MB | 2 | 1.0000 | 523.8 img/s |
| Transistor-3D | all | 0.5775 | 0.5134 | MISALIGNED:0.5950 MISSING:0.0000 TOMBSTONE:0.0950 | 800 | 79.92 MB | 2 | 1.0000 | 600.6 img/s |

<a id="model-transistor-3d-pt"></a>
### Transistor-3D.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| IC-16-3D | all | 0.5112 | 0.4004 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.9700 | 800 | 1.04 GB | 5 | 0.9667 | 52.2 img/s |
| QFN-3D | all | 0.6500 | 0.5784 | MISALIGNED:0.1100 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 614.92 MB | 5 | 0.9667 | 74.3 img/s |
| QFN32-2D | all | 0.2525 | 0.1186 | MISALIGNED:0.0100 MISSING:0.9900 TOMBSTONE:0.9900 | 800 | 1.29 GB | 5 | 0.9667 | 68.2 img/s |
| Resistor-2D | all | 0.2800 | 0.1566 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:0.8900 | 400 | 114.79 MB | 5 | 0.9667 | 487.6 img/s |
| Resistor-3D | all | 0.2500 | 0.1012 | MISALIGNED:1.0000 MISSING:0.0000 TOMBSTONE:1.0000 | 400 | 56.05 MB | 5 | 0.9667 | 531.0 img/s |
| Transistor-2D | all | 0.2650 | 0.1295 | MISALIGNED:0.0000 MISSING:1.0000 TOMBSTONE:0.9400 | 800 | 245.13 MB | 5 | 0.9667 | 528.6 img/s |
| Transistor-3D | all | 0.9838 | 0.9837 | MISALIGNED:0.0650 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 79.92 MB | 5 | 0.9667 | 589.6 img/s |

<a id="model-resistor-2d-pt"></a>
### Resistor-2D.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| IC-16-3D | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 1.04 GB | 7 | 0.9667 | 52.9 img/s |
| QFN-3D | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 614.92 MB | 7 | 0.9667 | 72.4 img/s |
| QFN32-2D | all | 0.4150 | 0.3548 | MISALIGNED:0.1050 MISSING:0.6650 TOMBSTONE:1.0000 | 800 | 1.29 GB | 7 | 0.9667 | 66.4 img/s |
| Resistor-2D | all | 0.9825 | 0.9825 | MISALIGNED:0.0300 MISSING:0.0300 TOMBSTONE:0.0000 | 400 | 114.79 MB | 7 | 0.9667 | 472.0 img/s |
| Resistor-3D | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:0.0000 TOMBSTONE:1.0000 | 400 | 56.05 MB | 7 | 0.9667 | 540.3 img/s |
| Transistor-2D | all | 0.6338 | 0.6416 | MISALIGNED:0.0300 MISSING:0.5500 TOMBSTONE:0.5050 | 800 | 245.13 MB | 7 | 0.9667 | 510.3 img/s |
| Transistor-3D | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:0.0000 TOMBSTONE:1.0000 | 800 | 79.92 MB | 7 | 0.9667 | 581.4 img/s |

<a id="model-qfn32-2d-pt"></a>
### QFN32-2D.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| IC-16-3D | all | 0.3013 | 0.1883 | MISALIGNED:0.7900 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 1.04 GB | 6 | 1.0000 | 52.4 img/s |
| QFN-3D | all | 0.2500 | 0.1000 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 614.92 MB | 6 | 1.0000 | 71.1 img/s |
| QFN32-2D | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 1.29 GB | 6 | 1.0000 | 67.1 img/s |
| Resistor-2D | all | 0.4775 | 0.3218 | MISALIGNED:0.0900 MISSING:0.0000 TOMBSTONE:1.0000 | 400 | 114.79 MB | 6 | 1.0000 | 439.2 img/s |
| Resistor-3D | all | 0.2500 | 0.1139 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:0.9700 | 400 | 56.05 MB | 6 | 1.0000 | 497.6 img/s |
| Transistor-2D | all | 0.4550 | 0.3084 | MISALIGNED:0.1750 MISSING:0.0050 TOMBSTONE:1.0000 | 800 | 245.13 MB | 6 | 1.0000 | 494.4 img/s |
| Transistor-3D | all | 0.2625 | 0.1769 | MISALIGNED:1.0000 MISSING:1.0000 TOMBSTONE:0.4900 | 800 | 79.92 MB | 6 | 1.0000 | 589.1 img/s |

<a id="model-resistor-3d-pt"></a>
### Resistor-3D.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| IC-16-3D | all | 0.2375 | 0.1386 | MISALIGNED:0.1500 MISSING:1.0000 TOMBSTONE:0.9000 | 800 | 1.04 GB | 9 | 0.9500 | 52.4 img/s |
| QFN-3D | all | 0.2087 | 0.1557 | MISALIGNED:0.4600 MISSING:1.0000 TOMBSTONE:0.7050 | 800 | 614.92 MB | 9 | 0.9500 | 72.4 img/s |
| QFN32-2D | all | 0.3800 | 0.2710 | MISALIGNED:0.4700 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 1.29 GB | 9 | 0.9500 | 67.9 img/s |
| Resistor-2D | all | 0.2575 | 0.1350 | MISALIGNED:0.0100 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 114.79 MB | 9 | 0.9500 | 488.6 img/s |
| Resistor-3D | all | 0.9800 | 0.9800 | MISALIGNED:0.0600 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 56.05 MB | 9 | 0.9500 | 550.1 img/s |
| Transistor-2D | all | 0.2575 | 0.1263 | MISALIGNED:0.0100 MISSING:1.0000 TOMBSTONE:0.9950 | 800 | 245.13 MB | 9 | 0.9500 | 533.2 img/s |
| Transistor-3D | all | 0.5000 | 0.3798 | MISALIGNED:0.0050 MISSING:0.0000 TOMBSTONE:0.9950 | 800 | 79.92 MB | 9 | 0.9500 | 596.9 img/s |

<a id="model-qfn-3d-pt"></a>
### QFN-3D.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Dataset Size | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| IC-16-3D | all | 0.4938 | 0.3725 | MISALIGNED:1.0000 MISSING:0.0050 TOMBSTONE:0.0200 | 800 | 1.04 GB | 8 | 0.9917 | 46.3 img/s |
| QFN-3D | all | 0.9875 | 0.9875 | MISALIGNED:0.0050 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 614.92 MB | 8 | 0.9917 | 64.4 img/s |
| QFN32-2D | all | 0.2650 | 0.1392 | MISALIGNED:1.0000 MISSING:0.9400 TOMBSTONE:0.0000 | 800 | 1.29 GB | 8 | 0.9917 | 66.9 img/s |
| Resistor-2D | all | 0.2350 | 0.2379 | MISALIGNED:0.7700 MISSING:0.7900 TOMBSTONE:0.5900 | 400 | 114.79 MB | 8 | 0.9917 | 471.9 img/s |
| Resistor-3D | all | 0.2600 | 0.1221 | MISALIGNED:0.9600 MISSING:1.0000 TOMBSTONE:1.0000 | 400 | 56.05 MB | 8 | 0.9917 | 517.0 img/s |
| Transistor-2D | all | 0.2437 | 0.1892 | MISALIGNED:0.8550 MISSING:0.9950 TOMBSTONE:0.3550 | 800 | 245.13 MB | 8 | 0.9917 | 514.5 img/s |
| Transistor-3D | all | 0.2925 | 0.1791 | MISALIGNED:0.0600 MISSING:1.0000 TOMBSTONE:1.0000 | 800 | 79.92 MB | 8 | 0.9917 | 582.7 img/s |

<a id="history"></a>
## History

- 2026-02-20: Report generated (9 models, 7 datasets)
