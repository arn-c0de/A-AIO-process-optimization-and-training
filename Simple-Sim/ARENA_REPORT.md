# Model Arena Report
> Last updated: 2026-02-14 12:30:18

## Overall Ranking

| Rank | Model | Type | Avg Accuracy | Avg F1 | Datasets Tested | Best Dataset | Worst Dataset | Last Run |
|---:|---|---|---:|---:|---:|---|---|---|
| 1 | Multi-Mind-v1_20260214_113757.bundle | Bundle | 0.9792 | 0.9790 | 3 | run_qfn32 (1.0000) | Resistor-v1 (0.9525) | 2026-02-14 11:54:37 |
| 2 | Transistor-v2.pt | Single | 0.7268 | 0.6876 | 3 | Transistor-v2 (0.9917) | run_qfn32 (0.4688) | 2026-02-14 11:11:11 |
| 3 | qfn32-v4.pt | Single | 0.6575 | 0.5779 | 3 | run_qfn32 (1.0000) | Transistor-v2 (0.4775) | 2026-02-14 11:39:40 |
| 4 | Resistor-v1.pt | Single | 0.6425 | 0.5811 | 3 | Resistor-v1 (0.9525) | run_qfn32 (0.3962) | 2026-02-14 10:58:25 |

## Per-Dataset Breakdown

### Dataset: Resistor-v1

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | Multi-Mind-v1_20260214_113757.bundle | 0.9525 | 0.9521 | all | 400 | 2026-02-14 11:53:31 |
| 2 | Resistor-v1.pt | 0.9525 | 0.9521 | all | 400 | 2026-02-14 10:57:58 |
| 3 | Transistor-v2.pt | 0.7200 | 0.7165 | all | 400 | 2026-02-14 10:59:54 |
| 4 | qfn32-v4.pt | 0.4950 | 0.3639 | all | 400 | 2026-02-14 11:38:49 |

### Dataset: Transistor-v2

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | Transistor-v2.pt | 0.9917 | 0.9917 | test | 120 | 2026-02-14 11:00:02 |
| 2 | Multi-Mind-v1_20260214_113757.bundle | 0.9850 | 0.9850 | all | 800 | 2026-02-14 11:53:14 |
| 3 | Resistor-v1.pt | 0.5787 | 0.5021 | all | 800 | 2026-02-14 10:58:05 |
| 4 | qfn32-v4.pt | 0.4775 | 0.3699 | all | 800 | 2026-02-14 11:39:06 |

### Dataset: run_qfn32

| Rank | Model | Accuracy | F1 | Split | Samples | Last Run |
|---:|---|---:|---:|---|---:|---|
| 1 | Multi-Mind-v1_20260214_113757.bundle | 1.0000 | 1.0000 | all | 800 | 2026-02-14 11:54:37 |
| 2 | qfn32-v4.pt | 1.0000 | 1.0000 | all | 800 | 2026-02-14 11:39:40 |
| 3 | Transistor-v2.pt | 0.4688 | 0.3545 | all | 800 | 2026-02-14 11:11:11 |
| 4 | Resistor-v1.pt | 0.3962 | 0.2891 | all | 800 | 2026-02-14 10:58:25 |

## Bundle Details

### Multi-Mind-v1_20260214_113757.bundle
- **Profiles:** chip_0603_resistor@1, qfn32_ic@1, sot23_transistor@1
- **Checkpoints:** 3
- **Created:** 2026-02-14T10:37:57Z

## Per-Model Detail Cards

### Multi-Mind-v1_20260214_113757.bundle (Bundle)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Resistor-v1 | all | 0.9525 | 0.9521 | MISALIGNED:0.1900 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 4 | 0.9667 | 443.1 img/s |
| Transistor-v2 | all | 0.9850 | 0.9850 | MISALIGNED:0.0500 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 4 | 0.9833 | 480.3 img/s |
| run_qfn32 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 13 | 1.0000 | 55.3 img/s |

### Transistor-v2.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Resistor-v1 | all | 0.7200 | 0.7165 | MISALIGNED:0.0600 MISSING:0.6300 TOMBSTONE:0.0000 | 400 | 4 | 0.9833 | 453.2 img/s |
| Transistor-v2 | test | 0.9917 | 0.9917 | MISALIGNED:0.0333 MISSING:0.0000 TOMBSTONE:0.0000 | 120 | - | - | - |
| run_qfn32 | all | 0.4688 | 0.3545 | MISALIGNED:0.0000 MISSING:0.1250 TOMBSTONE:1.0000 | 800 | 4 | 0.9833 | 63.1 img/s |

### qfn32-v4.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Resistor-v1 | all | 0.4950 | 0.3639 | MISALIGNED:0.0400 MISSING:0.0000 TOMBSTONE:0.9800 | 400 | 13 | 1.0000 | 443.7 img/s |
| Transistor-v2 | all | 0.4775 | 0.3699 | MISALIGNED:0.0650 MISSING:0.1800 TOMBSTONE:1.0000 | 800 | 6 | 1.0000 | 463.7 img/s |
| run_qfn32 | all | 1.0000 | 1.0000 | MISALIGNED:0.0000 MISSING:0.0000 TOMBSTONE:0.0000 | 800 | 13 | 1.0000 | 61.0 img/s |

### Resistor-v1.pt (Single)

| Dataset | Split | Accuracy | F1 | Critical FN | Samples | Epoch | Val Acc | Speed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Resistor-v1 | all | 0.9525 | 0.9521 | MISALIGNED:0.1900 MISSING:0.0000 TOMBSTONE:0.0000 | 400 | 4 | 0.9667 | 442.7 img/s |
| Transistor-v2 | all | 0.5787 | 0.5021 | MISALIGNED:0.1150 MISSING:0.1400 TOMBSTONE:0.4300 | 800 | 4 | 0.9667 | 479.9 img/s |
| run_qfn32 | all | 0.3962 | 0.2891 | MISALIGNED:0.0000 MISSING:0.4150 TOMBSTONE:1.0000 | 800 | 4 | 0.9667 | 62.2 img/s |

## History

- 2026-02-14: Report generated (4 models, 3 datasets)
