![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightblue)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![Status](https://img.shields.io/badge/status-Active-success)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![ResNet18](https://img.shields.io/badge/model-ResNet18-blue)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/arn-c0de/A-AIO-process-optimization-and-training)

# A-AIO-process-optimization-and-training

Testing/prototyping repository for AOI/AI concepts, focused on reproducible pipelines and practical model iteration.

## Quick Links

- Main project: [`Simple-Sim`](Simple-Sim/README.md)
- Documentation index: [`Simple-Sim/docs/INDEX.md`](Simple-Sim/docs/INDEX.md)
- Latest arena report: [`Simple-Sim/ARENA_REPORT.md`](Simple-Sim/ARENA_REPORT.md)
- Training logs index: [`Simple-Sim/docs/training_logs/INDEX.md`](Simple-Sim/docs/training_logs/INDEX.md)
- Sample gallery: [`Simple-Sim/SAMPLE_GALLERY.md`](Simple-Sim/SAMPLE_GALLERY.md)

## Contents

- [Current Status](#current-status)
- [Projects](#projects)
- [Arena](#arena)
- [Research Updates](#research-updates)
- [Model Strategy](#model-strategy)
- [License and Usage](#license-and-usage)

## Current Status

Last updated: **2026-02-20**

| Topic | Status | Link |
|---|---|---|
| Arena leaderboard | Current top ranking based on latest QFN crawler fine-tune boost | [`Simple-Sim/ARENA_REPORT.md`](Simple-Sim/ARENA_REPORT.md) |
| Active research | Extended image filter robustness + QFN-focused fine-tuning | [`Simple-Sim/docs/training_logs/INDEX.md`](Simple-Sim/docs/training_logs/INDEX.md) |
| Production recommendation | Prefer single cross-profile models for robustness/simplicity | [`Final Verdict`](Simple-Sim/docs/training_logs/2026-02-16-Training-Final-Conclusion-Phase1.md) |

## Projects

![Simple-Sim Pipeline Dashboard](Simple-Sim/images/pipeline-dashboard-simple-sim-v1.0.png)

| Project | Purpose | Start Here |
|---|---|---|
| [`Simple-Sim`](Simple-Sim/README.md) | Sandbox for synthetic data generation, training, evaluation, and model selection via arena workflows | [`Simple-Sim/README.md`](Simple-Sim/README.md) |

## Arena

- Latest stats: [`Simple-Sim/ARENA_REPORT.md`](Simple-Sim/ARENA_REPORT.md)
- Historical arena reports: [`Simple-Sim/docs/arena-reports-historie/INDEX.md`](Simple-Sim/docs/arena-reports-historie/INDEX.md)

![Top Avg Accuracy](Simple-Sim/ARENA_REPORT_assets/top_avg_accuracy.svg)

## Research Updates

### Current Focus (2026-02-20)

QFN crawler fine-tune boost with additional clean no-image-filter samples (`QFN32-2D +50`, `QFN-3D +50`) reached the current top arena ranking.  
Log: [`2026-02-20-Training-QFN-Crawler-NoImageFilter-Boost.md`](Simple-Sim/docs/training_logs/2026-02-20-Training-QFN-Crawler-NoImageFilter-Boost.md)

### Recent Logs

- Fine-tuned comparison (default-filter vs QFN-focused): [`2026-02-20-Training-Comparison-FineTuned-vs-QFN-FineTuned.md`](Simple-Sim/docs/training_logs/2026-02-20-Training-Comparison-FineTuned-vs-QFN-FineTuned.md)
- Extended filter restart (living log): [`2026-02-17-Training-Restart-Extended-Image-Filters.md`](Simple-Sim/docs/training_logs/2026-02-17-Training-Restart-Extended-Image-Filters.md)
- 90° random orientation + SOIC16 restart: [`2026-02-16-Training-Restart-90deg-SOIC16.md`](Simple-Sim/docs/training_logs/2026-02-16-Training-Restart-90deg-SOIC16.md)
- Filter settings reference: [`FILTER_SETTINGS.md`](Simple-Sim/docs/guides/FILTER_SETTINGS.md)

## Model Strategy

Current conclusions: [`Final Verdict`](Simple-Sim/docs/training_logs/2026-02-16-Training-Final-Conclusion-Phase1.md), [`Training Conclusions`](Simple-Sim/docs/guides/knowledge/ModelTrainingConclusions-2D-3D%20Mixed-vs.-Staged-Approach.md)

### Bundled Models (Multi-Profile)

- Require an upstream component/object classifier for routing.
- Higher per-component peak accuracy.
- Larger footprint (multiple sub-models).

### Single Models (Cross-Profile)

- Self-contained (no external classifier).
- Trained across all profiles and defect types.
- Smaller footprint.
- Slightly lower per-component peak accuracy, but stronger on randomized mixed/unknown component recognition.

Recommendation: use single cross-profile models for deployment when robustness and simplicity are prioritized.

## License and Usage

This repository is **proprietary** and **not open-source**.

- Full license: [`LICENSE`](LICENSE)
- Third-party dependency licenses: [`Simple-Sim/THIRD_PARTY_LICENSES.md`](Simple-Sim/THIRD_PARTY_LICENSES.md)

### Allowed

- Read code and documentation.
- Discuss concepts and provide feedback.
- Link to this repository.

### Not Allowed (without prior written permission)

- Use this code (whole or partial) in projects/products/services.
- Extract or reuse modules, files, snippets, or components.
- Copy, modify, distribute, sublicense, or republish.
- Use for commercial or production deployments.

Note: public visibility on GitHub does not grant reuse rights beyond viewing the repository.

Permission requests: `arn-c0de@protonmail.com`  
See also: [`CONTRIBUTING.md`](CONTRIBUTING.md)
