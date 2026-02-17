![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightblue)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![Status](https://img.shields.io/badge/status-Active-success)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![ResNet18](https://img.shields.io/badge/model-ResNet18-blue)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/arn-c0de/A-AIO-process-optimization-and-training)

# A-AIO-process-optimization-and-training

This is a **testing / prototyping repository** for experimenting with AOI/AI concepts and implementing them as clean, reproducible building blocks.

## Projects

![Simple-Sim Pipeline Dashboard](Simple-Sim/images/pipeline-dashboard-simple-sim-v1.0.png)

## - [`Simple-Sim`](Simple-Sim/README.md): 
A sandbox environment for pre-training and specializing models on synthetic datasets before passing them to main simulation and production training. Includes an arena system to evaluate and select the best-performing models for further training iterations.
- Sample images: [`Simple-Sim/SAMPLE_GALLERY.md`](Simple-Sim/SAMPLE_GALLERY.md)

---
## Latest Arena stats : [`Simple-Sim/ARENA_REPORT.md`](Simple-Sim/ARENA_REPORT.md)
>  Old Arena stats before 90° implementation : [`Simple-Sim/docs/arena-reports-historie/2026-02-ARENA_REPORT.md`](Simple-Sim/docs/arena-reports-historie/2026-02-ARENA_REPORT.md)

![Top Avg Accuracy](Simple-Sim/ARENA_REPORT_assets/top_avg_accuracy.svg)
---

## Current Research 2026-02-17
### Renewed training restart with extended image filter options to improve robustness/generalization (living research log): [`Simple-Sim/docs/training_logs/Training-Restart-Extended-Image-Filters-2026-02-17.md`](Simple-Sim/docs/training_logs/Training-Restart-Extended-Image-Filters-2026-02-17.md)  
>  Filter configuration and implementation reference: [`Simple-Sim/docs/guides/FILTER_SETTINGS.md`](Simple-Sim/docs/guides/FILTER_SETTINGS.md)


> ## Completed Testing Research
> - Completed training restart log (same datasets, with random 90° orientation per image + SOIC16 profile testing): [`Simple-Sim/docs/training_logs/Training-Restart-90deg-SOIC16-2026-02-16.md`](Simple-Sim/docs/training_logs/Training-Restart-90deg-SOIC16-2026-02-16.md)



---
## Production Notes: Bundled vs Single Models
> **📌 Current Conclusions 2026-02**
>
> [Final Verdict](Simple-Sim/docs/training_logs/Final-Conclusion-Training-Phase1.md)
> [Training Conclusions](Simple-Sim/docs/guides/knowledge/ModelTrainingConclusions-2D-3D%20Mixed-vs.-Staged-Approach.md)
> 
> **Bundled Models** (multi-profile, per-component-type):
> - Require an additional **object classification model** upstream to identify component type first
> - Higher per-component accuracy
> - Larger model footprint (multiple sub-models stored)
>
> **Single Models** (cross-profile, e.g. `random-datacrawler-v1`):
> - **No external classifier needed** — fully self-contained
> - Trained on **all datasets** (all profiles, all defect types combined)
> - **Much smaller model size** than bundles
> - Slightly lower per-component accuracy, but **significantly better at randomized recognition of mixed/unknown components**
> - Ideal for edge deployment and real-world PCB inspection where component type is unknown
>
> **→ Recommendation**: Use single models trained on all datasets for robustness, simplicity, and smaller footprint.

## License

This repository is **proprietary**. No permission is granted to use, copy, modify, or distribute this software without prior written permission.
See [`LICENSE`](LICENSE).

Third-party dependencies (Python packages, etc.) remain under their own licenses; see [`Simple-Sim/THIRD_PARTY_LICENSES.md`](Simple-Sim/THIRD_PARTY_LICENSES.md).

## What You May / May Not Do

This repository is public so others can understand the ideas and approach. It is **not** open-source.

Allowed:
- Read the code and documentation.
- Discuss concepts, provide feedback, and share high-level ideas.
- Link to this repository.

Not allowed (without prior written permission):
- Use this code (in whole or in part) in your own projects, products, or services.
- Extract or reuse individual modules, components, files, snippets, or other parts of this project.
- Copy, modify, merge, re-publish, distribute, or sublicense the code.
- Use it for commercial purposes or production deployments.

Note: On GitHub, others may be able to technically fork/clone public repositories. This does **not** grant permission to use the software beyond what is required to view it on GitHub; all other use remains strictly prohibited by [`LICENSE`](LICENSE).

Permission requests: arn-c0de@protonmail.com

See also: [`CONTRIBUTING.md`](CONTRIBUTING.md)
