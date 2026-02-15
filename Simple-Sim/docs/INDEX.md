# Documentation Index

Simple-Sim comprehensive documentation organized by topic and use case.

## Guides

Practical how-to guides for common tasks and workflows.

| Document | Purpose |
|----------|---------|
| [Command Cheatsheet](guides/CHEATSHEET.md) | Quick reference for common commands and operations |
| [3D Rendering Quickstart](guides/3D_RENDERING_QUICKSTART.md) | Guide to setting up and customizing 3D rendering with Blender |
| [Model Merge: Bundles vs Ensembles](guides/MODEL_MERGE_BUNDLES_ENSEMBLES.md) | Bundles vs ensembles, and what you need for unknown images (profile dispatch vs general models) |

## Technical Reference

In-depth technical documentation for advanced topics and system design.

| Document | Purpose |
|----------|---------|
| [Profile System Implementation](../PROFILE_SYSTEM_IMPLEMENTATION.md) | Technical details on versioned component profiles and profile hashing |
| [Sample Gallery](../SAMPLE_GALLERY.md) | Auto-generated reference images with defect overlays from all datasets |
| [Arena Report](../Simple-Sim/ARENA_REPORT.md) | Latest benchmark statistics and performance metrics |
| [Third-Party Licenses](../THIRD_PARTY_LICENSES.md) | Open-source dependencies and their licenses |

## Quick Navigation

### I want to...

- **Get started quickly**: Read [Command Cheatsheet](guides/CHEATSHEET.md)
- **Set up 3D rendering**: Read [3D Rendering Quickstart](guides/3D_RENDERING_QUICKSTART.md)
- **Understand component profiles**: Read [Profile System Implementation](../PROFILE_SYSTEM_IMPLEMENTATION.md)
- **See example outputs**: View [Sample Gallery](../SAMPLE_GALLERY.md)
- **Check performance benchmarks**: Review [Arena Report](../Simple-Sim/ARENA_REPORT.md)
- **View the main guide**: See [README](../README.md)

## Document Overview

### Guides

#### Command Cheatsheet
Essential commands for setup, generation, training, evaluation, and testing. Includes configuration tweaks, troubleshooting, and file locations.

#### 3D Rendering Quickstart
Complete guide to the Blender 3D rendering backend, including:
- Architecture overview
- Component profile and run configuration
- Pad positioning and component modifications
- Materials, lighting, and camera setup
- Debugging and testing workflows

### Technical Reference

#### Profile System Implementation
Technical documentation on the versioned component profile system, covering:
- Profile structure and versioning
- Profile hashing for determinism
- Per-profile checkpoints and multi-profile bundles
- Pipeline guards and manifest validation

#### Sample Gallery
Auto-generated image reference showing defects and variations across all available component types and datasets.

#### Arena Report
Performance benchmarks, accuracy metrics, and model comparison statistics.

#### Third-Party Licenses
Attribution and license information for all dependencies.
