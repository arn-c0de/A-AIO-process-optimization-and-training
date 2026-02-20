# Model Arena Research Log – Restart with Extended Image Filters (February 17, 2026)

**Research Type:** Current / ongoing (extendable log)  
**Status:** Restart initiated  
**Base Reference:** Previous restart with random 90-degree orientation + SOIC16 testing  
`Simple-Sim/docs/training_logs/2026-02-16-Training-Restart-90deg-SOIC16.md`

## Research Goal

Evaluate whether expanded image filter randomization can improve cross-profile robustness beyond the 90-degree restart baseline, without increasing training epochs as the main optimization lever.

## Current Research Scope

- Keep the realistic baseline from the previous run (random 90-degree base orientation).
- Add and validate new filter options for synthetic image generation.
- Focus on generalization gains in weak profiles (especially IC-16-3D and QFN-3D).
- Track impact per filter family, then combine the highest-value filters.

Full filter reference and parameter map:  
`Simple-Sim/docs/guides/FILTER_SETTINGS.md`

## New Filter Families Under Test

| Priority | Filter Family | Expected Benefit |
|----------|---------------|------------------|
| High | Perspective Transform | Better tolerance to camera angle deviation |
| High | Motion Blur | Robustness for conveyor/high-speed capture |
| High | Saturation / Hue Shift | Better lighting and white-balance invariance |
| High | Shadow / Occlusion | Better defect detection under partial shading |
| High | Reflection / Glare | Better handling of specular highlights |
| Medium | Vignetting / Chromatic Aberration | More realistic optical variation |
| Medium | JPEG Compression / Color Temperature | Better robustness to storage and lighting chain differences |
| Low | Lens Distortion / Dust / Sharpen | Fine-tuning realism for edge cases |

## Working Hypothesis

The next meaningful accuracy gains are expected from better data realism and randomized filter diversity, not from longer epoch schedules.

## Evaluation Plan (Extendable)

1. Build controlled filter profiles (single-factor and mixed-factor).
2. Run restart training with same core datasets and compare against the February 16, 2026 baseline.
3. Record arena deltas (avg accuracy + weakest-profile lift).
4. Keep only filters with measurable benefit; remove unstable/noisy transforms.

## Success Criteria

- Higher average arena accuracy than the February 16, 2026 restart.
- Clear improvement on weakest profiles (IC-16-3D, QFN-3D).
- Stable behavior across mixed-profile inference (no specialist collapse).

## Extension Log

Use this section for incremental updates so this document remains a living research log.

## Arena Comparison Table (Baseline vs. Filter Runs)

| Run ID | Date | Filter Profile | Avg Arena Acc | IC-16-3D Acc | QFN-3D Acc | Delta vs Baseline | Notes |
|--------|------|----------------|---------------|--------------|------------|-------------------|-------|
| Baseline-90deg-SOIC16 | 2026-02-16 | Random 90-degree orientation baseline | TBD | TBD | TBD | 0.0000 | Reference run (`2026-02-16-Training-Restart-90deg-SOIC16.md`) |
| FilterRun-01 | TBD | Perspective only | TBD | TBD | TBD | TBD | |
| FilterRun-02 | TBD | Motion blur only | TBD | TBD | TBD | TBD | |
| FilterRun-03 | TBD | Saturation + Hue shift only | TBD | TBD | TBD | TBD | |
| FilterRun-04 | TBD | Shadow + Reflection only | TBD | TBD | TBD | TBD | |
| FilterRun-05 | TBD | Mixed high-priority filters | TBD | TBD | TBD | TBD | |

## Filter Family Decision Tracker

| Filter Family | Keep/Remove | Evidence (Run IDs) | Reason |
|---------------|-------------|--------------------|--------|
| Perspective Transform | TBD | TBD | |
| Motion Blur | TBD | TBD | |
| Saturation / Hue Shift | TBD | TBD | |
| Shadow / Occlusion | TBD | TBD | |
| Reflection / Glare | TBD | TBD | |
| Vignetting / Chromatic Aberration | TBD | TBD | |
| JPEG Compression / Color Temperature | TBD | TBD | |
| Lens Distortion / Dust / Sharpen | TBD | TBD | |

### Update 2026-02-17
- Restart scope defined for extended filter research.
- Filter families and priorities aligned with `FILTER_SETTINGS.md`.
- Baseline and comparison strategy locked for next arena cycle.

## Next Planned Entry

- Add first arena comparison table after initial filter-profile batch finishes.
- Include per-profile deltas and a keep/remove decision list per filter family.
