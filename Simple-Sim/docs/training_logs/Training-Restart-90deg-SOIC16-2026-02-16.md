# Model Arena Report – Final Conclusion (February 16, 2026)

**New Dataset Regime**  
- 1200–1600 samples per dataset (balanced)  
- Random 90° base rotations (0/90/180/270) per image  
- Full pipeline restart from scratch

## Overall Performance Summary

Top models:  
1. **RandomCrawler-5X.pt** → **0.7170** avg acc  
2. **RandomCrawler-1X_MultiTrained.pt** → **0.7154** avg acc  

**Key new insight:**  
The 5X variant was trained **5× longer** (5 full passes / ~5× more epochs) than the 1X and all other specialist models.  
→ Result: **almost no improvement** (+0.0016 avg acc only).  

This proves **diminishing returns** from extra training epochs in the current setup.

## What Changed vs. Previous Run

- Absolute scores dropped (expected & desired) due to forced orientation randomization.  
- Generalization became harder → many cross-profile accuracies now hover at 0.25–0.44.  
- **RandomCrawler family still dominates**, but the gap between 5X and 1X is negligible.

## Critical Takeaways (Feb 2026)

| Observation                                      | Implication                                                                 |
|--------------------------------------------------|-----------------------------------------------------------------------------|
| 5X vs 1X: only +0.0016 avg acc despite 5× training | More epochs = almost no gain → training is saturated                        |
| Forced 90° rotations make task much harder       | Numbers look worse, but realism is now production-grade                     |
| RandomCrawler-5X & 1X still best overall         | Mixed training + broad data remains the winning strategy                    |
| Specialists collapse even more on cross tests    | Single-purpose models are increasingly obsolete under real variability     |
| Biggest remaining gaps: IC-16-3D & QFN-3D        | Tombstone / missing errors dominate failures                                |

## Realistic Verdict & Recommendation

**RandomCrawler-1X_MultiTrained.pt is now the preferred checkpoint.**  
It achieves **almost identical performance** to the 5X version while using **5× less training time / compute**.

**Extra training epochs are no longer a high-leverage lever.**

### Next High-Value Actions (priority order)

1. **Stop increasing epochs** – focus on data & augmentation instead  
2. Stronger domain augmentation (perspective warp, lighting jitter, synthetic tombstone/missing)  
3. Targeted data collection for IC-16-3D + QFN-3D (these are the only real blockers)  
4. Optional: fine-tune the 1X model only on the two weak profiles (cheap & effective)  
5. Re-run arena after above changes → expect 0.80–0.85+ avg acc with same realism level

**One-Liner Final Verdict**

The 5× longer trained model brings **zero meaningful gain** → we have reached the point of saturation.  
The **RandomCrawler-1X** (single pass) is the smartest, fastest, and equally strong model to take into production.  

More epochs won’t save us.  
Better data + smarter augmentation will.

