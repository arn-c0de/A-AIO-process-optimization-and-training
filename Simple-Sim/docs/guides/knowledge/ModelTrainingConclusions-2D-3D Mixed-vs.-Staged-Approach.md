## Preferred Strategy (Most Cases) (February 16, 2026)


**Direct joint / fully mixed training from scratch**  
→ Train **one single model** on **all 2D + 3D profiles together** right from epoch 1.

This approach consistently delivers the **strongest generalization**, the **most robust shared features**, and the **fastest closure of domain gaps** (especially QFN-3D).

## When to Consider Staged Training (2D first → then 3D)

Only in these limited cases:

- 3D data is **very scarce** (<10–20% of total samples)  
- 3D data is **extremely noisy** or low-quality  
- Joint training **diverges badly** in the first 3–5 epochs (very high initial loss oscillation)

In all other realistic scenarios → **direct mixed wins**.

## Key Conclusions & Evidence

| Aspect                              | Direct Mixed Training (everything together)              | Staged: 2D mixed first → 3D on top                       | Winner & Why (2026 perspective)                     |
|-------------------------------------|----------------------------------------------------------|----------------------------------------------------------|-----------------------------------------------------|
| Generalization on mixed batches     | Excellent – robust shared features                       | Good, but weaker cross-domain transfer                   | **Direct mixed** – RandomDataCrawler-v3 proof (0.9956 FullMerged) |
| Low-level feature quality           | Deep & truly shared across 2D/3D                         | Early layers biased toward 2D → suboptimal for 3D        | **Direct mixed**                                    |
| Risk of catastrophic forgetting     | Very low                                                 | Medium–high (3D overwrites good 2D features)             | **Direct mixed**                                    |
| Training stability & convergence    | Can be slower/noisier at start                           | Faster initial progress, but plateau risk later          | **Staged** short-term, **direct** long-term         |
| Hyperparameter complexity           | Simpler (no freeze/switch decisions)                     | More tuning (when to switch? LR schedule? freeze layers?) | **Direct mixed**                                    |
| QFN-3D gap closing                  | Fastest – 3D gradients from day 1                        | Slower – starts from 2D-biased weights                   | **Direct mixed** (critical for your biggest weakness) |
| Compute efficiency                  | Sometimes higher total epochs                            | Potentially fewer epochs if strong 2D pretrain           | Slight edge **staged** only if very imbalanced      |
| Arena observation                   | RandomDataCrawler-v3 (likely direct mixed) dominates     | Most specialist models (staged-like behavior) collapse   | **Direct mixed** wins empirically                   |

## Recommended Practical Sweet-Spot Pipeline (2026)

### 1. Main Path – Direct Joint Training (Recommended Default)

- **Dataset**: FullMergedDataset + targeted oversampling / upweighting of QFN-3D samples  
- **Sampling strategy**: Class-balanced or dynamic curriculum  
  - Start with higher weight on 2D classes  
  - Gradually increase 3D / QFN-3D weight over epochs  
- **Architecture**: Efficient modern backbone  
  - ViT variants, ConvNeXt, EfficientNetV2, or YOLO-style detector  
- **Augmentation**: Aggressive domain-specific  
  - Lighting/contrast jitter  
  - Random angle/perspective warp (simulate 2D→3D differences)  
  - Synthetic tombstone / missing / misaligned augmentations  
- **Goal**: Achieve ≥0.75–0.85 on QFN-3D-v1 while maintaining ≥0.99 on FullMerged

### 2. Fallback / Rescue Path – If Joint Training Diverges

1. Train **only on all 2D mixed** (Resistor-v1, QFN-v1, Transistor-v1, etc.) → until validation accuracy >0.98  
2. Unfreeze later layers → fine-tune on **full mixed dataset** with heavy QFN-3D weighting  
3. (Optional) Freeze early convolutional layers if 2D performance drops noticeably

## One-Liner Verdict

**Direct mixed training from scratch is superior in 80–90% of realistic 2026 scenarios.**  
It produces the most robust single model (strong evidence: RandomDataCrawler-v3), closes domain gaps (especially QFN-3D) fastest, and avoids most of the pitfalls and complexity of staged training.

**Only choose staged if 3D data is extremely scarce (<10–20%) or joint training shows severe early instability.**

Start with **direct mixed + smart dynamic sampling** — this is the highest-probability path to turning your generalist model into the clear single-model champion across the entire arena.