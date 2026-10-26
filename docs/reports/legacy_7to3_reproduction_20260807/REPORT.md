# Legacy 7-label to unseen 3-label reproduction

## Main result

On the frozen epoch-15 checkpoint, replacing cosine distance with a pooled
Ledoit-Wolf Mahalanobis distance raises test accuracy substantially.

| Embedding | Distance | Valid accuracy / macro-F1 | Test accuracy / macro-F1 | Test balanced accuracy |
|---|---|---:|---:|---:|
| Backbone (384-D) | Cosine | 28.86% / 28.33% | 27.52% / 26.87% | 51.34% |
| Backbone (384-D) | Mahalanobis | 66.67% / 55.52% | 63.30% / 48.23% | 50.77% |
| SupCon projection (128-D) | Cosine | 33.83% / 31.74% | 31.19% / 29.75% | 44.68% |
| SupCon projection (128-D) | Mahalanobis | 60.20% / 48.62% | 60.55% / 44.77% | 47.65% |

The primary 20%-range backbone result gains 35.78 percentage points in
accuracy and 21.36 points in macro-F1. Its balanced accuracy,
however, changes from 51.34% to 50.77%. The high raw accuracy mainly comes
from correcting many Bicon and ITI samples, while rare-class Bego recall falls
from 4/4 to 1/4. The target test set is imbalanced (Bego 4, Bicon 61, ITI 44),
so the accuracy gain should not be read as uniform three-class improvement.

The first projection result in this directory (34.86%) was evaluated against
the live `best.pt` immediately before epoch 15 replaced it. The table above is
the corrected apples-to-apples comparison: both distances use the same saved
epoch-15 file in one evaluation pass.

## Reproduced condition

- Source training labels: ADIN, DIONAVI, Dentium, MIS, NORIS, nobel, osstem.
- Source train/validation: 2,603 / 338 images. Source test is not used for
  training or checkpoint selection.
- Target labels: Bego, Bicon, ITI, absent from source training.
- Target train/validation/test: 1,765 / 201 / 109 images, matching the legacy
  manifest counts exactly.
- Model: `vit_small_patch16_224.augreg_in21k_ft_in1k`, ImageNet pretrained,
  224×224 input.
- Training: SupCon, batch 16, learning rate 3e-5, weight decay 0.05,
  temperature 0.07, projection dimension 128, dropout/drop-path 0.1, seed 42.
- Frozen checkpoint: epoch 15 of the fresh 30-epoch reconstruction.
- Target support: all labeled target-train images, one mean prototype per
  class.
- Query: target validation and test only.
- Cosine decision: nearest L2-normalized class-mean prototype.
- Mahalanobis decision: nearest class-mean prototype using a pooled covariance
  estimated from target-train within-class residuals with Ledoit-Wolf shrinkage.
- Target validation and test are not used to fit prototypes or covariance.
- No target image is used to train or fine-tune the encoder.

This is unseen-class transfer for the encoder, but not zero-shot: labeled
target-train images define the three prototypes.

## Test confusion matrices

Backbone cosine:

| True / predicted | Bego | Bicon | ITI |
|---|---:|---:|---:|
| Bego | 4 | 0 | 0 |
| Bicon | 37 | 8 | 16 |
| ITI | 26 | 0 | 18 |

Backbone Mahalanobis:

| True / predicted | Bego | Bicon | ITI |
|---|---:|---:|---:|
| Bego | 1 | 0 | 3 |
| Bicon | 4 | 43 | 14 |
| ITI | 8 | 11 | 25 |

No exact target-train/test image duplicate was found. The direct Mahalanobis
implementation agrees with SciPy to below `3.2e-14` absolute error.

## Provenance

- Frozen checkpoint SHA-256:
  `21d11b421eb33c9d23e7a0426d98b407ca91cf9c4613f5947abaac725748b820`
- Training manifest SHA-256:
  `bbada5e9ad642fa26da7da1630cc7a1fda5eb50916e71e69efb490c7daa184dd`
- Target manifest SHA-256:
  `d2381c0b27936fe047e22a5d460bb585be25226027ba9f98655079d8d31d7941`
- Projection artifacts:
  `20260807_101835_legacy_vitsmall_supcon7_to_3label/`
- Backbone-only artifacts:
  `20260807_102022_legacy_vitsmall_supcon7_to_3label_backbone_only/`
- Same-checkpoint distance comparison:
  `mahalanobis_projection/` and `mahalanobis_backbone/`

The target split is reconstructed from
`Dental Implants 2.0.v3i.yolov8/{train,valid,test}/images` by the canonical
filename prefixes `Bego`, `Bicon`, and `ITI`; unrelated filenames are excluded.
