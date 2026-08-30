# 42-class FT best.pt → local metric evaluation

## Protocol

- Frozen 42-class FT ViT-Large backbone (1024-D pre-logit embedding).
- GT = target valid mean; Mean = target train mean; SS = target train, up to 5 per label (seed 42).
- Query = target test only; Euclidean and cosine both evaluated.
- This is supervised target-gallery transfer, not zero-shot.

## Observed test results

| Dataset | Setting | Accuracy | Macro-F1 | Test coverage | Majority baseline |
|---|---|---:|---:|---:|---:|
| 3label | MEAN + cosine | 91.70% | 76.93% | 3/3 classes | 86.10% |
| 7label | SS + euclidean | 26.24% | 15.88% | 6/7 classes | 51.77% |
| 12label | MEAN + cosine | 36.72% | 36.08% | 12/12 classes | 34.57% |

> The table selects the highest observed Macro-F1 on test for compact display; it is descriptive, not a valid model-selection estimate.

## Data warnings

- 3label exact duplicates: train↔valid 6, train↔test 5, valid↔test 0.
- 7label test missing gallery label(s): nobel.
- No local 11-label dataset was found; 12label was evaluated intact.
- Internal 42-class training manifest is absent locally, so training-to-target image overlap cannot be ruled out.

## Attention maps

- ViT attention rollout over all transformer blocks and heads.
- 3label, 7label, and 12label each contain 100 matched test images for every GT/Mean/SS × Euclidean/Cosine setting.
- Saved overlays: 1,800; unique attention computations: 300.
- Attention is determined by the frozen backbone and image, so matched settings hard-link the same heatmap; their metric predictions are recorded separately in each folder's `predictions.csv`.
- Output: `attention_maps/`; full index: `attention_maps/attention_index.csv`.
