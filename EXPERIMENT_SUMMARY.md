# Experiment Summary

Generated from `outputs/` on 2026-05-07.

## Headline

| Category | Best/Relevant Run | Test Accuracy | Test Macro F1 | Notes |
|---|---:|---:|---:|---|
| 3-label plain ViT | `20260507_104805_plain_vit_3label` | 0.9449 | 0.7700 | Strong weighted F1, macro F1 limited by class imbalance/small classes. |
| 7-label plain ViT | `20260507_095713_plain_vit_7label` | 0.9440 | 0.8829 | Best completed plain 7-label classifier run. |
| 7-label SupCon metric | `20260507_101426_supcon_vit_7label` | 0.9587 | 0.7512 | Prototype evaluation; high weighted F1, lower macro F1. |
| 10-label plain ViT | `20260507_134640_plain_vit_10label` | 0.9554 | 0.8812 | Best overall supervised classifier among completed plain runs. |
| 10-label + NORIS/osstem two-stage | `20260507_134640_plain_vit_10label` + `NORIS_osstem` submodel | 0.9509 | 0.8786 | Two-stage was applied, but slightly reduced test accuracy. |
| 7-label metric -> 3-label | `20260507_142544_metric_7label_supcon_to_3label` | 0.3761 | 0.3475 | Direct transfer to 3-label prototype task is weak. |
| 7-label metric -> 3-label two-stage | `20260507_083124_twostage_metric_7label_to_3label` | 0.6881 | 0.5492 | Better than direct prototype transfer, still below plain 3-label classifier. |

## Plain Supervised ViT Runs

| Run | Labels / Dataset | Model | Main Settings | Valid Acc | Valid Macro F1 | Test Acc | Test Macro F1 | Artifacts |
|---|---|---|---|---:|---:|---:|---:|---|
| `20260507_104805_plain_vit_3label` | `plain_vit_3label.csv` | `vit_base_patch16_224` | 80 epochs, lr 5e-5, strong aug, class weights | 0.9353 | 0.8977 | 0.9450 | attention, t-SNE, confusion matrices |
| `20260507_095713_plain_vit_7label` | `plain_vit_7label.csv` | `vit_base_patch16_224` | 25 epochs, lr 3e-5, no strong aug, no class weights | 0.9379 | 0.8353 | 0.9440 | attention, t-SNE, confusion matrices |
| `20260507_134640_plain_vit_10label` | `plain_vit_10label.csv` | `vit_base_patch16_224` | 80 epochs, lr 5e-5, strong aug, no class weights | 0.9276 | 0.9202 | 0.9554 | attention, t-SNE, confusion matrices, NORIS/osstem two-stage |
| `20260507_090838_plain_vit_7label` | older/no-MIS style run | `vit_base_patch16_224` | 80 epochs, lr 5e-5, strong aug, class weights | 0.3000 | 0.4289 | 0.4123 | poor/obsolete |
| `20260507_094551_plain_vit_7label` | `plain_vit_7label.csv` | `vit_base_patch16_224` | 80 epochs, lr 5e-5, strong aug, class weights | 0.4615 | 0.5854 | N/A | incomplete or superseded |
| `20260507_051718_main_vit_7label` | args say `main_vit_6label`, `6label.csv` | `vit_base_patch16_224` | 80 epochs, lr 5e-5, strong aug, class weights | 1.0000 | 1.0000 | 0.9839 | naming mismatch; treat as older/special run |

## Metric Learning / SupCon

| Run | Dataset | Model | Loss / Eval | Valid Acc | Valid Macro F1 | Test Acc | Test Macro F1 | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| `20260507_101426_supcon_vit_7label` | `supcon_vit_7label.csv` | `vit_base_patch16_224` | SupCon, prototype cosine eval | 0.9320 | 0.7419 | 0.9587 | 0.7512 | High test accuracy; macro lower because minority classes remain hard. |
| `20260507_081650_metric_7label_to_3label` | `3label.csv` | `vit_small_patch16_224.augreg...` | prototype eval | 0.4776 | 0.3833 | 0.3486 | 0.2926 | weak direct transfer |
| `20260507_082013_metric_7label_to_3label_backbone_only` | `3label.csv` | `vit_small_patch16_224.augreg...` | backbone-only prototype eval | 0.4876 | 0.3895 | 0.3578 | 0.2988 | slightly better than projection in this setup, still weak |
| `20260507_142544_metric_7label_supcon_to_3label` | `3label.csv` | `vit_base_patch16_224` | prototype eval from 7-label SupCon ckpt | 0.3980 | 0.3620 | 0.3761 | 0.3475 | current 7-label SupCon -> 3-label test |
| `20260507_083124_twostage_metric_7label_to_3label` | `3label.csv` | `vit_small_patch16_224.augreg...` | two-stage metric eval | 0.6866 | 0.5399 | 0.6881 | 0.5492 | improves over direct prototype transfer |

## Two-Stage / Submodels

| Run | Purpose | Valid Acc | Valid Macro F1 | Test Acc | Test Macro F1 | Notes |
|---|---|---:|---:|---:|---:|---|
| `outputs/submodels/20260507_051743_NORIS_osstem` | binary NORIS vs osstem submodel | 0.9477 | 0.9473 | 0.9408 | 0.9396 | used for 10-label NORIS/osstem refinement |
| `outputs/submodels/20260507_051925_ADIN_MIS` | binary ADIN vs MIS submodel | 1.0000 | 1.0000 | 0.9839 | 0.9570 | available, not used in latest 10-label NORIS/osstem-only run |
| `20260507_134640_plain_vit_10label/two_stage_NORIS_osstem_test` | 10-label main + NORIS/osstem second stage | N/A | N/A | 0.9509 | 0.8786 | second stage evaluated 154 / 448 test samples |

Latest 10-label NORIS/osstem two-stage result:

- Main checkpoint: `outputs/20260507_134640_plain_vit_10label/checkpoints/best.pt`
- Submodel: `outputs/submodels/20260507_051743_NORIS_osstem/checkpoints/best.pt`
- Trigger: only first-stage predictions in `["NORIS", "osstem"]`
- Test samples: 448
- Second-stage samples: 154
- Baseline 10-label test accuracy: 0.9554
- Two-stage test accuracy: 0.9509
- Conclusion: applied successfully, but not beneficial on current test split.

## Visualization Artifacts

| Run | t-SNE | Attention |
|---|---|---|
| `20260507_095713_plain_vit_7label` | `embeddings/tsne_all.png` | `attention_maps/`, `metrics/attention_samples_test.csv` |
| `20260507_104805_plain_vit_3label` | `embeddings/tsne_all.png` | `attention_maps/`, `metrics/attention_samples_test.csv` |
| `20260507_134640_plain_vit_10label` | `embeddings/tsne_all.png` | `attention_maps/`, `metrics/attention_samples_test.csv` |
| `20260507_101426_supcon_vit_7label` | `embeddings/tsne_metric_7label.png`, `embeddings/tsne_metric_7to3.png` | metric attention script available; run-specific outputs may be under `attention_maps/<tag>/` when executed |

## Interpretation

- The strongest plain supervised model is the 10-label ViT by test accuracy: `0.9554`.
- The strongest 7-label plain classifier is `20260507_095713_plain_vit_7label`, not the earlier class-weight/strong-aug runs.
- SupCon metric learning has strong overall/prototype accuracy on 7 labels, but macro F1 is lower, suggesting class imbalance or minority-class confusion.
- Direct 7-label metric transfer to the canonical 3-label task is weak. The metric two-stage variant helps substantially but still underperforms the dedicated 3-label plain classifier.
- NORIS/osstem second-stage refinement did not improve the 10-label model on the current test split. The submodel is good in isolation, but main-model handoff plus submodel replacement introduced more harm than gain.

## Recommended Next Steps

1. Keep `20260507_134640_plain_vit_10label` as the current best 10-label supervised baseline.
2. Keep `20260507_095713_plain_vit_7label` as the current best plain 7-label baseline.
3. Keep `20260507_101426_supcon_vit_7label` for embedding/prototype analysis, not as a direct replacement for supervised classification.
4. Do not use NORIS/osstem two-stage by default unless thresholding is added.
5. If two-stage is revisited, trigger only low-confidence NORIS/osstem cases or only cases where main model’s top-2 labels are NORIS/osstem.
