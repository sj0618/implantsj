# Fresh 7-label SupCon → 3-label Mahalanobis 평가

## 결과 요약

7-label로만 SupCon 학습한 ViT-Small을 고정하고, 학습에 없던 `Bego·Bicon·ITI`의 target-train Mean prototype과 pooled covariance를 사용해 분류했다. 값은 `Accuracy / Macro-F1`(%)이다.

| Embedding | Euclidean | Manhattan | Cosine | Mahalanobis | Duplicate-free Mahalanobis |
|---|---:|---:|---:|---:|---:|
| SupCon projection 128-D | 71.97 / 50.17 | 71.75 / 51.51 | 71.97 / 50.17 | **93.27 / 81.46** | 93.20 / 81.45 |
| Backbone 384-D | 75.11 / 56.48 | 75.78 / 58.06 | 75.11 / 56.48 | **98.21 / 92.94** | 98.19 / 92.93 |

- SupCon이 직접 최적화한 projection 공간에서 Mahalanobis는 Cosine 대비 Accuracy `+21.30%p`, Macro-F1 `+31.29%p` 높았다.
- Backbone 공간에서는 Cosine 대비 Accuracy `+23.09%p`, Macro-F1 `+36.45%p` 높았다.
- test와 target train이 완전히 같은 이미지 5장을 제외해도 결과는 사실상 동일했다.

## 학습 조건

- Source: 7-label 3,279장 (`train 2,603 / valid 338 / test 338`)
- Source labels: `ADIN, DIONAVI, Dentium, MIS, NORIS, nobel, osstem`
- Model: `vit_small_patch16_224.augreg_in21k_ft_in1k`
- Loss: SupCon, 30 epoch, seed 42, projection 128-D
- Checkpoint selection: source validation Macro-F1 최고 epoch 26
- Source best validation: Accuracy `53.25%`, Macro-F1 `40.81%`
- Source test: Accuracy `54.73%`, Macro-F1 `41.19%`

## 3-label 평가 조건

- Target: cropped `Bego, Bicon, ITI` (`train 3,572 / valid 446 / test 446`)
- Prototype: target train의 클래스별 Mean embedding
- Covariance: target-train class residual만으로 추정한 pooled Ledoit-Wolf covariance
- Target valid/test는 prototype 및 covariance 추정에 사용하지 않음
- SciPy 직접 계산과 최대 절대오차: projection `3.20e-14`, backbone `2.84e-14`

## 해석 제한

Source 학습에 없던 label을 평가했으므로 unseen-label transfer에 해당한다. 다만 target train 정답으로 prototype과 covariance를 만들었기 때문에 strict zero-shot은 아니다. SupCon projection보다 backbone 성능이 높고 source 7-label 성능은 낮았다. 따라서 높은 3-label 점수는 SupCon 자체의 일반화보다 target-supervised Mahalanobis covariance의 영향이 컸다고 보는 편이 맞다.

과거 `34.86 / 29.26` 실험과는 checkpoint뿐 아니라 현재 target 데이터도 다르므로 직접적인 재현 비교가 아니다. 이번 실험은 사용자의 지시에 따라 checkpoint를 새로 학습한 fresh 평가다.
