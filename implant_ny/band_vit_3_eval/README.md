# band_vit_3_eval

`band_vit_3`을 수정하지 않는 독립 평가 패키지. `band_vit_3`과 같은 위치
(`/workspace/implant_seg`)에 폴더째 두고 실행하면 됩니다.
`segmentation.py`는 기존 `band_vit_3`의 것을 그대로 import해서 사용합니다.

저장된 체크포인트를 불러와 test 이미지로 추론/분석만 수행합니다(재학습 없음).

## input modes

- `--input-mode band` (기본): 학습 때와 동일한 밴드 전용 crop. 기존 평가와 같음.
- `--input-mode original`: 픽셀을 지우지 않은 원본 픽셀 입력.
  crop 기하(밴드 bbox)는 band 모드와 동일하게 맞춰서 임플란트 위치·스케일이
  학습 입력과 정렬됨 — attention 변화가 순수하게 "마스킹 제거" 효과가 되도록.

두 모드 모두 SAM2 밴드는 계산되며(original 모드에선 픽셀에 적용하지 않음),
attention/rollout 질량이 밴드에 놓인 비율을 정량화합니다:

- `attention_band_mass` / `rollout_band_mass`: 밴드 패치 위의 attention 비율
- `band_patch_fraction`: 밴드 면적 비율 (균등 분포 기준선)
- `focus_gain` = band mass / band fraction, **>1이면 경계에 집중**

샘플별 값은 `predictions.csv`/`index.csv`, 집계(평균·중앙값·gain>1 비율)는
`metrics.json`의 `band_focus`에 저장됩니다.

## 실행 예시

```bash
cd /workspace/implant_seg
source .venv-vit-sam2/bin/activate

# gradient rollout (모델이 보는 위치 분석에 권장)
python -m band_vit_3_eval.gradient_rollout \
  --checkpoint implant_outputs/band_vit3_11label/checkpoints/best.ckpt \
  --manifest /workspace/data/11label_crop_final_8_1_1/test.csv \
  --output-dir implant_outputs/band_vit3_11label/rollout_original \
  --input-mode original --rollout-samples -1

# attention 기반 평가
python -m band_vit_3_eval.evaluation \
  --checkpoint implant_outputs/band_vit3_11label/checkpoints/best.ckpt \
  --manifest /workspace/data/11label_crop_final_8_1_1/test.csv \
  --output-dir implant_outputs/band_vit3_11label/eval_original \
  --input-mode original --attention-samples -1
```

`--input-mode band`로 동일 명령을 한 번 더 실행해 성능 하락 폭과
`focus_gain` 변화를 비교하세요. focus_gain이 여전히 1을 크게 넘고 정확도
하락이 작으면 "원본에서도 경계에 집중한다"는 근거가 됩니다.
