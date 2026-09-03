# 멀티모달(ViT + VLM feature) 실험 현재 상태

측정 시각: 2026-09-03. 이 문서는 폐쇄망 안에서 같은 조사를 반복하지 않도록
바깥에서 확인한 사실만 적는다. 추정과 계획은 적지 않는다.

## 1. 결론

학습·평가 결과는 **아직 없다**. 코드와 입력 데이터는 이미지 안에 다 있다.

| 갈래 | 코드 | 결과 |
|---|---|---|
| VLM feature-only ML | 이 저장소에 없음 (이전 작업 환경) | run 디렉터리 없음. 아래 인용값만 존재 |
| Late fusion (5-feature) | `pipelines/vlm/train_vlm_fusion_5feature.py` | 0건 |
| Patch-token 주입 (메뉴 8 / 11) | `src/patch_feature.py` | 0건 |

`docs/reports/` 16개 디렉터리, `outputs*/`, git 전체 이력에서 `*_patch_*` run
디렉터리와 관련 REPORT 를 찾지 못했다.

문서에 남은 유일한 숫자는 `docs/analysis/PROJECT_ANALYSIS.md` 의 서술이다:
7-label VLM feature-only test accuracy 0.32~0.45, ViT image-only baseline 0.94+.
근거 run 디렉터리(`outputs/effective10_*`, `outputs/aihub_*`)는 이 저장소에
없으므로 재현으로 확인한 값이 아니다.

## 2. 이미지에 들어 있는 feature 파일

`configs/vlm/aihub_all_13feature_21reason_vlm.jsonl` (1.3MB)

`exp.conf` 의 `VLM_FEATURES` 는 `$DATA_ROOT` 에 같은 이름의 파일이 있으면
그것을 먼저 쓰고, 없으면 위 사본으로 넘어간다.

- 241 레코드 / provider `azure` / deployment `gpt-5.5` / 27 클래스
- `feature_vector` 63차원 = 21 feature x 3성분(value, assessable_mask, confidence)
- split 컬럼은 전부 비어 있다

## 3. feature 품질 (241건 집계)

```
feature_status    usable 102 (42%) | partial_usable 49 (20%) | low_confidence 90 (37%)
usable_for_vector True 151 (63%) | False 90 (37%)
failure_reason    none 102 | connection_obscured 49 | all_features_not_assessable 90
supporting 상태    present 1931 | absent 829 | not_assessable 2301 (45.5%)
평균 confidence    0.496
```

core 13 attribute 의 `unknown` 비율. `Company` / `Name` / `Connection_2` 는
전량 `unknown` 이다. 브랜드명을 묻지 않도록 설계한 결과이므로 정상이다.

```
Company 100%   Name 100%   Connection_2 100%
Connection_1 67.2%   Flange 64.7%   Thread_Shape 39.0%   System 38.6%
Collar 38.6%   Microthread 37.8%   Body_Shape 37.3%   Body_Type 37.3%
Apex_Shape 37.3%   Apex_Hole 37.3%
```

supporting 21 feature 중 `not_assessable` 이 많은 순서. connection 부위에
몰려 있다.

```
internal_small_void_present                62.7%
negative_internal_connection_space_present 62.2%
straight_screw_entry_space_present         60.6%
positive_connection_region_present         59.8%
platform_step_present                      58.9%
platform_switching_step_present            58.9%
connection_region_visible                  57.7%
```

## 4. 지금 그대로 메뉴 8 을 돌리면 막히는 지점

feature jsonl 의 경로는 `data/AI Hub Data/03_Osstem_SSII/...` 이고 클래스는
AI Hub 27종이다. 반면 manifest 의 경로는 `/workspace/data/<데이터셋>/<split>/<클래스>/...`
이고 클래스는 7 / 11-label 계열이다.

`src/patch_feature.py` 의 조회는 전체 경로뿐 아니라 파일명(`name`)과
확장자 뺀 이름(`stem`) 으로도 매칭하므로 경로 접두어가 달라도 파일명이 같으면
붙는다. 파일명 자체가 다르면 붙지 않는다.

커버리지는 학습 시작 전에 split 별로 출력되고, `--feature-mode patch` 에서
train 커버리지가 0 이면 학습을 시작하지 않고 종료한다. 그러니 메뉴 8 을
그냥 실행해서 출력되는 커버리지 숫자를 먼저 보면 된다. 조용히 전부 null
토큰으로 학습되는 일은 없다.

커버리지가 낮을 때 선택지는 두 가지다.

1. 현재 데이터셋에 VLM feature 추출을 다시 돌린다 (`pipelines/vlm/`).
2. jsonl 의 경로를 manifest 의 파일명 표기에 맞춰 다시 쓴다.

## 5. 규모 참고

feature 는 241건이다. 이 프로젝트의 데이터셋은 `7label_cropped` 8,185장,
`11label_crop_final_8_1_1` 2,797장 규모다.

별도로 3-label VLM 추출은 `VLM_3LABEL_APPLY_STATUS.md` 기준 62 / 2,075 에서
멈춰 있고, 그 62건의 `feature_status` 는 `low_confidence` 14 + `partial_usable` 48
로 `usable` 이 0건이다.

## 6. 대조군

`--feature-mode` 로 4종을 고른다. feature 를 넣어 좋아졌다는 주장은 대조군
없이 성립하지 않는다. 토큰이 21개 늘면 파라미터도 함께 늘기 때문이다.

```
patch     본 실험. feature 를 패치 토큰으로 주입
none      이미지만. 토큰 자체를 안 붙임
zero      토큰은 붙이되 입력을 0 으로. 늘어난 파라미터 효과만 분리
shuffle   feature 를 다른 이미지 것과 섞음. 진짜 쓰는지 검증
```

`./run.sh patch <데이터셋> <모델> [patch|none|zero|shuffle]`
