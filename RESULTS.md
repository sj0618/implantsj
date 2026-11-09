# implantsj 실험 결과 정리

마지막 갱신: 2026-11-06 KST (6~10장 추가: 9/28~10/01 서버 후속 결과, 9월 초 폐쇄망 결과, 5~8월 로컬 결과, 엑셀 속성 집계, 보고서 목록). 1~5장 서버 확인: 2026-09-26 13:50 KST. 6장 서버 확인: 2026-09-30 12:28–12:46 KST, 2026-10-01 07:58–08:25 KST. **실험 요약은 이 파일에서 관리합니다.**
기존 실험 공통: AI-Hub 임플란트 42클래스, test split, seed 1개, 신뢰구간 없음. 1~5장 수치는 정확도(0~1), 6장 이후는 %(표마다 단위 표시). 신규 속성 토큰 실험은 라벨을 통합한 41클래스 사용.
세부 근거: `docs/handoff_20260924/`. 보고서 원문 목록은 [10장](#10-보고서-목록). 이번 갱신에서 새로 학습하거나 재계산한 수치는 없다. 모두 아래 출처 문서의 값을 옮겼다.

---

## 0. 한눈에

| # | 실험 | 한 줄 결과 |
|---|---|---|
| 1 | 백본 분류 (closed-set) | 27클래스 최고 Band-DINOv2 0.952, 42클래스 최고 Band-ViT 0.959 |
| 2 | Metric learning, closed27 학습 → open15 인식 (서버 10개 실행) | 전체 Acc 최고 0.909 (학습 없는 백본 + Maha, galcov). unseen Acc 최고 0.821 / 전체 macro-F1 0.772 (SupCon 5ep + kmeans + Maha, galcov). CovAware·triplet 실패 |
| 3 | 속성 모델 (closed27 학습) | seen 속성 정확도 0.94~1.0, unseen 그룹 매칭 0.025 (무작위 수준) |
| 4 | 속성 + 임베딩 결합 | 속성을 더하면 seen↑ unseen↓ |
| 5 | 속성 unseen 실패 원인 | GS III(unseen의 41%)가 TS III(seen) 속성으로 예측됨. TS III 스펙 라벨 오류가 주원인 |
| 6 | [41클래스 DINOv2 + 8개 속성 토큰 공동 학습](docs/attribute_tokens_41class_20260927.md) | 이미지 패치와 속성 토큰을 모델 내부에서 결합. 9/27 12:33 KST 기준 학습 시작 확인. **9/30 확인 결과(6-4):** K-means OAS unseen 5ep 78.13%, 20ep 73.05%로 이미지 단독(79.03%)보다 높지 않음 |
| 7 | ViT·Band-ViT 백본 + OAS, gallery 공분산 (9/28 실행, 6-1) | ViT K-means 5: unseen 85.18%, seen 94.46%, 전체 93.77% (기록된 unseen 최고). Mean 중심 1개는 unseen 79.52% |
| 8 | ViT SupCon 5ep 그리드 (9/30 완료, 6-2) | K-means + OAS: unseen 75.35%, 전체 84.75%. 공분산 적합 범위 기록 없음 |
| 9 | Gap-SupCon, DINOv2 (9/29 실행, 6-3) | pooled OAS unseen 50.45%. 같은 조건 SupCon 단독 대조군이 중단돼 Gap 효과는 판단 불가 |
| 10 | 41클래스 예측 속성 융합 (6-4) | unseen 37.51~38.74%로 하락. 예측 속성 8개만으로는 전체 16.04% |
| 11 | 9월 초 폐쇄망 ViT 임베딩 평가 (7장) | 학습 없는 ViT Mean + Maha 전체 60.21%(27+15). 27클래스 FT ViT는 전체 87.60%, unseen 57.58% |
| 12 | 5~8월 로컬 실험 (8장) | closed-set plain ViT 95.54%. 7→3 전이에서 cosine을 공분산 거리로 바꾸면 크게 상승(예: 75.11% → 98.21%). 클래스당 5~10장으로 공분산까지 추정하면 60~70%대 |
| 13 | 엑셀 속성표 구분 불가 (11/2 집계, 9장) | 연세대 8/4판 기준 42클래스 중 4그룹 9개 클래스의 속성값이 완전히 같음 |

---

## 1. 백본 분류 (closed-set 분류기)

실험: 분류 헤드로 학습·평가. unseen 개념 없음. 출처 `implantsj/implant_experiment_summary_table.md`.

| 모델 | 27클래스 Acc | 27 Macro-F1 | 42클래스 Acc | 42 Macro-F1 |
|---|---:|---:|---:|---:|
| Band-DINOv2 | **0.9524** | **0.9024** | 0.9565 | 0.9188 |
| ViT | 0.9515 | 0.9012 | 0.9563 | 0.9194 |
| Band-ViT | 0.9494 | 0.8981 | **0.9589** | 0.9197 |
| DINOv2 | 0.9377 | 0.8754 | 0.9369 | 0.8900 |
| ViT-Large | – | – | 0.9544 | **0.9200** |

### 1-1. 클래스별 정답률 (서버 test 예측에서 직접 계산)

출처: 서버 `~/aihub_run_new/outputs/backbone/*/*/embeddings/*_test_head.npz`의 `labels`와 `features__logits` argmax (2026-09-26 추출). test 16,570장, 42클래스.
27클래스 모델은 closed-set이라 unseen(open15) 클래스는 맞힐 수 없으므로 "–". 42클래스 모델은 open15도 학습함(U는 참고 표시).
열: 27클래스 모델 = BD27 Band-DINOv2 · BV27 Band-ViT · D27 DINOv2 · V27 ViT / 42클래스 모델 = BD42 · BV42 Band-ViT · D42 · V42 · BVL42 Band-ViT-L · VL42 ViT-L. "BV42 최다 오답" = Band-ViT 42에서 가장 많이 틀린 예측과 장수. BV42 오름차순.

재계산 전체 정확도: BD42 0.9565 · BV42 0.9581 · D42 0.9369 · V42 0.9563 · BVL42 0.9581 · VL42 0.9544 (문서 값과 ±0.001 이내).

| 클래스 | U | n | BD27 | BV27 | D27 | V27 | BD42 | BV42 | D42 | V42 | BVL42 | VL42 | BV42 최다 오답 |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| it |  | 45 | .33 | .44 | .29 | .47 | .38 | .42 | .27 | .40 | .42 | .60 | internal 25 |
| iu |  | 6 | .50 | .33 | .17 | .50 | .50 | .50 | .33 | .50 | .50 | .33 | internal 2 |
| screw vent | U | 2 | – | – | – | – | .50 | .50 | .50 | .50 | 1.00 | 1.00 | TSV 1 |
| stella | U | 3 | – | – | – | – | .67 | .67 | .67 | 1.00 | .67 | .67 | SS III 1 |
| Anyone external |  | 256 | .72 | .67 | .74 | .70 | .80 | .74 | .76 | .80 | .75 | .76 | Anyone internal 65 |
| internal |  | 578 | .77 | .65 | .71 | .76 | .78 | .76 | .71 | .72 | .76 | .75 | Hexplant 124 |
| Simpleline | U | 5 | – | – | – | – | .80 | .80 | .80 | .60 | 1.00 | .80 | branemark 1 |
| SS III | U | 16 | – | – | – | – | .88 | .81 | .75 | .88 | .81 | .81 | SS II 3 |
| Hexplant |  | 884 | .83 | .87 | .81 | .80 | .81 | .85 | .79 | .85 | .84 | .82 | internal 116 |
| Anyone internal |  | 381 | .83 | .86 | .79 | .88 | .81 | .86 | .80 | .80 | .83 | .79 | Anyone external 48 |
| 3i | U | 23 | – | – | – | – | .91 | .91 | .91 | .91 | .91 | .91 | US II 1 |
| US III |  | 95 | .87 | .91 | .85 | .94 | .93 | .93 | .82 | .93 | .94 | .92 | US II 4 |
| IS II |  | 289 | .91 | .93 | .88 | .94 | .92 | .93 | .86 | .93 | .93 | .91 | IS I 16 |
| TS standard |  | 134 | .92 | .93 | .90 | .95 | .95 | .93 | .92 | .93 | .94 | .94 | Bone level 4 |
| SS II |  | 84 | .92 | .93 | .89 | .95 | .94 | .95 | .90 | .98 | .96 | .95 | US II 3 |
| IS III |  | 814 | .98 | .96 | .94 | .98 | .97 | .96 | .95 | .98 | .98 | .98 | IS I 11 |
| GS II |  | 174 | .98 | .97 | .98 | .98 | .99 | .97 | .95 | .96 | .98 | .97 | GS III 3 |
| TS standard plus |  | 106 | .95 | .98 | .96 | .92 | .96 | .97 | .96 | .94 | .98 | .95 | Bone level 1 |
| any ridge |  | 36 | .97 | .97 | .97 | .97 | .97 | .97 | .97 | .97 | .97 | .97 | Anyone external 1 |
| UF2 |  | 74 | .97 | .97 | .93 | .93 | .97 | .97 | .96 | .99 | .99 | .97 | IS I 1 |
| IS I |  | 785 | .97 | .96 | .95 | .95 | .97 | .97 | .95 | .96 | .97 | .97 | IS II 11 |
| Superline |  | 2195 | .97 | .97 | .96 | .96 | .98 | .98 | .96 | .98 | .98 | .97 | implantium 34 |
| GS III | U | 498 | – | – | – | – | .98 | .98 | .97 | .98 | .98 | .98 | GS II 6 |
| luna | U | 339 | – | – | – | – | .98 | .98 | .92 | .99 | .99 | .98 | TS III 5 |
| core1 | U | 46 | – | – | – | – | 1.00 | .98 | .96 | .98 | .96 | .96 | Superline 1 |
| Replace select | U | 191 | – | – | – | – | .97 | .98 | .95 | .97 | .98 | .96 | branemark 2 |
| EB |  | 241 | .99 | .99 | .95 | .98 | .99 | .98 | .95 | .99 | .99 | .99 | IS I 4 |
| exfeel internal | U | 54 | – | – | – | – | .98 | .98 | .96 | .98 | .98 | .96 | Anyone internal 1 |
| implantium |  | 1917 | .97 | .98 | .97 | .98 | .98 | .98 | .96 | .98 | .98 | .98 | Superline 21 |
| branemark |  | 366 | .99 | .99 | .97 | .99 | .98 | .98 | .97 | .98 | .98 | .98 | US II 3 |
| Bone level |  | 262 | .98 | .96 | .97 | .98 | .99 | .98 | .95 | .98 | .98 | .98 | TS standard 3 |
| US II |  | 967 | .99 | .99 | .98 | .99 | .99 | .99 | .98 | .99 | .99 | .99 | TS III 3 |
| exfeel external |  | 110 | 1.00 | .99 | .99 | .99 | 1.00 | .99 | .97 | .99 | .99 | .99 | US II 1 |
| astra |  | 1399 | .99 | .99 | .98 | .99 | .99 | .99 | .98 | .99 | .99 | .99 | Superline 3 |
| TS III |  | 2977 | .99 | .99 | .99 | .99 | .99 | .99 | .99 | 1.00 | .99 | .99 | US II 4 |
| ankylos | U | 4 | – | – | – | – | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | – |
| TSV | U | 20 | – | – | – | – | 1.00 | 1.00 | .95 | 1.00 | 1.00 | 1.00 | – |
| S clean tapered | U | 11 | – | – | – | – | 1.00 | 1.00 | .91 | 1.00 | 1.00 | 1.00 | – |
| atlas internal | U | 7 | – | – | – | – | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | – |
| Xive |  | 93 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .99 | 1.00 | 1.00 | 1.00 | – |
| Biohorizon external | U | 2 | – | – | – | – | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | – |
| UF |  | 81 | .99 | .99 | .99 | .98 | .99 | 1.00 | .95 | 1.00 | .99 | .98 | – |

**많이 틀리는 클래스 (백본)**
- 오답은 몇 개 쌍에 몰림: **internal↔Hexplant**(124/116장), **Anyone external↔Anyone internal**(65/48장), **it→internal**(25/45장), IS II→IS I(16장). 모델 10개 모두에서 같은 클래스가 하위권.
- it(0.27~0.60), iu(n=6), Anyone external·internal(0.67~0.88), internal(0.65~0.78), Hexplant(0.79~0.87)이 모든 백본에서 0.9 미만.
- 42클래스 모델(BV42)에서 open15 클래스는 15개 중 10개가 0.95 이상. 낮은 것은 screw vent·stella·Simpleline·SS III·3i로 모두 test 23장 이하. 표본 적은 클래스(screw vent 2, stella 3, Simpleline 5, iu 6)는 수치 흔들림 큼.

---

## 2. Metric learning: closed27 학습 → open15(unseen) 인식

**서버 기준 (2026-09-26 13:50 KST 확인)**: `~/aihub_run_new/outputs/metric/closed27_open15/*/*/eval/metrics_test_*.json` (실행 10개), 클래스별은 같은 폴더 `confusion_*.csv`.
실험: closed 27클래스로 임베딩 학습 → 42클래스 prototype 최근접 판정. test split, seed 1개.
Σ(Mahalanobis 공분산) 추정: **closed-only** = closed27 train만 / **galcov** = gallery(closed27+open15 train). 조건이 달라 두 그룹은 직접 비교 금지.
표기: Acc = 정확도, F1 = macro-F1. 전체 = 42클래스, seen = closed27, unseen = open15.

### 2-1. 실행 목록

| # | 실행 (서버 폴더) | 학습 | Σ | 상태 |
|---|---|---|---|---|
| M1 | `backbone_eval/dinov2_none` | 없음 (ft DINOv2 백본만) | closed-only | 완료 (kmeans 조합 없음) |
| M2 | `dinov2_full_finetune_5ep/supcon` | SupCon 5ep | closed-only | 완료 |
| M3 | `dinov2_full_finetune_5ep/arcface` | ArcFace 5ep | closed-only | 완료 |
| M4 | `dinov2_full_finetune_5ep/triplet` | Triplet 5ep | closed-only | 학습 붕괴 |
| M5 | `head_only/vit_supcon` | ViT head만 SupCon | closed-only | 완료 |
| M6 | `backbone_eval_galcov/dinov2_none` | 없음 (ft DINOv2 백본만) | galcov | 완료 |
| M7 | `dinov2_galcov_timm_5ep/supcon` | SupCon 5ep | galcov | 완료 |
| M8 | `dinov2_galcov_timm_5ep/arcface` | ArcFace 5ep | galcov | 완료 |
| M9 | `dinov2_galcov_timm_5ep/triplet` | Triplet 5ep | galcov | 학습 붕괴 |
| M10 | `dinov2_covaware_gradckpt_20260924_2124_r2/covaware_supcon` | CovAware SupCon | 폴더명으로 확인 불가 | 학습 실패 (로컬 문서에 없던 실행) |

### 2-2. 결과 (판정 방식 3종)

| 실행 | 판정 | 전체 Acc | 전체 F1 | seen Acc | seen F1 | unseen Acc | unseen F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| M1 백본, closed-only | mean+cosine | 0.879 | 0.611 | 0.929 | 0.554 | 0.246 | 0.091 |
| | mean+Maha | 0.886 | 0.616 | 0.935 | 0.555 | 0.269 | 0.094 |
| M6 백본, galcov | mean+cosine | 0.879 | 0.611 | 0.929 | 0.554 | 0.246 | 0.091 |
| | mean+Maha | 0.909 | 0.724 | 0.926 | 0.552 | 0.702 | 0.215 |
| | kmeans+Maha | 0.907 | 0.756 | 0.920 | 0.550 | 0.749 | 0.261 |
| M2 SupCon, closed-only | mean+cosine | 0.865 | 0.607 | 0.908 | 0.541 | 0.333 | 0.094 |
| | mean+Maha | 0.884 | 0.688 | 0.911 | 0.544 | 0.539 | 0.177 |
| | kmeans+Maha | 0.844 | 0.747 | 0.852 | 0.521 | 0.747 | 0.279 |
| **M7 SupCon, galcov** | mean+cosine | 0.866 | 0.613 | 0.909 | 0.545 | 0.327 | 0.095 |
| | mean+Maha | 0.898 | 0.733 | 0.911 | 0.548 | 0.731 | 0.221 |
| | **kmeans+Maha** | 0.854 | **0.772** | 0.857 | 0.526 | **0.821** | **0.298** |
| M3 ArcFace, closed-only | mean+cosine | 0.836 | 0.555 | 0.870 | 0.466 | 0.403 | 0.135 |
| | mean+Maha | 0.860 | 0.639 | 0.881 | 0.489 | 0.601 | 0.209 |
| | kmeans+Maha | 0.764 | 0.623 | 0.771 | 0.439 | 0.676 | 0.258 |
| M8 ArcFace, galcov | mean+cosine | 0.845 | 0.578 | 0.878 | 0.489 | 0.428 | 0.126 |
| | mean+Maha | 0.868 | 0.677 | 0.883 | 0.500 | 0.681 | 0.226 |
| | kmeans+Maha | 0.787 | 0.668 | 0.790 | 0.458 | 0.741 | 0.277 |
| M5 ViT head-only | mean+cosine | 0.480 | 0.362 | 0.489 | 0.325 | 0.368 | 0.125 |
| | mean+Maha | 0.557 | 0.435 | 0.558 | 0.352 | 0.539 | 0.215 |
| | kmeans+Maha | 0.544 | 0.479 | 0.536 | 0.335 | 0.640 | 0.258 |
| M4 Triplet, closed-only | mean+Maha | 0.124 | 0.092 | 0.115 | 0.076 | 0.243 | 0.075 |
| M9 Triplet, galcov | mean+Maha | 0.105 | 0.079 | 0.096 | 0.066 | 0.217 | 0.071 |
| M10 CovAware | mean+Maha | 0.335 | 0.245 | 0.326 | 0.191 | 0.454 | 0.175 |
| | kmeans+Maha | 0.353 | 0.292 | 0.337 | 0.206 | 0.546 | 0.221 |

Maha = Mahalanobis(Ledoit-Wolf). 38개 조합 전체 중 전체 Acc 최고 조합은 대부분 위 mean+Maha와 같거나 OAS 공분산(차이 ≤0.001).

- **전체 Acc 최고**: M6 학습 없는 백본 + mean+Maha(OAS) 0.909. **unseen Acc·전체 F1 최고**: M7 SupCon + kmeans+Maha (unseen 0.821, 전체 F1 0.772).
- cosine → Mahalanobis: unseen Acc가 크게 오름 (M7 0.33 → 0.73). kmeans 프로토타입: unseen↑, seen Acc↓.
- galcov: unseen만 오르고 seen은 거의 그대로 (M1→M6 unseen 0.27 → 0.70).
- seen F1(0.52~0.55)이 seen Acc(0.85~0.93)보다 크게 낮음 → 표본 적은 seen 클래스에서 많이 틀림 (2-3 표 참고).
- seed 1개라 조합 간 소수 둘째 자리 차이는 유의성 미확인.

> **2026-09-30 추가 확인 (F1 집계 범위):** 9/28 GalCov 평가에 쓴 원격 `evaluate.py`(`20260924-galcov-5ep`)의 `_metric_block`은 seen·unseen 부분집합을 계산할 때도 전체 클래스 목록에 unknown을 더해 넘긴다([remote_results.md 2장](docs/experiment_audit_20260930/remote_results.md)). 이 표의 실행이 같은 evaluator로 계산됐다면 seen F1·unseen F1은 해당 그룹 클래스만의 Macro-F1이 아니다. 위 "seen F1이 낮다"는 해석도 이 집계 범위 때문일 수 있으므로, 그룹 간 비교는 정확도로 한다.

### 2-3. 클래스별 정답률 (test, 서버 confusion 행 정규화)

열: **A** = M6 백본 kmeans+Maha · **B** = M7 SupCon kmeans+Maha (unseen 최고, 정렬 기준) · **C** = M7 SupCon mean+Maha · **D** = M2 SupCon kmeans+Maha (closed-only) · **E** = M1 백본 mean+Maha (closed-only).
U = unseen(open15). "B 최다 오답" = B에서 가장 많이 틀린 예측 클래스와 장수. B 정답률 오름차순.

| 클래스 | U | n | A | B | C | D | E | B 최다 오답 |
|---|:-:|---:|---:|---:|---:|---:|---:|---|
| stella | U | 3 | 0.33 | 0.33 | 0.67 | 0.67 | 0.33 | TS III 1 |
| Anyone internal | | 381 | 0.79 | **0.50** | 0.71 | 0.55 | 0.81 | Anyone external 187 |
| iu | | 6 | 0.50 | 0.50 | 0.50 | 0.33 | 0.50 | internal 3 |
| it | | 45 | 0.49 | 0.51 | 0.56 | 0.49 | 0.38 | internal 17 |
| Hexplant | | 884 | 0.78 | **0.55** | 0.70 | 0.55 | 0.80 | internal 353 |
| IS II | | 289 | 0.87 | 0.59 | 0.88 | 0.64 | 0.88 | IS I 52 |
| internal | | 578 | 0.69 | 0.60 | 0.57 | 0.63 | 0.70 | Hexplant 185 |
| SS III | U | 16 | 0.62 | 0.62 | 0.44 | 0.69 | 0.12 | SS II 3 |
| implantium | | 1917 | 0.96 | 0.76 | 0.96 | 0.71 | 0.97 | Superline 209 |
| Replace select | U | 191 | 0.72 | 0.76 | 0.57 | 0.70 | 0.18 | astra 9 |
| luna | U | 339 | 0.70 | 0.80 | 0.68 | 0.72 | 0.19 | TS III 28 |
| Anyone external | | 256 | 0.74 | 0.80 | 0.70 | 0.71 | 0.73 | Anyone internal 50 |
| 3i | U | 23 | 0.74 | 0.83 | 0.39 | 0.78 | 0.13 | US II 3 |
| GS III | U | 498 | 0.80 | 0.84 | 0.85 | 0.73 | 0.35 | GS II 47 |
| TSV | U | 20 | 0.60 | 0.85 | 0.65 | 0.80 | 0.25 | astra 2 |
| IS I | | 785 | 0.94 | 0.85 | 0.85 | 0.80 | 0.95 | IS II 50 |
| exfeel internal | U | 54 | 0.81 | 0.87 | 0.74 | 0.91 | 0.41 | Anyone external 3 |
| US III | | 95 | 0.80 | 0.89 | 0.93 | 0.93 | 0.86 | US II 7 |
| Superline | | 2195 | 0.95 | 0.90 | 0.92 | 0.91 | 0.96 | implantium 148 |
| S clean tapered | U | 11 | 0.91 | 0.91 | 0.73 | 1.00 | 0.27 | core1 1 |
| TS standard | | 134 | 0.89 | 0.91 | 0.96 | 0.91 | 0.90 | TS standard plus 6 |
| IS III | | 814 | 0.92 | 0.91 | 0.94 | 0.91 | 0.94 | IS II 39 |
| TS standard plus | | 106 | 0.96 | 0.93 | 0.94 | 0.91 | 0.96 | TS standard 4 |
| GS II | | 174 | 0.94 | 0.94 | 0.89 | 0.94 | 0.95 | GS III 7 |
| TS III | | 2977 | 0.96 | 0.94 | 0.97 | 0.95 | 0.98 | core1 58 |
| core1 | U | 46 | 0.78 | 0.96 | 0.85 | 1.00 | 0.33 | IS III 1 |
| astra | | 1399 | 0.96 | 0.96 | 0.99 | 0.98 | 0.98 | Superline 30 |
| US II | | 967 | 0.95 | 0.96 | 0.97 | 0.94 | 0.97 | 3i 18 |
| any ridge | | 36 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | Anyone external 1 |
| SS II | | 84 | 0.75 | 0.98 | 1.00 | 0.94 | 0.87 | SS III 2 |
| branemark | | 366 | 0.92 | 0.98 | 0.99 | 0.97 | 0.97 | Replace select 4 |
| EB | | 241 | 0.98 | 0.98 | 0.99 | 0.98 | 0.98 | IS I 4 |
| Bone level | | 262 | 0.97 | 0.98 | 0.98 | 0.97 | 0.97 | TS standard 2 |
| UF2 | | 74 | 0.93 | 0.99 | 1.00 | 0.96 | 0.93 | Anyone external 1 |
| exfeel external | | 110 | 0.98 | 0.99 | 1.00 | 1.00 | 0.99 | 3i 1 |
| UF | | 81 | 0.99 | 1.00 | 1.00 | 1.00 | 0.99 | – |
| Biohorizon external | U | 2 | 0.50 | 1.00 | 1.00 | 0.50 | 0.00 | – |
| Xive | | 93 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | – |
| ankylos | U | 4 | 0.75 | 1.00 | 0.75 | 1.00 | 0.00 | – |
| Simpleline | U | 5 | 0.40 | 1.00 | 0.60 | 0.80 | 0.20 | – |
| atlas internal | U | 7 | 0.71 | 1.00 | 0.57 | 0.71 | 0.43 | – |
| screw vent | U | 2 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | – |

**많이 틀리는 클래스 (B 기준)**
- seen 쌍 혼동이 가장 큼: Hexplant↔internal(353/185장), Anyone internal→external(187), IS II→IS I(52), implantium→Superline(209). 네 클래스 모두 백본만 쓴 A·E에서는 0.78~0.97이라 **SupCon+kmeans가 seen 혼동을 만든 것**.
- unseen 중 n이 큰 클래스: GS III 0.84(→GS II 47), luna 0.80, Replace select 0.76. closed-only 백본(E)에서는 0.18~0.35 → Σ·학습으로 크게 개선.
- unseen 15개 중 8개는 test가 23장 이하(stella 3, screw vent 2, Biohorizon 2, ankylos 4, Simpleline 5, atlas 7, S clean 11, SS III 16): 클래스별 수치 신뢰도 낮음.

### 2-3b. 백본 vs metric: 같은 클래스에서 틀림

| 클래스 | 백본 BV42 | metric M6 백본 kmeans | metric M7 SupCon kmeans | 주 혼동 |
|---|---:|---:|---:|---|
| it | 0.42 | 0.49 | 0.51 | → internal |
| Anyone internal | 0.86 | 0.79 | **0.50** | → Anyone external |
| Anyone external | 0.74 | 0.74 | 0.80 | → Anyone internal |
| internal | 0.76 | 0.69 | 0.60 | → Hexplant |
| Hexplant | 0.85 | 0.78 | **0.55** | → internal |
| IS II | 0.93 | 0.87 | **0.59** | → IS I |
| implantium | 0.98 | 0.96 | 0.76 | → Superline |

- 두 방식 모두 **같은 쌍(internal↔Hexplant, Anyone internal↔external, it→internal, IS II→IS I)**에서 틀림. 분류기가 이미 헷갈리는 쌍을 SupCon+kmeans가 더 크게 섞음(Anyone internal 0.86→0.50, Hexplant 0.85→0.55).
- [추정] 이 쌍들은 같은 제조사 계열의 외형이 비슷한 제품. 이미지로 구분할 단서가 적거나 라벨이 섞였을 가능성 → 해당 쌍의 오답 이미지 육안 확인 필요.

**스펙표 비교 (혼동 쌍)** — 출처: `docs/handoff_20260924/implant_labeling_original.xlsx`(원본), `implant_attributes_corrected_conservative.csv`(보수적 수정). 다른 값만 표시, 나머지 속성은 모두 같음.

| 쌍 | 다른 속성 | 같은 속성 |
|---|---|---|
| Hexplant(Warantec Explant) ↔ internal(Warantec Internal) | Connection1 External↔Internal, Connection2 HEX↔OCTA, Collar No↔Yes | Bone Level, Convergent, Microthread Yes, Tapered body, V-shaped, Rounded, Apex hole No |
| Anyone external ↔ Anyone internal (Megagen) | Connection1 External↔Internal, Flange Convergent↔Parallel, (원본만) Collar No↔Yes | Bone Level, HEX, Microthread No, Tapered apex, V-shaped, Flat, Apex hole No |
| it(Warantec IT) ↔ internal(Warantec Internal) | 원본: **차이 없음**. 수정본: Thread V-shaped→Square(IT만) | 나머지 전부 |

- 세 쌍 모두 같은 회사 제품이고, 차이는 대부분 연결부(내부/외부, HEX/OCTA)와 칼라뿐. [추정] 연결부는 픽스처 상단 안쪽 구조라 방사선 영상에서 잘 안 보여서 이미지로 구분이 어렵다.
- it와 internal은 원본 스펙이 완전히 같아 검증 문서에서도 같은 제품일 가능성을 지적함(`implant_attribute_verification_20260924.md` 9번). 라벨 중복 여부 확인 필요.
- 서버 모델이 쓴 스펙표(`spec_table.csv`)는 원본 값 기준으로 보임(TS III가 Non-tapered로 남아 있었음). 서버 표와 대조는 안 함.

**internal vs it 라벨 확인 (서버 `data/manifest_full42.csv` 파일명, 2026-09-26)** — 파일명 필드: `ID-P/S-제조사-제품-직경-길이-(나이?)-날짜1-날짜2-h코드` [필드 의미는 추정]

| 클래스 | 장수 | 제조사/제품 코드 | 직경 상위 | 길이 상위 | h코드 |
|---|---:|---|---|---|---|
| internal(in) | 5,193 | wa / in | 4.3, 5.3, 4.8, 6.3, 4.5 | 10.0, 8.5, 11.5, 13.0, 7.0 | h02 5,191 |
| it(it) | 401 | wa / it | 4.3, 5.3, 4.5, 4.8 | 8.5, 10.0, 11.5, 13.0, 7.0 | **h01 401 (전부)** |
| Hexplant(ex) | 7,948 | wa / ex | 4.3, 5.3, 4.5, 4.0, 4.8 | 10.0, 8.5, 11.5, 13.0, 7.0 | h02 7,622, h05 324 |

- internal과 it는 같은 제조사(wa=Warantec), 같은 직경·길이 규격, 겹치는 날짜 구간(2009~2020, 20120217 등 같은 날짜도 있음). 차이는 **h코드(기관 추정)뿐**: it는 전부 h01, internal은 거의 전부 h02.
- [추정] 같은 제품(Warantec IT, 내부 OCTA)을 기관마다 다른 이름으로 라벨링한 것일 가능성 높음. 모델이 it를 일부 맞히는 것도 제품 차이가 아니라 기관별 촬영 특성을 잡았을 수 있음.
- 결정 필요: internal+it 병합 여부(병합 시 42→41클래스, 기존 결과 재계산 필요).

**원본 라벨 JSON 확인 (서버 `~/workspace/data/raw/`, 2026-09-26)**

| 파일 | 일련번호 | 촬영방법 | 제조사 | 모델명 | 직경/길이 | 식립위치 | 식립일자 | 촬영일자 | 촬영병원 |
|---|---|---|---|---|---|---|---|---|---|
| `TL1/h02/h02_17/<ID>-S-wa-in-…-h02.json` | `<ID>` | 치근단 | 워렌텍(wa) | **internal(in)** | 4.3 / 10.0 | `<위치>` | `<날짜A>` | `<날짜B>` | 세브란스(강남) |
| `VL1/h01/h01_13/<ID>-S-wa-it-…-h01.json` | `<ID>` | 치근단 | 워렌텍(wa) | **it(it)** | 4.3 / 10.0 | `<위치>` | `<날짜A>` | `<날짜B>` | 세브란스(신촌) |

\* 2026-11-06 공유 저장소 게시 시 개별 기록을 특정할 수 있는 일련번호·식립위치·날짜는 `<…>`로 가렸다. 같은 기호는 두 파일에서 값이 같다는 뜻이다.

- **같은 일련번호·같은 규격·같은 식립/촬영일·같은 치아 위치인데 모델명만 다르게 달린 쌍을 발견.** 병원(h01=신촌, h02=강남)만 다름. 한 건이지만 같은 임플란트가 두 병원에서 서로 다른 이름으로 라벨링됐다는 직접 증거.
- 두 파일이 TL1(학습용)과 VL1(검증용)에 나뉘어 있어, 같은 이미지라면 split 간 중복(누수)도 됨. 이미지 동일 여부와 전체 중복 쌍 수는 아직 확인 안 함.
- h코드 = 촬영병원(h01 세브란스 신촌, h02 세브란스 강남), P/S = 촬영방법(파노라마/치근단) 확인.

### 2-4. 남은 일
- M10 covaware 실패 원인 확인(로그), 재실행 여부 결정.
- ViT·band-ViT 백본 galcov 평가, ViT SupCon 5ep 미실행.
- ft 체크포인트에서 metric 학습(`run_metric.py:39` assert) 결정 필요.
- seed·분할 반복 + 신뢰구간 없음.

---

## 3. 속성(attribute) 모델

실험: ConvNeXt-Tiny, 속성 10개 분류. closed27로 학습, open15는 test에서만 평가. `shared` = 백본 1개 + head 10개.

### 3-1. 속성별 정확도 (closed27 shared, test)

| 속성 | seen | unseen |
|---|---:|---:|
| System | 0.999 | 0.964 |
| Apex_Hole | 1.000 | 0.982 |
| Connection_1 | 0.959 | 0.867 |
| Connection_2 | 0.975 | 0.803 |
| Thread_Shape | 0.991 | 0.784 |
| Flange | 0.975 | 0.731 |
| Collar | 0.961 | 0.726 |
| Microthread | 0.987 | 0.484 |
| Apex_Shape | 0.993 | **0.287** |
| Body_Shape | 0.977 | **0.257** |

- 34그룹(속성 조합) 매칭: 전체 0.871 / seen 0.938 / unseen **0.025**
- 42클래스 단위 채점: 전체 0.798 / seen 0.860 / unseen 0.011
- 학습에 없는 값(맞힐 수 없음): `Connection_2=TRI-LOBE`, `Apex_Hole=Yes`

### 3-2. 속성별 독립 모델 (test 전체 정확도)

| 실험 | System | Connection_1 | Connection_2 | 비고 |
|---|---:|---:|---:|---|
| closed27 separate | 0.996 | 진행 중 | – | unseen 분리 수치 미집계 |
| full42 r3 separate | 0.998 | 0.967 | 0.975 | open15도 학습 → unseen 근거 불가 |

### 3-3. 속성 + 이미지 임베딩 결합

점수 = `-거리/2 + λ·Σ log P(속성)`, 거리 = galcov SupCon 5ep Maha. (전체 / seen / unseen 정확도)

| λ | kmeans5 | mean |
|---|---|---|
| 0 (속성 없음) | 0.854 / 0.857 / **0.821** | 0.898 / 0.911 / 0.731 |
| 0.03 | 0.901 / 0.919 / 0.681 | **0.906** / 0.924 / 0.682 |
| 0.1 | 0.893 / 0.934 / 0.375 | 0.894 / 0.933 / 0.404 |
| 1 | 0.874 / 0.939 / 0.060 | 0.874 / 0.939 / 0.059 |

- 결론: 속성을 더하면 seen↑ unseen↓. λ는 test에서 골라 낙관적.

---

## 4. 많이 틀리는 클래스 (속성, unseen)

unseen 이미지 1,221장 중 **71%가 seen 클래스의 속성 조합을 그대로 예측**, 자기 조합을 맞힌 비율 1.6%.

### 4-1. Apex_Shape (seen 클래스 값 분포 Flat 12 / Rounded 15)

| unseen 클래스 | 정답 | n | 정확도 | 주 예측 |
|---|---|---:|---:|---|
| **GS III** | Rounded | **498** | **0.01** | Flat 492 |
| Replace Select | Rounded | 191 | 0.19 | Flat 154 |
| luna | Flat | 339 | 0.54 | Flat 183 / Rounded 156 |
| exfeel internal | Flat | 54 | 0.63 | Flat 34 |
| core1 | Flat | 46 | 0.89 | Flat 41 |
| 3i | Flat | 23 | 0.83 | Flat 19 |
| TSV | Rounded | 20 | 0.00 | Flat 20 |
| SS III | Flat | 16 | 1.00 | Flat 16 |
| S clean tapered | Rounded | 11 | 0.00 | Flat 11 |
| atlas internal | Rounded | 7 | 0.14 | Flat 6 |
| Simpleline | Flat | 5 | 0.80 | Flat 4 |
| ankylos | Rounded | 4 | 1.00 | Rounded 4 |
| stella | Flat | 3 | 0.67 | Flat 2 |
| Biohorizon ext / screw vent | Flat | 2 / 2 | 1.00 | Flat |

### 4-2. Body_Shape

| unseen 클래스 | 정답 | n | 정확도 | 주 예측 |
|---|---|---:|---:|---|
| **GS III** | Tapered body | **498** | **0.04** | Non-tapered 467 |
| luna | Tapered body | 339 | 0.48 | Tapered body 164 / Non-tapered 100 |
| Replace Select | Tapered apex | 191 | 0.28 | Non-tapered 81 / Tapered body 57 |
| exfeel internal | Non-tapered | 54 | 0.54 | Non-tapered 29 |
| 3i | Non-tapered | 23 | 0.30 | Tapered apex 12 |
| TSV | Tapered body | 20 | 0.00 | Non-tapered 17 |
| SS III | Non-tapered | 16 | 1.00 | Non-tapered 16 |
| S clean tapered | Tapered body | 11 | 0.82 | Tapered body 9 |
| Simpleline | Tapered apex | 5 | 0.20 | Non-tapered 3 |
| ankylos | Tapered apex | 4 | 0.00 | Non-tapered 3 |
| stella | Tapered apex | 3 | 0.00 | Non-tapered 3 |
| Biohorizon ext | Tapered apex | 2 | 0.00 | Non-tapered 2 |
| screw vent | Non-tapered | 2 | 1.00 | Non-tapered 2 |
| core1, atlas internal | – | – | – | 화면 캡처 누락(서버 `~/fuse_attr/an.txt`) |

### 4-3. 원인 (확인)

- **GS III 하나가 unseen의 41%**이고, Body_Shape·Apex_Shape 오답의 대부분을 차지.
- 서버 스펙표: seen `TS III` = **Non-tapered / Flat**, unseen `GS III` = Tapered body / Rounded.
- 웹 검증(Osstem 카탈로그): TS III/SS III/US III의 Body Shape는 **Tapered body**(원본 라벨 오류, 확신 높음), Apex는 Rounded(확신 중간).
- 즉 모델은 GS III를 같은 계열인 TS III의 (잘못된) 속성으로 예측 → **라벨 오류 + 클래스 shortcut**.

---

## 5. 버그·미완료

| 항목 | 상태 |
|---|---|
| rda: logdet 지배, open 0 | 미수정, 제외 권고 |
| triplet 붕괴 | 미수정 |
| covaware_supcon (r2 실행) | 완료됐으나 학습 실패(전체 0.35) |
| `run_metric.py:39` assert (ft 체크포인트 metric 학습 차단) | 결정 필요 |
| `ss_count: 5pwd` 오타 | 미수정 |
| 속성 A/B 실험(클래스 hold-out 검증, 백본 고정) | 스크립트 준비 완료(`setup_ab.py`, `run_ab.sh`), 서버 미실행 |
| 스펙표 TS/SS/US III 라벨 수정 | 결정 필요 |

위 표는 9/27 기준이다. 이후 실행 상태는 6-6장에 있다.

---

## 6. 9/28~10/01 서버 후속 결과

확인: 2026-09-30 12:28–12:46 KST, 2026-10-01 07:58–08:25 KST. 둘 다 원격 화면을 읽어 전사한 값이며 서버 원본 파일은 로컬에 없다. 단위 %.
42클래스 = seen 27 + unseen 15. query 16,570장 = seen 15,349장 + unseen 1,221장. unseen이 전체의 7.37%뿐이라 전체 정확도만으로는 unseen 실패가 가려진다.
prototype과 공분산을 라벨 있는 gallery(unseen 포함)로 만들므로 zero-shot 결과가 아니다. K-means의 `5`는 클래스당 중심 수이며 support 5장이 아니다.

`*` 표시 F1은 원격 evaluator가 전체 클래스 목록에 unknown을 더해 계산한 원문 값이다. 해당 그룹 클래스만의 Macro-F1로 읽지 않는다.

### 6-1. ViT·Band-ViT 백본 + OAS 마할라노비스, gallery 공분산 (`vit_band_galcov_20260928`)

저장된 9/18 백본 checkpoint(`vit`/`band-vit_full_closed27_open15_c001`)로 특징만 추출했다(`embedding_loss=[none]`). 공분산은 seen + unseen train gallery로 OAS 추정.
출처: [mahalanobis_professor_20261006/report.md](docs/reports/mahalanobis_professor_20261006/report.md) 4장(Mean·K-means), [remote_experiments.csv](docs/experiment_audit_20260930/remote_experiments.csv) 11~34행(SS·Medoid 포함 전체), [remote_results.md](docs/experiment_audit_20260930/remote_results.md) 2장.

| 모델 | Prototype | 전체 Acc | seen Acc | unseen Acc | 전체 F1* |
|---|---|---:|---:|---:|---:|
| ViT | Mean | 93.54 | 94.66 | 79.52 | 81.85 |
| **ViT** | **K-means 5** | **93.77** | 94.46 | **85.18** | **82.99** |
| ViT | SS | 90.81 | **94.83** | 40.29 | 72.44 |
| ViT | Medoid | 90.45 | 94.53 | 39.15 | 68.05 |
| Band-ViT | Mean | 93.37 | 94.37 | 80.84 | 79.65 |
| Band-ViT | K-means 5 | 93.72 | 94.48 | 84.19 | 82.55 |
| Band-ViT | SS | 89.62 | 94.52 | 28.01 | 72.04 |
| Band-ViT | Medoid | 90.19 | 94.38 | 37.51 | 69.08 |

- ViT에서 Mean → K-means 5: unseen +5.65%p, seen −0.20%p(전사 CSV 6자리 값 기준).
- 대표점을 표본 1개(Medoid)나 일부 표본 평균(SS)으로 두면 seen은 유지되지만 unseen이 28~40%로 떨어진다.
- 10/01 확인에서 Band-ViT의 K-means + centered angular 조합은 전체 92.75 / seen 94.34 / unseen 72.81 / 전체 F1* 78.47이었다. 화면에 보인 마지막 조합이라 최고값이 아니다([server_results_20261001.md](docs/reports/server_results_20261001.md) 6.1).
- 9/8 Docker 이미지의 기본 설정은 Ledoit–Wolf·closed train 공분산이라, 보존된 이미지만으로 이 OAS·gallery 공분산 결과를 그대로 재현하지 못한다([DOCKER_REPRODUCTION_COMPARISON_20261026.md](docs/analysis/DOCKER_REPRODUCTION_COMPARISON_20261026.md), [mahalanobis_professor_20261006/report.md](docs/reports/mahalanobis_professor_20261006/report.md) 8장).
- 남은 확인: seen·unseen별 Macro-F1(sample별 예측 CSV가 로컬에 없음), support·query의 이미지·환자 중복, checkpoint 학습 이력.

### 6-2. ViT SupCon 5 epoch 그리드 (`vit_supcon5_20260928`)

ViT 전체 미세조정, SupCon 5 epoch, batch 128, 클래스당 8장, seed 42. 로그 완료 문구 `Metric learning grid 완료`(2026-09-30 04:28:35). 공분산 적합 범위는 기록에 없다. 아래 4개는 기록된 조합이며 전체 그리드가 아니다.
출처: [server_results_20261001.md](docs/reports/server_results_20261001.md) 2장.

| 조합 | 전체 Acc | 전체 F1* | seen Acc | seen F1* | unseen Acc | unseen F1* |
|---|---:|---:|---:|---:|---:|---:|
| Medoid + fractional | **87.35** | 65.07 | 90.21 | 54.55 | 51.35 | 12.97 |
| Medoid + centered angular | 87.04 | 64.82 | **90.23** | 54.56 | 46.85 | 12.74 |
| K-means + OAS Mahalanobis | 84.75 | **77.06** | 85.50 | 53.09 | **75.35** | **28.87** |
| K-means + centered angular | 84.70 | 71.59 | 86.46 | 53.04 | 62.65 | 22.98 |

### 6-3. Gap-SupCon (`gap-supcon-b256-20260929`)

DINOv2 base(일반 사전학습에서 시작), 입력 224, projection 128, batch 256, 클래스당 16장, 5 epoch, seed 42. 후보 4개 중 source에서 따로 둔 5개 클래스로 margin 0.05 / weight 0.01 선택. 공분산은 seen train 임베딩으로만 추정(6-1과 다름).
출처: [remote_results.md](docs/experiment_audit_20260930/remote_results.md) 1장, [remote_experiments.csv](docs/experiment_audit_20260930/remote_experiments.csv) 2~10행, [server_results_20261001.md](docs/reports/server_results_20261001.md) 1장. 값은 `Accuracy / Macro-F1`이며 F1은 그룹별 원문 값이다.

| 분류기 | 전체 | seen | unseen |
|---|---:|---:|---:|
| Cosine prototype | 84.97 / 62.93 | 89.00 / 85.85† | 34.32 / 31.14 |
| Euclidean prototype | 84.99 / 62.95 | 88.98 / 85.95 | 34.89 / 31.16 |
| **Pooled OAS Mahalanobis** | **87.67 / 69.65** | **90.63 / 86.66** | **50.45 / 48.33** |

† 9/30 전사(85.8461)와 10/01 전사(85.9461)가 다르다. 원본 파일이 로컬에 없어 어느 쪽이 맞는지 확인하지 못했다.

- 같은 초기값·데이터·조건의 SupCon 단독 대조군은 중단(Exited 143)돼 결과가 없다. **Gap loss의 효과는 판단할 수 없다.** 이 표가 보여주는 것은 분류기 차이다.
- 클래스 거리: unseen의 RMS 반지름 평균 0.4302로 seen(0.1910)의 약 2.25배. 최근접 유효 간격은 unseen 평균 −0.5763이며 unseen 15개 클래스 모두 음수.
- 선택 단계 기록: 학습 클래스 간격 +0.34259, 따로 둔 클래스 −0.42414. 학습 클래스에서 벌어진 간격이 새 클래스로 이어지지 않았다.

### 6-4. 41클래스(it + internal 병합, 26 seen + 15 unseen), gallery OAS

출처: [remote_experiments.csv](docs/experiment_audit_20260930/remote_experiments.csv) 35~58행, [remote_results.md](docs/experiment_audit_20260930/remote_results.md) 3장. 실험 설명: [attribute_tokens_41class_20260927.md](docs/attribute_tokens_41class_20260927.md), [predicted_attribute_tokens_41class_20260928.md](docs/predicted_attribute_tokens_41class_20260928.md).

| 모델 | Prototype | 전체 Acc | seen Acc | unseen Acc | 전체 F1* |
|---|---|---:|---:|---:|---:|
| 이미지 단독 (`new_41class`) | Mean | 89.62 | 91.09 | 71.17 | 72.80 |
| 이미지 단독 (`new_41class`) | K-means | 84.86 | 85.32 | **79.03** | 76.75 |
| 속성 토큰 5 epoch | Mean | 90.55 | 92.03 | 71.99 | 73.90 |
| 속성 토큰 5 epoch | K-means | 85.43 | 86.01 | 78.13 | **76.81** |
| 속성 토큰 20 epoch, patience 3 | Mean | **91.86** | **94.09** | 63.80 | 73.07 |
| 속성 토큰 20 epoch, patience 3 | K-means | 90.29 | 91.66 | 73.05 | 76.45 |
| 예측 속성 융합 | Mean | 88.81 | 92.79 | 38.74 | 60.47 |
| 예측 속성 융합 | K-means | 82.65 | 86.24 | 37.51 | 60.54 |

`*` 원문 `overall_macro_f1`.

- 속성 정보를 넣어도 unseen 정확도는 이미지 단독보다 높아지지 않았다. 20 epoch로 늘리면 seen은 오르고 unseen은 내려갔다. 모델마다 학습 조건과 속성 제공 방식이 달라 속성 자체를 원인으로 단정하지 않는다.
- 예측 속성 8개만으로 만든 prototype(이미지 없음)은 euclidean 기준 전체 16.04 / F1 5.14 / seen 17.01 / unseen 3.58이었다([server_results_20261001.md](docs/reports/server_results_20261001.md) 3장).

### 6-5. 속성 예측 반복 held-out: Apex Shape (10/01 확인)

클래스 5개를 학습에서 빼고(heldout5) 속성 정확도를 본 실험이다. 클래스 식별 성능이 아니다. 값은 `Accuracy / Macro-F1`.
출처: [server_results_20261001.md](docs/reports/server_results_20261001.md) 4장·7.1.

| seed / 설정 | heldout5 | other37 |
|---|---:|---:|
| 42, fine-tune | 76.35 / 76.35 | 82.66 / 76.15 |
| 42, frozen | 75.19 / 74.95 | 74.05 / 70.32 |
| 43, fine-tune | **94.23 / 93.84** | **92.43 / 90.50** |
| 43, frozen | 82.83 / 81.61 | 83.19 / 79.78 |
| 44, fine-tune | **96.77 / 95.87** | 92.01 / 90.74 |
| 44, frozen | 83.45 / 80.53 | 79.68 / 77.76 |

- heldout 클래스: seed 42 = astra, US II, Himplant, IS III, IS I / seed 43 = Bone level, astra, IS II, internal, branemark / seed 44 = TS standard plus, TS standard, astra, internal, US II.
- seed마다 빠진 클래스가 달라 seed 간 차이가 크다(heldout5 정확도 75.19~96.77). 3개 seed로 평균을 내 일반화 성능으로 보고하기에는 이르다.

### 6-6. 실행 상태 (2026-10-01 07:58–08:25 KST 확인)

출처: [server_results_20261001.md](docs/reports/server_results_20261001.md) 4·6·8장, [remote_results.md](docs/experiment_audit_20260930/remote_results.md) 4장.

| 작업 | 상태 |
|---|---|
| `galcov`, `heldout`, `heldout-frozen`, `vit-supcon5`, `dinov2-ft5`, `heldout(-frozen)-s43/s44` | `queue.log` 모두 `exit_code: 0`, `repeat_summary` `REPEAT_SUMMARY_OK 120`, 큐 `finished` |
| `vit-supcon5` | 9/30에는 실행 중 → 10/01에 완료 확인(6-2) |
| heldout 계열 일부 | 종료 코드 0이지만 `test_manifest.csv`가 없어 test 임베딩 저장을 건너뜀. 테스트 결과는 완료로 보지 않음 |
| `dinov2-ft5` | 화면에 보인 마지막 조합(covaware SupCon, K-means + centered angular) 전체 87.86 / F1* 65.82 / seen 90.48 / unseen 54.96. 최고값 아님 |
| Band-ViT-Large 42클래스 분류 | `summary.json` 없음. `history.csv` 마지막 epoch valid 95.80 / 91.83만 확인 |
| Gap-SupCon과 같은 조건의 SupCon 대조군 | 중단(Exited 143), 결과 없음 |

10/01 이후 원격 서버를 다시 확인한 기록은 없다. [발표.md](발표.md) 8장(9/30 기준)의 "`vit-supcon5` 실행 중"은 이 확인으로 갱신된다.

---

## 7. 9월 초 폐쇄망 서버: ViT 임베딩 평가 (9/12~9/16 기록)

`aihub_run` 기준. 백본을 추가 학습하지 않은 사전학습 ViT와, 27개 클래스로 전체 미세조정한 ViT(`vit_full_closed27_open15.pt`)를 같은 방식으로 평가했다. 공분산 적합 범위는 기록에 없다. 원문 0~1 값을 %로 옮겼다. `unknown_rate`는 모두 0.
출처: [closed_network_experiment_results_2026-09-12.md](docs/reports/closed_network_experiment_results_2026-09-12.md) 4장.

| 백본 | 평가 범위 | Prototype + 거리 | 전체 Acc | 전체 F1* | seen Acc | seen F1* | unseen Acc | unseen F1* |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 사전학습 ViT | 27 + 15 | Mean + Cosine | 20.59 | 14.39 | 18.75 | 11.35 | 43.65 | 14.09 |
| 사전학습 ViT | 27 + 15 | Mean + Mahalanobis | 60.21 | 46.87 | 59.43 | 32.28 | **69.94** | 24.66 |
| 27클래스 FT ViT | 27 + 15 | Mean + Cosine | 82.29 | 57.02 | 86.40 | 50.53 | 30.63 | 12.94 |
| 27클래스 FT ViT | 27 + 15 | Mean + Mahalanobis | **87.60** | **70.72** | **89.99** | 52.40 | 57.58 | 22.89 |
| 사전학습 ViT | 42 전체 | Mean + Mahalanobis | 60.34 | 47.41 | — | — | — | — |
| 사전학습 ViT | 42 전체 | K-means + Mahalanobis | 59.66 | 49.67 | — | — | — | — |

`*` 원문 그룹별 F1. 6장과 같은 evaluator 계열이면 같은 집계 범위 문제가 있을 수 있다.

- 27클래스로 미세조정하면 전체·seen은 크게 오르지만 unseen 정확도는 69.94 → 57.58로 내려갔다.

---

## 8. 5~8월 로컬 실험 (RunPod·로컬 Mac)

데이터·분할·평가 방식이 실험마다 달라 표끼리 순위를 매기지 않는다. 값은 `Accuracy / Macro-F1`, 단위 %.

### 8-1. closed-set 분류 기준선 (5월·7월)

출처: [EXPERIMENT_LEDGER_20260807.md](docs/reports/EXPERIMENT_LEDGER_20260807.md) 실험 표. 27-label ClosedFT는 [METRIC_DIAGNOSIS_FINAL_REPORT.md](docs/reports/METRIC_DIAGNOSIS_FINAL_REPORT.md)에도 있다. 5월 원자료 요약: [EXPERIMENT_SUMMARY.md](docs/analysis/EXPERIMENT_SUMMARY.md), [PRESENTATION_SUMMARY.md](docs/analysis/PRESENTATION_SUMMARY.md).

| 실험 | 데이터 | Test Acc / F1 |
|---|---|---:|
| 10-label plain ViT-Base, CE 80 epoch | 3-label + 7-label 통합 | **95.54** / 88.12 |
| 7-label plain ViT-Base, CE 25 epoch | 7-label | 94.40 / **88.29** |
| 3-label plain ViT-Base, CE 80 epoch | Bego·Bicon·ITI | 94.50 / 77.00 |
| 7-label SupCon, cosine prototype | 7-label | 95.87 / 75.12 |
| 10-label + NORIS/osstem 2단계 재분류 | 10-label test 448장 | 95.09 / 87.86 |
| 10-label → 3-label fine-tuning | 3-label | 91.74 / 71.73 |
| 11-label crop ViT-Base, 15 epoch | test 101장 | 82.18 / 75.67 |
| 27-label ClosedFT, CE 10 epoch | closed 27 | 95.43 / 92.67 |
| VLM 구조 feature + tabular ML | 7-label | Acc 대체로 32~45 |
| ImageNet ViT-Large(학습 없음), 클래스당 train 10장 평균 중심 | 42클래스 test 16,570장 | Euclidean 17.53 / 12.84, Cosine 17.59 / 12.75 |

plain CE ViT가 가장 안정적이었고 2단계 재분류·SupCon·상위 라벨 fine-tuning은 기준선을 넘지 못했다.

### 8-2. GT27 표현 진단 (7월)

27개 클래스 GT 이미지 241장 = support 189장 / query 52장. backbone 특징의 클래스 평균으로 27-way 분류. query 1장이 정확도 1.92%p를 움직인다.
출처: [METRIC_DIAGNOSIS_FINAL_REPORT.md](docs/reports/METRIC_DIAGNOSIS_FINAL_REPORT.md), [INDEPENDENT_EXPERIMENT_BASELINE_AND_GT27.md](docs/reports/INDEPENDENT_EXPERIMENT_BASELINE_AND_GT27.md).

| 모델 | Euclidean | Cosine |
|---|---:|---:|
| ViT-Base, 추가 학습 없음 | 48.08 / 51.28 | **51.92 / 52.80** |
| ViT-ClosedFT (27, CE) | 44.23 / 42.21 | 44.23 / 42.46 |
| ViT-ClosedFT-SupCon | 44.23 / 47.92 | 40.38 / 43.08 |
| ViT-FullFT (42, CE) | **51.92 / 52.44** | 50.00 / 50.59 |
| ViT-FullFT-SupCon | 42.31 / 42.79 | 42.31 / 41.55 |

- 27개 클래스로 fine-tuning하면 추가 학습하지 않은 Base보다 낮아졌다. SupCon은 closed·full 모두 top-1을 높이지 못했다.
- 별도 Band-loss checkpoint는 cosine top-1 61.54, top-3 84.62, top-5 92.31(Accuracy만).

### 8-3. 7-label 학습 → Bego·Bicon·ITI 3-label 전이 (8월, target train 전체로 중심·공분산)

encoder는 source 7-label로만 학습하고 target 3-label train 이미지로 중심과 공분산을 만든다. encoder 기준 unseen 라벨이지만 zero-shot은 아니다.
출처: 과거 조건 행 [legacy_7to3_reproduction_20260807](docs/reports/legacy_7to3_reproduction_20260807/REPORT.md), [mahalanobis_improvement_20260813](docs/reports/mahalanobis_improvement_20260813/REPORT.md) / 새 학습 SupCon 행 [vit7supcon_to_3label_mahalanobis_20260807](docs/reports/vit7supcon_to_3label_mahalanobis_20260807/REPORT.md) / CE 행 [vit7ce_to_3label_mahalanobis_20260806](docs/reports/vit7ce_to_3label_mahalanobis_20260806/REPORT.md) / ViT-Large 행 [vit42ft_local_metric_20260806](docs/reports/vit42ft_local_metric_20260806/REPORT.md), [vit42ft_local_mean_4dist_20260806](docs/reports/vit42ft_local_mean_4dist_20260806/REPORT.md) / RDA 열 [mahalanobis_method_selection_20260816](docs/reports/mahalanobis_method_selection_20260816/REPORT.md).

| Encoder | test 장수 | Cosine | Pooled LW Mahalanobis | RDA 0.75 + logdet |
|---|---:|---:|---:|---:|
| SupCon ViT-S backbone (과거 조건 재구성) | 109 | 27.52 / 26.87 | 63.30 / 48.23 | 71.56 / 57.60 |
| SupCon ViT-S projection (과거 조건 재구성) | 109 | 31.19 / 29.75 | 60.55 / 44.77 | 71.56 / 54.31 |
| SupCon ViT-S backbone (새 학습) | 446 | 75.11 / 56.48 | 98.21 / 92.94 | **99.55 / 98.21** |
| SupCon ViT-S projection (새 학습) | 446 | 71.97 / 50.17 | 93.27 / 81.46 | 94.84 / 87.91 |
| CE ViT-B, 30 epoch | 446 | 73.32 / 48.83 | 95.07 / 84.19 | — |
| 42클래스 FT ViT-Large (고정) | 446 | 91.70 / 76.93 | 98.43 / 95.65 | — |

- 새 학습 SupCon encoder의 source 7-label 성능은 test 54.73 / 41.19로 낮다. 높은 target 점수는 SupCon 학습보다 target 공분산의 효과로 본다.
- 446장 test에 train과 완전히 같은 이미지 5장이 있었다. 제외해도 CE Mahalanobis 95.01 / 84.18로 사실상 같았다.
- 42클래스 FT ViT-Large는 학습 manifest가 없어 target 이미지와의 중복을 배제할 수 없다.
- 5월 cosine 계열 직접 전이는 34.86 / 29.26 ~ 37.61 / 34.75, 2단계 metric 방식은 68.81 / 54.92였다(수치 [EXPERIMENT_LEDGER_20260807.md](docs/reports/EXPERIMENT_LEDGER_20260807.md), 시기 [발표.md](발표.md) 5.3).

### 8-4. 원본(uncropped) 7 → 3 전이, source group 분리 test 116장 (7월)

test 116장 중 Bicon이 95장이라 Bicon만 예측해도 정확도 81.90%다. Macro-F1을 함께 본다.
출처: ViT-Base 행 [mahalanobis_addendum_20260807/REPORT_TABLE.md](docs/reports/full_uncropped_7to3_metric/mahalanobis_addendum_20260807/REPORT_TABLE.md), 학습 모델 행 [full_uncropped_7to3_metric/report/REPORT.md](docs/reports/full_uncropped_7to3_metric/report/REPORT.md).

| 표현 | Prototype | Euclidean | Cosine | Pooled LW Mahalanobis |
|---|---|---:|---:|---:|
| ViT-Base, 추가 학습 없음 | Mean | 81.90 / 64.12 | 81.90 / 63.99 | **93.10 / 79.69** |
| ViT-Base, 추가 학습 없음 | GT | 80.17 / 56.26 | 82.76 / 64.16 | 90.52 / 70.57 |
| CE 7-label 15 epoch | 최고 조합 | 72.41 / 50.14 | — | — |
| Triplet backbone | 최고 조합 | 68.10 / 55.48 | — | — |
| SupCon backbone | 최고 조합 | 53.45 / 39.88 | — | — |
| SupCon projection | 최고 조합 | — | 44.83 / 35.93 | — |

7-label fine-tuning 모델은 source 성능이 높았지만 unseen 3-label 전이에서는 모두 ViT-Base보다 낮았다. "최고 조합"은 test에서 관측한 최댓값이라 모델 선택값으로 쓰지 않는다.

### 8-5. strict few-shot: support만으로 중심과 공분산을 모두 추정 (8월~9월)

값은 support seed 6개의 평균 `Accuracy / Macro-F1`.
출처: 3-label 행 [mahalanobis_strict_10shot_20260814](docs/reports/mahalanobis_strict_10shot_20260814/REPORT.md) / 5-label 10장 행 [vit6supcon_to_5label_strict10_20260814](docs/reports/vit6supcon_to_5label_strict10_20260814/REPORT.md) / 5-label 5장 행 [episodic_all_target_5shot](docs/reports/vit6supcon_to_5label_strict10_20260814/episodic_all_target_5shot/REPORT.md).

| 조건 | support / query | Cosine | Pooled LW Mahalanobis | 고정 RDA |
|---|---|---:|---:|---:|
| 3-label, 클래스당 10장 | 30 / 446 | — | 69.36 / 52.93 | 70.40 / 54.40 |
| 5-label, 클래스당 10장 | 50 / 17 | 48.04 / 41.84 | 69.61 / 59.60 | 65.69 / 57.28 |
| 5-label 반복 episode, 클래스당 5장 | 25 / 127 | 56.56 / 53.62 | **67.98 / 63.30** | 63.25 / 58.98 |

- **공분산 표본 수의 영향이 결정적이다.** 3-label seed 42에서 prototype 10장은 그대로 두고 공분산만 target train 3,572장 → 같은 10장으로 줄이면 pooled LW 93.05 → 69.06, validation으로 고른 RDA 98.88 → 69.06([mahalanobis_10shot_prototype_20260814](docs/reports/mahalanobis_10shot_prototype_20260814/REPORT.md), [mahalanobis_strict_10shot_20260814](docs/reports/mahalanobis_strict_10shot_20260814/REPORT.md)).
- 5-label 10-shot은 test가 17장뿐이라 1장이 정확도 5.88%p를 움직인다. NUCLEOSS는 5-shot에서도 F1 32.12로 가장 낮았다.
- 같은 5-label 5-shot에서 시도한 변형은 pooled LW(67.98 / 63.30)를 넘지 못했다: CovAware 학습 seed 42 66.40 / 62.65, pooled-full CovAware 학습 seed 2개 평균 66.60 / 63.32, 8-bit 균일 양자화 67.06 / 62.71, asinh 변환 67.59 / 62.84([local_results.md](docs/experiment_audit_20260930/local_results.md), [embedding_quantization_20260915/REPORT.md](docs/reports/embedding_quantization_20260915/REPORT.md), [CONTINUOUS_REPORT.md](docs/reports/embedding_quantization_20260915/CONTINUOUS_REPORT.md)).
- centered angular 거리는 3-label 10-shot에서 정확도를 69.36 → 74.22로 올렸지만 다른 조건에서는 낮았다. 이미 본 데이터에서 고른 후보라 탐색 결과로만 둔다([MAHALANOBIS_SPECTRUM_VALIDATION_FINAL_REPORT.md](docs/reports/MAHALANOBIS_SPECTRUM_VALIDATION_FINAL_REPORT.md)).

### 8-6. Track B: unseen 15-way few-shot (9월, 클래스당 support 10장)

42클래스(27 seen 학습 / 15 unseen) 중 unseen 15개끼리만 분류. 5개 그룹 holdout × 분할당 200 episode 평균.
출처: CE·ArcFace 행 [fewshot_trackB/REPORT.md](docs/reports/fewshot_trackB/REPORT.md), PCA 행 [local_results.md](docs/experiment_audit_20260930/local_results.md).

| 표현 | 거리 | Accuracy / Macro-F1 |
|---|---|---:|
| ImageNet ViT-B, 분할별 PCA 768 → 384 | Pooled Mahalanobis | **58.80 / 42.42** |
| ImageNet ViT-B 원본 768차원 | Pooled Mahalanobis | 52.40 / 41.08 |
| native 384차원 임베딩(원문 표기 `native384`) | Pooled Mahalanobis | 53.45 / 39.37 |
| CE ViT-S | Cosine | 52.3 / 31.8 |
| ArcFace ViT-S | Cosine | 48.9 / 30.7 |

PCA는 각 분할의 train 역할 데이터로만 적합했다. 모델 재학습이 아니라 임베딩 후처리다.

---

## 9. 엑셀 속성표만으로 구분되지 않는 클래스 (2026-11-02 집계)

연세대 측 `임플란트 라벨링.xlsx`의 형태 속성 11개로 42개 클래스를 비교한 결과다. 엑셀 원본은 저장소에 넣지 않았다. 출처와 상세: [발표.md](발표.md) 0장.

| 항목 | 8/4판 (45행, 연세대 측) | 9/24판 (49행, 직접 채움) |
|---|---:|---:|
| 42개 클래스 중 서로 다른 속성 조합 | 37 | 36 |
| 속성이 완전히 같은 그룹 / 포함 클래스 | 4그룹 / 9개 | 5그룹 / 11개 |
| `확인 불가` 칸 / 해당 클래스 | 12칸 / 3개 | 0칸 / 0개 |
| 속성을 완벽히 읽어도 표 조회로 얻는 Macro-Recall 상한 | 88.10% 이하 | 85.71% 이하 |

- 두 판 모두 같은 그룹: Osstem SSII / SSIII(제조사까지 넣어도 구분 불가), Osstem USII / MegaGen AnyOne External, MegaGen ExFeel External / Nobel Branemark / Biomet 3i, DIO UF / Dentsply Xive. 9/24판에서만 Dentium SuperLine / Cybermed Core1이 같다(직접 채운 값 때문).

---

## 10. 보고서 목록

이 저장소에 모은 실험 결과 보고서다. 날짜는 각 문서에 적힌 작성·확인 시점이다.

| 날짜 | 문서 | 내용 |
|---|---|---|
| 2026-05~06 | [docs/analysis/EXPERIMENT_SUMMARY.md](docs/analysis/EXPERIMENT_SUMMARY.md), [PRESENTATION_SUMMARY.md](docs/analysis/PRESENTATION_SUMMARY.md), [TSNE_UMAP_COMPARISON.md](docs/analysis/TSNE_UMAP_COMPARISON.md) | 5월 라벨 체계별 분류 기준선, t-SNE/UMAP |
| 2026-07-12 | [docs/reports/METRIC_DIAGNOSIS_CURRENT_RESULTS.md](docs/reports/METRIC_DIAGNOSIS_CURRENT_RESULTS.md) | metric learning 진단 중간 보고 |
| 2026-07-22 | [docs/reports/INDEPENDENT_EXPERIMENT_BASELINE_AND_GT27.md](docs/reports/INDEPENDENT_EXPERIMENT_BASELINE_AND_GT27.md) | 12개 checkpoint 기준선과 GT27 |
| 2026-07-23 | [docs/reports/full_uncropped_7to3_metric/report/REPORT.md](docs/reports/full_uncropped_7to3_metric/report/REPORT.md) | uncropped 7 → 3 전이 (CE·SupCon·Triplet·ViT-Base) |
| 2026-07-24 | [docs/reports/METRIC_DIAGNOSIS_FINAL_REPORT.md](docs/reports/METRIC_DIAGNOSIS_FINAL_REPORT.md) | metric learning 진단 최종 보고 (GT27) |
| 2026-08-06 | [vit42ft_local_metric_20260806](docs/reports/vit42ft_local_metric_20260806/REPORT.md), [vit42ft_local_mean_4dist_20260806](docs/reports/vit42ft_local_mean_4dist_20260806/REPORT.md), [vit7ce_to_3label_mahalanobis_20260806](docs/reports/vit7ce_to_3label_mahalanobis_20260806/REPORT.md) | 42클래스 FT ViT-Large·CE ViT-B의 7 → 3 전이 |
| 2026-08-07 | [vit7supcon_to_3label_mahalanobis_20260807](docs/reports/vit7supcon_to_3label_mahalanobis_20260807/REPORT.md), [legacy_7to3_reproduction_20260807](docs/reports/legacy_7to3_reproduction_20260807/REPORT.md), [mahalanobis_addendum_20260807](docs/reports/full_uncropped_7to3_metric/mahalanobis_addendum_20260807/REPORT_TABLE.md) | SupCon 7 → 3 전이, 과거 조건 재현, uncropped Mahalanobis 보충 |
| 2026-08-07 | [docs/reports/EXPERIMENT_LEDGER_20260807.md](docs/reports/EXPERIMENT_LEDGER_20260807.md) | 8/7까지 전체 실험 원장 |
| 2026-08-13 | [mahalanobis_improvement_20260813](docs/reports/mahalanobis_improvement_20260813/REPORT.md) | Mahalanobis 개선 비교 |
| 2026-08-14 | [mahalanobis_strict_10shot_20260814](docs/reports/mahalanobis_strict_10shot_20260814/REPORT.md), [mahalanobis_10shot_prototype_20260814](docs/reports/mahalanobis_10shot_prototype_20260814/REPORT.md), [vit6supcon_to_5label_strict10_20260814](docs/reports/vit6supcon_to_5label_strict10_20260814/REPORT.md) (+ [target 5-shot](docs/reports/vit6supcon_to_5label_strict10_20260814/episodic_all_target_5shot/REPORT.md), [source 6-shot](docs/reports/vit6supcon_to_5label_strict10_20260814/episodic_all_source_6shot/REPORT.md), [source train → test](docs/reports/vit6supcon_to_5label_strict10_20260814/episodic_source_train_to_test/REPORT.md)) | strict few-shot |
| 2026-08-16 | [mahalanobis_method_selection_20260816](docs/reports/mahalanobis_method_selection_20260816/REPORT.md), [mahalanobis_spectrum_validation_20260816/VALIDATION.md](docs/reports/mahalanobis_spectrum_validation_20260816/VALIDATION.md), [MAHALANOBIS_SPECTRUM_VALIDATION_FINAL_REPORT.md](docs/reports/MAHALANOBIS_SPECTRUM_VALIDATION_FINAL_REPORT.md) | 거리·공분산 방법 선택과 검증 |
| 2026-09-15 | [embedding_quantization_20260915](docs/reports/embedding_quantization_20260915/REPORT.md) (+ [CONTINUOUS_REPORT.md](docs/reports/embedding_quantization_20260915/CONTINUOUS_REPORT.md)), [fewshot_trackB/REPORT.md](docs/reports/fewshot_trackB/REPORT.md) | 임베딩 양자화·연속 변환, Track B few-shot 파일럿 |
| 2026-09-16 | [docs/reports/closed_network_experiment_results_2026-09-12.md](docs/reports/closed_network_experiment_results_2026-09-12.md) | 9월 초 폐쇄망 백본 학습·ViT 임베딩 평가 |
| 2026-09 | [implant_experiment_summary_table.md](implant_experiment_summary_table.md) | `aihub_run_new` 백본 분류 결과표 |
| 2026-09-23~25 | [docs/handoff_20260924/](docs/handoff_20260924/) (`backbone_eval_interim_20260923.md`, `metric_results_summary_20260924.md`, `metric_galcov_results_20260925.md`, `implant_attribute_verification_20260924.md`, `attribute_experiment_guide.md`, `metrics_test_*.txt`) | 1~5장 세부 근거 |
| 2026-09-27~28 | [attribute_tokens_41class_20260927.md](docs/attribute_tokens_41class_20260927.md), [predicted_attribute_tokens_41class_20260928.md](docs/predicted_attribute_tokens_41class_20260928.md) | 41클래스 속성 토큰 실험 설명 |
| 2026-09-28 | [professor_report_20260928/report.md](docs/reports/professor_report_20260928/report.md), [followup_20260928_status.md](docs/reports/followup_20260928_status.md) | 9/28 교수님 보고 초안, 후속 실험 상태 |
| 2026-09-29 | [docs/reports/remote_experiments_20260929.md](docs/reports/remote_experiments_20260929.md) | 원격 실험 종합 기록 |
| 2026-09-30 | [docs/experiment_audit_20260930/](docs/experiment_audit_20260930/report.md) (`report.md`, `remote_results.md`, `local_results.md`, `other_local_results.md`, `training_design.md`, CSV) | 로컬·원격 결과 감사와 다음 학습안 |
| 2026-10-01 | [docs/reports/server_results_20261001.md](docs/reports/server_results_20261001.md) | 10/01 서버 결과 정리 (6장 일부 출처) |
| 2026-10-06 | [docs/reports/mahalanobis_professor_20261006/report.md](docs/reports/mahalanobis_professor_20261006/report.md) | ViT·OAS 교수님 보고 |
| 2026-10-26 | [docs/analysis/DOCKER_REPRODUCTION_COMPARISON_20261026.md](docs/analysis/DOCKER_REPRODUCTION_COMPARISON_20261026.md) | 9/8 Docker 이미지와 저장소 코드 대조 |
| 2026-10-28 | [발표.md](발표.md), [발표_쉬운말.md](발표_쉬운말.md) | 5~10월 종합 결과 보고 (11/2 엑셀 집계 포함) |

**모은 경로:** 원래 로컬의 `outputs/local_only_20261027/docs/`(gitignore), Obsidian `Projects/implantsj/reports/`, 10/1 디스크 정리 때 `stash@{0}`에 들어간 미추적 파일(`cf68fb53`)에 흩어져 있던 문서를 위 경로로 옮겼다. 일부 문서 안의 절대 경로·저장소 밖 링크는 원문 그대로 두었다.

**넣지 않은 자료:** pptx·xlsx·pdf(`9월_실험결과.pdf`, `실험결과_전체.xlsx`, `엑셀_구분불가_보고*.pptx`, 연세대 라벨링 xlsx, `1차 임플란트 결과.pdf`), 원시 산출물(CSV·JSON·NPZ 등은 ImplantID `f961b3cb`의 `implantsj/docs/reports/`에 남아 있음), `../outputs_docker/`의 2026-08-28 Docker 1 epoch 동작 확인 실행(성능 비교용 아님), `.omx` QA 초안, 회의록(`../md파일모음/`), 논문 리서치 문서.
