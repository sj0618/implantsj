# 예측 속성 8개 + 이미지 패치 토큰 결합

## 확인 상태

- 서버 화면 확인: 2026-09-28 01:27 KST.
- 본 학습 시작 로그: 2026-09-27 16:26:50 UTC / 2026-09-28 01:26:50 KST.
- 실행 중이며 첫 에포크와 최종 성능은 아직 확인하지 않았다.
- V100 32GB 사용률 100%, GPU 메모리 27,928 MiB, Python PID 1283020.
- 설정: 최대 20epoch, patience=3, batch_size=128, samples_per_class=8, seed=42. 검증 Macro-F1 기준 기존 best checkpoint 저장 로직 사용.

## 재사용한 속성 모델

- 기존 실험: `20260925-attr-closed27`.
- 공통 모델: ConvNeXt-Tiny `convnext_tiny.fb_in22k_ft_in1k`, 10개 속성 헤드.
- 체크포인트: `/home/ubuntu/aihub_run_new/outputs/attributes/20260925-attr-closed27/shared/checkpoints/shared.pt` (서버에서 107 MiB 확인).
- 기존 속성별 독립 모델 10개도 같은 실험 디렉터리에 존재한다. 이번에는 공통 모델을 사용한다.
- `effective.json`의 학습 매니페스트는 `manifest_closed27.csv`, 프로토콜은 `closed27_open15`이다. 전체 클래스 학습 모델은 사용하지 않는다.
- 새 모델 생성 시 체크포인트 프로토콜과 선택한 속성의 어휘·순서 일치를 검사하고 strict load한다.
- 속성 모델은 읽기 전용으로 마운트하고 가중치를 고정한다. 매 forward에서 eval 및 no_grad로 실행한다.

## 구조

1. 동일한 224×224 RGB 입력을 고정된 속성 모델과 DINOv2 이미지 백본에 전달한다. 기존 두 코드의 직접 resize 및 ImageNet 정규화 경로를 확인했다.
2. 속성 모델의 출력 중 `System`, `Connection_1`, `Flange`, `Collar`, `Microthread`, `Body_Shape`, `Thread_Shape`, `Apex_Shape`를 선택한다.
3. 속성별 softmax 확률 벡터를 각각 선형 계층으로 DINOv2 토큰 차원에 투영한다.
4. 기존 8개 학습 가능한 속성 토큰에 이 투영값을 더하고, 이미지 패치 토큰과 함께 Transformer를 통과시킨다.
5. CLS의 128차원 임베딩은 기존 SupCon 및 프로토타입·거리 기반 평가 경로를 사용한다. 융합 후 속성 헤드의 8개 보조 CE 평균도 유지한다.

학습·검증·테스트 입력 모두 이미지에서 예측한 속성 확률이다. 정답 클래스의 속성은 보조 CE의 target에만 사용한다. 고정 속성 모델 자체에는 새 손실로 역전파하지 않는다. 이전 `attr8_tokens_41class_20260927` 실험과 달리 이미지별 예측 속성값이 실제 토큰 입력에 반영된다.

## 코드·출력 보존

- 새 컨테이너: `predattr8-41class-20260928`.
- 실험 사본: `/home/ubuntu/aihub_run_new/outputs/experiment_records/predattr8-41class-20260928/`.
- 결과: `/home/ubuntu/aihub_run_new/outputs/metric/predattr8-41class-20260928/closed27_open15_dinov2_supcon/`.
- 구현: 새 사본의 `src/metric/attribute_tokens.py`; 기존 setup을 호출한 뒤 고정 predictor, 8개 adapters, forward hooks를 추가했다.
- `src/run_metric.py`와 실행 인자는 기존 20epoch 실험에서 복사했다. `launch.json`에 새 이름·경로와 속성 모델 읽기 전용 마운트를 기록했다.
- 원본 코드·데이터·기존 결과를 수정하지 않았다. 서버 작업은 Codex Computer Use 및 화상키보드만 사용했다.

## 검증

- `predattr8-41class-20260928-smoke`: 실제 GPU 배치 4장, `SMOKE_OK`, 종료 코드 0.
- 유한한 손실, forward/backward, 128차원 출력, 8개 속성 출력 및 순서 검사.
- 속성 정답 -1 마스킹, 기존 토큰 gradient, 모든 새 adapter의 0보다 큰 gradient 확인.
- predictor의 모든 gradient가 None임을 확인.
- optimizer step, 체크포인트 저장 및 동일 인스턴스로 strict 복원 후 eval 출력 일치 확인. 별도 새 인스턴스 복원은 아직 확인하지 않았다.
- 본 학습 로그: trainable=86,458,516, frozen=27,839,353.

최종 성능 개선은 아직 확인하지 않았다. 이후 평가에서는 동일한 41클래스·프로토타입·거리·gallery 조건을 맞춰 비교해야 한다. 기존 gallery covariance 평가를 그대로 사용하므로 strict-5-shot 결과가 아니다.
