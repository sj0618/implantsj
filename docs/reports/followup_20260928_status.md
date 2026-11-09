# 반출 준비 및 후속 실험 실행 상태

2026-09-28 23:16 KST 원격 터미널 확인 기준. Windows App 화면으로 서버 파일과 실행 상태를 확인했다. 아래의 실행 중·대기 항목은 완료 결과가 아니다.

## 반출 폴더

- 경로: `/home/ubuntu/workspace/nasw/download/`
- CSV 3,876개 + 체크포인트 `.pt` 109개. 총 3,985개 개별 파일.
- 하위 폴더, NPZ, 기타 확장자 없음. `00_FILE_INDEX.csv`에 나머지 3,984개 파일 기록.
- 결과·임베딩·로짓·백본 피처를 CSV로 저장. 원본 파일명·경로 식별자는 일관된 임의 ID로 치환했다. 매핑은 서버 내부에만 보관했다.
- CSV 식별자 검사와 체크포인트 메타데이터 검사 완료. 정리 중 삭제된 클래스·손실 등 실험 메타데이터는 원본 대조 후 복구했다.
- 기존 반출 파일은 서버 내부 작업 폴더에 보관했다. 공식 반출신청 제출·승인은 완료하지 않았다.

## 후속 실험

기록 경로: `/home/ubuntu/aihub_run_new/outputs/experiment_records/followup-20260928-2228/`

| 항목 | 확인 상태 | 근거 파일 또는 실행 |
|---|---|---|
| 동일 OAS 조건 비교 | 완료. 4개 모델 × mean/kmeans 8개 결과를 동일 test 16,570장으로 대조 | `oas_comparison.csv`, `oas_verification.json` |
| ViT·Band-ViT galcov | 실행 중. ViT mean/LW 완료 후 OAS 계산 진행 | `followup-vit-band-galcov-20260928` 컨테이너 |
| 속성 A/B 클래스 홀드아웃·백본 고정 | 사전 검사 통과, 큐 대기 | `heldout`, `heldout-frozen` |
| ViT SupCon 5에포크 | 모델 로드·forward 검사 통과, 큐 대기 | `vit-supcon5` |
| fine-tuning 체크포인트 기반 metric 5에포크 | assert 제한을 별도 실행 사본에서 수정. 모델 로드·forward 검사 통과, 큐 대기 | `dinov2-ft5`; SupCon/Triplet/ArcFace/CovAware SupCon |
| 속성 A/B seed·분할 반복 및 CI | seed 42/43/44, A/B 총 6회 학습을 큐에 등록. 반복 종료 후 집계 | `repeat_holdouts.json`, `repeat_summary.py` |
| M10 CovAware | 9/24 재실행은 5에포크 및 평가 정상 종료(exit 0). 성능 저하였으며 동일 조건 재실행은 생략 | `m10_audit.txt` |
| 41클래스 Seen/Unseen 거리 분석 | 기존 분석 완료 확인. 원본과 16개 분할 캐시의 라벨·피처가 정확히 일치 | `geometry41_verified.csv` 32행 |
| M1/M6 백본 캐시 연결 | 같은 체크포인트 경로·SHA256, test 이미지 집합 및 cosine 점수 전체 일치 | `m1_m6_geometry_link.csv` 4행 |
| 속성 점수 결합 입력 연결 | 속성 출력·metric 캐시·거리 점수의 이미지 16,570개 및 정답 일치 | `fusion_inputs_verified.csv` |

큐 PID는 `1330161`. `queue_status.json`과 `queue.log`에 상태를 기록한다. galcov 종료를 기다린 뒤 8개 작업을 한 번에 하나씩 실행하며, 작업별 종료 코드와 로그를 남긴다. 큐 전체 완료는 아직 확인하지 않았다.

## 해석 시 주의

- OAS kmeans 전체 정확도: 이미지 기준 84.8582%, 속성 토큰 5에포크 85.4255%, 20에포크 90.2897%, 예측 속성 토큰 82.6494%. 같은 순서의 Unseen 정확도는 79.0336%, 78.1327%, 73.0549%, 37.5102%다. 전체 정확도 상승과 Unseen 개선을 구분해야 한다.
- 기존 Macro-F1에는 빈 UNKNOWN 클래스가 포함되어 있어 비교 CSV에 실제 41클래스 기준 Macro-F1도 별도 기록했다.
- OAS 정확도 CI는 고정 test 이미지의 Wilson 구간이다. 반복 실험 CI는 seed와 홀드아웃 클래스가 함께 바뀌는 3회 결과의 t 구간으로, 환자 단위 CI가 아니다.
- 속성 A/B는 원래 준비된 설정대로 백본 고정 여부와 head 학습률이 함께 다르다(A 1e-4, B 1e-3). 백본 고정만의 효과로 해석할 수 없다.
- M1/M6 연결은 체크포인트·이미지·점수·기존 거리 분석의 출처 검증이다. 두 평가 실행의 내부 피처 배열을 직접 비교한 것은 아니다.
- 이번 후속 실험의 새 결과는 위 기록 경로에 저장하며, 완료된 반출 폴더에 자동 추가하지 않는다.
