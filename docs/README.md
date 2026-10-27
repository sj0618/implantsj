# 문서 목차

실행은 [프로젝트 README](../README.md), 세부 절차는 아래 가이드를 확인합니다.
분석·결과·작업 기록은 작성 당시 기준이며 현재 서버 상태를 뜻하지 않습니다.

## 실행 가이드

- [시작 가이드](usage/QUICKSTART.md): Docker 반입, 실행 메뉴, 설정, 문제 해결
- [실험 가이드](usage/EXPERIMENT_GUIDE.md): 학습·평가 명령과 결과 비교
- [보고서 재현 실행기](../tools/README.md): 실행 설정, 입력, 검증 범위
- [명령 예시 출력](usage/USAGE.py): `python3 docs/usage/USAGE.py`
- [폐쇄망 데이터 구조](analysis/CLOSED_NETWORK_DATASET_LAYOUT.md)
- [데이터 구조 변경 검토](analysis/MODELS_DATASET_VIT_CHANGE_GUIDE.md)

## 분석과 실험 기록

- [ViT·OAS seen/unseen 교수님 보고서](reports/mahalanobis_professor_20261006/report.md)

- [프로젝트 분석](analysis/PROJECT_ANALYSIS.md)
- [실험 결과 요약](analysis/EXPERIMENT_SUMMARY.md)
- [발표 요약](analysis/PRESENTATION_SUMMARY.md)
- [Metric 진단 결과](reports/METRIC_DIAGNOSIS_CURRENT_RESULTS.md)
- [t-SNE·UMAP 비교](analysis/TSNE_UMAP_COMPARISON.md)
- [t-SNE·UMAP 연구 계획](analysis/RESEARCH_PLAN_TSNE_UMAP.md)

## VLM

- [13feature·21reason 정의](vlm/13feature_21reason.md)
- [5feature 프롬프트](vlm/vlm_feature_prompt_5feature.md)
- [5feature fusion 사용법](vlm/VLM_5FEATURE_FUSION_USAGE.md)
- [Azure 6-label 파이프라인](vlm/AZURE_VLM_6LABEL_FEATURE_PIPELINE.md)
- [3-label 실행 명령](vlm/VLM_3LABEL_RUN_COMMANDS.md)
- [3-label 적용 상태 기록](vlm/VLM_3LABEL_APPLY_STATUS.md)

## 과거 작업 기록

- [작업 기록 목차](context/README.md): 기존 `md/` 문서 모음
- [2026-05-26 인수인계](analysis/CURRENT_CONTEXT_HANDOFF.md)
- [이전 코드 보관소](../legacy/README.md)

이전 기록에 나오는 절대 경로·실행 상태·작업 우선순위는 당시 환경에 해당합니다.
현재 실행 명령은 `./run.sh --help`와 `python3 tools/run.py --help`에서 확인합니다.
