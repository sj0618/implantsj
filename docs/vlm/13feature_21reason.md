치과 임플란트 VLM Feature Extraction 실험 정리
1. 실험 목적
이번 실험의 목적은 VLM이 임플란트 방사선 ROI 이미지에서 브랜드명을 직접 맞히는 것이 아니라, 임플란트 구조적 특징을 추출할 수 있는지 확인하는 것이다.
최종적으로는 다음 구조를 사용한다.
ViT image feature
+ VLM structural feature
→ implant system classification 보조
단, VLM feature가 유의미하지 않으면 최종 판단은 ViT-only로 fallback한다.
***2. 기존 문제점
초기에는 VLM에게 아래와 같은 고수준 label을 직접 추출하게 했다.
connection_type = external / internal / tissue_level_internal / uncertain
platform_switching = switching / matching / uncertain
thread_pitch = fine / medium / coarse / uncertain
하지만 실제 결과에서 다음 문제가 발생했다.
connection_type = uncertain
confidence = 0.0
platform = uncertain
thread = uncertain
mean_vlm_confidence = 0.0
이 경우 JSON 생성은 성공했지만, feature extraction은 실패한 것이다.
따라서 status=ok는 API 호출 성공만 의미하고, feature 사용 가능 여부와 분리해야 한다.
api_status = ok
feature_status = usable / partial_usable / low_confidence / invalid_roi
usable_for_feature_vector = true / false
***3. 방향 전환: 분류가 아니라 존재 여부 검사
VLM에게 “connection type이 무엇인가?”를 묻는 방식은 불안정하다.
대신 다음처럼 바꾼다.
특정 구조 feature가 보이는가?
→ present / absent / not_assessable
상태 정의:
present
해당 구조가 실제로 보임
absent
관련 부위가 충분히 보이는데 해당 구조가 없음
not_assessable
가림, 흐림, crop 문제, 왜곡, 상부 구조물 때문에 판단 불가
중요한 점은 absent와 not_assessable을 반드시 구분하는 것이다.
***4. 하위 21개 supporting visual features
아래 21개 feature는 이미지에서 직접 관찰하는 하위 근거 feature다.
implant_visible
ROI 안에 임플란트 fixture가 보이는지입니다.
connection_region_visible
상부 platform/connection 부위가 보이는지입니다.
thread_region_visible
나사산 영역이 보이는지입니다.
abutment_or_prosthesis_present
abutment 또는 prosthesis 같은 상부 구조물이 연결되어 보이는지입니다.
positive_connection_region_present
connection 부위가 바깥으로 돌출된 positive/external 형태처럼 보이는지입니다.
straight_screw_entry_space_present
screw entry 공간이 직선형으로 보이는지입니다.
negative_internal_connection_space_present
내부로 파인 negative/internal connection 공간이 보이는지입니다.
platform_step_present
platform 부위에서 step이 보이는지입니다.
internal_small_void_present
platform 아래나 screw channel 주변에 작은 내부 void가 보이는지입니다.
smooth_no_thread_collar_present
상부에 나사산 없는 smooth collar가 보이는지입니다.
machined_surface_present
길고 매끈한 machined surface 영역이 보이는지입니다.
platform_boundary_visible
fixture platform 경계가 보이는지입니다.
platform_switching_step_present
abutment와 fixture 사이 폭 차이, 즉 platform switching step이 보이는지입니다.
thread_lines_visible
나사산 선들이 명확하게 보이는지입니다.
regular_thread_pitch_visible
나사산 간격이 규칙적으로 관찰되는지입니다.
deep_thread_relief_present
깊게 파인 thread relief/cut이 보이는지입니다.
micro_thread_zone_present
상부 쪽의 좁고 미세한 micro-thread zone이 보이는지입니다.
macro_thread_zone_present
더 굵고 큰 macro-thread zone이 보이는지입니다.
body_taper_present
임플란트 body가 apex 방향으로 좁아지는 taper 형태인지입니다.
parallel_body_wall_present
body 측벽이 거의 평행하게 보이는지입니다.
apex_boundary_visible
아래쪽 apex 경계가 보이는지입니다.
***5. 핵심 13개 core attributes
프로젝트 기준으로 아래 13개 항목은 반드시 출력해야 하는 핵심 attribute다.
Company
Name
System
Connection_1
Connection_2
Flange
Collar
Microthread
Body_Shape
Body_Type
Thread_Shape
Apex_Shape
Apex_Hole
규칙:
1. 어떤 이미지든 13개 항목은 반드시 출력한다.
2. 확실히 판단되면 값을 출력한다.
3. 판단 불가능하면 unknown을 출력한다.
4. 항목 자체를 누락하지 않는다.
5. 13개 attribute에는 confidence를 넣지 않는다.
6. confidence는 하위 21개 supporting feature에서 담당한다.
***6. 13개 attribute와 21개 feature의 관계
13개 core attributes는 상위 요약값이고, 21개 supporting features는 이를 뒷받침하는 시각적 근거다.
13개 core attributes = 최종 구조 요약
21개 supporting features = 근거 feature + confidence
매핑 관계:
13개 항목	관련 21개 feature	관계
Company	없음	직접 이미지 feature 아님
Name	없음	직접 이미지 feature 아님
System	smooth_no_thread_collar_present, machined_surface_present	Tissue Level / Bone Level 추론
Connection_1	positive_connection_region_present, straight_screw_entry_space_present, negative_internal_connection_space_present, platform_step_present, internal_small_void_present	External / Internal 추론
Connection_2	없음	HEX / OCTA 직접 판단 어려움
Flange	body_taper_present, parallel_body_wall_present, platform_boundary_visible	일부 간접 추론
Collar	smooth_no_thread_collar_present, machined_surface_present	강한 대응
Microthread	micro_thread_zone_present	직접 대응
Body_Shape	body_taper_present, parallel_body_wall_present	강한 대응
Body_Type	thread_lines_visible	직접 대응
Thread_Shape	deep_thread_relief_present, regular_thread_pitch_visible	부분 대응
Apex_Shape	apex_boundary_visible	apex 보임 여부만 직접 대응
Apex_Hole	없음	현재 21개 feature에는 직접 없음
***7. 최종 VLM 출력 구조
최종 프롬프트는 아래 구조를 사용한다.
{
  "api_status": "ok",
  "core_attributes": {
    "Company": {"value": "unknown"},
    "Name": {"value": "unknown"},
    "System": {"value": "Bone Level / Tissue Level / unknown"},
    "Connection_1": {"value": "External / Internal / unknown"},
    "Connection_2": {"value": "HEX / OCTA / unknown"},
    "Flange": {"value": "Convergent / Parallel / unknown"},
    "Collar": {"value": "Yes / No / unknown"},
    "Microthread": {"value": "Yes / No / unknown"},
    "Body_Shape": {"value": "Non-tapered / Tapered apex / Tapered body / unknown"},
    "Body_Type": {"value": "Threaded / unknown"},
    "Thread_Shape": {"value": "V-shaped / unknown"},
    "Apex_Shape": {"value": "Flat / Rounded / unknown"},
    "Apex_Hole": {"value": "Yes / No / unknown"}
  },
  "supporting_features": {
    "implant_visible": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "connection_region_visible": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "thread_region_visible": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "abutment_or_prosthesis_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "positive_connection_region_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "straight_screw_entry_space_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "negative_internal_connection_space_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "platform_step_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "internal_small_void_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "smooth_no_thread_collar_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "machined_surface_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "platform_boundary_visible": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "platform_switching_step_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "thread_lines_visible": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "regular_thread_pitch_visible": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "deep_thread_relief_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "micro_thread_zone_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "macro_thread_zone_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "body_taper_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "parallel_body_wall_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "apex_boundary_visible": {"state": "present / absent / not_assessable", "confidence": 0.0}
  },
  "feature_status": {
    "value": "usable / partial_usable / low_confidence / invalid_roi"
  },
  "usable_for_feature_vector": true,
  "failure_reason": "none / implant_not_visible / all_features_not_assessable / connection_obscured / thread_not_visible / apex_not_visible / low_confidence"
}
***8. Feature vector 처리 방식
21개 supporting feature는 다음 방식으로 vector화한다.
present
→ value = 1
→ mask = 1
absent
→ value = 0
→ mask = 1
not_assessable
→ value = 0
→ mask = 0
confidence는 별도 값으로 포함한다.
각 feature는 다음 세 요소를 가진다.
[value, assessable_mask, confidence]
21개 feature이므로 기본 vector 차원은 다음과 같다.
21 × 3 = 63차원
주의:
not_assessable을 one-hot feature로 그대로 넣으면 모델이 "판단 불가" 자체를 class signal로 학습할 수 있다.
따라서 not_assessable은 mask-out하는 것이 더 안전하다.
***9. 실험 구조
비교 실험군:
A. ViT-only
이미지 feature만 사용
B. VLM supporting feature only
21개 supporting feature vector만 사용
C. ViT + VLM supporting feature
ViT embedding과 VLM feature를 결합
D. ViT + shuffled VLM feature
VLM feature를 섞어서 leakage 또는 우연한 효과 확인
E. Confidence-filtered VLM feature
usable 또는 partial_usable 샘플만 VLM feature 사용
성공 기준:
ViT + VLM > ViT-only
단, shuffled VLM에서도 성능이 오르면 안 된다.
주요 평가 지표:
Top-1 accuracy
Top-3 accuracy
Macro-F1
Per-class recall
Confusion matrix
***10. VLM feature 사용 여부 판단
VLM feature가 다음 조건이면 사용하지 않는다.
implant_visible = absent
all supporting features = not_assessable
mean confidence가 너무 낮음
connection/thread/body/apex 중 사용 가능한 feature가 거의 없음
이 경우 최종 inference는 ViT-only를 사용한다.
usable_for_feature_vector = false
→ ViT-only prediction
일부만 사용 가능한 경우:
feature_status = partial_usable
→ assessable_mask가 1인 feature만 사용
***11. 실제 샘플 해석 예시
Dentium 샘플에서 다음과 같은 결과가 나왔다.
implant_visible = present
abutment_or_prosthesis_present = present
connection 관련 feature 대부분 = not_assessable
thread_lines_visible = present
regular_thread_pitch_visible = present
macro_thread_zone_present = present
body_taper_present = present
apex_boundary_visible = present
해석:
상부 구조물이 있어 connection/platform 판단은 어렵다.
하지만 thread/body/apex 계열 feature는 사용 가능하다.
따라서 feature_status는 usable이 아니라 partial_usable로 보는 것이 맞다.
***12. 최종 결론
이번 실험의 핵심 구조는 다음과 같다.
1. VLM은 브랜드명이나 시스템명을 직접 맞히지 않는다.
2. VLM은 13개 core attributes와 21개 supporting visual features를 출력한다.
3. 13개 core attributes는 항상 출력하되 confidence는 넣지 않는다.
4. 판단 불가능한 core attribute는 unknown으로 둔다.
5. 21개 supporting features는 state와 confidence를 포함한다.
6. 21개 feature가 13개 attribute의 판단 근거가 된다.
7. VLM feature가 불충분하면 ViT-only로 fallback한다.
최종 실험 관점에서 가장 중요한 원칙:
13개 = 최종 구조 요약
21개 = 근거와 confidence
ViT = 기본 판단 모델
VLM = 검증된 경우에만 보조 feature
***13. 최종 프롬프트 핵심 요약
You are analyzing a dental implant ROI radiograph.
Task:
Extract both:
1. 13 required core implant attributes
2. 21 supporting visual features
Do not use file path, folder name, dataset label, split name, or metadata.
Use only visible evidence from the radiograph.
Do not guess from prior knowledge.
Return only valid JSON.
Do not include explanatory text outside JSON.
Do not use null.
Important:
- The 13 core attributes must always be returned.
- If a core attribute cannot be determined, use "unknown".
- The 13 core attributes do not need numeric confidence.
- The 21 supporting visual features must include state and confidence.
- The 21 supporting features are the evidence used to derive the 13 core attributes.
- Do not confuse "absent" with "not_assessable".
- Use "absent" only when the relevant region is clearly visible and the feature is clearly not present.
- Use "not_assessable" when the region is hidden, cropped, blurred, distorted, or obscured by abutment/prosthesis.
Allowed states for 21 supporting features:
- present
- absent
- not_assessable
Allowed value for unknown core attributes:
- unknown
