# 임플란트 라벨링 표 웹 검증 결과 (2026-09-24)

- 대상: `임플란트 라벨링 (2).xlsx`. 49행이며, Company·Name을 뺀 11개 속성을 검증했습니다.
- 방법: 제조사별 5개 그룹으로 나눠 웹 조사 에이전트가 병렬로 확인했습니다. 출처 우선순위는 제조사 카탈로그·IFU·제품 페이지, 그다음 식별 DB(주로 SpotImplant), 그다음 논문입니다. 에이전트가 실제로 연 URL만 근거로 인정했습니다.
- 한계: 제가 원문과 다시 대조하지는 않았습니다. SpotImplant는 가입 요구로, Straumann·Nobel 웹은 403으로 막힌 곳이 많습니다. DIO와 Cowellmedi는 제조사 공식 문서를 하나도 열지 못했습니다.

## 1. 집계

| 그룹 | 셀 | 일치 | 불일치 | 정의모호 | 확인불가 |
|---|---|---|---|---|---|
| Osstem 7 | 77 | 44 | 12 | 7 | 14 |
| Dentium·NeoBiotech·Shinhung 9 | ~99 | ~42 | 5(+3 저확신) | ~9 | ~43 |
| DIO·Megagen·Dentis·Cowell 12 | 132 | 71 | 7 | 20 | 34 |
| Warantec·Cybermed·Straumann·Nobel 13 | — | — | 8 | 다수 | 다수 |
| Dentsply·Zimmer·3i·BioHorizons·BEGO·Bicon 11 | 120 | ~76 | 28 | 13 | 3 |

## 2. 수정 제안: 높음·중상 확신 (보수적 수정본에 반영)

| 행 | 제품 | 속성 | 원본 → 제안 | 근거 |
|---|---|---|---|---|
| 4 | Osstem TSIII | Body Shape | Non-tapered → Tapered body | 공식 카탈로그 "1.5° tapered body" |
| 6 | Osstem SSIII | Body Shape | Non-tapered → Tapered body | 공식 카탈로그 (SpotImplant와 상충) |
| 7 | Osstem USIII | Body Shape | Non-tapered → Tapered body | 공식 카탈로그 "Taper body design" |
| 10 | Dentium SimpleLine(II) | System / Conn2 | Bone Level→Tissue Level / HEX→OCTA | 공식 카탈로그 G/H 2.0mm, "8° Morse Taper Octagon" |
| 21 | Megagen ExFeel_Internal | Collar | No → Yes | 공식 브로셔 transgingival cuff |
| 24 | Megagen AnyOne_Internal | Collar | Yes → No | 공식 도면: 나사산이 플랫폼까지 이어짐 |
| 31 | Straumann TS Standard | Collar | No → Yes | 공식 PDF: 2.8mm smooth neck |
| 32 | Straumann Standard Plus | Collar | No → Yes | 공식 PDF: 1.8mm machined neck. 45행과 같은 제품이므로 병합 |
| 33 | Straumann Bone Level | Conn2 | HEX → CrossFit(15° 원뿔 + 홈 4개) | 공식 PDF 2건 |
| 34 | Nobel Brånemark | Microthread / Collar | Yes→No / No→Yes | 식별 DB 2건 |
| 35 | Nobel Replace Select | Collar | No → Yes | Nobel 공식 "Collar: Machined" |
| 36 | Astra (OsseoSpeed TX) | Conn2 | HEX → Double hex(12-point) | 공식 카탈로그 |
| 37 | XiVE | Collar | No → Yes | 공식 매뉴얼 "structure-polished collar" |
| 38 | Zimmer SwissPlus | Collar | No → Yes | ZimVie "fluted machined neck" |
| 39 | Zimmer Screw-Vent | Collar | No → Yes | 3.6mm machined collar |
| 40 | Zimmer TSV | Collar | No → Yes | TSV 1.0mm machined collar (TSVT 변형만 No) |
| 42 | BioHorizons External | Thread | V-shaped → Square | 제조사 "square thread design" |
| 44 | Bicon Integra-CP | Body Type / Thread | Threaded→Plateau(fin) / V→N/A | 제조사·논문: 스크루형이 아닌 plateau fin, press-fit 식립 |
| 47 | Warantec IT | Thread | V-shaped → Square | Warantec 공식 |

## 3. 중간 이하 확신 (추가 확인 필요, 전체 제안본에만 반영)

| 행 | 제품 | 속성 | 원본 → 제안 | 근거와 확신 |
|---|---|---|---|---|
| 4·6·7 | Osstem TSIII/SSIII/USIII | Apex | Flat → Rounded | 식별 DB 단독, 중간 |
| 5 | Osstem USII | Body Shape | Tapered apex → Non-tapered | 공식 "straight body". USII와 USIII 값이 뒤바뀌어 입력된 것으로 보임 |
| 18·19 | DIO UF / UF2 | Microthread | Yes → No | DB 단독, 0.65–0.70 |
| 21·22 | ExFeel Int/Ext | Microthread | Yes → No | DB 단독, 0.65 |
| 20 | AnyRidge | Collar | No → Yes | 공식 "thread-less collar", 0.70 |
| 9 | SuperLine | Thread | V → Buttress | DB 단독, 0.55 |
| 49 | Stella | Body Shape | Tapered apex → Tapered body | DB 단독, 0.50 |
| 36 | Astra | Body Shape | Non-tapered → Tapered apex | DB, 중간 |
| 35 | Replace Select | Body Shape | Tapered apex → Tapered body | DB와 제조사 설명 일치, 중간 |
| 37·38·41·42 | XiVE, SwissPlus, 3i, BioHorizons | Microthread | Yes → No | DB, 중간 |
| 41 | 3i | Collar | No → Yes | Hybrid 모델 기준, 변형 의존 |
| 44 | Bicon | Apex | Rounded → Flat | DB 단독 |
| 46 | Ankylos | Conn2 | Hex → Six-spline(X) / 없음(C) | C/X 구분 필요 |

## 4. 반복해서 드러난 라벨 정의 문제 (먼저 정해야 함)

1. **Collar**: 원본의 Collar=No 가운데 10건 이상이 제조사 자료로는 machined 또는 smooth collar가 있는 제품이었습니다. 라벨링 기준이 제조사 설계보다 좁게 적용된 것으로 보입니다.
   - 제안: "나사산이 없는 관상부 높이가 X mm 이상이면 Yes"처럼 높이 기준을 명시합니다. 방사선에서 보이는지를 기준으로 할지, 설계 사양을 기준으로 할지도 함께 정해야 합니다.
2. **Flange "Convergent"**: tissue-level 목은 실제로 위로 벌어지는 flared 형태라 정의와 방향이 반대입니다.
   - 제안: 값을 Parallel / Flared / Convergent 세 가지로 나눕니다.
3. **Body Shape**:
   - Tapered body와 Tapered apex의 경계 기준이 없습니다.
   - taper-straight-taper 같은 복합형(EB, Luna S, Core1)을 넣을 칸이 없습니다.
   - 외형 기준인지 코어 기준인지 정해야 합니다(Ankylos, XiVE).
4. **Microthread와 microgroove**: 홈(NeoBiotech IS, TSVM)과 미세 나사산을 구분해야 합니다.
5. **Thread**: 한 제품에 형태가 섞인 경우(Warantec IU: 상부 square, 하부 V)의 기준 부위를 정해야 합니다. 문헌끼리 상충하는 제품(Straumann, Replace)도 있습니다.
6. **Conn2 어휘 확장**: CrossFit, double hex, six-spline, No anti-rotational 값이 필요합니다. HEX/Hex/Hexagon, OCTA/Octagon 표기도 통일해야 합니다.
7. **Body Type**: Plateau/fin 값을 추가합니다(Bicon).
8. **모델 변형 특정**: 행 이름만으로는 변형을 알 수 없는 제품이 있습니다.
   - SimpleLine I/II, Luna1(LUNA/Luna S), ExFeel·Atlas(Straight/Taper), TSV(TSV/TSVM/TSVT), 3i(Hybrid/Full, Parallel/Tapered), Ankylos(C/X), BEGO(RS/RSX), SwissPlus(플랫폼별 hex/octa)
9. **중복**: 32행과 45행은 같은 제품(Standard Plus)이므로 45를 32에 병합합니다. 29행("Warantec Internal")과 47행(IT)도 같은 제품일 가능성이 있습니다.

## 5. 속성 판별력 재계산 (Company·Name 제외 11개 속성)

| 버전 | 42종 고유 조합 | 충돌 그룹 |
|---|---|---|
| 원본 | 36/42 | SSII=SSIII, USII=AnyOne_Ext, SuperLine=Core1, UF=XiVE, ExFeel_Ext=Brånemark=3i |
| 보수적 수정 (§2) | 37/42 | TSIII=SuperLine=Core1, USII=AnyOne_Ext, ExFeel_Ext=3i, SClean=XiVE |
| 전체 제안 (§2+§3) | 39/42 | ISII=UF2, ISIII=AnyRidge, Brånemark=3i |

- 48종 기준(45를 32에 병합)으로는 원본 39/48, 보수적 40/48, 전체 43/48입니다.
- 어느 버전이든 속성만으로 모든 클래스를 구분할 수는 없습니다. 속성은 이미지 임베딩을 보조하는 용도로 쓰는 것이 맞다는 결론은 바뀌지 않습니다.
- 충돌 그룹의 구성원은 버전마다 달라집니다. 논문에서 "속성으로 구분할 수 없는 쌍"을 분석 축으로 쓰려면 라벨을 먼저 확정해야 합니다.

## 6. 파일

- `docs/implant_attributes_corrected_conservative.csv`: §2의 수정을 반영하고 표기를 정규화한 버전입니다. 수정 내역은 `changed` 열에 있습니다. 원본 xlsx는 수정하지 않았습니다.
