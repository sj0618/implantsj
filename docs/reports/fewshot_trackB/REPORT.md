# few-shot 파일럿 결과 보고서

- 생성 2026-09-15 07:51, 결과 루트 `/Users/yunseojune/TRIA_LAB/implant_project/implantsj/outputs/fewshot`, 데이터 루트 `/Users/yunseojune/TRIA_LAB/implant_project/data/fewshot_local`. 명세: `docs/paper/FEWSHOT_ATTRIBUTE_SPEC.md`.
- 셀 = 분할 평균 ± 95% CI 반폭(t, df = n−1). 분할 값은 그 분할의 에피소드 평균이다(§2.4). 정확도·F1·특징 정확도 단위는 %. 분할이 5개보다 적으면 `(n=k)`. `—` = 결과 없음.
- 분할별 원값은 `tables/long.csv`, 표마다 `tables/<이름>.csv`.

## 1. 데이터와 분할

### 트랙 B 클래스별 장수 (`class_counts.csv`)

| label           |   n_items |   n_groups |   n_full |   n_panoramic |   n_periapical |   n_catalog | datasets                                                                                    |
|:----------------|----------:|-----------:|---------:|--------------:|---------------:|------------:|:--------------------------------------------------------------------------------------------|
| bicon           |      3803 |        542 |     1732 |          3768 |              0 |          35 | Dental Implants 2.0.v3i.yolov8;FULL_Bego_Bicon_ITI_3label;realyolo26                        |
| straumann       |      1016 |        409 |      582 |           841 |            137 |          38 | Dental Implants 2.0.v3i.yolov8;FULL_Bego_Bicon_ITI_3label;implant_yolo26_12label;realyolo26 |
| osstem          |      3232 |        217 |      857 |          3132 |              0 |         100 | dddd;realyolo26                                                                             |
| bego            |       796 |        189 |      472 |           298 |            498 |           0 | Dental Implants 2.0.v3i.yolov8;FULL_Bego_Bicon_ITI_3label;implant_yolo26_12label            |
| adin            |       724 |        151 |      420 |           724 |              0 |           0 | dddd                                                                                        |
| noris           |       645 |         87 |      314 |           645 |              0 |           0 | dddd                                                                                        |
| nobel           |       210 |         77 |      213 |             5 |             46 |         159 | dddd;implant_yolo26_12label;realyolo26                                                      |
| cowellmedi      |       216 |         76 |      228 |             0 |              0 |         216 | realyolo26                                                                                  |
| ags_medikal     |       161 |         59 |      127 |             0 |            137 |          24 | implant_yolo26_12label;realyolo26                                                           |
| camlog          |       103 |         48 |      165 |             0 |              0 |         103 | realyolo26                                                                                  |
| biomet_3i       |       150 |         47 |      177 |             0 |            112 |          38 | implant_yolo26_12label;realyolo26                                                           |
| implant_direct  |        72 |         41 |      153 |             0 |              0 |          72 | realyolo26                                                                                  |
| mis             |       237 |         40 |       89 |           228 |              0 |           9 | dddd;realyolo26                                                                             |
| biohorizons     |        96 |         40 |      112 |             0 |             26 |          70 | implant_yolo26_12label;realyolo26                                                           |
| nucleoss        |        83 |         35 |       71 |             0 |             55 |          28 | implant_yolo26_12label;realyolo26                                                           |
| medentika       |        48 |         34 |       38 |             0 |             48 |           0 | implant_yolo26_12label                                                                      |
| bilim           |        56 |         32 |       40 |             0 |             56 |           0 | implant_yolo26_12label                                                                      |
| euroteknika     |        70 |         29 |       90 |             0 |              0 |          70 | realyolo26                                                                                  |
| dentsply        |        45 |         27 |      101 |             0 |              0 |          45 | realyolo26                                                                                  |
| novodent        |        53 |         22 |       75 |             0 |              0 |          53 | realyolo26                                                                                  |
| keystone_dental |        34 |         22 |      107 |             0 |              0 |          34 | realyolo26                                                                                  |
| dentium         |       472 |         19 |      224 |           469 |              0 |           3 | dddd;realyolo26                                                                             |
| sterngold       |        26 |         17 |       60 |             0 |              0 |          26 | realyolo26                                                                                  |
| zimmer          |        22 |         17 |       79 |             0 |              0 |          22 | realyolo26                                                                                  |
| 3m_espe         |        40 |         16 |       70 |             0 |              0 |          40 | realyolo26                                                                                  |
| anthogyr        |        26 |         16 |       65 |             0 |              0 |          26 | realyolo26                                                                                  |
| neoss           |        26 |         15 |       48 |             0 |              0 |          26 | realyolo26                                                                                  |
| neodent         |        26 |         13 |      102 |             0 |              0 |          26 | realyolo26                                                                                  |
| megagen         |        24 |         11 |       19 |             0 |             23 |           1 | implant_yolo26_12label;realyolo26                                                           |
| blue_sky_bio    |        17 |         10 |       42 |             0 |              0 |          17 | realyolo26                                                                                  |
| oco_biomedical  |        18 |          9 |       33 |             0 |              0 |          18 | realyolo26                                                                                  |
| swiss           |        17 |          9 |       12 |             0 |             17 |           0 | implant_yolo26_12label                                                                      |
| leone           |        15 |          6 |       18 |             0 |              0 |          15 | realyolo26                                                                                  |
| dionavi         |        17 |          5 |        6 |            17 |              0 |           0 | dddd                                                                                        |
| nta             |         8 |          5 |        7 |             0 |              8 |           0 | implant_yolo26_12label                                                                      |
| titan_implants  |         8 |          4 |       12 |             0 |              0 |           8 | realyolo26                                                                                  |
| biolife         |         5 |          3 |        9 |             0 |              0 |           5 | realyolo26                                                                                  |
| dentatus        |         5 |          2 |        6 |             0 |              0 |           5 | realyolo26                                                                                  |
| osseolink       |         3 |          2 |        9 |             0 |              0 |           3 | realyolo26                                                                                  |
| frontier        |         2 |          1 |        3 |             0 |              0 |           2 | realyolo26                                                                                  |
| ameross         |         1 |          1 |        3 |             0 |              0 |           1 | realyolo26                                                                                  |
| dentis          |         1 |          1 |        3 |             0 |              0 |           1 | realyolo26                                                                                  |

### 트랙 B 분할 요약 (plan)

way_K = 그 K 에서 평가 가능한 unseen 클래스 수(group ≥ K+1). groups_moved_to_unseen / groups_added = 다른 manifest 를 덮으려고 바꾼 group 수(_cover).

|   split | manifest                 |   n_unseen_classes |   n_seen_classes |   n_train |   n_val |   n_test |   n_unseen |   n_excluded |   way_K1 |   way_K3 |   way_K5 |   way_K10 |   groups_moved_to_unseen |   groups_added | unseen_classes                                                                                                                          |
|--------:|:-------------------------|-------------------:|-----------------:|----------:|--------:|---------:|-----------:|-------------:|---------:|---------:|---------:|----------:|-------------------------:|---------------:|:----------------------------------------------------------------------------------------------------------------------------------------|
|       0 | trackB/manifest_crop.csv |                 15 |               27 |      4951 |     431 |      489 |       6577 |          181 |       15 |       15 |       15 |        15 |                        2 |             47 | 3m_espe adin anthogyr bego bicon biomet_3i camlog dentium euroteknika keystone_dental medentika neodent nobel novodent zimmer           |
|       0 | trackB/manifest_full.csv |                 15 |               27 |      2313 |     255 |      237 |       4029 |          159 |       15 |       15 |       15 |        15 |                        2 |             47 | 3m_espe adin anthogyr bego bicon biomet_3i camlog dentium euroteknika keystone_dental medentika neodent nobel novodent zimmer           |
|       1 | trackB/manifest_crop.csv |                 15 |               27 |      5250 |     818 |      600 |       5901 |           60 |       15 |       15 |       15 |        15 |                        0 |             82 | 3m_espe ags_medikal bicon bilim biohorizons cowellmedi dentsply medentika neoss nobel novodent nucleoss sterngold straumann zimmer      |
|       1 | trackB/manifest_full.csv |                 15 |               27 |      2629 |     401 |      340 |       3576 |           47 |       15 |       15 |       15 |        15 |                        0 |             82 | 3m_espe ags_medikal bicon bilim biohorizons cowellmedi dentsply medentika neoss nobel novodent nucleoss sterngold straumann zimmer      |
|       2 | trackB/manifest_crop.csv |                 15 |               27 |      1653 |     159 |      212 |      10509 |           96 |       15 |       15 |       15 |        15 |                        0 |             58 | adin anthogyr bicon biohorizons camlog dentium euroteknika keystone_dental neodent neoss nobel noris osstem sterngold straumann         |
|       2 | trackB/manifest_full.csv |                 15 |               27 |      1465 |     180 |      177 |       5091 |           80 |       15 |       15 |       15 |        15 |                        0 |             58 | adin anthogyr bicon biohorizons camlog dentium euroteknika keystone_dental neodent neoss nobel noris osstem sterngold straumann         |
|       3 | trackB/manifest_crop.csv |                 15 |               27 |      6801 |     789 |      914 |       3986 |          139 |       15 |       15 |       15 |        15 |                        1 |             69 | 3m_espe adin ags_medikal anthogyr bego biohorizons cowellmedi dentium dentsply keystone_dental medentika mis novodent straumann zimmer  |
|       3 | trackB/manifest_full.csv |                 15 |               27 |      3240 |     374 |      473 |       2789 |          117 |       15 |       15 |       15 |        15 |                        1 |             69 | 3m_espe adin ags_medikal anthogyr bego biohorizons cowellmedi dentium dentsply keystone_dental medentika mis novodent straumann zimmer  |
|       4 | trackB/manifest_crop.csv |                 15 |               27 |      5671 |     691 |      755 |       5450 |           62 |       15 |       15 |       15 |        15 |                        1 |             50 | adin bilim biomet_3i camlog cowellmedi euroteknika implant_direct keystone_dental medentika neodent neoss noris osstem sterngold zimmer |
|       4 | trackB/manifest_full.csv |                 15 |               27 |      3244 |     410 |      398 |       2878 |           63 |       15 |       15 |       15 |        15 |                        1 |             50 | adin bilim biomet_3i camlog cowellmedi euroteknika implant_direct keystone_dental medentika neodent neoss noris osstem sterngold zimmer |

### 트랙 S 클래스별 장수 (`class_counts.csv`)

| label                    |   n_items |   n_groups |   n_phantom | datasets     |
|:-------------------------|----------:|-----------:|------------:|:-------------|
| osstem_usii              |         9 |          6 |           9 | aihub_sample |
| dentium_superline        |         7 |          4 |           7 | aihub_sample |
| osstem_usiii             |         6 |          4 |           6 | aihub_sample |
| neobiotech_isii          |        10 |          3 |          10 | aihub_sample |
| osstem_ssii              |         9 |          3 |           9 | aihub_sample |
| dentium_implantium       |         7 |          3 |           7 | aihub_sample |
| osstem_tsiii             |         6 |          3 |           6 | aihub_sample |
| osstem_ssiii             |         5 |          3 |           5 | aihub_sample |
| dentium_simpleline       |         6 |          2 |           6 | aihub_sample |
| zimmer_swissplus         |         6 |          2 |           6 | aihub_sample |
| neobiotech_isiii         |         5 |          2 |           5 | aihub_sample |
| megagen_anyoneinternal   |         6 |          1 |           6 | aihub_sample |
| megagen_anyridge         |         5 |          1 |           5 | aihub_sample |
| biohorizons_external     |         3 |          1 |           3 | aihub_sample |
| cowellmedi_atlasexternal |         3 |          1 |           3 | aihub_sample |
| cowellmedi_atlasinternal |         3 |          1 |           3 | aihub_sample |
| megagen_anyoneexternal   |         3 |          1 |           3 | aihub_sample |
| megagen_exfeelinternal   |         3 |          1 |           3 | aihub_sample |
| straumann_bonelevel      |         3 |          1 |           3 | aihub_sample |
| dentsply_xive            |         2 |          1 |           2 | aihub_sample |
| megagen_exfeelexternal   |         2 |          1 |           2 | aihub_sample |
| neobiotech_eb            |         2 |          1 |           2 | aihub_sample |
| neobiotech_isi           |         2 |          1 |           2 | aihub_sample |
| osstem_gsii              |         2 |          1 |           2 | aihub_sample |
| osstem_gsiii             |         2 |          1 |           2 | aihub_sample |
| shinhung_luna1           |         2 |          1 |           2 | aihub_sample |
| straumann_tsstandard     |         2 |          1 |           2 | aihub_sample |

### 트랙 S 분할 요약 (plan)

way_K = 그 K 에서 평가 가능한 unseen 클래스 수(group ≥ K+1). groups_moved_to_unseen / groups_added = 다른 manifest 를 덮으려고 바꾼 group 수(_cover).

|   split | manifest            |   n_unseen_classes |   n_seen_classes |   n_train |   n_val |   n_test |   n_unseen |   n_excluded |   way_K1 |   way_K2 |   groups_moved_to_unseen |   groups_added | unseen_classes                                                                                                                                                                         |
|--------:|:--------------------|-------------------:|-----------------:|----------:|--------:|---------:|-----------:|-------------:|---------:|---------:|-------------------------:|---------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|       0 | trackS/manifest.csv |                 10 |               17 |        67 |       6 |        6 |         42 |            0 |        4 |        2 |                        0 |              0 | biohorizons_external cowellmedi_atlasexternal dentium_simpleline dentium_superline dentsply_xive megagen_exfeelexternal neobiotech_eb neobiotech_isiii osstem_usii straumann_bonelevel |
|       1 | trackS/manifest.csv |                 10 |               17 |        61 |       9 |        7 |         44 |            0 |        4 |        2 |                        0 |              0 | biohorizons_external dentium_implantium dentsply_xive megagen_anyoneinternal megagen_anyridge neobiotech_isiii osstem_ssiii shinhung_luna1 straumann_bonelevel zimmer_swissplus        |
|       2 | trackS/manifest.csv |                 10 |               17 |        69 |       6 |        3 |         43 |            0 |        3 |        1 |                        0 |              0 | cowellmedi_atlasinternal dentium_simpleline dentsply_xive megagen_anyoneinternal megagen_anyridge megagen_exfeelexternal neobiotech_isiii osstem_gsii osstem_ssii straumann_bonelevel  |
|       3 | trackS/manifest.csv |                 10 |               17 |        55 |       5 |        3 |         58 |            0 |        6 |        5 |                        0 |              0 | cowellmedi_atlasexternal cowellmedi_atlasinternal dentium_implantium dentium_simpleline neobiotech_isii osstem_ssii osstem_tsiii osstem_usii straumann_bonelevel straumann_tsstandard  |
|       4 | trackS/manifest.csv |                 10 |               17 |        75 |       7 |        6 |         33 |            0 |        3 |        3 |                        0 |              0 | cowellmedi_atlasinternal megagen_exfeelexternal megagen_exfeelinternal neobiotech_isi osstem_gsii osstem_gsiii osstem_ssiii osstem_tsiii osstem_usiii shinhung_luna1                   |

## 2. unseen K-shot 정확도와 macro-F1 (표현 × 분류기)

`n` = 결과가 있는 분할 수, `way` = 분할별 N-way 범위.

### 트랙 B, K=1, 정확도

| method       |   n |   way | cos               | euc               | ss                | mt                | mtr               | mw               |
|:-------------|----:|------:|:------------------|:------------------|:------------------|:------------------|:------------------|:-----------------|
| in_vitb      |   5 |    15 | 25.9 ± 3.3        | 24.5 ± 3.1        | 27.0 ± 4.0        | 25.9 ± 3.3        | 25.5 ± 3.2        | 7.6 ± 2.8        |
| dino_vitb    |   2 |    15 | 24.3 ± 22.4 (n=2) | 23.4 ± 23.5 (n=2) | 24.6 ± 20.0 (n=2) | 24.3 ± 22.3 (n=2) | 24.7 ± 19.8 (n=2) | 14.0 ± 3.2 (n=2) |
| in_vits      |   5 |    15 | 26.4 ± 3.3        | 26.8 ± 3.3        | 27.1 ± 3.6        | 26.3 ± 3.3        | 26.4 ± 3.4        | 10.2 ± 4.4       |
| ce_vits      |   5 |    15 | 33.8 ± 7.1        | 33.1 ± 5.7        | 30.9 ± 5.6        | 33.8 ± 7.1        | 31.5 ± 7.2        | 28.7 ± 5.1       |
| proto_vits   |   1 |    15 | 36.0 ± 0.0 (n=1)  | 34.8 ± 0.0 (n=1)  | 33.0 ± 0.0 (n=1)  | 35.9 ± 0.0 (n=1)  | 35.0 ± 0.0 (n=1)  | 27.4 ± 0.0 (n=1) |
| arc_vits_m05 |   5 |    15 | 31.7 ± 6.8        | 29.5 ± 3.7        | 30.0 ± 6.3        | 31.7 ± 6.8        | 30.5 ± 5.0        | 25.9 ± 5.8       |

### 트랙 B, K=1, macro-F1

| method       |   n |   way | cos               | euc               | ss                | mt                | mtr               | mw                |
|:-------------|----:|------:|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| in_vitb      |   5 |    15 | 17.4 ± 2.3        | 17.0 ± 2.5        | 16.7 ± 1.6        | 17.4 ± 2.3        | 17.1 ± 2.4        | 9.8 ± 2.5         |
| dino_vitb    |   2 |    15 | 18.2 ± 26.1 (n=2) | 18.2 ± 26.7 (n=2) | 16.6 ± 22.7 (n=2) | 18.2 ± 26.1 (n=2) | 18.5 ± 29.7 (n=2) | 12.6 ± 19.2 (n=2) |
| in_vits      |   5 |    15 | 18.2 ± 2.4        | 18.9 ± 2.5        | 17.4 ± 1.9        | 18.1 ± 2.4        | 18.0 ± 2.5        | 11.6 ± 3.5        |
| ce_vits      |   5 |    15 | 21.6 ± 2.4        | 21.4 ± 2.8        | 19.0 ± 1.4        | 21.6 ± 2.4        | 19.5 ± 2.9        | 20.6 ± 1.6        |
| proto_vits   |   1 |    15 | 22.7 ± 0.0 (n=1)  | 24.1 ± 0.0 (n=1)  | 20.2 ± 0.0 (n=1)  | 22.6 ± 0.0 (n=1)  | 24.1 ± 0.0 (n=1)  | 20.7 ± 0.0 (n=1)  |
| arc_vits_m05 |   5 |    15 | 20.2 ± 1.9        | 19.6 ± 3.4        | 18.2 ± 2.4        | 20.2 ± 1.9        | 19.8 ± 3.2        | 18.4 ± 2.3        |

### 트랙 B, K=3, 정확도

| method       |   n |   way | cos               | euc               | ss                | mt                | mtr               | mw                |
|:-------------|----:|------:|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| in_vitb      |   5 |    15 | 37.3 ± 4.9        | 36.9 ± 4.6        | 38.3 ± 5.6        | 37.9 ± 5.0        | 39.5 ± 5.4        | 24.4 ± 5.0        |
| dino_vitb    |   2 |    15 | 33.8 ± 25.0 (n=2) | 32.8 ± 19.7 (n=2) | 33.3 ± 16.2 (n=2) | 34.2 ± 24.2 (n=2) | 39.6 ± 27.5 (n=2) | 24.4 ± 24.4 (n=2) |
| in_vits      |   5 |    15 | 36.3 ± 3.5        | 36.5 ± 3.6        | 37.2 ± 4.3        | 37.1 ± 3.7        | 40.2 ± 5.1        | 28.7 ± 5.4        |
| ce_vits      |   5 |    15 | 42.8 ± 11.9       | 42.9 ± 11.0       | 41.2 ± 10.3       | 43.5 ± 11.2       | 44.1 ± 9.3        | 43.5 ± 6.9        |
| proto_vits   |   1 |    15 | 46.2 ± 0.0 (n=1)  | 44.4 ± 0.0 (n=1)  | 43.8 ± 0.0 (n=1)  | 45.6 ± 0.0 (n=1)  | 46.9 ± 0.0 (n=1)  | 42.6 ± 0.0 (n=1)  |
| arc_vits_m05 |   5 |    15 | 40.8 ± 10.3       | 38.9 ± 7.6        | 38.9 ± 8.8        | 41.1 ± 8.9        | 41.6 ± 7.3        | 41.8 ± 7.5        |

### 트랙 B, K=3, macro-F1

| method       |   n |   way | cos               | euc               | ss                | mt                | mtr               | mw                |
|:-------------|----:|------:|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| in_vitb      |   5 |    15 | 26.6 ± 3.3        | 27.2 ± 3.3        | 27.3 ± 2.8        | 28.0 ± 3.3        | 26.9 ± 3.3        | 24.6 ± 5.8        |
| dino_vitb    |   2 |    15 | 26.2 ± 43.9 (n=2) | 26.7 ± 44.5 (n=2) | 26.3 ± 40.1 (n=2) | 27.3 ± 45.9 (n=2) | 28.5 ± 50.0 (n=2) | 25.5 ± 46.4 (n=2) |
| in_vits      |   5 |    15 | 27.0 ± 3.7        | 28.0 ± 3.9        | 27.6 ± 3.5        | 28.2 ± 3.8        | 27.4 ± 3.7        | 25.4 ± 5.2        |
| ce_vits      |   5 |    15 | 28.0 ± 3.2        | 28.4 ± 3.1        | 26.2 ± 1.4        | 29.1 ± 2.9        | 29.8 ± 2.9        | 32.5 ± 3.7        |
| proto_vits   |   1 |    15 | 31.0 ± 0.0 (n=1)  | 33.5 ± 0.0 (n=1)  | 30.0 ± 0.0 (n=1)  | 32.9 ± 0.0 (n=1)  | 35.9 ± 0.0 (n=1)  | 34.7 ± 0.0 (n=1)  |
| arc_vits_m05 |   5 |    15 | 27.0 ± 2.7        | 26.9 ± 4.5        | 25.5 ± 3.0        | 28.2 ± 3.4        | 29.2 ± 4.7        | 31.5 ± 3.1        |

### 트랙 B, K=5, 정확도

| method       |   n |   way | cos               | euc               | ss                | mt                | mtr               | mw                |
|:-------------|----:|------:|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| in_vitb      |   5 |    15 | 42.3 ± 6.7        | 41.7 ± 6.0        | 42.9 ± 7.0        | 42.8 ± 6.6        | 47.2 ± 6.7        | 37.1 ± 4.5        |
| dino_vitb    |   2 |    15 | 38.0 ± 34.2 (n=2) | 36.4 ± 31.6 (n=2) | 36.8 ± 28.2 (n=2) | 38.2 ± 34.7 (n=2) | 47.0 ± 35.2 (n=2) | 34.9 ± 31.7 (n=2) |
| in_vits      |   5 |    15 | 40.8 ± 4.7        | 40.6 ± 4.7        | 41.2 ± 5.2        | 41.5 ± 4.8        | 47.3 ± 6.5        | 40.2 ± 5.4        |
| ce_vits      |   5 |    15 | 46.8 ± 12.3       | 46.8 ± 11.4       | 45.3 ± 11.0       | 47.6 ± 11.5       | 49.0 ± 9.2        | 50.7 ± 6.8        |
| proto_vits   |   1 |    15 | 50.3 ± 0.0 (n=1)  | 48.5 ± 0.0 (n=1)  | 48.7 ± 0.0 (n=1)  | 49.9 ± 0.0 (n=1)  | 52.1 ± 0.0 (n=1)  | 50.3 ± 0.0 (n=1)  |
| arc_vits_m05 |   5 |    15 | 44.3 ± 11.3       | 42.3 ± 8.2        | 42.7 ± 10.0       | 44.8 ± 9.7        | 45.8 ± 7.8        | 49.0 ± 8.3        |

### 트랙 B, K=5, macro-F1

| method       |   n |   way | cos               | euc               | ss                | mt                | mtr               | mw                |
|:-------------|----:|------:|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| in_vitb      |   5 |    15 | 30.2 ± 3.3        | 30.6 ± 3.3        | 30.7 ± 3.0        | 31.5 ± 3.3        | 32.8 ± 3.7        | 32.9 ± 6.3        |
| dino_vitb    |   2 |    15 | 29.8 ± 54.2 (n=2) | 30.0 ± 55.4 (n=2) | 29.7 ± 52.2 (n=2) | 30.8 ± 56.6 (n=2) | 34.1 ± 57.8 (n=2) | 33.1 ± 54.2 (n=2) |
| in_vits      |   5 |    15 | 30.4 ± 4.0        | 31.1 ± 4.2        | 30.8 ± 3.9        | 31.5 ± 4.2        | 33.1 ± 4.2        | 32.9 ± 5.8        |
| ce_vits      |   5 |    15 | 30.5 ± 3.5        | 30.8 ± 3.2        | 28.5 ± 1.6        | 31.6 ± 3.2        | 33.6 ± 2.9        | 37.2 ± 4.6        |
| proto_vits   |   1 |    15 | 34.1 ± 0.0 (n=1)  | 36.8 ± 0.0 (n=1)  | 33.6 ± 0.0 (n=1)  | 36.5 ± 0.0 (n=1)  | 40.3 ± 0.0 (n=1)  | 40.0 ± 0.0 (n=1)  |
| arc_vits_m05 |   5 |    15 | 29.5 ± 2.7        | 29.1 ± 4.5        | 27.9 ± 3.1        | 30.9 ± 3.4        | 33.0 ± 5.1        | 36.2 ± 3.3        |

### 트랙 B, K=10, 정확도

| method       |   n |   way | cos               | euc               | ss                | mt                | mtr               | mw                |
|:-------------|----:|------:|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| in_vitb      |   5 |    15 | 49.1 ± 8.8        | 48.5 ± 7.7        | 49.6 ± 9.0        | 49.9 ± 8.6        | 57.4 ± 8.3        | 52.4 ± 4.3        |
| dino_vitb    |   2 |    15 | 43.9 ± 24.3 (n=2) | 41.3 ± 24.5 (n=2) | 41.4 ± 22.1 (n=2) | 43.8 ± 28.3 (n=2) | 58.7 ± 39.1 (n=2) | 53.0 ± 45.9 (n=2) |
| in_vits      |   5 |    15 | 47.4 ± 6.1        | 47.0 ± 6.1        | 47.8 ± 6.9        | 48.3 ± 6.2        | 57.2 ± 7.8        | 53.5 ± 6.8        |
| ce_vits      |   5 |    15 | 52.3 ± 12.7       | 51.9 ± 12.2       | 50.7 ± 12.2       | 53.1 ± 12.2       | 54.5 ± 8.0        | 59.7 ± 7.5        |
| proto_vits   |   1 |    15 | 55.6 ± 0.0 (n=1)  | 54.2 ± 0.0 (n=1)  | 55.9 ± 0.0 (n=1)  | 56.2 ± 0.0 (n=1)  | 59.5 ± 0.0 (n=1)  | 58.4 ± 0.0 (n=1)  |
| arc_vits_m05 |   5 |    15 | 48.9 ± 10.7       | 46.4 ± 8.4        | 47.3 ± 10.7       | 49.4 ± 9.8        | 51.7 ± 7.5        | 57.2 ± 8.0        |

### 트랙 B, K=10, macro-F1

| method       |   n |   way | cos               | euc               | ss                | mt                | mtr               | mw                |
|:-------------|----:|------:|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| in_vitb      |   5 |    15 | 32.7 ± 3.6        | 32.9 ± 3.6        | 33.0 ± 3.4        | 33.9 ± 3.7        | 39.2 ± 4.1        | 41.1 ± 6.1        |
| dino_vitb    |   2 |    15 | 31.8 ± 53.7 (n=2) | 31.5 ± 54.3 (n=2) | 31.2 ± 51.6 (n=2) | 32.5 ± 56.5 (n=2) | 39.6 ± 61.0 (n=2) | 40.3 ± 67.7 (n=2) |
| in_vits      |   5 |    15 | 32.6 ± 4.0        | 33.1 ± 4.1        | 33.0 ± 4.1        | 33.7 ± 4.2        | 39.3 ± 4.7        | 39.4 ± 6.0        |
| ce_vits      |   5 |    15 | 31.8 ± 3.4        | 31.9 ± 3.1        | 29.7 ± 2.3        | 33.0 ± 3.3        | 36.7 ± 3.6        | 40.3 ± 5.6        |
| proto_vits   |   1 |    15 | 35.2 ± 0.0 (n=1)  | 37.4 ± 0.0 (n=1)  | 35.2 ± 0.0 (n=1)  | 37.6 ± 0.0 (n=1)  | 42.8 ± 0.0 (n=1)  | 42.5 ± 0.0 (n=1)  |
| arc_vits_m05 |   5 |    15 | 30.7 ± 3.0        | 30.0 ± 4.8        | 28.9 ± 3.4        | 32.0 ± 3.7        | 36.1 ± 5.6        | 39.2 ± 3.7        |

![트랙 B K-곡선](kcurve_B.png)

### 트랙 S, K=1, 정확도

| method           |   n | way   | cos         | euc         | ss          | mt          | mtr         | mw          |
|:-----------------|----:|:------|:------------|:------------|:------------|:------------|:------------|:------------|
| in_vitb          |   5 | 3–6   | 50.7 ± 16.6 | 48.9 ± 16.3 | 50.3 ± 14.7 | 50.7 ± 16.6 | 49.8 ± 14.2 | 61.8 ± 8.1  |
| dino_vitb        |   5 | 3–6   | 49.3 ± 17.3 | 49.4 ± 17.5 | 49.2 ± 16.1 | 49.4 ± 17.3 | 48.9 ± 16.8 | 65.1 ± 14.6 |
| in_vits          |   5 | 3–6   | 45.1 ± 17.6 | 44.8 ± 17.6 | 46.6 ± 17.0 | 45.1 ± 17.5 | 45.2 ± 14.9 | 58.9 ± 10.2 |
| arc_vits_m05     |   5 | 3–6   | 70.0 ± 20.4 | 68.0 ± 19.2 | 70.1 ± 19.2 | 69.9 ± 20.2 | 69.2 ± 18.4 | 67.5 ± 19.7 |
| arcattr_vits_m05 |   5 | 3–6   | 70.0 ± 14.5 | 68.2 ± 17.0 | 70.9 ± 15.5 | 70.0 ± 14.4 | 68.9 ± 17.2 | 69.0 ± 16.4 |

### 트랙 S, K=1, macro-F1

| method           |   n | way   | cos         | euc         | ss          | mt          | mtr         | mw          |
|:-----------------|----:|:------|:------------|:------------|:------------|:------------|:------------|:------------|
| in_vitb          |   5 | 3–6   | 49.4 ± 18.1 | 46.1 ± 17.8 | 49.4 ± 16.6 | 49.3 ± 18.1 | 47.5 ± 15.7 | 57.4 ± 10.2 |
| dino_vitb        |   5 | 3–6   | 48.5 ± 19.9 | 48.6 ± 20.0 | 48.9 ± 18.6 | 48.5 ± 19.9 | 48.1 ± 19.4 | 62.0 ± 17.9 |
| in_vits          |   5 | 3–6   | 44.5 ± 18.4 | 44.3 ± 18.6 | 46.0 ± 18.4 | 44.4 ± 18.3 | 44.6 ± 16.1 | 55.6 ± 12.5 |
| arc_vits_m05     |   5 | 3–6   | 67.8 ± 21.0 | 66.4 ± 20.1 | 68.4 ± 19.6 | 67.6 ± 20.7 | 67.5 ± 19.2 | 63.9 ± 21.5 |
| arcattr_vits_m05 |   5 | 3–6   | 68.9 ± 16.3 | 68.3 ± 18.5 | 70.1 ± 17.6 | 68.8 ± 16.3 | 69.1 ± 18.3 | 67.0 ± 19.1 |

### 트랙 S, K=2, 정확도

| method           |   n | way   | cos               | euc               | ss                | mt                | mtr               | mw                |
|:-----------------|----:|:------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| in_vitb          |   4 | 1–5   | 61.1 ± 31.1 (n=4) | 61.9 ± 30.5 (n=4) | 65.0 ± 31.0 (n=4) | 61.1 ± 29.7 (n=4) | 63.5 ± 26.3 (n=4) | 80.5 ± 21.4 (n=4) |
| dino_vitb        |   4 | 1–5   | 58.9 ± 27.7 (n=4) | 58.6 ± 27.7 (n=4) | 59.8 ± 27.6 (n=4) | 59.1 ± 27.7 (n=4) | 61.4 ± 25.2 (n=4) | 80.0 ± 27.1 (n=4) |
| in_vits          |   4 | 1–5   | 53.3 ± 18.0 (n=4) | 53.3 ± 18.1 (n=4) | 52.9 ± 17.6 (n=4) | 53.2 ± 18.0 (n=4) | 56.0 ± 14.6 (n=4) | 78.5 ± 23.7 (n=4) |
| arc_vits_m05     |   4 | 1–5   | 82.2 ± 28.1 (n=4) | 82.5 ± 22.5 (n=4) | 83.2 ± 25.2 (n=4) | 83.2 ± 26.6 (n=4) | 82.1 ± 22.0 (n=4) | 81.8 ± 28.8 (n=4) |
| arcattr_vits_m05 |   4 | 1–5   | 81.5 ± 21.4 (n=4) | 81.6 ± 21.3 (n=4) | 83.3 ± 21.5 (n=4) | 82.9 ± 21.6 (n=4) | 83.3 ± 20.3 (n=4) | 81.2 ± 21.3 (n=4) |

### 트랙 S, K=2, macro-F1

| method           |   n | way   | cos               | euc               | ss                | mt                | mtr               | mw                |
|:-----------------|----:|:------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| in_vitb          |   4 | 1–5   | 55.1 ± 34.7 (n=4) | 55.1 ± 33.7 (n=4) | 59.6 ± 36.9 (n=4) | 55.1 ± 33.7 (n=4) | 56.5 ± 30.3 (n=4) | 75.8 ± 27.4 (n=4) |
| dino_vitb        |   4 | 1–5   | 52.8 ± 30.3 (n=4) | 52.6 ± 30.6 (n=4) | 54.7 ± 31.7 (n=4) | 53.0 ± 30.5 (n=4) | 55.6 ± 28.1 (n=4) | 75.0 ± 34.3 (n=4) |
| in_vits          |   4 | 1–5   | 47.5 ± 21.3 (n=4) | 47.3 ± 21.1 (n=4) | 46.8 ± 21.0 (n=4) | 47.2 ± 21.2 (n=4) | 50.2 ± 18.7 (n=4) | 73.8 ± 28.6 (n=4) |
| arc_vits_m05     |   4 | 1–5   | 79.1 ± 32.0 (n=4) | 80.3 ± 24.8 (n=4) | 80.9 ± 28.6 (n=4) | 80.5 ± 30.3 (n=4) | 79.5 ± 25.8 (n=4) | 78.9 ± 32.7 (n=4) |
| arcattr_vits_m05 |   4 | 1–5   | 77.9 ± 26.9 (n=4) | 79.5 ± 24.0 (n=4) | 80.1 ± 27.3 (n=4) | 79.9 ± 26.8 (n=4) | 80.8 ± 25.5 (n=4) | 77.5 ± 26.3 (n=4) |

![트랙 S K-곡선](kcurve_S.png)

## 3. 기준선 표 (§3)

Simple CNAPS 헤드 = 각 표현 + `mtr`(정규화 전 특징 공간의 계층 정규화 Mahalanobis 헤드만; FiLM 적응은 구현하지 않았다. L2 공간 변형 `mt` 는 βI 가 지배해 euc 와 거의 같아지므로 §2 표에만 싣는다). 트랙 B 는 특징표가 없어 제안 = ArcFace(`arc_vits_m05`)이다. 정확도만 싣는다(F1 은 §2).

### 트랙 B

| 기준선            | method            | classifier   | K=1               | K=3               | K=5               | K=10              |
|:------------------|:------------------|:-------------|:------------------|:------------------|:------------------|:------------------|
| ImageNet ViT-B    | in_vitb           | cos          | 25.9 ± 3.3        | 37.3 ± 4.9        | 42.3 ± 6.7        | 49.1 ± 8.8        |
| DINOv2 ViT-B      | dino_vitb         | cos          | 24.3 ± 22.4 (n=2) | 33.8 ± 25.0 (n=2) | 38.0 ± 34.2 (n=2) | 43.9 ± 24.3 (n=2) |
| ProtoNet          | proto_vits        | euc          | 34.8 ± 0.0 (n=1)  | 44.4 ± 0.0 (n=1)  | 48.5 ± 0.0 (n=1)  | 54.2 ± 0.0 (n=1)  |
| SimpleShot (CE)   | ce_vits           | ss           | 30.9 ± 5.6        | 41.2 ± 10.3       | 45.3 ± 11.0       | 50.7 ± 12.2       |
| ArcFace 갤러리    | arc_vits_m05      | cos          | 31.7 ± 6.8        | 40.8 ± 10.3       | 44.3 ± 11.3       | 48.9 ± 10.7       |
| Simple CNAPS 헤드 | in_vitb           | mtr          | 25.5 ± 3.2        | 39.5 ± 5.4        | 47.2 ± 6.7        | 57.4 ± 8.3        |
| Simple CNAPS 헤드 | dino_vitb         | mtr          | 24.7 ± 19.8 (n=2) | 39.6 ± 27.5 (n=2) | 47.0 ± 35.2 (n=2) | 58.7 ± 39.1 (n=2) |
| Simple CNAPS 헤드 | in_vits           | mtr          | 26.4 ± 3.4        | 40.2 ± 5.1        | 47.3 ± 6.5        | 57.2 ± 7.8        |
| Simple CNAPS 헤드 | ce_vits           | mtr          | 31.5 ± 7.2        | 44.1 ± 9.3        | 49.0 ± 9.2        | 54.5 ± 8.0        |
| Simple CNAPS 헤드 | proto_vits        | mtr          | 35.0 ± 0.0 (n=1)  | 46.9 ± 0.0 (n=1)  | 52.1 ± 0.0 (n=1)  | 59.5 ± 0.0 (n=1)  |
| Simple CNAPS 헤드 | arc_vits_m05      | mtr          | 30.5 ± 5.0        | 41.6 ± 7.3        | 45.8 ± 7.8        | 51.7 ± 7.5        |
| Simple CNAPS 헤드 | arc_vits_m02      | mtr          | —                 | —                 | —                 | —                 |
| Simple CNAPS 헤드 | arc_vits_m00      | mtr          | —                 | —                 | —                 | —                 |
| Simple CNAPS 헤드 | arc_vitb_m05      | mtr          | —                 | —                 | —                 | —                 |
| Simple CNAPS 헤드 | arc_vits_m05_full | mtr          | —                 | —                 | —                 | —                 |

### 트랙 S

| 기준선            | method           | classifier   | K=1         | K=2               | 0-shot (head)   |
|:------------------|:-----------------|:-------------|:------------|:------------------|:----------------|
| ImageNet ViT-B    | in_vitb          | cos          | 50.7 ± 16.6 | 61.1 ± 31.1 (n=4) |                 |
| DINOv2 ViT-B      | dino_vitb        | cos          | 49.3 ± 17.3 | 58.9 ± 27.7 (n=4) |                 |
| ArcFace 갤러리    | arc_vits_m05     | cos          | 70.0 ± 20.4 | 82.2 ± 28.1 (n=4) |                 |
| 제안              | arcattr_vits_m05 | cos          | 70.0 ± 14.5 | 81.5 ± 21.4 (n=4) | 40.1 ± 24.1     |
| Simple CNAPS 헤드 | in_vitb          | mtr          | 49.8 ± 14.2 | 63.5 ± 26.3 (n=4) |                 |
| Simple CNAPS 헤드 | dino_vitb        | mtr          | 48.9 ± 16.8 | 61.4 ± 25.2 (n=4) |                 |
| Simple CNAPS 헤드 | in_vits          | mtr          | 45.2 ± 14.9 | 56.0 ± 14.6 (n=4) |                 |
| Simple CNAPS 헤드 | arc_vits_m05     | mtr          | 69.2 ± 18.4 | 82.1 ± 22.0 (n=4) |                 |
| Simple CNAPS 헤드 | arcattr_vits_m05 | mtr          | 68.9 ± 17.2 | 83.3 ± 20.3 (n=4) |                 |

## 4. Ablation (§4, cos 분류기 정확도)

### 트랙 B: margin (m = 0.0 / 0.2 / 0.5)

| method       | K=1        | K=3         | K=5         | K=10        |
|:-------------|:-----------|:------------|:------------|:------------|
| arc_vits_m00 | —          | —           | —           | —           |
| arc_vits_m02 | —          | —           | —           | —           |
| arc_vits_m05 | 31.7 ± 6.8 | 40.8 ± 10.3 | 44.3 ± 11.3 | 48.9 ± 10.7 |

### 트랙 B: 백본 크기 (ViT-S/16 vs ViT-B/16)

| method       | K=1        | K=3         | K=5         | K=10        |
|:-------------|:-----------|:------------|:------------|:------------|
| in_vits      | 26.4 ± 3.3 | 36.3 ± 3.5  | 40.8 ± 4.7  | 47.4 ± 6.1  |
| in_vitb      | 25.9 ± 3.3 | 37.3 ± 4.9  | 42.3 ± 6.7  | 49.1 ± 8.8  |
| arc_vits_m05 | 31.7 ± 6.8 | 40.8 ± 10.3 | 44.3 ± 11.3 | 48.9 ± 10.7 |
| arc_vitb_m05 | —          | —           | —           | —           |

### 트랙 B: crop 유무 (크롭 vs 원본 전체)

`arc_vits_m05_full` 은 manifest_full 로 학습·평가해 query 집합이 다르다. 짝지은 비교가 아니다.

| method            | K=1        | K=3         | K=5         | K=10        |
|:------------------|:-----------|:------------|:------------|:------------|
| arc_vits_m05      | 31.7 ± 6.8 | 40.8 ± 10.3 | 44.3 ± 11.3 | 48.9 ± 10.7 |
| arc_vits_m05_full | —          | —           | —           | —           |

### 트랙 S: 특징 헤드 유무 (w = 0 / 1)

| method           | K=1         | K=2               | 0-shot probe   | 0-shot head   | 특징 채우기 probe   | 특징 채우기 head   |
|:-----------------|:------------|:------------------|:---------------|:--------------|:--------------------|:-------------------|
| arc_vits_m05     | 70.0 ± 20.4 | 82.2 ± 28.1 (n=4) | 44.1 ± 15.0    | —             | 67.5 ± 4.1          | —                  |
| arcattr_vits_m05 | 70.0 ± 14.5 | 81.5 ± 21.4 (n=4) | 42.7 ± 15.8    | 40.1 ± 24.1   | 68.2 ± 5.5          | 68.9 ± 4.3         |

## 5. 트랙 S 특징표

`mean` = 상수 아닌 feature 10개의 평균. 제외(이 트랙 모든 클래스에서 값이 같음): Body_Type, Apex_Hole.

### 이미지 단위 특징표 채우기 (unseen 이미지, feature 별 정확도)

`head` = 제안 모델의 attribute 헤드, `probe` = seen train L2 임베딩의 feature 별 로지스틱 회귀, `majority` = seen 클래스 최빈값, `nearest_seen` = 가장 가까운 seen centroid 의 행. `unreachable` = seen train 에 없던 값이 정답인 unseen 항목 수(probe 는 도달 불가, 분할 합).

#### in_vitb

| feature      | head   | probe       | majority    | nearest_seen   |   unreachable |
|:-------------|:-------|:------------|:------------|:---------------|--------------:|
| Company      | —      | 22.6 ± 17.6 | 11.7 ± 14.7 | 39.8 ± 4.3     |            33 |
| System       | —      | 74.6 ± 7.5  | 77.9 ± 16.0 | 76.1 ± 3.3     |             0 |
| Connection_1 | —      | 80.1 ± 21.0 | 79.7 ± 20.3 | 75.4 ± 4.8     |             0 |
| Connection_2 | —      | 69.4 ± 5.4  | 72.7 ± 14.2 | 69.7 ± 3.5     |            12 |
| Flange       | —      | 59.7 ± 19.7 | 41.4 ± 7.0  | 72.5 ± 8.2     |             0 |
| Collar       | —      | 56.2 ± 20.7 | 55.9 ± 19.9 | 50.3 ± 7.1     |             0 |
| Microthread  | —      | 69.2 ± 13.3 | 56.6 ± 34.5 | 69.1 ± 7.4     |             0 |
| Body_Shape   | —      | 24.2 ± 18.1 | 16.3 ± 22.6 | 38.6 ± 10.4    |             0 |
| Thread_Shape | —      | 77.7 ± 4.4  | 77.1 ± 4.8  | 72.3 ± 7.0     |             0 |
| Apex_Shape   | —      | 63.8 ± 12.3 | 51.9 ± 22.2 | 68.1 ± 6.1     |             0 |
| mean         | —      | 59.7 ± 4.3  | 54.1 ± 8.2  | 63.2 ± 2.2     |               |

#### dino_vitb

| feature      | head   | probe       | majority    | nearest_seen   |   unreachable |
|:-------------|:-------|:------------|:------------|:---------------|--------------:|
| Company      | —      | 24.9 ± 21.1 | 11.7 ± 14.7 | 40.3 ± 8.4     |            33 |
| System       | —      | 71.7 ± 4.8  | 77.9 ± 16.0 | 66.7 ± 12.7    |             0 |
| Connection_1 | —      | 79.2 ± 19.6 | 79.7 ± 20.3 | 68.7 ± 10.4    |             0 |
| Connection_2 | —      | 67.9 ± 2.7  | 72.7 ± 14.2 | 65.7 ± 12.5    |            12 |
| Flange       | —      | 60.5 ± 19.5 | 41.4 ± 7.0  | 70.0 ± 5.8     |             0 |
| Collar       | —      | 55.4 ± 19.4 | 55.9 ± 19.9 | 54.0 ± 11.7    |             0 |
| Microthread  | —      | 65.2 ± 15.0 | 56.6 ± 34.5 | 64.1 ± 8.8     |             0 |
| Body_Shape   | —      | 19.9 ± 17.0 | 16.3 ± 22.6 | 36.4 ± 12.3    |             0 |
| Thread_Shape | —      | 77.1 ± 4.8  | 77.1 ± 4.8  | 68.1 ± 4.4     |             0 |
| Apex_Shape   | —      | 62.2 ± 9.5  | 51.9 ± 22.2 | 56.8 ± 4.9     |             0 |
| mean         | —      | 58.4 ± 4.8  | 54.1 ± 8.2  | 59.1 ± 2.8     |               |

#### in_vits

| feature      | head   | probe       | majority    | nearest_seen   |   unreachable |
|:-------------|:-------|:------------|:------------|:---------------|--------------:|
| Company      | —      | 22.6 ± 18.2 | 11.7 ± 14.7 | 43.7 ± 7.6     |            33 |
| System       | —      | 76.0 ± 11.0 | 77.9 ± 16.0 | 74.4 ± 7.6     |             0 |
| Connection_1 | —      | 79.2 ± 19.6 | 79.7 ± 20.3 | 67.0 ± 15.6    |             0 |
| Connection_2 | —      | 70.8 ± 9.1  | 72.7 ± 14.2 | 71.5 ± 8.9     |            12 |
| Flange       | —      | 53.7 ± 15.2 | 41.4 ± 7.0  | 65.0 ± 7.6     |             0 |
| Collar       | —      | 57.3 ± 21.7 | 55.9 ± 19.9 | 55.8 ± 14.6    |             0 |
| Microthread  | —      | 66.6 ± 13.6 | 56.6 ± 34.5 | 65.0 ± 11.1    |             0 |
| Body_Shape   | —      | 23.9 ± 16.8 | 16.3 ± 22.6 | 36.1 ± 9.4     |             0 |
| Thread_Shape | —      | 77.1 ± 4.8  | 77.1 ± 4.8  | 75.9 ± 8.1     |             0 |
| Apex_Shape   | —      | 62.8 ± 10.8 | 51.9 ± 22.2 | 69.8 ± 10.5    |             0 |
| mean         | —      | 59.0 ± 4.3  | 54.1 ± 8.2  | 62.4 ± 3.3     |               |

#### arc_vits_m05

| feature      | head   | probe       | majority    | nearest_seen   |   unreachable |
|:-------------|:-------|:------------|:------------|:---------------|--------------:|
| Company      | —      | 43.8 ± 10.7 | 11.7 ± 14.7 | 46.2 ± 10.2    |            33 |
| System       | —      | 82.4 ± 14.8 | 77.9 ± 16.0 | 79.6 ± 14.9    |             0 |
| Connection_1 | —      | 85.5 ± 18.7 | 79.7 ± 20.3 | 82.7 ± 9.8     |             0 |
| Connection_2 | —      | 79.7 ± 13.8 | 72.7 ± 14.2 | 77.3 ± 11.8    |            12 |
| Flange       | —      | 69.2 ± 18.2 | 41.4 ± 7.0  | 70.9 ± 7.5     |             0 |
| Collar       | —      | 53.6 ± 17.2 | 55.9 ± 19.9 | 54.9 ± 14.8    |             0 |
| Microthread  | —      | 72.6 ± 14.1 | 56.6 ± 34.5 | 70.8 ± 6.8     |             0 |
| Body_Shape   | —      | 30.3 ± 14.1 | 16.3 ± 22.6 | 32.9 ± 16.2    |             0 |
| Thread_Shape | —      | 85.4 ± 11.1 | 77.1 ± 4.8  | 84.3 ± 6.9     |             0 |
| Apex_Shape   | —      | 72.7 ± 7.1  | 51.9 ± 22.2 | 73.3 ± 7.1     |             0 |
| mean         | —      | 67.5 ± 4.1  | 54.1 ± 8.2  | 67.3 ± 4.8     |               |

#### arcattr_vits_m05

| feature      | head        | probe       | majority    | nearest_seen   |   unreachable |
|:-------------|:------------|:------------|:------------|:---------------|--------------:|
| Company      | 46.6 ± 9.4  | 40.3 ± 9.6  | 11.7 ± 14.7 | 48.7 ± 11.3    |            33 |
| System       | 82.5 ± 10.8 | 85.0 ± 12.8 | 77.9 ± 16.0 | 82.2 ± 12.1    |             0 |
| Connection_1 | 87.5 ± 18.4 | 86.2 ± 22.6 | 79.7 ± 20.3 | 82.9 ± 13.2    |             0 |
| Connection_2 | 79.5 ± 12.6 | 79.6 ± 14.2 | 72.7 ± 14.2 | 78.0 ± 11.8    |            12 |
| Flange       | 71.8 ± 13.0 | 73.1 ± 6.9  | 41.4 ± 7.0  | 68.9 ± 6.8     |             0 |
| Collar       | 57.8 ± 23.8 | 55.6 ± 21.9 | 55.9 ± 19.9 | 56.1 ± 15.0    |             0 |
| Microthread  | 73.0 ± 15.6 | 72.2 ± 17.1 | 56.6 ± 34.5 | 74.3 ± 13.1    |             0 |
| Body_Shape   | 31.1 ± 4.1  | 32.2 ± 13.6 | 16.3 ± 22.6 | 33.9 ± 8.5     |             0 |
| Thread_Shape | 84.4 ± 9.7  | 82.6 ± 10.5 | 77.1 ± 4.8  | 83.5 ± 8.5     |             0 |
| Apex_Shape   | 75.2 ± 9.9  | 74.7 ± 8.3  | 51.9 ± 22.2 | 72.8 ± 8.7     |             0 |
| mean         | 68.9 ± 4.3  | 68.2 ± 5.5  | 54.1 ± 8.2  | 68.1 ± 3.2     |               |

### 클래스 단위 특징표 채우기 (K 서포트 확률 평균 → 클래스 행, 상수 아닌 feature 평균)

| method           | source   | K=1        | K=2               |
|:-----------------|:---------|:-----------|:------------------|
| in_vitb          | probe    | 66.1 ± 6.8 | 71.1 ± 10.2 (n=4) |
| dino_vitb        | probe    | 65.0 ± 6.5 | 71.4 ± 11.3 (n=4) |
| in_vits          | probe    | 64.9 ± 6.6 | 70.7 ± 8.6 (n=4)  |
| arc_vits_m05     | probe    | 74.2 ± 5.9 | 78.7 ± 9.4 (n=4)  |
| arcattr_vits_m05 | head     | 74.0 ± 5.4 | 79.3 ± 5.9 (n=4)  |
| arcattr_vits_m05 | probe    | 75.0 ± 9.0 | 80.6 ± 9.8 (n=4)  |

### 0-shot (특징표 한 행만, §2.3)

`ceiling` = 알려진 feature 로 다른 모든 unseen 클래스와 구분되는 클래스 비율(원리상 상한).

| method           | source   | acc         | macro-F1    | n_classes   | ceiling     |
|:-----------------|:---------|:------------|:------------|:------------|:------------|
| in_vitb          | probe    | 18.8 ± 12.8 | 7.4 ± 5.8   | 10.0 ± 0.0  | 100.0 ± 0.0 |
| dino_vitb        | probe    | 21.7 ± 13.3 | 8.7 ± 8.4   | 10.0 ± 0.0  | 100.0 ± 0.0 |
| in_vits          | probe    | 20.9 ± 11.2 | 7.5 ± 6.1   | 10.0 ± 0.0  | 100.0 ± 0.0 |
| arc_vits_m05     | probe    | 44.1 ± 15.0 | 26.3 ± 12.1 | 10.0 ± 0.0  | 100.0 ± 0.0 |
| arcattr_vits_m05 | head     | 40.1 ± 24.1 | 23.6 ± 15.4 | 10.0 ± 0.0  | 100.0 ± 0.0 |
| arcattr_vits_m05 | probe    | 42.7 ± 15.8 | 25.9 ± 10.6 | 10.0 ± 0.0  | 100.0 ± 0.0 |

## 6. 진단 (§5)

Δ_W(K) = acc_mw − acc_cos (%p). trace ratio = unseen query 의 tr(S_W)/tr(S_B). eff rank = seen train 의 축소 없는 pooled Σ_W 유효 순위. LW shrinkage = `mw` 가 쓰는 Ledoit-Wolf 축소 강도.

### 트랙 B

| method       | Δ_W K=1            | Δ_W K=3          | Δ_W K=5          | Δ_W K=10         | trace ratio         | eff rank           | LW shrinkage        |
|:-------------|:-------------------|:-----------------|:-----------------|:-----------------|:--------------------|:-------------------|:--------------------|
| in_vitb      | -18.3 ± 5.1        | -12.9 ± 7.9      | -5.2 ± 8.6       | 3.3 ± 7.1        | 3.573 ± 0.253       | 111.4 ± 15.0       | 0.008 ± 0.005       |
| dino_vitb    | -10.2 ± 19.2 (n=2) | -9.4 ± 0.6 (n=2) | -3.1 ± 2.5 (n=2) | 9.1 ± 21.6 (n=2) | 2.893 ± 1.703 (n=2) | 128.5 ± 14.8 (n=2) | 0.007 ± 0.007 (n=2) |
| in_vits      | -16.2 ± 4.9        | -7.6 ± 5.6       | -0.6 ± 5.2       | 6.1 ± 4.9        | 2.661 ± 0.298       | 86.5 ± 11.1        | 0.007 ± 0.004       |
| ce_vits      | -5.1 ± 2.6         | 0.7 ± 5.9        | 3.9 ± 7.4        | 7.5 ± 8.7        | 1.814 ± 0.228       | 19.6 ± 1.5         | 0.007 ± 0.005       |
| proto_vits   | -8.6 ± 0.0 (n=1)   | -3.6 ± 0.0 (n=1) | -0.0 ± 0.0 (n=1) | 2.8 ± 0.0 (n=1)  | 2.724 ± 0.000 (n=1) | 47.0 ± 0.0 (n=1)   | 0.009 ± 0.000 (n=1) |
| arc_vits_m05 | -5.8 ± 4.0         | 1.0 ± 4.1        | 4.7 ± 5.7        | 8.3 ± 6.6        | 2.600 ± 0.365       | 30.5 ± 5.2         | 0.012 ± 0.012       |

### 트랙 S

| method           | Δ_W K=1     | Δ_W K=2           | trace ratio   | eff rank   | LW shrinkage   |
|:-----------------|:------------|:------------------|:--------------|:-----------|:---------------|
| in_vitb          | 11.1 ± 12.0 | 19.4 ± 12.3 (n=4) | 1.304 ± 0.323 | 14.4 ± 1.3 | 0.139 ± 0.019  |
| dino_vitb        | 15.8 ± 10.6 | 21.1 ± 7.8 (n=4)  | 1.237 ± 0.352 | 13.3 ± 1.8 | 0.123 ± 0.024  |
| in_vits          | 13.7 ± 14.2 | 25.1 ± 7.0 (n=4)  | 1.274 ± 0.345 | 14.4 ± 1.1 | 0.135 ± 0.017  |
| arc_vits_m05     | -2.5 ± 4.7  | -0.4 ± 8.4 (n=4)  | 0.582 ± 0.160 | 21.5 ± 4.2 | 0.346 ± 0.027  |
| arcattr_vits_m05 | -1.0 ± 4.6  | -0.3 ± 2.1 (n=4)  | 0.526 ± 0.093 | 21.9 ± 2.6 | 0.350 ± 0.020  |

## 7. GFSL (§2.2 보조, cos)

seen 프로토타입 ∪ unseen K-shot 프로토타입, query = seen test ∪ unseen query. 분할의 H = 그 분할 에피소드 평균 acc_seen·acc_unseen 의 조화평균(에피소드별 H 의 평균이 아니다).

### 트랙 B

| method       |   K | acc_seen          | acc_unseen       | H                 |
|:-------------|----:|:------------------|:-----------------|:------------------|
| in_vitb      |   1 | 48.2 ± 18.5       | 5.9 ± 1.3        | 10.4 ± 2.3        |
| in_vitb      |   3 | 47.3 ± 18.4       | 16.4 ± 3.3       | 23.9 ± 4.9        |
| in_vitb      |   5 | 46.5 ± 18.2       | 23.7 ± 5.2       | 30.6 ± 7.0        |
| in_vitb      |  10 | 45.6 ± 18.1       | 32.2 ± 7.7       | 36.5 ± 9.2        |
| dino_vitb    |   1 | 50.9 ± 80.0 (n=2) | 5.2 ± 2.7 (n=2)  | 9.4 ± 5.8 (n=2)   |
| dino_vitb    |   3 | 50.1 ± 77.7 (n=2) | 12.8 ± 3.9 (n=2) | 20.4 ± 11.4 (n=2) |
| dino_vitb    |   5 | 49.4 ± 77.7 (n=2) | 18.6 ± 5.6 (n=2) | 26.9 ± 17.6 (n=2) |
| dino_vitb    |  10 | 48.4 ± 77.0 (n=2) | 26.6 ± 2.8 (n=2) | 34.2 ± 21.8 (n=2) |
| in_vits      |   1 | 46.5 ± 18.3       | 5.6 ± 0.4        | 9.9 ± 1.0         |
| in_vits      |   3 | 45.6 ± 18.0       | 15.2 ± 0.6       | 22.3 ± 3.0        |
| in_vits      |   5 | 44.8 ± 17.8       | 21.8 ± 1.3       | 28.6 ± 5.0        |
| in_vits      |  10 | 44.1 ± 17.8       | 29.4 ± 2.8       | 34.3 ± 7.3        |
| ce_vits      |   1 | 76.5 ± 13.5       | 15.4 ± 6.9       | 25.2 ± 8.9        |
| ce_vits      |   3 | 75.5 ± 13.8       | 24.3 ± 11.3      | 35.8 ± 11.9       |
| ce_vits      |   5 | 74.7 ± 14.3       | 28.5 ± 12.0      | 40.3 ± 11.7       |
| ce_vits      |  10 | 74.1 ± 14.6       | 33.7 ± 12.9      | 45.2 ± 11.3       |
| proto_vits   |   1 | 74.7 ± 0.0 (n=1)  | 9.1 ± 0.0 (n=1)  | 16.2 ± 0.0 (n=1)  |
| proto_vits   |   3 | 73.6 ± 0.0 (n=1)  | 18.9 ± 0.0 (n=1) | 30.1 ± 0.0 (n=1)  |
| proto_vits   |   5 | 72.9 ± 0.0 (n=1)  | 25.1 ± 0.0 (n=1) | 37.4 ± 0.0 (n=1)  |
| proto_vits   |  10 | 72.0 ± 0.0 (n=1)  | 32.5 ± 0.0 (n=1) | 44.8 ± 0.0 (n=1)  |
| arc_vits_m05 |   1 | 79.6 ± 9.3        | 12.1 ± 4.1       | 20.8 ± 6.0        |
| arc_vits_m05 |   3 | 79.4 ± 9.4        | 18.5 ± 6.9       | 29.7 ± 8.5        |
| arc_vits_m05 |   5 | 79.0 ± 9.5        | 22.5 ± 7.8       | 34.6 ± 9.0        |
| arc_vits_m05 |  10 | 78.3 ± 10.0       | 27.6 ± 7.9       | 40.4 ± 8.3        |

### 트랙 S

| method           |   K | acc_seen          | acc_unseen        | H                 |
|:-----------------|----:|:------------------|:------------------|:------------------|
| in_vitb          |   1 | 24.5 ± 23.3       | 19.2 ± 7.1        | 16.4 ± 12.0       |
| in_vitb          |   2 | 30.7 ± 23.5 (n=4) | 13.4 ± 15.0 (n=4) | 17.9 ± 18.7 (n=4) |
| dino_vitb        |   1 | 24.7 ± 23.3       | 19.7 ± 7.0        | 17.1 ± 12.7       |
| dino_vitb        |   2 | 35.1 ± 23.8 (n=4) | 14.5 ± 10.6 (n=4) | 20.2 ± 14.2 (n=4) |
| in_vits          |   1 | 21.3 ± 25.1       | 17.5 ± 5.2        | 14.2 ± 12.1       |
| in_vits          |   2 | 31.9 ± 22.6 (n=4) | 11.2 ± 13.0 (n=4) | 15.2 ± 17.0 (n=4) |
| arc_vits_m05     |   1 | 31.7 ± 31.0       | 20.8 ± 13.0       | 20.6 ± 18.7       |
| arc_vits_m05     |   2 | 29.1 ± 41.4 (n=4) | 24.1 ± 21.2 (n=4) | 18.0 ± 27.9 (n=4) |
| arcattr_vits_m05 |   1 | 35.3 ± 35.5       | 22.0 ± 9.4        | 21.6 ± 16.9       |
| arcattr_vits_m05 |   2 | 34.0 ± 49.2 (n=4) | 23.6 ± 30.5 (n=4) | 21.6 ± 31.6 (n=4) |

## 8. 기준 방법 대비 짝지은 차이 (cos 정확도, %p)

셀 = 같은 분할 위 차이(방법 − 기준)의 평균 ± 95% CI 반폭 (이긴 분할 수 / 공통 분할 수).

### 트랙 B (기준: `arc_vits_m05`)

`arc_vits_m05_full` 은 query 집합이 달라(원본 전체) 같은 에피소드 비교가 아니다.

| method     | K=1               | K=3                | K=5                | K=10               |
|:-----------|:------------------|:-------------------|:-------------------|:-------------------|
| in_vitb    | -5.8 ± 6.2 (1/5)  | -3.5 ± 7.6 (2/5)   | -2.0 ± 8.1 (2/5)   | +0.2 ± 7.0 (3/5)   |
| dino_vitb  | -13.3 ± 8.9 (0/2) | -15.7 ± 29.9 (0/2) | -15.7 ± 40.9 (0/2) | -14.1 ± 33.4 (0/2) |
| in_vits    | -5.3 ± 7.4 (1/5)  | -4.5 ± 9.7 (1/5)   | -3.5 ± 10.7 (1/5)  | -1.5 ± 9.8 (2/5)   |
| ce_vits    | +2.1 ± 5.1 (4/5)  | +2.1 ± 6.7 (3/5)   | +2.5 ± 5.7 (4/5)   | +3.4 ± 5.4 (3/5)   |
| proto_vits | -2.6 ± 0.0 (0/1)  | -2.9 ± 0.0 (0/1)   | -2.9 ± 0.0 (0/1)   | -1.6 ± 0.0 (0/1)   |

### 트랙 S (기준: `arcattr_vits_m05`)

| method       | K=1                | K=2                |
|:-------------|:-------------------|:-------------------|
| in_vitb      | -19.2 ± 15.1 (0/5) | -20.4 ± 17.4 (0/4) |
| dino_vitb    | -20.6 ± 18.6 (1/5) | -22.6 ± 17.6 (0/4) |
| in_vits      | -24.9 ± 16.9 (0/5) | -28.2 ± 8.1 (0/4)  |
| arc_vits_m05 | -0.0 ± 10.9 (2/5)  | +0.7 ± 10.4 (2/4)  |

## 9. 트랙 B 도메인별 정확도 (cos)

분할 값 = 그 분할 모든 에피소드의 도메인별 query 정확도(합산).

| method       |   K | catalog           | panoramic         | periapical        |
|:-------------|----:|:------------------|:------------------|:------------------|
| in_vitb      |   1 | 23.0 ± 5.1        | 26.8 ± 4.0        | 19.3 ± 4.8        |
| in_vitb      |   3 | 32.2 ± 7.7        | 39.2 ± 6.0        | 25.0 ± 7.6        |
| in_vitb      |   5 | 36.3 ± 8.2        | 44.3 ± 8.0        | 28.1 ± 8.1        |
| in_vitb      |  10 | 40.8 ± 9.2        | 51.6 ± 10.3       | 30.4 ± 8.1        |
| dino_vitb    |   1 | 23.8 ± 21.2 (n=2) | 25.0 ± 21.7 (n=2) | 18.5 ± 36.0 (n=2) |
| dino_vitb    |   3 | 33.8 ± 29.5 (n=2) | 35.2 ± 25.7 (n=2) | 21.8 ± 32.4 (n=2) |
| dino_vitb    |   5 | 39.0 ± 38.0 (n=2) | 39.5 ± 36.0 (n=2) | 24.2 ± 40.0 (n=2) |
| dino_vitb    |  10 | 44.1 ± 27.1 (n=2) | 45.8 ± 25.7 (n=2) | 26.5 ± 45.7 (n=2) |
| in_vits      |   1 | 22.8 ± 4.8        | 27.3 ± 3.7        | 20.5 ± 6.4        |
| in_vits      |   3 | 32.2 ± 7.5        | 38.0 ± 3.7        | 25.3 ± 8.4        |
| in_vits      |   5 | 36.3 ± 8.0        | 42.5 ± 5.1        | 27.8 ± 8.7        |
| in_vits      |  10 | 40.5 ± 8.5        | 49.5 ± 6.5        | 29.9 ± 8.4        |
| ce_vits      |   1 | 25.1 ± 5.0        | 36.7 ± 9.6        | 24.6 ± 9.0        |
| ce_vits      |   3 | 33.0 ± 8.1        | 46.4 ± 15.6       | 30.9 ± 13.5       |
| ce_vits      |   5 | 36.2 ± 9.3        | 50.5 ± 15.9       | 33.3 ± 13.9       |
| ce_vits      |  10 | 39.4 ± 10.4       | 56.4 ± 15.9       | 35.7 ± 14.9       |
| proto_vits   |   1 | 29.6 ± 0.0 (n=1)  | 38.8 ± 0.0 (n=1)  | 20.2 ± 0.0 (n=1)  |
| proto_vits   |   3 | 41.1 ± 0.0 (n=1)  | 49.7 ± 0.0 (n=1)  | 24.5 ± 0.0 (n=1)  |
| proto_vits   |   5 | 45.5 ± 0.0 (n=1)  | 53.7 ± 0.0 (n=1)  | 28.0 ± 0.0 (n=1)  |
| proto_vits   |  10 | 49.4 ± 0.0 (n=1)  | 59.3 ± 0.0 (n=1)  | 31.7 ± 0.0 (n=1)  |
| arc_vits_m05 |   1 | 23.6 ± 3.1        | 34.5 ± 9.1        | 22.4 ± 9.0        |
| arc_vits_m05 |   3 | 31.9 ± 5.0        | 44.2 ± 13.8       | 29.3 ± 17.5       |
| arc_vits_m05 |   5 | 35.0 ± 5.5        | 47.8 ± 15.0       | 32.3 ± 20.2       |
| arc_vits_m05 |  10 | 37.7 ± 6.3        | 52.7 ± 13.9       | 35.6 ± 24.8       |

## 10. 커버리지

### 실행 설정 (결과 JSON 별)

`protonet_excluded` = train 항목이 K+Q=10장 미만이라 ProtoNet 에피소드 학습에 한 번도 뽑히지 않은 seen 클래스(ce·arc 는 모든 seen 클래스로 학습한다).

| track   | method           |   split |   episodes |   dim |   n_unseen_items |   epochs |   epoch_size |   best_epoch |   best_val_acc |   train_s | protonet_excluded                                                     |
|:--------|:-----------------|--------:|-----------:|------:|-----------------:|---------:|-------------:|-------------:|---------------:|----------:|:----------------------------------------------------------------------|
| B       | in_vitb          |       0 |        200 |   768 |             6577 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| B       | in_vitb          |       1 |        200 |   768 |             5901 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| B       | in_vitb          |       2 |        200 |   768 |            10509 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| B       | in_vitb          |       3 |        200 |   768 |             3986 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| B       | in_vitb          |       4 |        200 |   768 |             5450 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| B       | dino_vitb        |       0 |        200 |   768 |             6577 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| B       | dino_vitb        |       1 |        200 |   768 |             5901 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| B       | in_vits          |       0 |        200 |   384 |             6577 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| B       | in_vits          |       1 |        200 |   384 |             5901 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| B       | in_vits          |       2 |        200 |   384 |            10509 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| B       | in_vits          |       3 |        200 |   384 |             3986 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| B       | in_vits          |       4 |        200 |   384 |             5450 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| B       | ce_vits          |       0 |        200 |   128 |             6577 |       10 |         3000 |            7 |       0.721578 |       394 |                                                                       |
| B       | ce_vits          |       1 |        200 |   128 |             5901 |       10 |         3000 |            7 |       0.838631 |       407 |                                                                       |
| B       | ce_vits          |       2 |        200 |   128 |            10509 |       10 |         3000 |           10 |       0.779874 |       372 |                                                                       |
| B       | ce_vits          |       3 |        200 |   128 |             3986 |       10 |         3000 |           10 |       0.917617 |       545 |                                                                       |
| B       | ce_vits          |       4 |        200 |   128 |             5450 |       10 |         3000 |            9 |       0.855282 |       393 |                                                                       |
| B       | proto_vits       |       0 |        200 |   128 |             6577 |       10 |         3000 |           10 |       0.617169 |       369 | ameross biolife dentatus dentis frontier nta osseolink titan_implants |
| B       | arc_vits_m05     |       0 |        200 |   128 |             6577 |       10 |         3000 |            9 |       0.700696 |       382 |                                                                       |
| B       | arc_vits_m05     |       1 |        200 |   128 |             5901 |       10 |         3000 |           10 |       0.815403 |       397 |                                                                       |
| B       | arc_vits_m05     |       2 |        200 |   128 |            10509 |       10 |         3000 |           10 |       0.786164 |       372 |                                                                       |
| B       | arc_vits_m05     |       3 |        200 |   128 |             3986 |       10 |         3000 |            4 |       0.920152 |       396 |                                                                       |
| B       | arc_vits_m05     |       4 |        200 |   128 |             5450 |       10 |         3000 |           10 |       0.865412 |       392 |                                                                       |
| S       | in_vitb          |       0 |        200 |   768 |               42 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | in_vitb          |       1 |        200 |   768 |               44 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | in_vitb          |       2 |        200 |   768 |               43 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | in_vitb          |       3 |        200 |   768 |               58 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | in_vitb          |       4 |        200 |   768 |               33 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | dino_vitb        |       0 |        200 |   768 |               42 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | dino_vitb        |       1 |        200 |   768 |               44 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | dino_vitb        |       2 |        200 |   768 |               43 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | dino_vitb        |       3 |        200 |   768 |               58 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | dino_vitb        |       4 |        200 |   768 |               33 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | in_vits          |       0 |        200 |   384 |               42 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | in_vits          |       1 |        200 |   384 |               44 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | in_vits          |       2 |        200 |   384 |               43 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | in_vits          |       3 |        200 |   384 |               58 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | in_vits          |       4 |        200 |   384 |               33 |      nan |          nan |          nan |     nan        |       nan |                                                                       |
| S       | arc_vits_m05     |       0 |        200 |   128 |               42 |       10 |          600 |           10 |       0.666667 |        78 |                                                                       |
| S       | arc_vits_m05     |       1 |        200 |   128 |               44 |       10 |          600 |           10 |       0.666667 |        78 |                                                                       |
| S       | arc_vits_m05     |       2 |        200 |   128 |               43 |       10 |          600 |           10 |       0.666667 |       148 |                                                                       |
| S       | arc_vits_m05     |       3 |        200 |   128 |               58 |       10 |          600 |           10 |       0.6      |        78 |                                                                       |
| S       | arc_vits_m05     |       4 |        200 |   128 |               33 |       10 |          600 |           10 |       0.714286 |        80 |                                                                       |
| S       | arcattr_vits_m05 |       0 |        200 |   128 |               42 |       10 |          600 |           10 |       0.666667 |        82 |                                                                       |
| S       | arcattr_vits_m05 |       1 |        200 |   128 |               44 |       10 |          600 |           10 |       0.666667 |        82 |                                                                       |
| S       | arcattr_vits_m05 |       2 |        200 |   128 |               43 |       10 |          600 |           10 |       0.333333 |        82 |                                                                       |
| S       | arcattr_vits_m05 |       3 |        200 |   128 |               58 |       10 |          600 |           10 |       0.8      |        83 |                                                                       |
| S       | arcattr_vits_m05 |       4 |        200 |   128 |               33 |       10 |          600 |           10 |       0.571429 |        83 |                                                                       |

### 없는 결과 JSON (있음 48 / 기대 75)

| track   | method            | missing splits   | expected JSON                                                                                                                                                       |
|:--------|:------------------|:-----------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| B       | dino_vitb         | 2, 3, 4          | B/dino_vitb/split2.json, B/dino_vitb/split3.json, B/dino_vitb/split4.json                                                                                           |
| B       | proto_vits        | 1, 2, 3, 4       | B/proto_vits/split1.json, B/proto_vits/split2.json, B/proto_vits/split3.json, B/proto_vits/split4.json                                                              |
| B       | arc_vits_m02      | 0, 1, 2, 3, 4    | B/arc_vits_m02/split0.json, B/arc_vits_m02/split1.json, B/arc_vits_m02/split2.json, B/arc_vits_m02/split3.json, B/arc_vits_m02/split4.json                          |
| B       | arc_vits_m00      | 0, 1, 2, 3, 4    | B/arc_vits_m00/split0.json, B/arc_vits_m00/split1.json, B/arc_vits_m00/split2.json, B/arc_vits_m00/split3.json, B/arc_vits_m00/split4.json                          |
| B       | arc_vitb_m05      | 0, 1, 2, 3, 4    | B/arc_vitb_m05/split0.json, B/arc_vitb_m05/split1.json, B/arc_vitb_m05/split2.json, B/arc_vitb_m05/split3.json, B/arc_vitb_m05/split4.json                          |
| B       | arc_vits_m05_full | 0, 1, 2, 3, 4    | B/arc_vits_m05_full/split0.json, B/arc_vits_m05_full/split1.json, B/arc_vits_m05_full/split2.json, B/arc_vits_m05_full/split3.json, B/arc_vits_m05_full/split4.json |

### 오류 (`errors.jsonl`, 0건)

_(결과 없음)_

## 부록 A. 에피소드 수준 95% CI (분할별, cos)

셀 = 그 분할 E개 에피소드 평균 ± 95% CI 반폭(t, df = E−1). 본문 셀의 CI 는 분할 간 변동이다(§2.4).

| track   | method           |   K |   split | acc        | macro-F1   |
|:--------|:-----------------|----:|--------:|:-----------|:-----------|
| B       | in_vitb          |   1 |       0 | 29.7 ± 1.3 | 19.2 ± 0.5 |
| B       | in_vitb          |   1 |       1 | 24.9 ± 1.3 | 15.7 ± 0.4 |
| B       | in_vitb          |   1 |       2 | 25.9 ± 1.2 | 15.7 ± 0.4 |
| B       | in_vitb          |   1 |       3 | 22.4 ± 0.8 | 19.6 ± 0.5 |
| B       | in_vitb          |   1 |       4 | 26.8 ± 2.0 | 17.0 ± 0.5 |
| B       | in_vitb          |   3 |       0 | 43.3 ± 1.2 | 29.5 ± 0.4 |
| B       | in_vitb          |   3 |       1 | 37.9 ± 1.3 | 23.3 ± 0.4 |
| B       | in_vitb          |   3 |       2 | 37.7 ± 1.2 | 24.9 ± 0.4 |
| B       | in_vitb          |   3 |       3 | 32.8 ± 0.6 | 29.0 ± 0.4 |
| B       | in_vitb          |   3 |       4 | 34.8 ± 1.8 | 26.5 ± 0.4 |
| B       | in_vitb          |   5 |       0 | 49.7 ± 1.2 | 33.3 ± 0.4 |
| B       | in_vitb          |   5 |       1 | 42.8 ± 1.2 | 26.3 ± 0.3 |
| B       | in_vitb          |   5 |       2 | 44.6 ± 1.2 | 29.3 ± 0.4 |
| B       | in_vitb          |   5 |       3 | 36.0 ± 0.5 | 31.9 ± 0.3 |
| B       | in_vitb          |   5 |       4 | 38.4 ± 1.9 | 30.1 ± 0.4 |
| B       | in_vitb          |  10 |       0 | 57.7 ± 1.2 | 35.8 ± 0.4 |
| B       | in_vitb          |  10 |       1 | 51.2 ± 1.2 | 27.9 ± 0.3 |
| B       | in_vitb          |  10 |       2 | 52.7 ± 1.1 | 32.8 ± 0.4 |
| B       | in_vitb          |  10 |       3 | 39.7 ± 0.5 | 33.8 ± 0.3 |
| B       | in_vitb          |  10 |       4 | 44.3 ± 1.7 | 33.2 ± 0.4 |
| B       | dino_vitb        |   1 |       0 | 26.0 ± 1.2 | 20.2 ± 0.5 |
| B       | dino_vitb        |   1 |       1 | 22.5 ± 1.1 | 16.1 ± 0.4 |
| B       | dino_vitb        |   3 |       0 | 35.8 ± 1.1 | 29.7 ± 0.5 |
| B       | dino_vitb        |   3 |       1 | 31.8 ± 1.2 | 22.8 ± 0.4 |
| B       | dino_vitb        |   5 |       0 | 40.7 ± 1.1 | 34.1 ± 0.4 |
| B       | dino_vitb        |   5 |       1 | 35.4 ± 1.1 | 25.6 ± 0.3 |
| B       | dino_vitb        |  10 |       0 | 45.8 ± 1.1 | 36.1 ± 0.4 |
| B       | dino_vitb        |  10 |       1 | 42.0 ± 1.2 | 27.6 ± 0.3 |
| B       | in_vits          |   1 |       0 | 28.8 ± 1.2 | 20.1 ± 0.5 |
| B       | in_vits          |   1 |       1 | 24.7 ± 1.2 | 15.9 ± 0.4 |
| B       | in_vits          |   1 |       2 | 25.9 ± 1.3 | 16.8 ± 0.4 |
| B       | in_vits          |   1 |       3 | 23.1 ± 0.7 | 20.1 ± 0.5 |
| B       | in_vits          |   1 |       4 | 29.4 ± 2.4 | 17.8 ± 0.5 |
| B       | in_vits          |   3 |       0 | 40.2 ± 1.2 | 30.2 ± 0.4 |
| B       | in_vits          |   3 |       1 | 35.4 ± 1.1 | 23.1 ± 0.4 |
| B       | in_vits          |   3 |       2 | 35.5 ± 1.2 | 25.0 ± 0.4 |
| B       | in_vits          |   3 |       3 | 32.7 ± 0.6 | 29.5 ± 0.3 |
| B       | in_vits          |   3 |       4 | 37.6 ± 2.2 | 27.3 ± 0.4 |
| B       | in_vits          |   5 |       0 | 46.3 ± 1.1 | 34.0 ± 0.4 |
| B       | in_vits          |   5 |       1 | 39.0 ± 1.1 | 25.8 ± 0.3 |
| B       | in_vits          |   5 |       2 | 41.2 ± 1.3 | 28.7 ± 0.4 |
| B       | in_vits          |   5 |       3 | 35.9 ± 0.5 | 32.5 ± 0.3 |
| B       | in_vits          |   5 |       4 | 41.6 ± 2.3 | 30.9 ± 0.4 |
| B       | in_vits          |  10 |       0 | 52.9 ± 1.1 | 35.5 ± 0.3 |
| B       | in_vits          |  10 |       1 | 46.7 ± 1.1 | 27.4 ± 0.3 |
| B       | in_vits          |  10 |       2 | 49.1 ± 1.3 | 32.2 ± 0.4 |
| B       | in_vits          |  10 |       3 | 39.6 ± 0.5 | 34.5 ± 0.3 |
| B       | in_vits          |  10 |       4 | 48.6 ± 2.1 | 33.7 ± 0.4 |
| B       | ce_vits          |   1 |       0 | 34.7 ± 1.6 | 21.6 ± 0.5 |
| B       | ce_vits          |   1 |       1 | 42.6 ± 2.4 | 21.1 ± 0.5 |
| B       | ce_vits          |   1 |       2 | 32.5 ± 1.4 | 21.4 ± 0.5 |
| B       | ce_vits          |   1 |       3 | 32.3 ± 0.9 | 24.7 ± 0.6 |
| B       | ce_vits          |   1 |       4 | 26.9 ± 1.6 | 19.2 ± 0.4 |
| B       | ce_vits          |   3 |       0 | 43.4 ± 1.2 | 28.4 ± 0.4 |
| B       | ce_vits          |   3 |       1 | 57.4 ± 1.6 | 26.8 ± 0.4 |
| B       | ce_vits          |   3 |       2 | 42.3 ± 1.2 | 27.8 ± 0.4 |
| B       | ce_vits          |   3 |       3 | 40.4 ± 0.6 | 31.9 ± 0.4 |
| B       | ce_vits          |   3 |       4 | 30.7 ± 1.5 | 25.0 ± 0.4 |
| B       | ce_vits          |   5 |       0 | 48.9 ± 1.1 | 31.4 ± 0.4 |
| B       | ce_vits          |   5 |       1 | 61.1 ± 1.3 | 28.6 ± 0.3 |
| B       | ce_vits          |   5 |       2 | 47.5 ± 1.0 | 30.9 ± 0.4 |
| B       | ce_vits          |   5 |       3 | 43.0 ± 0.5 | 34.4 ± 0.3 |
| B       | ce_vits          |   5 |       4 | 33.8 ± 1.4 | 27.1 ± 0.4 |
| B       | ce_vits          |  10 |       0 | 55.4 ± 0.6 | 33.2 ± 0.3 |
| B       | ce_vits          |  10 |       1 | 66.7 ± 0.7 | 28.7 ± 0.2 |
| B       | ce_vits          |  10 |       2 | 53.3 ± 0.8 | 32.9 ± 0.3 |
| B       | ce_vits          |  10 |       3 | 46.7 ± 0.5 | 35.1 ± 0.3 |
| B       | ce_vits          |  10 |       4 | 39.3 ± 1.2 | 29.2 ± 0.3 |
| B       | proto_vits       |   1 |       0 | 36.0 ± 1.3 | 22.7 ± 0.4 |
| B       | proto_vits       |   3 |       0 | 46.2 ± 1.1 | 31.0 ± 0.4 |
| B       | proto_vits       |   5 |       0 | 50.3 ± 0.9 | 34.1 ± 0.4 |
| B       | proto_vits       |  10 |       0 | 55.6 ± 0.8 | 35.2 ± 0.3 |
| B       | arc_vits_m05     |   1 |       0 | 38.6 ± 1.9 | 22.5 ± 0.5 |
| B       | arc_vits_m05     |   1 |       1 | 36.5 ± 1.8 | 18.8 ± 0.5 |
| B       | arc_vits_m05     |   1 |       2 | 27.2 ± 1.4 | 19.1 ± 0.5 |
| B       | arc_vits_m05     |   1 |       3 | 29.5 ± 0.7 | 21.0 ± 0.4 |
| B       | arc_vits_m05     |   1 |       4 | 26.7 ± 1.4 | 19.6 ± 0.5 |
| B       | arc_vits_m05     |   3 |       0 | 49.2 ± 1.4 | 30.3 ± 0.4 |
| B       | arc_vits_m05     |   3 |       1 | 49.9 ± 1.2 | 24.6 ± 0.4 |
| B       | arc_vits_m05     |   3 |       2 | 36.2 ± 1.1 | 25.6 ± 0.4 |
| B       | arc_vits_m05     |   3 |       3 | 37.1 ± 0.5 | 27.3 ± 0.4 |
| B       | arc_vits_m05     |   3 |       4 | 31.5 ± 1.3 | 26.9 ± 0.4 |
| B       | arc_vits_m05     |   5 |       0 | 53.2 ± 1.2 | 32.9 ± 0.3 |
| B       | arc_vits_m05     |   5 |       1 | 54.3 ± 1.0 | 26.8 ± 0.3 |
| B       | arc_vits_m05     |   5 |       2 | 41.2 ± 0.9 | 28.9 ± 0.3 |
| B       | arc_vits_m05     |   5 |       3 | 39.5 ± 0.4 | 29.5 ± 0.3 |
| B       | arc_vits_m05     |   5 |       4 | 33.4 ± 1.3 | 29.6 ± 0.4 |
| B       | arc_vits_m05     |  10 |       0 | 57.2 ± 0.8 | 33.5 ± 0.3 |
| B       | arc_vits_m05     |  10 |       1 | 58.7 ± 0.7 | 26.8 ± 0.2 |
| B       | arc_vits_m05     |  10 |       2 | 46.4 ± 0.9 | 30.8 ± 0.3 |
| B       | arc_vits_m05     |  10 |       3 | 42.7 ± 0.4 | 30.9 ± 0.3 |
| B       | arc_vits_m05     |  10 |       4 | 39.5 ± 1.1 | 31.5 ± 0.3 |
| S       | in_vitb          |   1 |       0 | 58.4 ± 1.4 | 55.4 ± 1.4 |
| S       | in_vitb          |   1 |       1 | 48.3 ± 1.3 | 47.9 ± 1.4 |
| S       | in_vitb          |   1 |       2 | 67.4 ± 1.6 | 69.2 ± 1.5 |
| S       | in_vitb          |   1 |       3 | 31.6 ± 1.2 | 29.3 ± 1.1 |
| S       | in_vitb          |   1 |       4 | 48.0 ± 2.3 | 45.0 ± 2.3 |
| S       | in_vitb          |   2 |       0 | 83.7 ± 1.9 | 79.3 ± 2.7 |
| S       | in_vitb          |   2 |       1 | 67.7 ± 3.5 | 64.8 ± 3.9 |
| S       | in_vitb          |   2 |       3 | 37.3 ± 1.9 | 28.7 ± 1.6 |
| S       | in_vitb          |   2 |       4 | 55.8 ± 2.4 | 47.7 ± 2.7 |
| S       | dino_vitb        |   1 |       0 | 50.7 ± 1.2 | 47.9 ± 1.1 |
| S       | dino_vitb        |   1 |       1 | 53.5 ± 1.5 | 52.6 ± 1.7 |
| S       | dino_vitb        |   1 |       2 | 69.5 ± 1.7 | 72.8 ± 1.5 |
| S       | dino_vitb        |   1 |       3 | 32.9 ± 1.4 | 30.7 ± 1.3 |
| S       | dino_vitb        |   1 |       4 | 40.1 ± 2.3 | 38.4 ± 2.3 |
| S       | dino_vitb        |   2 |       0 | 75.8 ± 2.3 | 68.5 ± 3.2 |
| S       | dino_vitb        |   2 |       1 | 71.2 ± 3.8 | 68.7 ± 4.2 |
| S       | dino_vitb        |   2 |       3 | 39.8 ± 1.7 | 30.2 ± 1.6 |
| S       | dino_vitb        |   2 |       4 | 48.7 ± 2.5 | 43.9 ± 2.6 |
| S       | in_vits          |   1 |       0 | 45.5 ± 1.3 | 45.7 ± 1.4 |
| S       | in_vits          |   1 |       1 | 46.9 ± 1.5 | 46.6 ± 1.4 |
| S       | in_vits          |   1 |       2 | 65.8 ± 1.9 | 66.0 ± 1.9 |
| S       | in_vits          |   1 |       3 | 26.3 ± 1.1 | 25.0 ± 1.1 |
| S       | in_vits          |   1 |       4 | 41.2 ± 2.8 | 39.3 ± 2.8 |
| S       | in_vits          |   2 |       0 | 61.8 ± 2.5 | 57.9 ± 2.7 |
| S       | in_vits          |   2 |       1 | 61.9 ± 3.9 | 57.1 ± 4.4 |
| S       | in_vits          |   2 |       3 | 38.0 ± 1.9 | 29.2 ± 1.7 |
| S       | in_vits          |   2 |       4 | 51.7 ± 2.8 | 45.8 ± 2.8 |
| S       | arc_vits_m05     |   1 |       0 | 67.2 ± 1.7 | 65.3 ± 1.8 |
| S       | arc_vits_m05     |   1 |       1 | 78.7 ± 1.2 | 75.2 ± 1.4 |
| S       | arc_vits_m05     |   1 |       2 | 82.3 ± 2.1 | 82.9 ± 2.0 |
| S       | arc_vits_m05     |   1 |       3 | 42.4 ± 1.4 | 39.6 ± 1.4 |
| S       | arc_vits_m05     |   1 |       4 | 79.2 ± 1.0 | 75.9 ± 1.4 |
| S       | arc_vits_m05     |   2 |       0 | 91.6 ± 1.8 | 90.9 ± 2.0 |
| S       | arc_vits_m05     |   2 |       1 | 97.6 ± 1.2 | 97.5 ± 1.2 |
| S       | arc_vits_m05     |   2 |       3 | 57.5 ± 1.5 | 52.2 ± 1.8 |
| S       | arc_vits_m05     |   2 |       4 | 82.0 ± 1.5 | 75.7 ± 2.4 |
| S       | arcattr_vits_m05 |   1 |       0 | 73.7 ± 1.7 | 75.8 ± 1.6 |
| S       | arcattr_vits_m05 |   1 |       1 | 75.1 ± 1.8 | 74.6 ± 1.9 |
| S       | arcattr_vits_m05 |   1 |       2 | 68.8 ± 2.3 | 68.9 ± 2.2 |
| S       | arcattr_vits_m05 |   1 |       3 | 50.7 ± 1.2 | 46.2 ± 1.2 |
| S       | arcattr_vits_m05 |   1 |       4 | 81.5 ± 1.2 | 78.8 ± 1.5 |
| S       | arcattr_vits_m05 |   2 |       0 | 89.0 ± 1.9 | 88.6 ± 1.9 |
| S       | arcattr_vits_m05 |   2 |       1 | 88.5 ± 2.9 | 87.5 ± 3.2 |
| S       | arcattr_vits_m05 |   2 |       3 | 61.4 ± 1.4 | 52.8 ± 1.6 |
| S       | arcattr_vits_m05 |   2 |       4 | 87.1 ± 1.5 | 82.7 ± 2.2 |
