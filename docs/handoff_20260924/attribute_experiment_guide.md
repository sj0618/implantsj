# Attribute(속성) 실험 구조와 실행 방법 (2026-09-25 09:05 KST 갱신)

서버에서 attribute 실험을 바로 이어서 돌리기 위한 문서입니다. 인터넷 없이 볼 수 있게 썼습니다.
- **[확인]**: 서버 파일을 직접 열어 확인했습니다.
- **[추정]**: 확인하지 않은 판단입니다.
- 서버 로그 시각은 UTC입니다. KST는 +9시간입니다.

## 요약

- 서버에는 attribute 실행 경로가 **두 개** 있습니다.
  - **(A) 파이프라인 경로:** `scripts/05_attributes/train_attributes.sh`가 `configs/05_attributes.yaml`을 읽고 `src/run_attributes.py`를 실행합니다.
  - **(B) 기록 폴더 경로:** 기록 폴더 안의 `driver.py`와 `effective.json`으로 돌립니다. yaml을 쓰지 않습니다. **지금 돌고 있는 실험이 이 방식입니다.**
- 지금 실행 중인 실험은 `20260924-2303-attributes-tiny-queue-r3`입니다.
  - ConvNeXt-Tiny로 42클래스 전체(`full42`)를 학습합니다.
  - 10개 속성을 **각각 따로 학습**하는 큐입니다.
  - 01:30 KST 기준 첫 속성(`System`)을 학습하는 중입니다.
- **2026-09-25 02:49 KST에 closed27 학습 + open15 평가 실험 `20260925-attr-closed27`을 띄웠습니다(0절).** r3와 병렬로 돕니다.
- **속성+이미지 임베딩 결합(0-6절, 09:00 KST): 속성을 더하면 seen은 오르고 unseen은 떨어집니다. H 최고는 속성을 쓰지 않을 때(0.838)입니다.**
- **주의:** r3 실험은 `full42`라서 open15 클래스 이미지도 학습에 들어갑니다. **unseen 클래스 인식(논문 핵심 질문)의 근거로는 쓸 수 없습니다.** 그 목적이라면 closed27로 다시 학습해야 합니다(4절).

---

## 0. closed27 attribute 실험: `20260925-attr-closed27` (2026-09-25 02:49 KST 시작) [확인]

r3 드라이버를 복사해서 **closed27로 학습하고 open15로 unseen을 평가**하도록 고친 실험입니다. r3와 병렬로 GPU를 나눠 씁니다.

### 0-1. 무엇을 바꿨나

기록 폴더는 `aihub_run_new/outputs/experiment_records/20260925-attr-closed27/`입니다. r3 폴더에서 `driver.py`, `effective.json`, `spec_table.csv`, `pretrained_hash.txt`, `src/`를 복사했습니다.

`effective.json`에서 바꾼 값:

| 키 | r3 | closed27 |
|---|---|---|
| `manifest` | `data/manifest_full42.csv` | `data/manifest_closed27.csv` |
| `open_manifest` | 없음 | `data/manifest_open15.csv` (새 키) |
| `protocol` | `full42` | `closed27_open15` |
| `output` | `outputs/attributes/20260924-2303-attributes-tiny-queue-r3` | `outputs/attributes/20260925-attr-closed27` |

나머지 학습 설정(ConvNeXt-Tiny, 30 epoch, batch 32, lr 1e-5/1e-4 등)은 r3와 같습니다.

`driver.py` 패치(원본 대비 변경 4곳). 서버에서 `diff ../20260924-2303-attributes-tiny-queue-r3/driver.py driver.py`로 확인할 수 있습니다.
1. `frames = ...` 다음 줄에 추가:
   ```python
   OPEN=set()
   if cfg.get('open_manifest'):
       ot=core.read_manifest(cfg['open_manifest'],'test'); OPEN=set(ot['label']); frames['test']=pd.concat([frames['test'],ot],ignore_index=True)
   ```
   - train과 valid는 closed27만 씁니다. 모델 선택(early stopping)과 fuse 가중치 학습은 closed27 valid만 봅니다.
   - test에만 open15 test를 붙입니다. 스펙표는 test 라벨까지 포함하므로 42클래스, 34개 그룹 그대로입니다.
2. smoke/preflight의 `assert all(min(v)>0 for v in counts.values())`를 `print('ZERO_COUNTS', ...)`로 바꿨습니다. closed27 train에는 일부 속성값이 아예 없어서 assert가 실패하기 때문입니다.
3. `protocol='full42'`를 `protocol=cfg['protocol']`로 바꿨습니다(체크포인트 contract에 기록).
4. fuse 단계에서 test 행마다 open 클래스 여부(`opens[s]`)를 계산하고, `summary.csv`에 `n_seen`, `n_unseen`, `top1_seen`, `top1_unseen` 열을 추가했습니다.

패치는 로컬의 가짜 드라이버로 먼저 시험했고, 서버에서 `python3 -m py_compile driver.py`를 통과했습니다(COMPILE_OK).

### 0-2. 실행 큐

`queue_runner.py` 대신 bash 반복문으로 돌립니다. 해시 검증은 하지 않습니다. 기록 폴더에서 실행했습니다.

```bash
A=/home/ubuntu/aihub_run_new nohup bash -c 'for j in smoke shared fuse_shared System Connection_1 Connection_2 Flange Collar Microthread Body_Shape Thread_Shape Apex_Shape Apex_Hole fuse_separate; do c=$(docker run -d --gpus all --network none --ipc host -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/experiment:ro -v $A/data:/workspace/ai_hub_pipeline/data -v $A/outputs:/workspace/ai_hub_pipeline/outputs -v $A/checkpoints:/workspace/ai_hub_pipeline/checkpoints -v $A/seg_data:/workspace/ai_hub_pipeline/seg_data -w /workspace/ai_hub_pipeline --name attr-c27-$j sha256:9b7683277acc6bfef03914aa3c10aacbad7a658e43f7cf4994c59d1c18251fd0 python -u /experiment/driver.py $j); echo $(date -u +%FT%T) START $j $c; r=$(docker wait $c); echo $(date -u +%FT%T) END $j $r; docker logs $c > log_$j.txt 2>&1; if [ $j = smoke ] && [ $r != 0 ]; then exit 1; fi; done' > queue.log 2>&1 &
```

- 순서: smoke → **shared**(백본 1개 + head 10개, 한 번에 학습) → fuse_shared → 속성별 독립 모델 10개 → fuse_separate.
- shared를 먼저 둔 이유: 모델 하나로 10개 속성을 학습하므로 가장 빨리 전체 결과(seen/unseen top1)가 나옵니다.
- 컨테이너 이름은 `attr-c27-<job>`, 각 job의 로그는 기록 폴더의 `log_<job>.txt`입니다.
- 같은 이름의 컨테이너가 남아 있으면 다시 실행할 때 실패합니다. 재실행 전에 `docker rm attr-c27-<job>`으로 지웁니다.

### 0-3. 진행 상황

| 시각(KST) | 내용 |
|---|---|
| 02:48:34 | smoke 시작 |
| 02:49:14 | smoke 종료, 종료 코드 0. `VERIFIED_PRETRAINED 31fa48e9…`, `PREFLIGHT_OK smoke` |
| 02:49:15 | shared 시작 |
| 02:49:50 | shared 데이터 로드: train 107,573장, valid 15,348장, test 16,570장(closed27 test + open15 test). head 10개: System 2, Connection_1 2, Connection_2 3, Flange 3, Collar 2, Microthread 2, Body_Shape 3, Thread_Shape 4, Apex_Shape 2, Apex_Hole 2 |
| 02:52 | GPU 99%, 6.2GB (r3 큐와 함께 사용 중) |
| 07:50 | shared 29/30 epoch 완료. epoch당 약 600초. valid 평균 macro-F1 0.935→0.941(epoch 27→29), 평균 정확도 0.981 (`history_shared.csv`) |
| 07:50 | r3 큐: System 완료(01:55 KST), Connection_1 완료(05:14 KST), Connection_2 학습 중. r3는 속성 하나에 약 2.4–3.3시간 |

### 0-4. smoke에서 확인한 사실: closed27 train에 없는 속성값

```
ZERO_COUNTS {'System': [], 'Connection_1': [], 'Connection_2': [2], 'Flange': [], 'Collar': [], 'Microthread': [], 'Body_Shape': [], 'Thread_Shape': [], 'Apex_Shape': [], 'Apex_Hole': [1]}
```

- vocab: `Connection_2` = [HEX, OCTA, **TRI-LOBE**], `Apex_Hole` = [No, **Yes**] (`smoke/spec_table.json`에서 확인).
- 즉 **`Connection_2=TRI-LOBE`와 `Apex_Hole=Yes`는 closed27 train 이미지에 한 장도 없습니다.** 이 값을 가진 클래스는 open15에만 있습니다.
- 모델은 이 두 값을 학습에서 한 번도 정답으로 보지 못하므로 거의 예측하지 못합니다. 그러면 해당 open 클래스는 이 속성에서 항상 틀리게 매칭됩니다.
- [추정] `Apex_Hole=Yes`는 원래 표에서 2종뿐이었고(`implant_attribute_verification_20260924.md`), 두 종 모두 open15에 있는 것으로 보입니다. TRI-LOBE는 Nobel 계열의 삼엽형 연결로 보이지만 해당 클래스 이름은 확인하지 않았습니다.
- 논문에서는 "학습에 없는 속성값" 문제로 따로 보고해야 합니다. 속성 기반 zero-shot의 알려진 한계입니다.

### 0-4b. shared 결과 (2026-09-25 08:20 KST 확인) [확인]

shared는 07:53 KST에 30 epoch를 마쳤고(종료 코드 0), fuse_shared는 07:54 KST에 끝났습니다(종료 코드 0). 이어서 속성별 독립 모델(System부터)이 돌고 있습니다.

**34개 그룹 매칭 결과** (`fuse_shared/summary.csv`)

| 가중치 | split | top1 | top2 | macro-F1(34그룹) | n_seen | n_unseen | top1_seen | top1_unseen |
|---|---|---|---|---|---|---|---|---|
| learned | valid | 0.9367 | 0.9632 | 0.5880 | 15,348 | 0 | 0.9367 | – |
| learned | test | 0.8708 | 0.8979 | 0.5531 | 15,349 | 1,221 | **0.9380** | **0.0254** |
| equal | test | 0.8705 | 0.8983 | 0.5540 | 15,349 | 1,221 | 0.9378 | 0.0254 |

- **속성만으로 처음 보는 클래스를 맞힌 비율은 2.5%(31/1,221장)입니다.** 34개 그룹에서 무작위로 고를 때의 기댓값(약 2.9%)과 비슷합니다.
- 학습한 27종은 93.8%입니다. 가중치를 학습해도(learned) 같은 가중치(equal)와 차이가 없습니다.

**속성별 정확도: seen(closed27 test 15,349장) / unseen(open15 test 1,221장)**

`shared/test.npz`의 예측 argmax와, `shared/groups.json`의 그룹 signature로 복원한 정답을 비교했습니다.

| 속성 | 값 개수 | seen | unseen |
|---|---|---|---|
| System | 2 | 0.999 | 0.964 |
| Connection_1 | 2 | 0.959 | 0.867 |
| Connection_2 | 3 | 0.975 | 0.803 |
| Flange | 3 | 0.975 | 0.731 |
| Collar | 2 | 0.961 | 0.726 |
| Microthread | 2 | 0.987 | 0.484 |
| Body_Shape | 3 | 0.977 | **0.257** |
| Thread_Shape | 4 | 0.991 | 0.784 |
| Apex_Shape | 2 | 0.993 | **0.287** |
| Apex_Hole | 2 | 1.000 | 0.982 |

해석:
- 10개 속성을 모두 맞혀야 그룹이 맞습니다. unseen 정확도를 곱하면 약 0.5%이므로, 그룹 정확도가 2.5%로 낮은 것은 속성별 오류가 누적된 결과로 설명됩니다.
- **Apex_Shape(값 2개)의 unseen 정확도 0.287은 무작위(0.5)보다 낮습니다.** Body_Shape 0.257, Microthread 0.484도 낮습니다.
  - 무작위보다 낮다는 것은 모델이 형태를 보고 판단하지 않고, 겉모습이 비슷한 학습 클래스의 값을 그대로 가져온다는 신호입니다. 클래스 정체성을 경유하는 shortcut으로 보입니다[추정].
  - 스펙표 라벨 오류도 원인일 수 있습니다. 웹 검증에서 Body Shape(Osstem TS/SS/USIII)와 Collar에 오류가 많았습니다(`implant_attribute_verification_20260924.md`). 서버 `spec_table.csv`가 21:52에 수정됐지만 어떤 값이 바뀌었는지는 확인하지 않았습니다.
- Apex_Hole unseen 0.982: `Yes`는 학습에 없던 값입니다. 대부분의 unseen 이미지 정답이 `No`라서 정확도가 높게 나온 것이지, `Yes`를 맞힌 것이 아닙니다.
- 결론[추정]: 지금 방식(속성을 한 모델로 학습해서 34개 그룹에 매칭)만으로는 unseen 인식이 되지 않습니다. 속성은 이미지 임베딩(galcov supcon kmeans_multi+Maha, unseen 0.82)을 보조하는 신호로 결합해 봐야 합니다.

### 0-5. 결과 확인 명령

```bash
R=~/aihub_run_new/outputs/experiment_records/20260925-attr-closed27
O=~/aihub_run_new/outputs/attributes/20260925-attr-closed27
cat $R/queue.log
docker logs --tail 5 attr-c27-shared
tail -3 $O/shared/history_shared.csv
cat $O/fuse_shared/summary.csv      # method, split, top1, top2, macro_f1, n_seen, n_unseen, top1_seen, top1_unseen
```

- `summary.csv`의 test 행에서 `top1_unseen`이 attribute만으로 처음 보는 클래스를 맞힌 비율입니다. 34개 그룹 단위이므로 42클래스 정확도와 직접 비교하면 안 됩니다.
- closed27 train(107,573장)은 full42 train(115,988장)보다 8,415장 적습니다. open15의 이미지가 전체의 약 7%라는 앞선 추정과 맞습니다.
- [추정] 소요 시간: r3의 System 한 속성이 GPU를 혼자 쓸 때 2시간 23분 걸렸습니다. closed27은 학습 이미지가 적지만 GPU를 r3와 나눠 쓰므로, shared가 2–4시간, 전체 큐가 하루 이상 걸릴 것으로 봅니다.

### 0-6. 속성과 이미지 임베딩 결합 실험 (2026-09-25 09:00 KST) [확인]

방법:
- 클래스 c의 점수는 `score(c) = -D(c)/2 + λ·Σ_a log P_a(spec[c,a])`입니다. 42클래스 중 점수가 가장 큰 것을 예측으로 씁니다.
  - `D`는 galcov supcon(timm 5ep) metric 평가의 test Mahalanobis(Ledoit-Wolf) 거리입니다. 파일은 `metric/closed27_open15/dinov2_galcov_timm_5ep/closed27_open15_dinov2_supcon/eval/scores_test_<proto>_mahalanobis_ledoit_wolf.csv`입니다.
  - `P_a`는 shared 속성 모델의 test 확률(`shared/test.npz`)입니다. 클래스별 속성값은 `groups.json`의 signatures에서 가져옵니다.
  - seen과 unseen은 `data/manifest_open15.csv`의 라벨로 나눕니다. H는 seen과 unseen 정확도의 조화평균입니다.
- 스크립트는 서버 `~/fuse_attr/fa.py`, 결과는 `~/fuse_attr/res.txt`에 있습니다. 전문은 이 절 끝에 있습니다.

정합성 확인:
- λ=0(거리만) 결과가 기존 보고값과 같습니다. kmeans_multi는 0.8542/0.8569/0.8206, mean은 0.8976/0.9109/0.7305입니다. 이미지 경로 정렬, 클래스 열 순서, 거리 방향(작을수록 가깝다)이 맞다는 뜻입니다.
- 속성만으로 34그룹 단위로 채점하면 0.8705/0.9378/0.0254입니다. `driver.py`의 `fuse_shared`(0.8708/0.938/0.0254)와 같습니다. 42클래스 단위로 채점하면 0.7975/0.8601/0.0106입니다.

결과 (test, overall / seen / unseen / H; seen 15,349장, unseen 1,221장):

| λ | kmeans_multi + Maha LW | mean + Maha LW |
|---|---|---|
| 속성만 | 0.7975 / 0.8601 / 0.0106 / 0.021 | 같음 |
| 0 (거리만) | 0.8542 / 0.8569 / **0.8206** / **0.838** | 0.8976 / 0.9109 / 0.7305 / 0.811 |
| 0.03 | 0.9011 / 0.9186 / 0.6806 / 0.782 | **0.9063** / 0.9241 / 0.6822 / 0.785 |
| 0.1 | 0.8929 / 0.9341 / 0.3751 / 0.535 | 0.8940 / 0.9330 / 0.4038 / 0.564 |
| 0.3 | 0.8815 / 0.9380 / 0.1712 / 0.290 | 0.8794 / 0.9374 / 0.1507 / 0.260 |
| 1 | 0.8737 / 0.9385 / 0.0598 / 0.112 | 0.8744 / 0.9393 / 0.0590 / 0.111 |
| 3 | 0.8710 / 0.9373 / 0.0385 / 0.074 | 0.8721 / 0.9387 / 0.0344 / 0.066 |

해석:
- λ를 0보다 크게 하면 seen 정확도는 오르고 unseen 정확도는 항상 내려갑니다. λ가 클수록 unseen은 무너집니다. **H가 가장 높은 것은 λ=0, 즉 속성을 쓰지 않을 때입니다.**
- overall 최고값은 mean + λ=0.03의 0.9063입니다(거리만일 때보다 +0.009). test의 93%가 seen이라서 overall은 seen을 따라갑니다. λ를 test에서 골랐으므로 이 값은 낙관적입니다.
- λ=1(두 확률을 그대로 곱하는 베이즈식 결합)에서 unseen은 0.06입니다.
- 원인[추정]:
  - 속성 모델은 unseen 이미지에도 비슷한 seen 클래스의 속성 조합을 확신 있게 예측합니다. 그래서 unseen 클래스의 조합은 확률이 낮고, 결합하면 예측이 seen 클래스 쪽으로 끌려갑니다.
  - 학습에 없던 값(`Connection_2=TRI-LOBE`, `Apex_Hole=Yes`)을 가진 클래스는 확률이 거의 0이라서, 결합하면 사실상 후보에서 빠집니다.
- 결론[추정]: 지금 속성 모델은 unseen 인식에 도움이 되지 않습니다. seen 정확도를 올리는 효과는 있습니다. 이 용도로 쓰려면 λ를 valid에서 골라야 합니다.

다음 후보(결정 필요):
- 속성별 독립 모델(separate 큐)이 끝나면 같은 스크립트로 다시 돌립니다. separate 출력이 `test.npz` 같은 형식인지는 확인하지 않았습니다.
- unseen 정확도가 높은 속성(System 0.964, Apex_Hole, Conn1 0.867)만 쓰거나, 확률에 temperature를 걸어 보정합니다.
- valid 거리 CSV로 λ를 고릅니다. valid CSV가 있는지는 확인하지 않았습니다.

버그 점검 결과:
- 결합 스크립트: 위 정합성 확인 두 가지를 통과했습니다. 버그는 찾지 못했습니다.
- `driver.py`의 `fuse_*/summary.csv`에 있는 top1은 **34그룹 단위**입니다. 42클래스 metric 결과와 직접 비교하면 안 됩니다. 버그는 아니지만 보고할 때 주의해야 합니다.
- 속성만의 42클래스 unseen 정확도 0.0106은 무작위(1/42≈0.024)보다 낮습니다. 같은 속성 조합을 공유하는 클래스끼리는 점수가 같아서, argmax가 앞쪽 열(대부분 seen 클래스)을 고르는 효과가 섞여 있습니다.
- 이전에 확인한 버그(README 5절, `metric_galcov_results_20260925.md`)는 그대로입니다: rda logdet 지배, triplet collapse, covaware OOM, `run_metric.py:39` assert, `ss_count: 5pwd`.

실행 명령 (서버 `~/fuse_attr`):

```bash
M=~/aihub_run_new/outputs/metric/closed27_open15/dinov2_galcov_timm_5ep/closed27_open15_dinov2_supcon/eval
for d in kmeans_multi mean; do echo == $d; python3 fa.py ~/aihub_run_new/outputs/attributes/20260925-attr-closed27/shared $M/scores_test_${d}_mahalanobis_ledoit_wolf.csv ~/aihub_run_new/data/manifest_open15.csv; done 2>&1 | tee res.txt
```

`fa.py` 전문:

```python
import json,csv,sys,numpy as np
A,E,M=sys.argv[1:4]
t=json.load(open(A+'/spec_table.json'));g=json.load(open(A+'/groups.json'));z=np.load(A+'/test.npz')
C=t['classes'];F=t['attributes']
S={c:s for m,s in zip(g['members'],g['signatures']) for c in m}
T=np.array([S[c] for c in C])
L=sum(np.log(z[a][:,T[:,j]]+1e-6) for j,a in enumerate(F))
ix={p:i for i,p in enumerate(z['paths'])}
r=csv.reader(open(E));K=next(r)[1:];R=list(r)
D=np.array([x[1:] for x in R],float);ii=np.array([ix[x[0]] for x in R])
L=L[ii][:,[C.index(c) for c in K]]
y=np.array([K.index(C[k]) for k in z['class_index'][ii]])
op={x['label'] for x in csv.DictReader(open(M))};o=np.array([K[k] in op for k in y])
print('n',len(y),'unseen',o.sum(),'argmin',(D.argmin(1)==y).mean(),'argmax',(D.argmax(1)==y).mean())
def rep(n,s):
 c=s.argmax(1)==y;a,b=c[~o].mean(),c[o].mean();print(n,round(c.mean(),4),round(a,4),round(b,4),round(2*a*b/(a+b+1e-9),4))
rep('attr_only',L)
for w in [0,0.03,0.1,0.3,1,3]:rep('lam%g'%w,-D/2+w*L)
```

---

## 1. 서버 파일 위치 [확인]

코드 (`/home/ubuntu/implant_pipeline_new`)

| 경로 | 내용 |
|---|---|
| `src/attributes/core.py` | 공통 함수: manifest 읽기, 백본 생성, masked CE, seed |
| `src/attributes/train.py` | `train_attributes()`: 속성별 학습, 확률 예측, 임베딩 내보내기 |
| `src/attributes/spec_table.py` | `build_spec_table()`: 스펙표 CSV를 클래스 × 속성 정답표로 변환 |
| `src/attributes/fuse.py` | `match_probability_tensor()`, `learn_feature_weights()`: 속성 확률을 클래스 점수로 결합 |
| `src/attributes/visualize.py` | 그림 생성 |
| `src/run_attributes.py` | (A) 경로의 드라이버 |
| `src/check_spec_table.py` | 스펙표와 데이터 폴더명 매칭 점검 |
| `configs/05_attributes.yaml` | (A) 경로 설정 |
| `configs/spec_table.csv` | 클래스 → 속성 정답표. **2026-09-24 21:52에 다른 세션이 수정했습니다.** 백업이 `spec_table.before_20260924_2152.csv`와 `spec_table.csv.bak`에 있고, 변경 내역은 `spec_table.update_20260924_2152.json`에 있습니다. 어떤 값이 바뀌었는지는 확인하지 않았습니다 |
| `scripts/05_attributes/train_attributes.sh`, `README.md` | (A) 경로 실행 스크립트와 설명 |

결과 (`/home/ubuntu/aihub_run_new/outputs`)

| 경로 | 내용 |
|---|---|
| `experiment_records/20260924-2303-attributes-tiny-queue-r3/` | 실행 중인 실험의 기록 폴더. `record.md`, `effective.json`, `driver.py`, `queue_config.json`, `queue_runner.py`, `queue_state.json`, `queue_r3.log`, `spec_table.csv` 사본, `src/` 사본 |
| `experiment_records/20260924-2303-attributes-tiny-queue`, `-r2` | 같은 큐의 이전 시도. r3가 세 번째입니다 |
| `experiment_records/20260924-2250-attributes-cnn-groups/` | 그 직전 기록. 내용은 확인하지 않았습니다 |
| `attributes/20260924-2303-attributes-tiny-queue-r3/<job>/` | r3 결과. 속성마다 폴더가 하나씩 생깁니다 |

## 2. 실행 중인 실험: `attributes-tiny-queue-r3` [확인]

`record.md` 요약:
- Tiny(ConvNeXt-Tiny) full42 attributes 큐이고, 2026-09-24에 준비했습니다.
- 실제 설정은 `effective.json`이 정합니다. 드라이버는 yaml 상속을 쓰지 않습니다.
- 큐 순서: covaware 컨테이너가 끝나기를 기다린 뒤 smoke(batch 32), 속성별 모델 10개, fuse, shared, fuse 순서로 돕니다.
- 속성 하나가 실패해도 큐는 계속 돕니다. smoke가 실패하면 멈춥니다. fuse는 모델 10개가 모두 있어야 합니다.
- 이미지는 sha256으로 고정했고, 오프라인 pretrained 가중치의 해시를 모델마다 검증합니다.
- r3에서 고친 것: `/experiment` 마운트 방식, `queue_runner`가 표준 라이브러리 이름과 겹치던 문제.

`effective.json`의 주요 값:

| 항목 | 값 |
|---|---|
| model | `convnext_tiny.fb_in22k_ft_in1k` (pretrained, `checkpoint: null`) |
| protocol / manifest | `full42` / `data/manifest_full42.csv` |
| features (10개) | System, Connection_1, Connection_2, Flange, Collar, Microthread, Body_Shape, Thread_Shape, Apex_Shape, Apex_Hole. Body_Type은 값이 하나뿐이라 빠졌습니다 |
| 학습 | epochs 30, batch 32, AdamW, backbone_lr 1e-5, head_lr 1e-4, weight_decay 0.05, warmup 3, cosine, patience 7, amp |
| loss | cross_entropy, smoothing 0, class_weights = train 빈도의 역수(최대 비율 20) |
| 정규화 | dropout 0.1, drop_path 0.1 |
| 증강 | flip 0.5, rotation 10, brightness 0.1, contrast 0.1, resize direct 224 |
| fuse | `learned_probability`, 300 step, lr 0.05. `equal`(속성 가중치 동일)도 함께 계산 |
| group_count | 34 |
| output | `outputs/attributes/20260924-2303-attributes-tiny-queue-r3` |

`driver.py`의 동작:
1. `spec_table.csv`로 42클래스 × 10속성 정답표를 만듭니다.
   - `Connection_2`는 `Hex→HEX`, `Octagon→OCTA`로 합칩니다.
   - 값이 하나뿐인 속성은 뺍니다.
2. 속성 조합이 같은 클래스를 한 그룹으로 묶습니다. **42클래스는 34개 그룹이 되고, 평가는 이 34개 그룹 단위로 합니다.** 같은 그룹 안의 클래스는 속성만으로 구분할 수 없기 때문입니다. 그룹 구성은 `groups.json`에 있습니다.
3. job이 속성 이름이면 그 속성 하나만 학습합니다(`separate`). `shared`이면 백본 하나에 head 10개를 붙여 학습합니다.
   - 학습이 끝나면 valid와 test의 확률을 `<split>.npz`로 저장합니다.
   - 속성별 classification report, 혼동행렬, 학습 곡선, 추론 시간도 저장합니다.
4. job이 `fuse_separate` 또는 `fuse_shared`이면 속성 확률을 그룹 점수로 바꿉니다.
   - 속성 가중치는 valid에서만 학습합니다.
   - `learned`와 `equal` 두 방식의 top1, top2, macro-F1(34그룹)을 `summary.csv`에 씁니다.
5. 매 실행마다 pretrained 가중치의 해시를 `pretrained_hash.txt`와 비교합니다.

진행 상황 (2026-09-25 01:30 KST):
- `smoke`: 완료. 23:31–23:32 KST, peak memory 2.43GB.
- `System`: 23:32 KST 시작. 01:26에 `history_System.csv`와 `checkpoints/System.pt`(111MB)가 갱신됐습니다. GPU 사용량은 79–84%, 약 3.3GB입니다.
- 남은 job: 속성 9개, `fuse_separate`, `shared`, `fuse_shared`.
- **[추정] 속성 하나에 2시간 이상 걸리므로 전체가 끝나려면 하루 이상 걸립니다.** early stopping(patience 7)이 걸리면 더 빨리 끝납니다.

## 3. 상태 확인 명령

```bash
R=~/aihub_run_new/outputs/experiment_records/20260924-2303-attributes-tiny-queue-r3
O=~/aihub_run_new/outputs/attributes/20260924-2303-attributes-tiny-queue-r3
docker ps --format '{{.Names}} {{.Status}}' | grep attr
tail -5 $R/queue_r3.log
python3 -c "import json;s=json.load(open('$R/queue_state.json'));print(s['status'],[(j['job'],j['status']) for j in s['jobs']])"
ls $O; tail -3 $O/*/history_*.csv
cat $O/fuse_separate/summary.csv 2>/dev/null
```

## 4. 새 attribute 실험을 띄우는 방법

### 4-1. (B) 기록 폴더 방식: 검증된 경로

r3 기록 폴더를 복사해서 설정만 바꿉니다.

```bash
cd ~/aihub_run_new/outputs/experiment_records
NEW=$(date +%Y%m%d-%H%M)-attributes-<이름>
cp -r 20260924-2303-attributes-tiny-queue-r3 $NEW
cd $NEW
rm -rf jobs prior_attempt* queue.* queue_r2.log queue_r3.log queue_launch*.json queue_state.json predecessor_final.json
# effective.json에서 output을 새 폴더로 바꾸고, 필요하면 epochs·features 등을 바꿉니다.
#   "output": "/workspace/ai_hub_pipeline/outputs/attributes/<NEW와 같은 이름>"
```

속성 하나를 직접 띄우는 명령입니다. `queue_runner.py`가 job마다 실행하는 명령과 같습니다.

```bash
A=/home/ubuntu/aihub_run_new
R=$A/outputs/experiment_records/$NEW
docker run -d --gpus all --network none --ipc host \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v $R:/experiment:ro \
  -v $A/data:/workspace/ai_hub_pipeline/data \
  -v $A/outputs:/workspace/ai_hub_pipeline/outputs \
  -v $A/checkpoints:/workspace/ai_hub_pipeline/checkpoints \
  -v $A/seg_data:/workspace/ai_hub_pipeline/seg_data \
  -w /workspace/ai_hub_pipeline --name attr-<이름>-<속성> \
  sha256:9b7683277acc6bfef03914aa3c10aacbad7a658e43f7cf4994c59d1c18251fd0 \
  python -u /experiment/driver.py <job>
```

- `<job>`은 `smoke`, 속성 이름(예: `Collar`), `shared`, `fuse_separate`, `fuse_shared` 중 하나입니다.
- **먼저 `smoke`를 돌립니다.** CUDA로 2 step만 학습해서 설정 오류를 잡습니다.
- 큐로 돌리려면 `queue_config.json`에서 `predecessor`와 `jobs`, 마운트 경로(`/experiment`)를 새 폴더에 맞게 고친 뒤 `nohup python3 queue_runner.py > queue.log 2>&1 &`로 실행합니다.
  - `queue_runner.py`는 시작할 때 `hashes`에 적힌 파일 해시를 검증합니다. `effective.json`이나 `driver.py`를 고쳤으면 해시도 새로 계산해 넣어야 합니다.
  - `predecessor`는 기다릴 컨테이너입니다. 기다릴 것이 없으면 이미 끝난 컨테이너 ID를 넣습니다. 비워도 되는지는 확인하지 않았습니다.
- **closed27로 학습하려면 `driver.py`를 고쳐야 합니다.**
  - `assert len(table.classes)==42`와 `assert len(sigs)==34`, fuse의 `range(34)`, `success.json`의 `groups=34, classes=42`가 42클래스를 가정합니다.
  - `manifest`를 `data/manifest_closed27.csv`로 바꾸고 위의 숫자를 데이터에서 계산하도록 바꿔야 합니다.
  - [추정] 그 뒤에 open15 이미지에 대해 속성 예측과 매칭을 따로 평가하는 job을 추가해야 unseen 평가가 됩니다. 이 부분은 아직 구현되어 있지 않습니다.

### 4-2. (A) 파이프라인 방식: 설정 파일로 실행 (실행 확인 안 됨)

`scripts/05_attributes/README.md`에 적힌 사용법입니다.

```bash
cd ~/implant_pipeline_new/scripts/05_attributes
bash train_attributes.sh                                   # protocol: closed27_open15 (기본값)
PROTOCOL=full42 bash train_attributes.sh                   # 42클래스 전체
SET="grid.backbone=cnn" bash train_attributes.sh           # 백본 하나만
SET="grid.backbone=cnn run.features=[System,Connection_1]" bash train_attributes.sh
```

`configs/05_attributes.yaml`의 기본값:
- `protocol: closed27_open15`: 27클래스만 학습해서 누수를 막습니다.
- `backbones.cnn`: `convnext_tiny.fb_in22k_ft_in1k`, 224px. `grid.backbone: [cnn, dinov2]`로 두 백본을 모두 돌립니다.
- `spec.csv: configs/spec_table.csv`. `unknown_values`(`확인 불가`, `unknown`, `n/a`, `-`, `?` 등)는 학습과 평가에서 제외합니다. `drop_constant: true`입니다.
- `features`: 11개 속성마다 `{train, use_in_fuse}`를 켜고 끌 수 있습니다.
- `run.features: all`, `skip_existing: true`: 이미 학습된 속성은 건너뛰므로 며칠에 나눠 돌릴 수 있습니다.
- `train.mode: shared`, epochs 60, batch 32, lr 5e-5, weight_decay 0.05, label_smoothing 0.1, `use_class_weights: false`, patience 12, seed 42.
- `fuse.mode: learned`, `learn_epochs` 300, `learn_lr` 0.05, `min_match` 8, `top_k` 2.
- 출력: `outputs/attributes/<protocol>_<backbone>_<mode>/`
  - `checkpoints/`, `embeddings/`, `history_<feature>.csv`, `attribute_metrics.json`, `feature_weights.json`
  - `predictions_{valid,test}.csv`, `scores_{valid,test}.csv`(N×C, 클수록 좋음), `spec_table.json`, `visualization/`

주의 [확인되지 않은 위험]:
- 메모리 기록에 따르면 `configs/base.yaml`의 `paths`가 `/home/ubuntu/aihub_run/...` 절대경로입니다. `train_attributes.sh`가 컨테이너 안에서 도는지, 결과가 `aihub_run_new`에 쌓이는지는 **확인하지 않았습니다.** 처음 실행할 때 `SET="train.epochs=1 visualize.enabled=false"`로 짧게 돌려 출력 경로부터 확인하세요.
- 다른 세션이 (A) 대신 (B)를 만든 이유는 기록에 없습니다. [추정] 설정 상속과 코드 변경을 통제하고 재현성을 확보하려던 것으로 보입니다.

## 5. 스펙표(속성 정답표) 점검

```bash
cd ~/implant_pipeline_new
python src/check_spec_table.py --config configs/05_attributes.yaml          # 매칭 안 되는 클래스 찾기
python src/check_spec_table.py --config configs/05_attributes.yaml --fill   # dataset_label 열 자동 채우기
```

- 스펙표 값의 오류와 라벨 정의 문제는 `implant_attribute_verification_20260924.md`에 정리했습니다(Collar 10건 이상 등).
- 서버의 `spec_table.csv`는 21:52에 수정됐습니다. 이 수정이 웹 검증 결과를 반영한 것인지는 **확인하지 않았습니다.** 다음 접속 때 `spec_table.update_20260924_2152.json`을 먼저 봅니다.

## 6. 결정이 필요한 것

| 결정 | 선택지 | 권고 | 근거 | 확신 |
|---|---|---|---|---|
| 실행 중인 r3 큐 | (1) 끝까지 돌림 (2) 중단하고 closed27로 다시 설계 | (1) | 속성 예측 자체의 정확도와 34그룹 매칭 성능을 보는 데는 full42도 유효합니다. GPU를 3GB만 써서 다른 실험과 함께 돌릴 수 있습니다 | 0.6. 하루 이상 걸린다는 추정이 맞다면 closed27 실험이 그만큼 늦어집니다 |
| 논문용 attribute 실험 | (1) (B)의 `driver.py`를 closed27용으로 고침 (2) (A) 파이프라인을 closed27_open15로 실행 | (1) | (B)는 이미 해시 검증, 오프라인 가중치 검증, smoke까지 통과한 경로입니다. (A)는 출력 경로 문제가 확인되지 않았습니다 | 0.55. (A)의 `run_attributes.py`가 open15 평가를 이미 구현했다면 (2)가 빠릅니다. 이 코드는 읽지 않았습니다 |
