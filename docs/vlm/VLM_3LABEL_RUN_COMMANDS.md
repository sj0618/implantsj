# 3-label 최신 VLM 분석 직접 실행 명령어

이 문서는 `plain_vit_3label.csv` 전체 2,075장에 최신 VLM 분석 schema인 **13 core attributes + 21 supporting visual features**를 직접 적용하기 위한 터미널 명령어 모음입니다.

## 0. 작업 위치

```bash
cd /workspace/implant_python_only_final
```

## 1. 의존성 설치

최초 1회만 실행합니다.

```bash
python -m pip install --upgrade openai python-dotenv
```

## 2. Azure endpoint root 추출

현재 `.env`의 `AZURE_OPENAI_ENDPOINT`가 `/openai/v1`까지 포함되어 있으면 Azure SDK에서 404가 날 수 있습니다. 아래처럼 root endpoint만 뽑아서 사용합니다.

```bash
ROOT_ENDPOINT=$(python - <<'PY'
from pathlib import Path
from urllib.parse import urlparse
endpoint=''
for line in Path('.env').read_text(errors='replace').splitlines():
    s=line.strip()
    if s.startswith('AZURE_OPENAI_ENDPOINT='):
        endpoint=s.split('=', 1)[1].strip().strip('"').strip("'")
        break
u=urlparse(endpoint)
print(f'{u.scheme}://{u.netloc}')
PY
)

echo "$ROOT_ENDPOINT"
```

## 3. Dry-run: 대상 확인

API 호출 없이 manifest, label 수, pending 개수만 확인합니다.

```bash
python generate_13feature_21reason_7label.py \
  --input data/manifests/plain_vit_3label.csv \
  --expected-label-count 3 \
  --output data/manifests/13feature_21reason_3label_all.jsonl \
  --json-output data/manifests/13feature_21reason_3label_all.json \
  --summary-csv data/manifests/13feature_21reason_3label_all.summary.csv \
  --prompt-output data/manifests/13feature_21reason_3label.prompt.md \
  --dry-run
```

정상이라면 대략 다음처럼 나옵니다.

```text
Rows: total=2075 done=... pending=...
Labels: ['Bego', 'Bicon', 'ITI']
Splits: ['test', 'train', 'valid']
Dry run only. No API calls were made.
```

## 4. 1장 smoke test

전체를 돌리기 전에 API/endpoint/deployment가 맞는지 1장만 테스트합니다.

```bash
python generate_13feature_21reason_7label.py \
  --input data/manifests/plain_vit_3label.csv \
  --expected-label-count 3 \
  --output data/manifests/13feature_21reason_3label_all.jsonl \
  --json-output data/manifests/13feature_21reason_3label_all.json \
  --summary-csv data/manifests/13feature_21reason_3label_all.summary.csv \
  --prompt-output data/manifests/13feature_21reason_3label.prompt.md \
  --azure-endpoint "$ROOT_ENDPOINT" \
  --max-images 1 \
  --summary-every 1 \
  --workers 1 \
  --resume
```

## 5. 전체 실행: tmux 백그라운드 권장

긴 작업이므로 tmux session에서 실행합니다. 이미 생성된 JSONL은 `--resume`으로 건너뜁니다.

```bash
SESSION=vlm3label
LOG=logs/13feature_21reason_3label_tmux_$(date +%Y%m%d_%H%M%S)_workers12.log
mkdir -p logs

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION" || true

tmux new-session -d -s "$SESSION" "cd /workspace/implant_python_only_final && python -u generate_13feature_21reason_7label.py \
  --input data/manifests/plain_vit_3label.csv \
  --expected-label-count 3 \
  --output data/manifests/13feature_21reason_3label_all.jsonl \
  --json-output data/manifests/13feature_21reason_3label_all.json \
  --summary-csv data/manifests/13feature_21reason_3label_all.summary.csv \
  --prompt-output data/manifests/13feature_21reason_3label.prompt.md \
  --azure-endpoint '$ROOT_ENDPOINT' \
  --workers 12 \
  --summary-every 25 \
  --resume 2>&1 | tee -a '$LOG'"

echo "$SESSION" > logs/13feature_21reason_3label.tmux_session
echo "$LOG" > logs/13feature_21reason_3label.latest_log

echo "started session=$SESSION log=$LOG"
```

## 6. 진행 상황 확인

### 6.1 실시간 로그 보기

```bash
tmux capture-pane -t vlm3label -p | tail -80
```

또는 attach:

```bash
tmux attach -t vlm3label
```

attach 후 빠져나오기: `Ctrl-b` 누른 뒤 `d`.

### 6.2 처리된 줄 수만 확인

```bash
wc -l data/manifests/13feature_21reason_3label_all.jsonl
```

### 6.3 상태별/라벨별 집계 확인

```bash
python - <<'PY'
from pathlib import Path
import json, collections, time
p = Path('data/manifests/13feature_21reason_3label_all.jsonl')
status = collections.Counter()
feature_status = collections.Counter()
labels = collections.Counter()
splits = collections.Counter()
failures = collections.Counter()
last_path = ''

if p.exists():
    for line in p.read_text(errors='replace').splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        status[rec.get('status', '')] += 1
        labels[rec.get('label', '')] += 1
        splits[rec.get('split', '')] += 1
        feature = rec.get('feature', {})
        feature_status[feature.get('feature_status', {}).get('value', '')] += 1
        failures[feature.get('failure_reason', '')] += 1
        last_path = rec.get('path', '')

    st = p.stat()
    print('records:', sum(status.values()), '/ 2075')
    print('status:', dict(status))
    print('feature_status:', dict(feature_status))
    print('labels:', dict(labels))
    print('splits:', dict(splits))
    print('failure_reason:', dict(failures))
    print('file_mtime:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)))
    print('last_path:', last_path)
else:
    print('JSONL not found')
PY
```

## 7. 중단 / 재개

### 중단

```bash
tmux kill-session -t vlm3label
```

### 재개

5번의 전체 실행 명령을 다시 실행하면 됩니다. `--resume` 옵션 때문에 이미 `status=ok`로 저장된 이미지는 건너뜁니다.

## 8. 완료 후 최종 산출물

전체 완료 시 아래 파일들이 최종 산출물입니다.

```text
data/manifests/13feature_21reason_3label_all.jsonl
고속 append/재개용 원본 결과. 한 줄에 이미지 1개 JSON.

data/manifests/13feature_21reason_3label_all.summary.csv
분석/집계용 CSV. core attribute, supporting feature state/confidence 포함.

data/manifests/13feature_21reason_3label_all.json
JSONL을 배열로 변환한 최종 JSON. 스크립트가 정상 종료될 때 갱신됨.

data/manifests/13feature_21reason_3label.prompt.md
실제 사용한 VLM prompt.
```

## 9. 흔한 문제

### 9.1 404 Resource not found

대부분 endpoint가 `/openai/v1`까지 포함됐거나 deployment 이름이 Azure Portal의 실제 deployment와 다를 때 발생합니다.

- `--azure-endpoint "$ROOT_ENDPOINT"`처럼 root endpoint만 넘기세요.
- `.env`의 `AZURE_OPENAI_DEPLOYMENT`가 실제 vision-capable chat deployment 이름인지 확인하세요.

### 9.2 `.env` parse warning

현재 `.env`에 shell command 형태의 줄이 섞여 있으면 아래 경고가 나올 수 있습니다.

```text
python-dotenv could not parse statement starting at line ...
```

필수 변수만 정상적으로 읽히면 실행은 계속됩니다.

필수 변수:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT
```

### 9.3 속도가 느림

`--workers 12`는 병렬 API 호출 수입니다. rate limit이 나면 낮추고, 안정적이면 16 정도까지 올려볼 수 있습니다.

```bash
--workers 8
# 또는
--workers 16
```

## 10. 현재 실행 중인 작업 확인용 한 줄 명령

```bash
cd /workspace/implant_python_only_final && \
echo "tmux_alive=$(tmux has-session -t vlm3label 2>/dev/null && echo yes || echo no)" && \
pgrep -af 'generate_13feature_21reason_7label.py' && \
wc -l data/manifests/13feature_21reason_3label_all.jsonl
```
