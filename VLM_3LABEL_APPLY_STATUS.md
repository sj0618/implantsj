# 3-label 최신 VLM 분석 적용 상태

- 갱신 시각: 2026-05-26T11:56:03Z
- 적용 schema: 13 core attributes + 21 supporting visual features
- 대상 manifest: `data/manifests/plain_vit_3label.csv`
- 총 대상: 2,075 images
- 실행 상태: tmux session `vlm3label` alive=`yes`
- 로그: `logs/13feature_21reason_3label_tmux_20260526_115230_workers12.log`

## 출력 파일

- JSONL streaming output: `data/manifests/13feature_21reason_3label_all.jsonl`
- Summary CSV: `data/manifests/13feature_21reason_3label_all.summary.csv` (25건마다 갱신)
- Final JSON array: `data/manifests/13feature_21reason_3label_all.json` (전체 완료 시 최종 갱신)
- Prompt: `data/manifests/13feature_21reason_3label.prompt.md`

## 현재 진행률

- records: 62 / 2075
- status: `{'ok': 62}`
- feature_status: `{'low_confidence': 14, 'partial_usable': 48}`
- failure_reason: `{'all_features_not_assessable': 14, 'connection_obscured': 47, 'apex_not_visible': 1}`
- labels processed: `{'Bicon': 62}`
- splits processed: `{'train': 62}`

## 모니터링 명령

```bash
cd /workspace/implant_python_only_final
tmux capture-pane -t vlm3label -p | tail -80
python - <<'PY2'
from pathlib import Path
import json, collections
p=Path('data/manifests/13feature_21reason_3label_all.jsonl')
c=collections.Counter()
for line in p.read_text(errors='replace').splitlines():
    if line.strip(): c[json.loads(line).get('status','')]+=1
print(c, 'total=', sum(c.values()), '/ 2075')
PY2
```
