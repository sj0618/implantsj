# ImplantsJ 독립 실험 baseline

이 폴더는 상위 저장소의 `src/`, `pipelines/`, `tests/`를 import하거나 수정하지 않는
독립 프로젝트다. 매니페스트 생성, 학습, 임베딩 추출, validation/test 평가,
t-SNE, 결과 집계를 각각 별도 명령으로 실행한다.

## 1. 준비

```bash
cd implantsj
python3 -m pip install -r requirements.txt
```

폐쇄망에서는 필요한 wheel과 초기 checkpoint를 먼저 반입하고
`--no-pretrained --checkpoint /local/path/model.pt`를 사용한다. 로컬 checkpoint가
지정되면 `timm` pretrained 다운로드를 시도하지 않는다.

실제 42-system GT 기본 경로는 `/workspace/data/GT_data_42label`이다. `DATA_DIR`을
`/workspace/data`로 지정하면 wrapper가 이 경로를 자동으로 사용한다.

## 2. 매니페스트 생성

원본 CSV를 사용하는 방법이다. CSV 필수 열은 `path,label,split`이고 `split`은
`train`, `valid`, `val`, `validation`, `test` 중 하나다. CSV 안 상대 경로는 CSV
파일 위치를 기준으로 해석한다.

```bash
python3 -m implantsj.manifests \
  --source-manifest /data/source.csv \
  --gt-manifest /data/gt.csv \
  --closed-labels configs/closed_labels.txt \
  --output-dir data/manifests \
  --valid-fraction 0.2 \
  --seed 2026
```

ImageFolder를 직접 스캔할 수도 있다.

```bash
python3 -m implantsj.manifests \
  --train-root /data/original_train \
  --test-root /data/original_test \
  --gt-root /workspace/data/GT_data_42label \
  --closed-labels configs/closed_labels.txt \
  --output-dir data/manifests
```

생성물은 `train.csv`, `valid.csv`, `test.csv`, `support.csv`, `full.csv`,
`gt.csv`, `labels.json`, `audit.json`이다. 원본 test는 재분할하지
않으며, validation은 원본 train pool에서만 만든다. 라벨 별칭은 fuzzy matching
없이 `--aliases configs/label_aliases.example.csv`의 정확 일치만 적용한다.

GT는 42개 라벨을 모두 포함하고 라벨마다 독립 이미지가 2장 이상이어야 한다.
train/valid/test와 같은 이미지 경로가 하나라도 있으면 생성 단계에서 실패한다.

소형 개발 데이터는 사용자 정의 closed-label 파일을 전달하면 된다. 정식 42-label
계약 자체를 완화해야 할 때만 `--allow-incomplete`를 사용한다.

## 3. 초기 weight 저장

온라인 환경에서 한 번 저장한 뒤 모든 실험이 같은 파일을 사용하게 할 수 있다.

```bash
python3 -m implantsj.models save-initial \
  --architecture vit_base_patch16_224 \
  --output outputs/_shared/initial/vit_base_patch16_224.pt
```

## 4. 학습 실행

CE 예시:

```bash
python3 -m implantsj.training \
  --experiment-name my-ce-run \
  --model vit_base_patch16_224 \
  --no-pretrained \
  --checkpoint outputs/_shared/initial/vit_base_patch16_224.pt \
  --train-data data/manifests/train.csv \
  --valid-data data/manifests/valid.csv \
  --test-data data/manifests/test.csv \
  --scope all \
  --loss ce \
  --epochs 15 \
  --batch-size 32 \
  --output-dir outputs/my-ce-run
```

SupCon은 필요한 인자만 바꾼다.

```bash
python3 -m implantsj.training \
  --experiment-name my-supcon-run \
  --model resnet50 \
  --no-pretrained \
  --checkpoint /checkpoints/resnet50.pt \
  --train-data /manifests/train_v2.csv \
  --valid-data /manifests/valid_v2.csv \
  --test-data /manifests/test_v2.csv \
  --scope closed \
  --loss supcon \
  --samples-per-class 4 \
  --batch-size 32 \
  --temperature 0.07 \
  --output-dir outputs/my-supcon-run
```

`--model`은 `timm` architecture 이름, `--loss`는 `ce`, `supcon`, `triplet`,
`--optimizer`는 `adamw`, `sgd`, `--scheduler`는 `cosine`, `step`, `none`을
받는다. 새 loss는 `implantsj/losses.py`의 `LOSS_STRATEGIES`에 한 클래스만
등록하면 CLI 선택지에 자동으로 나타난다. 새 checkpoint 형식은
`implantsj/models.py`의 adapter registry에 등록한다.

학습 출력:

- `checkpoints/last.pt`: 마지막 epoch, 평가용 기본 checkpoint
- `checkpoints/best.pt`: validation으로만 선택한 best checkpoint
- `history.csv`: train/valid/test curve 값
- `curves.png`, `run_config.json`, `model_spec.json`

test loss/accuracy는 기록만 한다. checkpoint 선택이나 hyperparameter 분기에는
사용하지 않는다.

## 5. 임베딩 추출

학습이 만든 model spec과 원하는 매니페스트를 명시한다. 따라서 support/query를
서로 독립적으로 교체할 수 있다.

```bash
python3 -m implantsj.evaluation embed \
  --model-spec outputs/my-ce-run/model_spec.json \
  --manifest /manifests/support_v3.csv \
  --output outputs/my-ce-run/embeddings/support.npz \
  --device auto \
  --overwrite

python3 -m implantsj.evaluation embed \
  --model-spec outputs/my-ce-run/model_spec.json \
  --checkpoint outputs/my-ce-run/checkpoints/best.pt \
  --manifest /manifests/test_v4.csv \
  --output outputs/my-ce-run/embeddings/test.npz \
  --overwrite
```

`--checkpoint`를 주면 model spec 안의 checkpoint만 해당 실행에서 교체한다.

## 6. validation 또는 test 평가

```bash
python3 -m implantsj.evaluation evaluate \
  --model-name my-ce-run \
  --support-embeddings outputs/my-ce-run/embeddings/support.npz \
  --gt-embeddings outputs/my-ce-run/embeddings/gt.npz \
  --query-embeddings outputs/my-ce-run/embeddings/test.npz \
  --labels-json data/manifests/labels.json \
  --query-name test \
  --prototypes gt,mean,ss \
  --distances euclidean,cosine \
  --output-dir outputs/my-ce-run/eval/test
```

validation은 `--query-embeddings`와 `--query-name valid`, 출력 폴더만 바꾼다.
평가는 checkpoint 안의 과거 manifest를 자동 사용하지 않는다. support, GT,
query cache를 항상 명시하므로 서로 다른 매니페스트 실험이 섞이지 않는다.

실제 42-system GT만 사용한 test는 폐쇄망에서 다음처럼 단계별로 실행한다.

```bash
DATA_DIR=/workspace/data ./run.sh manifests
./run.sh embed ViT-ClosedFT gt
./run.sh embed ViT-ClosedFT test
PROTOTYPES=gt ./run.sh test ViT-ClosedFT
```

GT 폴더가 없거나 42-label coverage가 맞지 않으면 실행은 실패하며, 다른 이미지로
자동 대체하지 않는다.

## 7. t-SNE 생성

```bash
python3 -m implantsj.visualization tsne \
  --model-name my-ce-run \
  --embedding train=outputs/my-ce-run/embeddings/support.npz \
  --embedding valid=outputs/my-ce-run/embeddings/valid.npz \
  --embedding test=outputs/my-ce-run/embeddings/test.npz \
  --labels-json data/manifests/labels.json \
  --output-dir outputs/my-ce-run/tsne \
  --max-points 5000 \
  --seed 2026
```

`tsne.png`, `tsne_coordinates.csv`, `sampled_paths.txt`, `tsne_metadata.json`이
생긴다. 샘플은 정렬된 path와 seed로 결정하므로 같은 manifest를 쓰는 모델끼리는
동일한 subset을 사용한다.

## 8. 5-model 결과 집계

각 모델의 `eval/valid/metrics.csv`, `eval/test/metrics.csv`가 준비된 뒤 실행한다.

```bash
python3 -m implantsj.reporting \
  --eval-root outputs \
  --models 'ViT-Base,ViT-ClosedFT,ViT-ClosedFT-SupCon,ViT-FullFT,ViT-FullFT-SupCon' \
  --roles 'base=ViT-Base,closed_ft=ViT-ClosedFT,closed_ft_supcon=ViT-ClosedFT-SupCon,full_ft=ViT-FullFT,full_ft_supcon=ViT-FullFT-SupCon' \
  --output-dir outputs/report \
  --expected-rows 180
```

운영점 선택은 validation만 사용하고, 선택된 조건의 test 결과를 보고서에 남긴다.

## 9. 짧은 wrapper 명령

`config.sh` 값은 모두 외부 환경변수로 덮어쓸 수 있다. 각 단계는 독립 실행된다.
예를 들어 `CANONICAL_EPOCH_CKPT=best.pt`를 주면 wrapper의 임베딩 단계만
validation-best checkpoint로 바뀐다.

```bash
./run.sh manifests
./run.sh save_initial
./run.sh train ViT-ClosedFT
./run.sh embed ViT-ClosedFT support
./run.sh embed ViT-ClosedFT test
./run.sh eval ViT-ClosedFT valid
./run.sh test ViT-ClosedFT
./run.sh tsne ViT-ClosedFT
./run.sh report
```

명령 확인만 할 때:

```bash
DRY_RUN=1 ./run.sh all
```

`all`은 위 단계 함수를 순서대로 호출하는 선택 기능일 뿐이다. 개별 명령 실행이
기본 사용법이다.

## 10. 빠른 검증

```bash
python3 -m pytest tests/test_pipeline.py -q
python3 -m pytest tests/test_smoke.py -q
bash -n run.sh config.sh
```

## 11. GT 27-label checkpoint 비교 결과

2026-07-22에 `/workspace/data/GT_data_42label/AI Hub Data`를 검사했다. 42개
라벨 폴더 중 이미지가 없는 15개는 제외하고, 이미지가 있는 27개 라벨의 241장을
seed 2026으로 support 189장과 query 52장으로 고정 분할했다. 각 checkpoint의
backbone embedding에서 라벨별 support 평균을 GT prototype으로 만들고, 동일한
query를 Euclidean/Cosine 최근접 prototype으로 분류했다. 아래 값은 모두 27-way
accuracy와 macro-F1이며 단위는 `%`다.

| 모델/checkpoint | Euclidean Acc. | Euclidean Macro-F1 | Cosine Acc. | Cosine Macro-F1 |
|---|---:|---:|---:|---:|
| ViT-Base | 48.08 | 51.28 | 51.92 | 52.80 |
| ViT-ClosedFT | 44.23 | 42.21 | 44.23 | 42.46 |
| ViT-ClosedFT-SupCon | 44.23 | 47.92 | 40.38 | 43.08 |
| ViT-FullFT | 51.92 | **52.44** | 50.00 | 50.59 |
| ViT-FullFT-SupCon | 42.31 | 42.79 | 42.31 | 41.55 |
| `resnet50_best.pt` | 51.92 | 50.12 | 55.77 | **53.82** |
| `supcon_closedlabel.pt` | 51.92 | 45.13 | 50.00 | 42.91 |
| `vit42class_best.pt` | 42.31 | 41.80 | 40.38 | 39.47 |
| `vit_12block_band_best.pt` | 26.92 | 25.87 | 30.77 | 29.94 |
| `vit_a=4_best.pt` | 23.08 | 21.61 | 21.15 | 19.81 |
| `vit_band_6b_lam0.8.pt` | **59.62** | 51.37 | **61.54** | 49.63 |
| `vit_band_a8_lam8.pt` | 23.08 | 21.61 | 21.15 | 19.81 |

전체 24개 결과 행은
`/workspace/metric_diagnosis/work_gt27_metric_all/results_summary.csv`에 저장했다.
각 모델별 confusion matrix, prediction, per-class metric은 같은 경로의
`work_gt27_metric_*` 결과 폴더에 있다.

`vit_a=4_best.pt`와 `vit_band_a8_lam8.pt`는 SHA-256이
`ebe2eeb19a7136638cd0f23e63fc3c85af31aea14a052ad65800906e5c8c0818`로 완전히
같다. checkpoint 내부 설정도 `band_alpha=8.0`, `band_loss_weight=0.8`이므로
현재 파일 기준으로는 독립적인 alpha 4 실험이 아니며 결과도 동일하다.
`supcon_closedlabel.pt`의 `class_to_idx`는 26개지만, 이 표는 classifier head가
아니라 backbone과 27개 GT prototype을 사용한 결과다.
