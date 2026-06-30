# Current Context Handoff

- Saved at UTC: 2026-05-26T12:58:26Z
- Project dir: `/workspace/implant_python_only_final`
- Purpose: Preserve current session context, running jobs, outputs, commands, and conclusions.

## 1. Active / Recent User Requests

1. Analyze all files in `implant_python_only_final` and save as Markdown.
2. Apply latest VLM analysis to 3-label data.
3. Save terminal commands so the user can run it manually next time.
4. Run 3-label fine-tuning while VLM analysis is running.
5. Analyze completed 3-label fine-tuning result.
6. Save current context.

## 2. Important Saved Documents

- Full project analysis: `PROJECT_ANALYSIS.md`
- 3-label VLM run commands: `VLM_3LABEL_RUN_COMMANDS.md`
- 3-label VLM status snapshot: `VLM_3LABEL_APPLY_STATUS.md`
- Fine-tune result analysis: `outputs/20260526_123524_3label_ft_from_10label_vit_base/RESULT_ANALYSIS.md`
- This handoff: `CURRENT_CONTEXT_HANDOFF.md`

## 3. Latest VLM 3-label Application

### Schema

- Latest schema applied: **13 core attributes + 21 supporting visual features**
- Source script reused: `generate_13feature_21reason_7label.py`
- Target manifest: `data/manifests/plain_vit_3label.csv`
- Total images: 2,075

### Output files

- Streaming JSONL: `data/manifests/13feature_21reason_3label_all.jsonl`
- Summary CSV: `data/manifests/13feature_21reason_3label_all.summary.csv`
- Final JSON array: `data/manifests/13feature_21reason_3label_all.json`
- Prompt: `data/manifests/13feature_21reason_3label.prompt.md`
- tmux session marker: `logs/13feature_21reason_3label.tmux_session`
- Latest log marker: `logs/13feature_21reason_3label.latest_log`
- Latest log: `logs/13feature_21reason_3label_tmux_20260526_115230_workers12.log`

### Current execution state at save time

- tmux session: `vlm3label`
- tmux alive: `yes`
- matching process sample:

```text
40548 bash -c cd /workspace/implant_python_only_final && python -u generate_13feature_21reason_7label.py --input data/manifests/plain_vit_3label.csv --expected-label-count 3 --output data/manifests/13feature_21reason_3label_all.jsonl --json-output data/manifests/13feature_21reason_3label_all.json --summary-csv data/manifests/13feature_21reason_3label_all.summary.csv --prompt-output data/manifests/13feature_21reason_3label.prompt.md --azure-endpoint 'https://ktx06-mowm0s0q-eastus2.cognitiveservices.azure.com' --workers 12 --summary-every 25 --resume 2>&1 | tee -a 'logs/13feature_21reason_3label_tmux_20260526_115230_workers12.log'
40550 python -u generate_13feature_21reason_7label.py --input data/manifests/plain_vit_3label.csv --expected-label-count 3 --output data/manifests/13feature_21reason_3label_all.jsonl --json-output data/manifests/13feature_21reason_3label_all.json --summary-csv data/manifests/13feature_21reason_3label_all.summary.csv --prompt-output data/manifests/13feature_21reason_3label.prompt.md --azure-endpoint https://ktx06-mowm0s0q-eastus2.cognitiveservices.azure.com --workers 12 --summary-every 25 --resume
```

### Current VLM progress snapshot

```text
VLM_JSONL_RECORDS=1277
VLM_STATUS={"ok": 1276, "error": 1}
VLM_FEATURE_STATUS={"low_confidence": 211, "partial_usable": 900, "invalid_roi": 76, "usable": 90}
VLM_LABELS={"Bicon": 959, "ITI": 318}
VLM_SPLITS={"train": 1277}
VLM_FAILURES={"all_features_not_assessable": 210, "connection_obscured": 896, "apex_not_visible": 4, "implant_not_visible": 76, "none": 90, "low_confidence": 1}
VLM_JSONL_MTIME=2026-05-26 12:58:26
VLM_LAST_PATH=/workspace/3label/train/ITI/ITI-298_jpg.rf.0e3104bcb476e73ccdc69c2c6f20c96f.jpg
```

### Check VLM progress

```bash
cd /workspace/implant_python_only_final
tmux capture-pane -t vlm3label -p | tail -80
wc -l data/manifests/13feature_21reason_3label_all.jsonl
```

### Stop / resume VLM

Stop:

```bash
cd /workspace/implant_python_only_final
tmux kill-session -t vlm3label
```

Resume: use commands in `VLM_3LABEL_RUN_COMMANDS.md`. The generator uses `--resume`, so already saved `status=ok` lines are skipped.

## 4. 3-label Fine-tuning Run

### Run details

- Run dir: `outputs/20260526_123524_3label_ft_from_10label_vit_base`
- Log: `logs/gpu_3label_ft_from_10label_20260526_123521.log`
- Initial checkpoint: `outputs/20260507_134640_plain_vit_10label/checkpoints/best.pt`
- Manifest: `data/manifests/plain_vit_3label.csv`
- Model: `vit_base_patch16_224`
- Command used: `train_supervised.py`
- Key options:
  - `--pretrained false`
  - `--init-checkpoint outputs/20260507_134640_plain_vit_10label/checkpoints/best.pt`
  - `--epochs 60`
  - `--batch-size 16`
  - `--lr 1e-5`
  - `--label-smoothing 0.05`
  - `--use-class-weights`
  - `--patience 12`

### Execution state

- Fine-tuning completed.
- tmux session `gpu_3label_ft10` is gone / done.
- matching process sample:

```text

```

### Metrics snapshot

```text
FT_VALID_METRICS={"accuracy": 0.9353233830845771, "macro_precision": 0.9583918813427011, "macro_recall": 0.9480143045360437, "macro_f1": 0.9522374350584116, "weighted_f1": 0.9347386035464629}
FT_TEST_METRICS={"accuracy": 0.9174311926605505, "macro_precision": 0.7298117789921067, "macro_recall": 0.7087680079483358, "macro_f1": 0.7172679349875887, "weighted_f1": 0.9146185223909169}
FT_EPOCHS=38
FT_BEST_EPOCH={"epoch": "26", "train_loss": "0.31865298043060575", "valid_accuracy": "0.9353233830845771", "valid_macro_f1": "0.9522374350584116", "valid_weighted_f1": "0.9347386035464629", "valid_macro_precision": "0.9583918813427011", "valid_macro_recall": "0.9480143045360437", "lr": "6.039558454088796e-06", "seconds": "10.379119873046875"}
BASE_VALID_METRICS={"accuracy": 0.9353233830845771, "macro_precision": 0.9196573751451801, "macro_recall": 0.879737331911245, "macro_f1": 0.8976924102974523, "weighted_f1": 0.934180637429101}
BASE_TEST_METRICS={"accuracy": 0.944954128440367, "macro_precision": 0.9597222222222223, "macro_recall": 0.7314952806756084, "macro_f1": 0.7699964067553, "weighted_f1": 0.9377801659485672}
BASE_EPOCHS=41
BASE_BEST_EPOCH={"epoch": "29", "train_loss": "0.5431575411102252", "valid_accuracy": "0.9353233830845771", "valid_macro_f1": "0.8976924102974523", "valid_weighted_f1": "0.934180637429101", "valid_macro_precision": "0.9196573751451801", "valid_macro_recall": "0.879737331911245", "lr": "3.5466493438435696e-05", "seconds": "16.42832350730896"}
```

### Key conclusion

The fine-tuned model **should not replace** the existing 3-label baseline on the current test split.

Comparison:

| Model | Valid Acc | Valid Macro-F1 | Test Acc | Test Macro-F1 | Test Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Existing 3-label baseline `20260507_104805_plain_vit_3label` | 0.9353 | 0.8977 | **0.9450** | **0.7700** | **0.9378** |
| 10-label → 3-label fine-tune `20260526_123524_...` | 0.9353 | **0.9522** | 0.9174 | 0.7173 | 0.9146 |

Interpretation:

- Validation macro-F1 improved significantly.
- Test performance worsened.
- Existing baseline errors: 6.
- Fine-tune errors: 9.
- Fine-tune fixed 0 existing errors and introduced 3 new ITI-related errors.
- Main issue: Bego/ITI boundary instability; test Bego support is only 4.

Preferred current model:

```text
outputs/20260507_104805_plain_vit_3label/checkpoints/best.pt
```

Fine-tune checkpoint kept for reference:

```text
outputs/20260526_123524_3label_ft_from_10label_vit_base/checkpoints/best.pt
```

## 5. Environment / Dependencies Changed

Installed into current Python during this session:

```text
openai
python-dotenv
timm
pandas
scikit-learn
matplotlib
```

Torch stack already existed:

```text
torch 2.4.1+cu124
torchvision 0.19.1+cu124
CUDA available: NVIDIA RTX 4000 Ada Generation
```

## 6. Azure Endpoint Note

The `.env` endpoint includes `/openai/v1`, which caused 404 with `AzureOpenAI(azure_endpoint=...)`. The working pattern is to pass only the root endpoint with:

```bash
ROOT_ENDPOINT=https://ktx06-mowm0s0q-eastus2.cognitiveservices.azure.com
```

Use `--azure-endpoint ""`.

## 7. Suggested Next Steps

1. Let VLM 3-label generation finish.
2. After VLM completion, validate:
   - JSONL lines = 2,075
   - summary CSV rows = 2,075
   - final JSON array length = 2,075
3. Do not deploy the 10-label→3-label fine-tuned checkpoint as baseline.
4. If more GPU experiments are needed:
   - rerun fine-tune with `--no-strong-aug`
   - rerun with `--no-class-weights`
   - try LR `5e-6`
   - try head-only warmup then full fine-tune
   - use k-fold/repeated split due tiny Bego test support.
5. Once VLM 3-label completes, consider feature quality analysis by class/split before any VLM-fusion training.

## 8. One-line Health Check

```bash
cd /workspace/implant_python_only_final && echo "VLM tmux_alive=yes" && wc -l data/manifests/13feature_21reason_3label_all.jsonl && ls -lh outputs/20260526_123524_3label_ft_from_10label_vit_base/RESULT_ANALYSIS.md PROJECT_ANALYSIS.md VLM_3LABEL_RUN_COMMANDS.md CURRENT_CONTEXT_HANDOFF.md
```
