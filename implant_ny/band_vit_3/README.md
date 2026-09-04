# band_vit_3

`band_vit_3` trains a ViT classifier on boundary-band-only inputs generated at
runtime by the fine-tuned SAM2.1 Hiera Tiny + Adapter + U-Net model.

Pipeline:

1. Load each train/valid/test image.
2. Predict an implant mask with the frozen segmentation model.
3. Keep the largest filled component.
4. Build a boundary band using dilation minus erosion.
5. Black out every pixel outside the band.
6. Tight-crop, square-pad and resize the band image to 224x224.
7. Cache only the band image in RAM for the current run.
8. Release SAM2 from GPU and fine-tune the ViT with cross-entropy.

No predicted mask PNG files are written. A CSV report containing mask/band area
ratios and crop boxes is saved for auditability.

Example:

```bash
cd /workspace/implant_seg
source .venv-vit-sam2/bin/activate

python -m band_vit_3.training \
  --model vit_base_patch16_224 \
  --train-data /workspace/data/11label_crop_final_8_1_1/train.csv \
  --valid-data /workspace/data/11label_crop_final_8_1_1/valid.csv \
  --test-data /workspace/data/11label_crop_final_8_1_1/test.csv \
  --output-dir /workspace/implant_seg/implant_outputs/band_vit3_11label \
  --epochs 70 \
  --batch-size 8 \
  --num-workers 0 \
  --device cuda
```
