---
title: "Park 2023 KAOMI Dental Implant Classification Benchmark"
tags: ["paper", "benchmark", "KAOMI", "dental-implant", "classification", "Park2023"]
created: 2026-07-26T02:52:14.335Z
updated: 2026-07-26T02:52:53.342Z
sources: ["10.1038/s41598-023-32118-1"]
links: []
category: reference
confidence: high
schemaVersion: 1
---

# Park 2023 KAOMI Dental Implant Classification Benchmark

Canonical detailed record: `docs/references/kaomi_park_2023_benchmark.md`
Machine-readable tables: `docs/references/data/kaomi_park_2023_system_performance.csv` and `docs/references/data/kaomi_park_2023_dataset_counts.csv`.

Comparison ID: `PARK2023_KAOMI_27CLASS_ROI`

Headline: ROI-cropped 27-way dental implant system classification on 156,965 images from 10 manufacturers. Panoramic + periapical accuracy 88.53%, precision 85.70%, recall 82.30%, F1 84.00%, AUC 0.885. Panoramic-only n=116,756: 87.89/85.20/81.10/83.10%, AUC 0.878. Periapical-only n=40,209: 86.87/84.40/81.70/83.00%, AUC 0.868.

Dataset: raw NIA 165,700 images and 42 DIS types; retained 27 types having at least 100 images in each modality. One implant fixture ROI per image, 512x512 or smaller. Random 80/10/10 image split, 10x training augmentation, Neuro-T 3.0.1, two NVIDIA A6000 48GB GPUs, maximum 500 epochs, early-stop patience 20.

Comparison directive: use as an external clinical-scale literature reference, not a direct baseline for the current uncropped 3-way manufacturer prototype experiment. Always state ROI-cropped vs uncropped, 27-way system vs 3-way manufacturer, random image split vs source-group-clean split, and unspecified final architecture/metric averaging. The paper reports no confidence intervals, repeated seeds, or p-value for its no-difference claim.

Quality notes: Table 1 prose calls per-system precision "classification accuracy." Published Megagen Anyone external periapical recall is 65.909%. Two published periapical F1 entries do not equal the harmonic mean of their printed precision/recall; preserve the published values and cite the detailed record.
