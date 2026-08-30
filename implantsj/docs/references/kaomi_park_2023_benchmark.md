# Park et al. (2023) KAOMI dental implant classification benchmark

## Source and verification

- Paper: Wonse Park, Jong-Ki Huh, and Jae-Hong Lee, "Automated deep learning for classification of dental implant radiographs using a large multi-center dataset"
- Journal: *Scientific Reports* 13, 4862 (2023)
- DOI: `10.1038/s41598-023-32118-1`
- Local source: `/Users/yunseojune/TRIA_LAB/implant_project/KAOMI 박원서 교수님 논문.pdf`
- Comparison ID: `PARK2023_KAOMI_27CLASS_ROI`
- Verification: all 8 PDF pages rendered and visually checked on 2026-07-26. Tables 1 and 2 were transcribed from the rendered pages and checked against extracted PDF text.
- Scope: experimental dataset, protocol, performance, and paper-stated comparison numbers. Bibliographic page ranges, IRB identifiers, and equation numbering are not treated as benchmark values.

## Benchmark headline

This is a 27-way dental implant system classification benchmark using a large, multicenter, single-fixture ROI-cropped radiograph dataset. It is not an uncropped image benchmark, a manufacturer-level 3-class benchmark, or a prototype/metric-learning benchmark.

| Image group | Images | Accuracy | Precision | Recall | F1 | AUC | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Panoramic | 116,756 | 87.89% | 85.20% | 81.10% | 83.10% | 0.878 | pp. 1-3 |
| Periapical | 40,209 | 86.87% | 84.40% | 81.70% | 83.00% | 0.868 | pp. 1-3 |
| Panoramic + periapical | 156,965 | **88.53%** | **85.70%** | **82.30%** | **84.00%** | **0.885** | pp. 1-3 |

The paper states that accuracy did not differ significantly among the three image groups, but it does not report the statistical test, p-value, effect-size interval, or confidence interval.

Derived, not directly reported:

- Panoramic share: 74.38%; periapical share: 25.62%.
- Combined accuracy is +0.64 percentage points over panoramic-only and +1.66 points over periapical-only.
- Panoramic-only accuracy is +1.02 points over periapical-only.

## Dataset construction

| Stage | Images | DIS types | Manufacturers | Rule/source |
| --- | ---: | ---: | ---: | --- |
| Raw NIA dataset | 165,700 | 42 | not stated | pp. 4-5 |
| Paper analysis dataset | 156,965 | 27 | 10 | retained systems with at least 100 panoramic and 100 periapical images |

- Collection sites: 5 college dental hospitals and 10 private dental clinics.
- Source images: panoramic and periapical radiographs after implant surgery.
- Preprocessing: DICOM images were de-identified, converted to JPEG/PNG at 512 x 512 pixels or smaller, and cropped to one implant fixture ROI per image.
- Quality and labeling: cropping, resolution, and sharpness reviewed by 10 dental professionals; final images validated by one board-certified oral and maxillofacial radiologist.
- Labels included manufacturer, brand/system, diameter, length, placement position, surgery date, age, and sex. The reported classification target was the 27-class implant system.

### Manufacturer distribution

| Manufacturer | Total images | Share |
| --- | ---: | ---: |
| Osstem | 42,920 | 27.34% |
| Dentium | 41,096 | 26.18% |
| Neobiotech | 21,260 | 13.54% |
| Dentsply | 15,296 | 9.74% |
| Warantec | 15,065 | 9.60% |
| Megagen | 7,801 | 4.97% |
| Straumann | 4,977 | 3.17% |
| Nobel biocare | 3,644 | 2.32% |
| Shinhung | 3,376 | 2.15% |
| Dioimplant | 1,530 | 0.97% |

The full 27-system distribution is in `data/kaomi_park_2023_dataset_counts.csv`.

### Class imbalance

- Largest system: Osstem TS III, 29,758 images (18.96%).
- Smallest system: Megagen Any ridge, 352 images (0.22%).
- Derived largest-to-smallest ratio: 84.54x.
- Warantec IT contained 238 panoramic and 208 periapical images, 446 total.

## Split, augmentation, and training

- Each of the panoramic, periapical, and combined groups was randomly divided 80% training / 10% validation / 10% testing.
- The paper reports 105,080 panoramic and 36,188 periapical images in training + validation.
- Derived held-out counts from those reported totals: 11,676 panoramic and 4,021 periapical, 15,697 total.
- Training data were augmented 10x.
- Augmentations:
  - rotation: 90 degrees
  - hue: -0.1 to 0.1
  - brightness: -0.12 to 0.12
  - saturation: 0.6 to 1.5
  - contrast: 0.6 to 1.5
  - noise: 0.05
  - horizontal and vertical flips
- Automated engine: Neuro-T 3.0.1.
- Compute: 2 NVIDIA A6000 GPUs, 48 GB each.
- Maximum training: 500 epochs.
- Early stopping: validation loss not improving for more than 20 epochs.
- Statistical software: Neuro-T 3.0.1 and R 4.2.0.
- The automated engine selected the resize method, architecture/depth, decay, learning rate, dropout, batch size, epochs, patience, and optimizer. The final selected architecture and final hyperparameter values are not reported in the main paper.

## System-level performance

The complete 27-system x 3 image-group precision/recall/F1 table is in `data/kaomi_park_2023_system_performance.csv`.

### Combined panoramic + periapical highlights

- Highest precision: Nobel biocare Branemark, Dioimplant UF, and Megagen Exfeel external at 100.0%.
- Highest recall: Dentsply Astra and Xive at 100.0%.
- Highest F1: Dentsply Xive at 99.20%, Dioimplant UF at 98.10%, and Osstem TS III at 97.60%.
- Lowest precision: Warantec IT at 35.30%.
- Lowest recall and F1: Neobiotech IS II at 20.00% recall and 30.80% F1.
- Warantec IT: 35.30% precision, 28.60% recall, 31.60% F1.

### Modality-specific extrema reported in the paper

- Panoramic: Osstem US III and Megagen Exfeel external were described as highest at 100.0%; Warantec IT was lowest at 19.0%.
- Periapical: Megagen Exfeel external and Dentsply Xive were described as highest at 100.0%; Warantec IT was lowest at 19.2%.
- Combined: Nobel biocare Branemark and Megagen Exfeel external were described as highest at 100.0%; Warantec IT was lowest at 35.3%.

These prose values correspond to the per-system precision column in Table 1, although the paper calls them "classification accuracy." Do not treat them as a separate per-class accuracy metric.

## Prior-study numbers quoted by this paper

- Prior implant-classification studies: generally 70-100% accuracy, usually fewer than a few thousand images and fewer than 10 DIS types.
- ResNet study: 9,767 panoramic images, 12 types, at least 98% accuracy.
- Authors' prior pilot: 11,980 images, 6 types, 95.4% accuracy, 95.5% sensitivity, 85.3% specificity.
- YOLOv3 study: 1,282 panoramic images, accuracy in the 70% range.
- Paper claim for varying angles/confounders: prior studies including this one reported over 80% accuracy.

These are secondary quotations, not independently verified values from the cited source papers.

## Comparison rules for this project

Use the headline combined benchmark as a literature reference only when clearly labeled:

`Park2023: ROI-cropped, 27-way DIS, n=156,965, accuracy 88.53%, precision 85.70%, recall 82.30%, F1 84.00%.`

Do not directly rank a project result against this paper without noting all of the following:

1. The paper uses one-fixture ROI crops; the current `FULL_Bego_Bicon_ITI_3label` experiment uses uncropped images.
2. The paper is 27-way implant-system classification; the current target experiment is 3-way manufacturer classification.
3. The paper uses a random 80/10/10 image split; the current experiment uses source-group-clean splits.
4. The paper's final architecture is unspecified; current experiments use explicitly named ViT representations.
5. The paper does not state whether headline precision/recall/F1 are macro, micro, or weighted averages.
6. No confidence intervals, repeated seeds, or p-value for the modality comparison are reported.

Accordingly, the paper is an external clinical-scale reference, not an apples-to-apples baseline for current prototype results.

## Data quality notes

- Table 1 prints Megagen Anyone external periapical recall as `65.909%`, unlike the surrounding two-decimal formatting. This transcription preserves the published value.
- Two periapical F1 entries are inconsistent with the harmonic mean of the printed precision and recall: Megagen Anyone internal reports 72.00% / 71.90% / 72.40% (harmonic mean 71.95%), and Warantec Internal reports 52.50% / 71.90% / 61.30% (harmonic mean 60.69%). The CSV preserves the published F1 values.
- The paper alternates between "classification accuracy" and per-system precision when discussing Table 1 extrema.
- Figure 3 is called a "multi-label classification confusion matrix," while the described task and 27 mutually exclusive systems are presented as multiclass classification.
- The paper does not describe patient-level, clinic-level, or source-image group isolation across splits.
- Selection of only systems with at least 100 images per modality improves minimum class size but changes the benchmark from the raw 42-system dataset to a selected 27-system subset.
