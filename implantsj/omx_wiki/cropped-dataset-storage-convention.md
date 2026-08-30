---
title: "Cropped Dataset Storage Convention"
tags: ["dataset", "yolo", "crop", "classification", "split", "naming"]
created: 2026-07-21T12:00:35.540Z
updated: 2026-07-21T12:00:35.540Z
sources: []
links: []
category: convention
confidence: medium
schemaVersion: 1
---

# Cropped Dataset Storage Convention

All YOLO-to-classification dataset conversions in this project use the following default contract. Pool labeled objects from every available source split, crop one classification image per YOLO box or per polygon bounding rectangle, then split independently within each class using deterministic seed 42 and an 8:1:1 train/valid/test ratio. Save outputs under `<source_name>_cropped/train|valid|test/<class>/`. Normalize unsafe class-folder characters such as spaces to underscores. Name files with one class-wide continuous sequence across train, valid, and test: `<class>_001.jpg`, `<class>_002.jpg`, and so on; never use per-source suffixes such as `5-1` or `5-2`. Cropped outputs are ImageFolder-style classification datasets and do not contain YOLO label files. Keep original crop resolution and aspect ratio; do not upscale unless explicitly requested. Very small classes may have no valid or test item because ratios cannot create fractional samples. Preserve source datasets except when the owner explicitly requests cleanup.
