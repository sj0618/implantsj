#!/usr/bin/env python3
"""
VLM 5-feature fusion experiment for implant system classification.

This script intentionally keeps the fusion simple:
  z_img = frozen image encoder ROI embedding
  z_vlm = LinearProjection(one-hot VLM feature vector)
  z_fused = L2Normalize(z_img) + L2Normalize(z_vlm)
  prediction = Classifier(z_fused)

Expected manifest CSV:
  path,label,split

Expected VLM feature file:
  JSONL: one object per image. Each object should contain path/image_path/roi_path/id
         plus the structured VLM JSON fields.
  CSV:   either a vlm_json column containing the structured JSON, or flattened columns
         such as connection_type.value, platform.platform_switching, etc.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import build_transforms, class_weights_from_manifest, label_mapping, load_manifest
from src.models import load_matching_weights, save_checkpoint
from src.utils import count_parameters, log, make_run_dir, seed_everything, write_json


VLM_PROMPT = """You are analyzing a dental implant ROI radiograph.
Do not identify or guess the implant brand or system name.
Return only valid JSON. Extract only the requested structural features.

Allowed JSON schema:
{
  "connection_type": {
    "value": "external / internal / tissue_level_internal / uncertain",
    "confidence": 0.0
  },
  "platform": {
    "platform_visible": true,
    "platform_position_relative_y": 0.0,
    "platform_switching": "switching / matching / uncertain",
    "confidence": 0.0
  },
  "thread": {
    "pitch": "fine / medium / coarse / uncertain",
    "depth": "shallow / medium / deep / uncertain",
    "micro_thread_zone": "present / absent / uncertain",
    "macro_thread_zone": "present / absent / uncertain",
    "confidence": 0.0
  },
  "outline": {
    "body_shape": "straight / tapered / reverse_tapered / mixed / uncertain",
    "diameter_length_ratio": "low / medium / high / uncertain",
    "confidence": 0.0
  },
  "image_quality": {
    "thread_visibility": "clear / partial / blurred",
    "abutment_present": true,
    "projection_distortion": "none / mild / severe",
    "confidence": 0.0
  }
}

Rules:
- Use uncertain when the feature is not visually reliable.
- Use confidence values from 0.0 to 1.0.
- Temperature must be 0 when calling the VLM.
- Output JSON only, with no explanatory text.
"""


FEATURE_SCHEMAS: Dict[str, List[Dict[str, Any]]] = {
    "connection": [
        {
            "json_path": ("connection_type", "value"),
            "name": "connection_type",
            "type": "cat",
            "choices": ["external", "internal", "tissue_level_internal", "uncertain"],
        },
    ],
    "platform": [
        {
            "json_path": ("platform", "platform_visible"),
            "name": "platform_visible",
            "type": "boolcat",
            "choices": ["true", "false", "uncertain"],
        },
        {
            "json_path": ("platform", "platform_switching"),
            "name": "platform_switching",
            "type": "cat",
            "choices": ["switching", "matching", "uncertain"],
        },
        {
            "json_path": ("platform", "platform_position_relative_y"),
            "name": "platform_position_relative_y",
            "type": "numeric",
        },
    ],
    "thread": [
        {
            "json_path": ("thread", "pitch"),
            "name": "thread_pitch",
            "type": "cat",
            "choices": ["fine", "medium", "coarse", "uncertain"],
        },
        {
            "json_path": ("thread", "depth"),
            "name": "thread_depth",
            "type": "cat",
            "choices": ["shallow", "medium", "deep", "uncertain"],
        },
        {
            "json_path": ("thread", "micro_thread_zone"),
            "name": "micro_thread_zone",
            "type": "cat",
            "choices": ["present", "absent", "uncertain"],
        },
        {
            "json_path": ("thread", "macro_thread_zone"),
            "name": "macro_thread_zone",
            "type": "cat",
            "choices": ["present", "absent", "uncertain"],
        },
    ],
    "outline": [
        {
            "json_path": ("outline", "body_shape"),
            "name": "body_shape",
            "type": "cat",
            "choices": ["straight", "tapered", "reverse_tapered", "mixed", "uncertain"],
        },
        {
            "json_path": ("outline", "diameter_length_ratio"),
            "name": "diameter_length_ratio",
            "type": "cat",
            "choices": ["low", "medium", "high", "uncertain"],
        },
    ],
    "image_quality": [
        {
            "json_path": ("image_quality", "thread_visibility"),
            "name": "thread_visibility",
            "type": "cat",
            "choices": ["clear", "partial", "blurred", "uncertain"],
        },
        {
            "json_path": ("image_quality", "abutment_present"),
            "name": "abutment_present",
            "type": "boolcat",
            "choices": ["true", "false", "uncertain"],
        },
        {
            "json_path": ("image_quality", "projection_distortion"),
            "name": "projection_distortion",
            "type": "cat",
            "choices": ["none", "mild", "severe", "uncertain"],
        },
    ],
}

GROUP_CONF_PATHS = {
    "connection": ("connection_type", "confidence"),
    "platform": ("platform", "confidence"),
    "thread": ("thread", "confidence"),
    "outline": ("outline", "confidence"),
    "image_quality": ("image_quality", "confidence"),
}

DEFAULT_FEATURE_GROUPS = ["connection", "platform", "thread", "outline", "image_quality"]


@dataclass
class FusionArgs:
    manifest: str
    vlm_features: str
    output_root: str = str(ROOT / "outputs")
    experiment_name: str = "vlm_5feature_fusion"
    mode: str = "fusion_sum"
    feature_groups: str = "connection,platform,thread,outline,image_quality"
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    init_checkpoint: Optional[str] = None
    image_size: int = 224
    epochs: int = 40
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 3e-4
    image_lr: float = 1e-5
    weight_decay: float = 0.05
    label_smoothing: float = 0.05
    dropout: float = 0.1
    drop_path_rate: float = 0.0
    seed: int = 42
    amp: bool = True
    patience: int = 10
    monitor: str = "macro_f1"
    use_class_weights: bool = True
    strong_aug: bool = True
    freeze_image_encoder: bool = True
    uncertain_confidence_threshold: float = 0.0
    confidence_split_threshold: float = 0.7
    allow_cpu: bool = False
    write_prompt_only: bool = False


def parse_feature_groups(text: str) -> List[str]:
    groups = [x.strip() for x in text.split(",") if x.strip()]
    unknown = sorted(set(groups) - set(DEFAULT_FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}. Allowed: {DEFAULT_FEATURE_GROUPS}")
    return groups


def nested_get(obj: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def normalize_category(value: Any) -> str:
    if value is None:
        return "uncertain"
    value = str(value).strip().lower()
    value = value.replace("-", "_").replace(" ", "_")
    if value in {"", "nan", "none", "null", "unknown"}:
        return "uncertain"
    if value in {"yes", "present", "true", "1"}:
        return value
    if value in {"no", "absent", "false", "0"}:
        return value
    return value


def bool_to_category(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    value = normalize_category(value)
    if value in {"true", "yes", "present", "1"}:
        return "true"
    if value in {"false", "no", "absent", "0"}:
        return "false"
    return "uncertain"


def one_hot(value: str, choices: Sequence[str]) -> List[float]:
    if value not in choices:
        value = "uncertain" if "uncertain" in choices else choices[-1]
    return [1.0 if value == c else 0.0 for c in choices]


def feature_names(groups: Sequence[str], include_confidence: bool = True) -> List[str]:
    names: List[str] = []
    for group in groups:
        for spec in FEATURE_SCHEMAS[group]:
            if spec["type"] in {"cat", "boolcat"}:
                names.extend([f'{group}.{spec["name"]}={choice}' for choice in spec["choices"]])
            elif spec["type"] == "numeric":
                names.append(f'{group}.{spec["name"]}')
            else:
                raise ValueError(spec["type"])
        if include_confidence:
            names.append(f"{group}.confidence")
    return names


def vectorize_vlm_features(
    raw: Dict[str, Any],
    groups: Sequence[str],
    uncertain_confidence_threshold: float = 0.0,
    include_confidence: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    values: List[float] = []
    flat: Dict[str, Any] = {}
    for group in groups:
        confidence = to_float(nested_get(raw, GROUP_CONF_PATHS[group], 0.0), 0.0)
        low_conf = uncertain_confidence_threshold > 0 and confidence < uncertain_confidence_threshold

        for spec in FEATURE_SCHEMAS[group]:
            key = f'{group}.{spec["name"]}'
            if spec["type"] == "cat":
                value = "uncertain" if low_conf else normalize_category(nested_get(raw, spec["json_path"]))
                if value not in spec["choices"]:
                    value = "uncertain" if "uncertain" in spec["choices"] else spec["choices"][-1]
                values.extend(one_hot(value, spec["choices"]))
                flat[key] = value
            elif spec["type"] == "boolcat":
                value = "uncertain" if low_conf else bool_to_category(nested_get(raw, spec["json_path"]))
                values.extend(one_hot(value, spec["choices"]))
                flat[key] = value
            elif spec["type"] == "numeric":
                value_float = 0.0 if low_conf else to_float(nested_get(raw, spec["json_path"], 0.0), 0.0)
                value_float = float(np.clip(value_float, 0.0, 1.0))
                values.append(value_float)
                flat[key] = value_float
            else:
                raise ValueError(spec["type"])

        if include_confidence:
            values.append(float(np.clip(confidence, 0.0, 1.0)))
            flat[f"{group}.confidence"] = float(np.clip(confidence, 0.0, 1.0))

    return np.asarray(values, dtype=np.float32), flat


def default_uncertain_features() -> Dict[str, Any]:
    return {
        "connection_type": {"value": "uncertain", "confidence": 0.0},
        "platform": {
            "platform_visible": None,
            "platform_position_relative_y": 0.0,
            "platform_switching": "uncertain",
            "confidence": 0.0,
        },
        "thread": {
            "pitch": "uncertain",
            "depth": "uncertain",
            "micro_thread_zone": "uncertain",
            "macro_thread_zone": "uncertain",
            "confidence": 0.0,
        },
        "outline": {
            "body_shape": "uncertain",
            "diameter_length_ratio": "uncertain",
            "confidence": 0.0,
        },
        "image_quality": {
            "thread_visibility": "uncertain",
            "abutment_present": None,
            "projection_distortion": "uncertain",
            "confidence": 0.0,
        },
    }


def merge_default_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    merged = default_uncertain_features()
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def parse_json_maybe(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return {}
    text = str(text).strip()
    if not text:
        return {}
    return json.loads(text)


def flatten_csv_row_to_feature(row: pd.Series) -> Dict[str, Any]:
    if "vlm_json" in row and pd.notna(row["vlm_json"]):
        return parse_json_maybe(row["vlm_json"])

    out = default_uncertain_features()
    aliases = {
        ("connection_type", "value"): ["connection_type", "connection_type.value"],
        ("connection_type", "confidence"): ["connection_confidence", "connection_type.confidence"],
        ("platform", "platform_visible"): ["platform_visible", "platform.platform_visible"],
        ("platform", "platform_position_relative_y"): [
            "platform_position_relative_y",
            "platform.platform_position_relative_y",
        ],
        ("platform", "platform_switching"): ["platform_switching", "platform.platform_switching"],
        ("platform", "confidence"): ["platform_confidence", "platform.confidence"],
        ("thread", "pitch"): ["thread_pitch", "thread.pitch"],
        ("thread", "depth"): ["thread_depth", "thread.depth"],
        ("thread", "micro_thread_zone"): ["micro_thread_zone", "thread.micro_thread_zone"],
        ("thread", "macro_thread_zone"): ["macro_thread_zone", "thread.macro_thread_zone"],
        ("thread", "confidence"): ["thread_confidence", "thread.confidence"],
        ("outline", "body_shape"): ["body_shape", "outline.body_shape"],
        ("outline", "diameter_length_ratio"): ["diameter_length_ratio", "outline.diameter_length_ratio"],
        ("outline", "confidence"): ["outline_confidence", "outline.confidence"],
        ("image_quality", "thread_visibility"): ["thread_visibility", "image_quality.thread_visibility"],
        ("image_quality", "abutment_present"): ["abutment_present", "image_quality.abutment_present"],
        ("image_quality", "projection_distortion"): [
            "projection_distortion",
            "image_quality.projection_distortion",
        ],
        ("image_quality", "confidence"): ["image_quality_confidence", "image_quality.confidence"],
    }
    for (group, field), cols in aliases.items():
        for col in cols:
            if col in row and pd.notna(row[col]):
                out[group][field] = row[col]
                break
    return out


def add_feature_lookup_keys(lookup: Dict[str, Dict[str, Any]], rec: Dict[str, Any], feature: Dict[str, Any]) -> None:
    key_candidates = [
        rec.get("path"),
        rec.get("image_path"),
        rec.get("roi_path"),
        rec.get("file"),
        rec.get("filename"),
        rec.get("id"),
        rec.get("image_id"),
    ]
    for key in key_candidates:
        if key is None:
            continue
        key_s = str(key)
        lookup[key_s] = feature
        p = Path(key_s)
        lookup[p.name] = feature
        lookup[p.stem] = feature
        try:
            lookup[str(p.resolve())] = feature
        except Exception:
            pass


def load_vlm_feature_lookup(path: str | Path) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    lookup: Dict[str, Dict[str, Any]] = {}
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                feature = rec.get("vlm_json", rec.get("features", rec.get("feature", rec)))
                if isinstance(feature, str):
                    feature = json.loads(feature)
                add_feature_lookup_keys(lookup, rec, merge_default_features(feature))
    elif path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        records = obj if isinstance(obj, list) else obj.get("records", obj.get("features", []))
        if isinstance(records, dict):
            for key, feature in records.items():
                rec = {"path": key}
                add_feature_lookup_keys(lookup, rec, merge_default_features(feature))
        else:
            for rec in records:
                feature = rec.get("vlm_json", rec.get("features", rec.get("feature", rec)))
                if isinstance(feature, str):
                    feature = json.loads(feature)
                add_feature_lookup_keys(lookup, rec, merge_default_features(feature))
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            rec = row.to_dict()
            feature = flatten_csv_row_to_feature(row)
            add_feature_lookup_keys(lookup, rec, merge_default_features(feature))
    else:
        raise ValueError(f"Unsupported VLM feature file: {path}")
    return lookup


def find_feature_for_path(lookup: Dict[str, Dict[str, Any]], image_path: str) -> Optional[Dict[str, Any]]:
    p = Path(str(image_path))
    candidates = [str(image_path), str(p), p.name, p.stem]
    try:
        candidates.append(str(p.resolve()))
    except Exception:
        pass
    for key in candidates:
        if key in lookup:
            return lookup[key]
    return None


class ImplantVLMFusionDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        class_to_idx: Dict[str, int],
        vlm_lookup: Dict[str, Dict[str, Any]],
        feature_groups: Sequence[str],
        image_size: int,
        train: bool,
        strong_aug: bool,
        uncertain_confidence_threshold: float,
        shuffle_vlm: bool = False,
        seed: int = 42,
    ):
        self.df = load_manifest(manifest, split=split)
        self.class_to_idx = class_to_idx
        self.transform = build_transforms(image_size=image_size, train=train, strong_aug=strong_aug)
        self.feature_groups = list(feature_groups)
        self.raw_features: List[Dict[str, Any]] = []
        self.flat_features: List[Dict[str, Any]] = []
        self.vectors: List[np.ndarray] = []
        self.missing_count = 0

        for _, row in self.df.iterrows():
            raw = find_feature_for_path(vlm_lookup, str(row["path"]))
            if raw is None:
                self.missing_count += 1
                raw = default_uncertain_features()
            raw = merge_default_features(raw)
            vec, flat = vectorize_vlm_features(
                raw,
                groups=self.feature_groups,
                uncertain_confidence_threshold=uncertain_confidence_threshold,
            )
            self.raw_features.append(raw)
            self.flat_features.append(flat)
            self.vectors.append(vec)

        if shuffle_vlm and len(self.vectors) > 1:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(self.vectors))
            self.vectors = [self.vectors[i].copy() for i in perm]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row["path"])
        y = self.class_to_idx[str(row["label"])]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        vlm = torch.from_numpy(self.vectors[idx])
        return img, vlm, int(y), str(path), idx


class SimpleSumFusionModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        vlm_dim: int,
        mode: str,
        pretrained: bool,
        dropout: float,
        drop_path_rate: float,
        freeze_image_encoder: bool,
    ):
        super().__init__()
        import timm

        self.mode = mode
        self.freeze_image_encoder = freeze_image_encoder
        self.image_encoder: Optional[nn.Module] = None
        self.img_dim = 0

        if mode in {"image_only", "fusion_sum", "shuffled_fusion"}:
            self.image_encoder = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                drop_rate=dropout,
                drop_path_rate=drop_path_rate,
            )
            self.img_dim = int(getattr(self.image_encoder, "num_features"))
            if freeze_image_encoder:
                for p in self.image_encoder.parameters():
                    p.requires_grad = False
        else:
            self.img_dim = 512

        if mode == "image_only":
            self.classifier = nn.Linear(self.img_dim, num_classes)
        elif mode in {"fusion_sum", "shuffled_fusion"}:
            self.vlm_projection = nn.Linear(vlm_dim, self.img_dim)
            self.classifier = nn.Linear(self.img_dim, num_classes)
        elif mode == "vlm_only":
            hidden = max(128, min(512, vlm_dim * 4))
            self.vlm_mlp = nn.Sequential(
                nn.LayerNorm(vlm_dim),
                nn.Linear(vlm_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.image_encoder is None:
            raise RuntimeError("Image encoder is not available for this mode.")
        if self.freeze_image_encoder:
            self.image_encoder.eval()
            with torch.no_grad():
                return self.image_encoder(image)
        return self.image_encoder(image)

    def forward(self, image: torch.Tensor, vlm: torch.Tensor) -> torch.Tensor:
        if self.mode == "vlm_only":
            return self.vlm_mlp(vlm)

        z_img = self.encode_image(image)
        z_img = nn.functional.normalize(z_img, dim=1)

        if self.mode == "image_only":
            return self.classifier(z_img)

        z_vlm = self.vlm_projection(vlm)
        z_vlm = nn.functional.normalize(z_vlm, dim=1)
        z = z_img + z_vlm
        return self.classifier(z)


def compute_metrics_with_topk(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    labels = list(range(len(class_names)))
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    topk = min(3, len(class_names))
    top3_idx = np.argsort(-y_prob, axis=1)[:, :topk] if len(y_prob) else np.zeros((0, topk))
    top3_acc = float(np.mean([t in row for t, row in zip(y_true_arr, top3_idx)])) if len(y_true_arr) else 0.0
    per_class_recall = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    return {
        "top1_accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "top3_accuracy": top3_acc,
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": {class_names[i]: float(per_class_recall[i]) for i in range(len(class_names))},
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            target_names=list(class_names),
            zero_division=0,
            output_dict=True,
        ),
    }


def save_confusion_matrix_files(out_dir: Path, split: str, y_true, y_pred, class_names: Sequence[str]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    csv_path = out_dir / "confusion_matrices" / f"{split}.csv"
    png_path = out_dir / "confusion_matrices" / f"{split}.png"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(csv_path)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(max(6, len(class_names) * 0.9), max(5, len(class_names) * 0.8)))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{split} confusion matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
    except Exception as e:
        log(f"Warning: could not save confusion matrix png: {repr(e)}")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    idx_to_class: Dict[int, str],
    split: str,
    out_dir: Optional[Path],
    confidence_split_threshold: float,
) -> Dict[str, Any]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    subgroup_fields = [
        "connection.connection_type",
        "image_quality.thread_visibility",
        "image_quality.abutment_present",
        "platform.platform_switching",
    ]
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    dataset = loader.dataset

    with torch.no_grad():
        for image, vlm, y, paths, indices in loader:
            image = image.to(device, non_blocking=True)
            vlm = vlm.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(image, vlm)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            y_true.extend(y.cpu().numpy().astype(int).tolist())
            y_pred.extend(pred.cpu().numpy().astype(int).tolist())
            y_prob.extend(probs.cpu().numpy())
            topk = torch.topk(probs, k=min(3, probs.shape[1]), dim=1)
            for i, p, t, pr, prob, top_idx, top_val in zip(
                indices.cpu().numpy().tolist(),
                paths,
                y.cpu().numpy().astype(int).tolist(),
                pred.cpu().numpy().astype(int).tolist(),
                probs.cpu().numpy(),
                topk.indices.cpu().numpy(),
                topk.values.cpu().numpy(),
            ):
                flat = dataset.flat_features[int(i)]
                raw = dataset.raw_features[int(i)]
                row = {
                    "path": p,
                    "true_idx": int(t),
                    "true_label": idx_to_class[int(t)],
                    "pred_idx": int(pr),
                    "pred_label": idx_to_class[int(pr)],
                    "confidence": float(np.max(prob)),
                    "correct": bool(int(t) == int(pr)),
                    "top3_labels": "|".join(idx_to_class[int(k)] for k in top_idx),
                    "top3_probs": "|".join(f"{float(v):.6f}" for v in top_val),
                    "mean_vlm_confidence": float(
                        np.mean([to_float(nested_get(raw, GROUP_CONF_PATHS[g], 0.0)) for g in DEFAULT_FEATURE_GROUPS])
                    ),
                }
                for field in subgroup_fields:
                    row[field] = flat.get(field, "missing")
                rows.append(row)

    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    metrics = compute_metrics_with_topk(y_true, y_pred, y_prob_arr, class_names)

    if out_dir is not None:
        metrics_dir = out_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"{split}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(rows).to_csv(metrics_dir / f"{split}_predictions.csv", index=False)
        save_confusion_matrix_files(out_dir, split, y_true, y_pred, class_names)
        save_subgroup_metrics(
            rows=rows,
            class_names=class_names,
            out_path=metrics_dir / f"{split}_subgroup_metrics.csv",
            confidence_split_threshold=confidence_split_threshold,
        )
    return metrics


def save_subgroup_metrics(
    rows: List[Dict[str, Any]],
    class_names: Sequence[str],
    out_path: Path,
    confidence_split_threshold: float,
) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    groups = [
        "connection.connection_type",
        "image_quality.thread_visibility",
        "image_quality.abutment_present",
        "platform.platform_switching",
    ]
    df["vlm_confidence_bucket"] = np.where(
        df["mean_vlm_confidence"].astype(float) >= confidence_split_threshold,
        f">={confidence_split_threshold}",
        f"<{confidence_split_threshold}",
    )
    groups.append("vlm_confidence_bucket")
    out_rows = []
    for group in groups:
        if group not in df.columns:
            continue
        for value, part in df.groupby(group):
            y_true = part["true_idx"].astype(int).to_numpy()
            y_pred = part["pred_idx"].astype(int).to_numpy()
            if len(part) == 0:
                continue
            out_rows.append(
                {
                    "group": group,
                    "value": value,
                    "n": int(len(part)),
                    "top1_accuracy": float(accuracy_score(y_true, y_pred)),
                    "macro_f1": float(
                        f1_score(
                            y_true,
                            y_pred,
                            labels=list(range(len(class_names))),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)


def create_loaders(
    args: FusionArgs,
    class_to_idx: Dict[str, int],
    vlm_lookup: Dict[str, Dict[str, Any]],
    feature_groups: Sequence[str],
    shuffle_vlm: bool,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], int, Dict[str, int]]:
    train_ds = ImplantVLMFusionDataset(
        args.manifest,
        "train",
        class_to_idx,
        vlm_lookup,
        feature_groups,
        args.image_size,
        train=True,
        strong_aug=args.strong_aug,
        uncertain_confidence_threshold=args.uncertain_confidence_threshold,
        shuffle_vlm=shuffle_vlm,
        seed=args.seed,
    )
    valid_ds = ImplantVLMFusionDataset(
        args.manifest,
        "valid",
        class_to_idx,
        vlm_lookup,
        feature_groups,
        args.image_size,
        train=False,
        strong_aug=False,
        uncertain_confidence_threshold=args.uncertain_confidence_threshold,
        shuffle_vlm=shuffle_vlm,
        seed=args.seed + 1,
    )
    test_ds: Optional[ImplantVLMFusionDataset] = None
    try:
        test_ds = ImplantVLMFusionDataset(
            args.manifest,
            "test",
            class_to_idx,
            vlm_lookup,
            feature_groups,
            args.image_size,
            train=False,
            strong_aug=False,
            uncertain_confidence_threshold=args.uncertain_confidence_threshold,
            shuffle_vlm=shuffle_vlm,
            seed=args.seed + 2,
        )
    except Exception:
        test_ds = None

    missing = {
        "train": train_ds.missing_count,
        "valid": valid_ds.missing_count,
        "test": test_ds.missing_count if test_ds is not None else -1,
    }
    if any(v > 0 for v in missing.values() if v >= 0):
        log(f"Warning: missing VLM features by split: {missing}. Missing rows use uncertain/0.0 confidence.")

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs) if test_ds is not None else None
    return train_loader, valid_loader, test_loader, len(train_ds.vectors[0]), missing


def train_one_run(args: FusionArgs, suite_tag: Optional[str] = None) -> Dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow-cpu only for debugging.")

    full_df = load_manifest(args.manifest, split=None)
    class_to_idx, idx_to_class = label_mapping(full_df)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    feature_groups = parse_feature_groups(args.feature_groups)
    vlm_lookup = load_vlm_feature_lookup(args.vlm_features)
    shuffle_vlm = args.mode == "shuffled_fusion"

    exp_name = args.experiment_name
    if suite_tag:
        exp_name = f"{exp_name}_{suite_tag}"
    run_dir = make_run_dir(args.output_root, exp_name)
    write_json(run_dir / "args.json", asdict(args))
    write_json(run_dir / "class_to_idx.json", class_to_idx)
    write_json(
        run_dir / "vlm_feature_schema.json",
        {
            "feature_groups": feature_groups,
            "feature_names": feature_names(feature_groups),
            "uncertain_confidence_threshold": args.uncertain_confidence_threshold,
        },
    )
    (run_dir / "vlm_feature_prompt_5feature.md").write_text(VLM_PROMPT, encoding="utf-8")

    train_loader, valid_loader, test_loader, vlm_dim, missing = create_loaders(
        args,
        class_to_idx,
        vlm_lookup,
        feature_groups,
        shuffle_vlm=shuffle_vlm,
    )
    write_json(run_dir / "vlm_missing_counts.json", missing)

    log(f"Run dir: {run_dir}")
    log(f"Mode: {args.mode} | groups={feature_groups} | vlm_dim={vlm_dim}")
    log(f"Classes ({len(class_names)}): {class_names}")
    log(f"Device: {device}")

    model = SimpleSumFusionModel(
        model_name=args.model_name,
        num_classes=len(class_names),
        vlm_dim=vlm_dim,
        mode=args.mode,
        pretrained=args.pretrained,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        freeze_image_encoder=args.freeze_image_encoder,
    )
    if args.init_checkpoint:
        loaded, skipped = load_matching_weights(model, args.init_checkpoint, strict=False)
        log(f"Initialized from checkpoint: loaded={loaded}, skipped={skipped}, path={args.init_checkpoint}")
    model.to(device)
    log(f"Trainable parameters: {count_parameters(model):,}")

    weights = class_weights_from_manifest(args.manifest, class_to_idx, split="train").to(device) if args.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    image_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("image_encoder."):
            image_params.append(param)
        else:
            other_params.append(param)
    param_groups = [{"params": other_params, "lr": args.lr}]
    if image_params:
        param_groups.append({"params": image_params, "lr": args.image_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    best_score = -1.0
    best_metrics: Dict[str, Any] = {}
    patience_count = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        t0 = time.time()
        for image, vlm, y, _paths, _indices in train_loader:
            image = image.to(device, non_blocking=True)
            vlm = vlm.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(image, vlm)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            bs = image.size(0)
            epoch_loss += float(loss.item()) * bs
            n_seen += bs
        scheduler.step()

        train_loss = epoch_loss / max(1, n_seen)
        valid_metrics = evaluate(
            model,
            valid_loader,
            device,
            idx_to_class,
            "valid_live",
            out_dir=None,
            confidence_split_threshold=args.confidence_split_threshold,
        )
        score = float(valid_metrics.get(args.monitor, valid_metrics.get("macro_f1", 0.0)))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_top1_accuracy": valid_metrics["top1_accuracy"],
            "valid_top3_accuracy": valid_metrics["top3_accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_macro_recall": valid_metrics["macro_recall"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "metrics" / "history.csv", index=False)
        log(
            f"epoch={epoch:03d} loss={train_loss:.4f} "
            f"valid_top1={valid_metrics['top1_accuracy']:.4f} "
            f"valid_top3={valid_metrics['top3_accuracy']:.4f} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f}"
        )

        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, valid_metrics, asdict(args), class_to_idx)
        if score > best_score:
            best_score = score
            best_metrics = valid_metrics
            patience_count = 0
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, valid_metrics, asdict(args), class_to_idx)
            (run_dir / "metrics" / "best_valid_metrics.json").write_text(
                json.dumps(valid_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log(f"New best {args.monitor}={best_score:.4f} at epoch={epoch}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log(f"Early stopping at epoch={epoch}. Best {args.monitor}={best_score:.4f}")
                break

    best_ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    valid_final = evaluate(
        model,
        valid_loader,
        device,
        idx_to_class,
        "valid",
        out_dir=run_dir,
        confidence_split_threshold=args.confidence_split_threshold,
    )
    test_final: Optional[Dict[str, Any]] = None
    if test_loader is not None:
        test_final = evaluate(
            model,
            test_loader,
            device,
            idx_to_class,
            "test",
            out_dir=run_dir,
            confidence_split_threshold=args.confidence_split_threshold,
        )
        log(
            f"TEST top1={test_final['top1_accuracy']:.4f} "
            f"top3={test_final['top3_accuracy']:.4f} macro_f1={test_final['macro_f1']:.4f}"
        )

    summary = {
        "run_dir": str(run_dir),
        "suite_tag": suite_tag or "",
        "mode": args.mode,
        "feature_groups": feature_groups,
        "best_valid": best_metrics,
        "valid": valid_final,
        "test": test_final,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def suite_variants(args: FusionArgs, suite: str) -> List[Tuple[str, FusionArgs]]:
    all_groups = "connection,platform,thread,outline,image_quality"
    variants: List[Tuple[str, FusionArgs]] = []
    if suite in {"single"}:
        variants.append((args.mode, args))
    if suite in {"core", "all"}:
        variants.extend(
            [
                ("A_image_only", replace(args, mode="image_only", feature_groups=all_groups)),
                ("B_vlm_only", replace(args, mode="vlm_only", feature_groups=all_groups)),
                ("C_image_vlm_sum", replace(args, mode="fusion_sum", feature_groups=all_groups)),
                ("D_shuffled_vlm", replace(args, mode="shuffled_fusion", feature_groups=all_groups)),
            ]
        )
    if suite in {"cumulative", "all"}:
        cumulative = [
            ("ablation_01_image_only", "image_only", all_groups),
            ("ablation_02_connection", "fusion_sum", "connection"),
            ("ablation_03_connection_platform", "fusion_sum", "connection,platform"),
            ("ablation_04_connection_platform_thread", "fusion_sum", "connection,platform,thread"),
            (
                "ablation_05_connection_platform_thread_outline",
                "fusion_sum",
                "connection,platform,thread,outline",
            ),
            ("ablation_06_all_5_features", "fusion_sum", all_groups),
        ]
        variants.extend((tag, replace(args, mode=mode, feature_groups=groups)) for tag, mode, groups in cumulative)
    if suite in {"leave_one_out", "all"}:
        for held_out in DEFAULT_FEATURE_GROUPS:
            groups = ",".join([g for g in DEFAULT_FEATURE_GROUPS if g != held_out])
            variants.append((f"loo_minus_{held_out}", replace(args, mode="fusion_sum", feature_groups=groups)))
    return variants


def save_suite_summary(output_root: Path, experiment_name: str, summaries: List[Dict[str, Any]]) -> None:
    rows = []
    for item in summaries:
        for split in ["valid", "test"]:
            metrics = item.get(split)
            if metrics is None:
                continue
            rows.append(
                {
                    "suite_tag": item.get("suite_tag", ""),
                    "mode": item.get("mode", ""),
                    "feature_groups": ",".join(item.get("feature_groups", [])),
                    "split": split,
                    "top1_accuracy": metrics.get("top1_accuracy"),
                    "top3_accuracy": metrics.get("top3_accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "macro_recall": metrics.get("macro_recall"),
                    "run_dir": item.get("run_dir", ""),
                }
            )
    out_dir = output_root / f"{experiment_name}_suite_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Suite summary: {out_dir / 'summary.csv'}")


def parse_args() -> Tuple[FusionArgs, str]:
    p = argparse.ArgumentParser(description="VLM 5-feature simple-sum fusion experiment.")
    p.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "plain_vit_7label.csv"))
    p.add_argument("--vlm-features", required=False, default="")
    p.add_argument("--output-root", default=str(ROOT / "outputs"))
    p.add_argument("--experiment-name", default="vlm_5feature_fusion")
    p.add_argument("--suite", choices=["single", "core", "cumulative", "leave_one_out", "all"], default="all")
    p.add_argument("--mode", choices=["image_only", "vlm_only", "fusion_sum", "shuffled_fusion"], default="fusion_sum")
    p.add_argument("--feature-groups", default="connection,platform,thread,outline,image_quality")
    p.add_argument("--model-name", default="vit_base_patch16_224")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--init-checkpoint", default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--drop-path-rate", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--monitor", default="macro_f1")
    p.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    p.add_argument("--no-strong-aug", dest="strong_aug", action="store_false")
    p.add_argument("--finetune-image-encoder", dest="freeze_image_encoder", action="store_false")
    p.add_argument("--uncertain-confidence-threshold", type=float, default=0.0)
    p.add_argument("--confidence-split-threshold", type=float, default=0.7)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--write-prompt-only", action="store_true")
    ns = p.parse_args()

    args = FusionArgs(
        manifest=ns.manifest,
        vlm_features=ns.vlm_features,
        output_root=ns.output_root,
        experiment_name=ns.experiment_name,
        mode=ns.mode,
        feature_groups=ns.feature_groups,
        model_name=ns.model_name,
        pretrained=ns.pretrained,
        init_checkpoint=ns.init_checkpoint,
        image_size=ns.image_size,
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        num_workers=ns.num_workers,
        lr=ns.lr,
        image_lr=ns.image_lr,
        weight_decay=ns.weight_decay,
        label_smoothing=ns.label_smoothing,
        dropout=ns.dropout,
        drop_path_rate=ns.drop_path_rate,
        seed=ns.seed,
        amp=ns.amp,
        patience=ns.patience,
        monitor=ns.monitor,
        use_class_weights=ns.use_class_weights,
        strong_aug=ns.strong_aug,
        freeze_image_encoder=ns.freeze_image_encoder,
        uncertain_confidence_threshold=ns.uncertain_confidence_threshold,
        confidence_split_threshold=ns.confidence_split_threshold,
        allow_cpu=ns.allow_cpu,
        write_prompt_only=ns.write_prompt_only,
    )
    return args, ns.suite


def main() -> None:
    args, suite = parse_args()
    prompt_path = Path(args.output_root) / "vlm_feature_prompt_5feature.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(VLM_PROMPT, encoding="utf-8")
    if args.write_prompt_only:
        log(f"Wrote VLM prompt: {prompt_path}")
        return
    if not args.vlm_features:
        raise RuntimeError("Pass --vlm-features path/to/vlm_features.jsonl or use --write-prompt-only.")

    summaries: List[Dict[str, Any]] = []
    for tag, variant in suite_variants(args, suite):
        log(f"Starting variant: {tag}")
        summaries.append(train_one_run(variant, suite_tag=tag if suite != "single" else None))
    if len(summaries) > 1:
        save_suite_summary(Path(args.output_root), args.experiment_name, summaries)


if __name__ == "__main__":
    main()
