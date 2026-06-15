#!/usr/bin/env python3
"""
Generate 13-core-attribute + 21-supporting-feature JSON/JSONL for the 7-label
dental implant ROI dataset with Azure OpenAI vision-capable chat deployments.

The experiment is not run by importing this file. Run it explicitly, for example:

  python generate_13feature_21reason_7label.py \
    --env-file .env \
    --input data/7label \
    --output data/manifests/13feature_21reason_7label_all.jsonl \
    --json-output data/manifests/13feature_21reason_7label_all.json \
    --summary-csv data/manifests/13feature_21reason_7label_all.summary.csv \
    --azure-deployment gpt-5.5 \
    --workers 4 \
    --resume

Required Azure settings can be supplied by CLI args or environment variables:
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION,
  AZURE_OPENAI_DEPLOYMENT or AZURE_OPENAI_CHAT_DEPLOYMENT.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

STATE_VALUES = {"present", "absent", "not_assessable"}
FEATURE_STATUS_VALUES = {"usable", "partial_usable", "low_confidence", "invalid_roi"}
FAILURE_REASONS = {
    "none",
    "implant_not_visible",
    "all_features_not_assessable",
    "connection_obscured",
    "thread_not_visible",
    "apex_not_visible",
    "low_confidence",
}

CORE_ATTRIBUTES = [
    "Company",
    "Name",
    "System",
    "Connection_1",
    "Connection_2",
    "Flange",
    "Collar",
    "Microthread",
    "Body_Shape",
    "Body_Type",
    "Thread_Shape",
    "Apex_Shape",
    "Apex_Hole",
]

STRICT_CORE_VALUES = {
    "System": {"Bone Level", "Tissue Level", "unknown"},
    "Connection_1": {"External", "Internal", "unknown"},
    "Connection_2": {"HEX", "OCTA", "unknown"},
    "Flange": {"Convergent", "Parallel", "unknown"},
    "Collar": {"Yes", "No", "unknown"},
    "Microthread": {"Yes", "No", "unknown"},
    "Body_Shape": {"Non-tapered", "Tapered apex", "Tapered body", "unknown"},
    "Body_Type": {"Threaded", "unknown"},
    "Thread_Shape": {"V-shaped", "unknown"},
    "Apex_Shape": {"Flat", "Rounded", "unknown"},
    "Apex_Hole": {"Yes", "No", "unknown"},
}

FEATURE_NAMES = [
    "implant_visible",
    "connection_region_visible",
    "thread_region_visible",
    "abutment_or_prosthesis_present",
    "positive_connection_region_present",
    "straight_screw_entry_space_present",
    "negative_internal_connection_space_present",
    "platform_step_present",
    "internal_small_void_present",
    "smooth_no_thread_collar_present",
    "machined_surface_present",
    "platform_boundary_visible",
    "platform_switching_step_present",
    "thread_lines_visible",
    "regular_thread_pitch_visible",
    "deep_thread_relief_present",
    "micro_thread_zone_present",
    "macro_thread_zone_present",
    "body_taper_present",
    "parallel_body_wall_present",
    "apex_boundary_visible",
]

CONNECTION_DEPENDENT_FEATURES = [
    "positive_connection_region_present",
    "straight_screw_entry_space_present",
    "negative_internal_connection_space_present",
    "platform_step_present",
    "internal_small_void_present",
]

THREAD_DEPENDENT_FEATURES = [
    "thread_lines_visible",
    "regular_thread_pitch_visible",
    "deep_thread_relief_present",
    "micro_thread_zone_present",
    "macro_thread_zone_present",
]


VLM_PROMPT = """You are analyzing a dental implant ROI radiograph.

Task:
Extract both:
1. 13 required core implant attributes
2. 21 supporting visual features

Do not use file path, folder name, dataset label, split name, or metadata.
Use only visible evidence from the radiograph.
Do not guess from prior knowledge.
Return only valid JSON.
Do not include explanatory text outside JSON.
Do not use null.

Important:
- The 13 core attributes must always be returned.
- If a core attribute cannot be determined, use "unknown".
- The 13 core attributes do not need numeric confidence.
- The 21 supporting visual features must include state and confidence.
- The 21 supporting features are the evidence used to derive the 13 core attributes.
- Do not confuse "absent" with "not_assessable".
- Use "absent" only when the relevant region is clearly visible and the feature is clearly not present.
- Use "not_assessable" when the region is hidden, cropped, blurred, distorted, or obscured by abutment/prosthesis.

Allowed states for 21 supporting features:
- present
- absent
- not_assessable

Allowed value for unknown core attributes:
- unknown

13 required core attributes:

Company:
- possible value: known company name or unknown
- Company should be determined only if reference-table matching is strongly supported.
- Otherwise use unknown.

Name:
- possible value: known implant system name or unknown
- Name is not directly visible from radiograph alone.
- Use unknown unless reference-table matching is strongly supported.

System:
- Bone Level
- Tissue Level
- unknown

Connection_1:
- External
- Internal
- unknown

Connection_2:
- HEX
- OCTA
- unknown

Flange:
- Convergent
- Parallel
- unknown

Collar:
- Yes
- No
- unknown

Microthread:
- Yes
- No
- unknown

Body_Shape:
- Non-tapered
- Tapered apex
- Tapered body
- unknown

Body_Type:
- Threaded
- unknown

Thread_Shape:
- V-shaped
- unknown

Apex_Shape:
- Flat
- Rounded
- unknown

Apex_Hole:
- Yes
- No
- unknown

21 supporting visual features:

1. implant_visible
2. connection_region_visible
3. thread_region_visible
4. abutment_or_prosthesis_present
5. positive_connection_region_present
6. straight_screw_entry_space_present
7. negative_internal_connection_space_present
8. platform_step_present
9. internal_small_void_present
10. smooth_no_thread_collar_present
11. machined_surface_present
12. platform_boundary_visible
13. platform_switching_step_present
14. thread_lines_visible
15. regular_thread_pitch_visible
16. deep_thread_relief_present
17. micro_thread_zone_present
18. macro_thread_zone_present
19. body_taper_present
20. parallel_body_wall_present
21. apex_boundary_visible

Derivation rules:

System:
- Tissue Level if smooth_no_thread_collar_present or machined_surface_present is present.
- Bone Level if the coronal/platform area is visible and no smooth collar is present.
- Otherwise unknown.

Connection_1:
- External if positive_connection_region_present or straight_screw_entry_space_present is present.
- Internal if negative_internal_connection_space_present, platform_step_present, or internal_small_void_present is present.
- If connection region is obscured, use unknown.

Connection_2:
- HEX or OCTA only if the detailed connection geometry is directly visible.
- Otherwise unknown.

Flange:
- Convergent if the coronal/flange contour visibly narrows.
- Parallel if the coronal/flange contour is visibly parallel.
- Otherwise unknown.

Collar:
- Yes if smooth_no_thread_collar_present or machined_surface_present is present.
- No only if the coronal region is clearly visible and no collar exists.
- Otherwise unknown.

Microthread:
- Yes if micro_thread_zone_present is present.
- No only if the upper thread region is clearly visible and no microthread zone exists.
- Otherwise unknown.

Body_Shape:
- Non-tapered if parallel_body_wall_present is present and body_taper_present is absent.
- Tapered body if the whole body narrows apically.
- Tapered apex if mainly the apical portion narrows.
- Otherwise unknown.

Body_Type:
- Threaded if thread_lines_visible is present.
- Otherwise unknown.

Thread_Shape:
- V-shaped if thread relief/cut pattern is clearly V-like.
- Otherwise unknown.

Apex_Shape:
- Flat if the apex end is clearly flat.
- Rounded if the apex end is clearly rounded.
- Otherwise unknown.

Apex_Hole:
- Yes if an apical hole/opening is directly visible.
- No only if the apex is clearly visible and no hole/opening exists.
- Otherwise unknown.

Decision rules:
- If implant_visible is absent, all 13 core attributes should be unknown.
- If connection_region_visible is not_assessable, Connection_1, Connection_2, and Flange should usually be unknown.
- If thread_region_visible is not_assessable, Microthread, Body_Type, and Thread_Shape should usually be unknown.
- If apex_boundary_visible is not_assessable, Apex_Shape and Apex_Hole should usually be unknown.
- If abutment_or_prosthesis_present is present and connection_region_visible is not_assessable, connection-related attributes should usually be unknown.
- Output every required field even when unknown.

Output JSON schema:

{
  "api_status": "ok",
  "core_attributes": {
    "Company": {"value": "unknown"},
    "Name": {"value": "unknown"},
    "System": {"value": "Bone Level / Tissue Level / unknown"},
    "Connection_1": {"value": "External / Internal / unknown"},
    "Connection_2": {"value": "HEX / OCTA / unknown"},
    "Flange": {"value": "Convergent / Parallel / unknown"},
    "Collar": {"value": "Yes / No / unknown"},
    "Microthread": {"value": "Yes / No / unknown"},
    "Body_Shape": {"value": "Non-tapered / Tapered apex / Tapered body / unknown"},
    "Body_Type": {"value": "Threaded / unknown"},
    "Thread_Shape": {"value": "V-shaped / unknown"},
    "Apex_Shape": {"value": "Flat / Rounded / unknown"},
    "Apex_Hole": {"value": "Yes / No / unknown"}
  },
  "supporting_features": {
    "implant_visible": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "connection_region_visible": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "thread_region_visible": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "abutment_or_prosthesis_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "positive_connection_region_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "straight_screw_entry_space_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "negative_internal_connection_space_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "platform_step_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "internal_small_void_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "smooth_no_thread_collar_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "machined_surface_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "platform_boundary_visible": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "platform_switching_step_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "thread_lines_visible": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "regular_thread_pitch_visible": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "deep_thread_relief_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "micro_thread_zone_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "macro_thread_zone_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "body_taper_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "parallel_body_wall_present": {"state": "present / absent / not_assessable", "confidence": 0.0},
    "apex_boundary_visible": {"state": "present / absent / not_assessable", "confidence": 0.0}
  },
  "feature_status": {"value": "usable / partial_usable / low_confidence / invalid_roi"},
  "usable_for_feature_vector": true,
  "failure_reason": "none / implant_not_visible / all_features_not_assessable / connection_obscured / thread_not_visible / apex_not_visible / low_confidence"
}
"""


def load_dotenv_if_available(path: Optional[str]) -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    if path:
        load_dotenv(path)
    else:
        load_dotenv(Path.cwd() / ".env")


def image_to_data_url(path: Path, max_side: int, jpeg_quality: int) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if max_side > 0:
            width, height = img.size
            scale = min(1.0, float(max_side) / float(max(width, height)))
            if scale < 1.0:
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_state(value: Any) -> str:
    if value is None:
        return "not_assessable"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in STATE_VALUES:
        return normalized
    if normalized in {"unknown", "uncertain", "not_visible", "hidden", "blurred", "na", "n/a"}:
        return "not_assessable"
    if normalized in {"yes", "true", "1", "visible"}:
        return "present"
    if normalized in {"no", "false", "0", "not_present"}:
        return "absent"
    return "not_assessable"


def clamp_confidence(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, out))


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_core_value(name: str, value: Any) -> str:
    if name in {"Company", "Name"}:
        return "unknown"
    if value is None:
        return "unknown"
    text = str(value).strip()
    if not text:
        return "unknown"
    return text if text in STRICT_CORE_VALUES[name] else "unknown"


def feature_default(state: str = "not_assessable", confidence: float = 0.0) -> Dict[str, Any]:
    return {"state": state, "confidence": float(confidence)}


def default_response(api_status: str = "ok", failure_reason: str = "low_confidence") -> Dict[str, Any]:
    return {
        "api_status": api_status,
        "core_attributes": {name: {"value": "unknown"} for name in CORE_ATTRIBUTES},
        "supporting_features": {name: feature_default() for name in FEATURE_NAMES},
        "feature_status": {"value": "low_confidence"},
        "usable_for_feature_vector": False,
        "failure_reason": failure_reason,
    }


def mean_confidence(feature: Dict[str, Any]) -> float:
    features = feature.get("supporting_features", {})
    values = [float(features.get(name, {}).get("confidence", 0.0)) for name in FEATURE_NAMES]
    return sum(values) / max(1, len(values))


def all_structural_not_assessable(feature: Dict[str, Any]) -> bool:
    features = feature["supporting_features"]
    return all(features[name]["state"] == "not_assessable" for name in FEATURE_NAMES)


def first_partial_failure_reason(feature: Dict[str, Any]) -> str:
    features = feature["supporting_features"]
    if features["connection_region_visible"]["state"] == "not_assessable":
        return "connection_obscured"
    if features["thread_region_visible"]["state"] == "not_assessable":
        return "thread_not_visible"
    if features["apex_boundary_visible"]["state"] == "not_assessable":
        return "apex_not_visible"
    return "none"


def apply_decision_rules(feature: Dict[str, Any], min_mean_confidence: float) -> Dict[str, Any]:
    features = feature["supporting_features"]
    core = feature["core_attributes"]

    if (
        features["smooth_no_thread_collar_present"]["state"] == "present"
        or features["machined_surface_present"]["state"] == "present"
    ):
        core["System"]["value"] = "Tissue Level"
        core["Collar"]["value"] = "Yes"
    elif (
        features["connection_region_visible"]["state"] == "present"
        and features["smooth_no_thread_collar_present"]["state"] == "absent"
        and features["machined_surface_present"]["state"] == "absent"
    ):
        core["System"]["value"] = "Bone Level"
        core["Collar"]["value"] = "No"

    if (
        features["positive_connection_region_present"]["state"] == "present"
        or features["straight_screw_entry_space_present"]["state"] == "present"
    ):
        core["Connection_1"]["value"] = "External"
    elif (
        features["negative_internal_connection_space_present"]["state"] == "present"
        or features["platform_step_present"]["state"] == "present"
        or features["internal_small_void_present"]["state"] == "present"
    ):
        core["Connection_1"]["value"] = "Internal"

    if features["micro_thread_zone_present"]["state"] == "present":
        core["Microthread"]["value"] = "Yes"
    elif (
        features["thread_region_visible"]["state"] == "present"
        and features["micro_thread_zone_present"]["state"] == "absent"
    ):
        core["Microthread"]["value"] = "No"

    if (
        features["parallel_body_wall_present"]["state"] == "present"
        and features["body_taper_present"]["state"] == "absent"
    ):
        core["Body_Shape"]["value"] = "Non-tapered"

    if features["thread_lines_visible"]["state"] == "present":
        core["Body_Type"]["value"] = "Threaded"

    if features["connection_region_visible"]["state"] == "not_assessable":
        for name in CONNECTION_DEPENDENT_FEATURES:
            features[name]["state"] = "not_assessable"
        core["Connection_1"]["value"] = "unknown"
        core["Connection_2"]["value"] = "unknown"
        core["Flange"]["value"] = "unknown"

    if (
        features["abutment_or_prosthesis_present"]["state"] == "present"
        and features["connection_region_visible"]["state"] == "not_assessable"
    ):
        core["Connection_1"]["value"] = "unknown"
        core["Connection_2"]["value"] = "unknown"
        core["Flange"]["value"] = "unknown"

    if features["thread_region_visible"]["state"] == "not_assessable":
        for name in THREAD_DEPENDENT_FEATURES:
            features[name]["state"] = "not_assessable"
        core["Microthread"]["value"] = "unknown"
        core["Body_Type"]["value"] = "unknown"
        core["Thread_Shape"]["value"] = "unknown"

    if features["apex_boundary_visible"]["state"] == "not_assessable":
        core["Apex_Shape"]["value"] = "unknown"
        core["Apex_Hole"]["value"] = "unknown"

    if features["implant_visible"]["state"] == "absent":
        for name in CORE_ATTRIBUTES:
            core[name]["value"] = "unknown"
        feature["feature_status"]["value"] = "invalid_roi"
        feature["usable_for_feature_vector"] = False
        feature["failure_reason"] = "implant_not_visible"
        return feature

    if all_structural_not_assessable(feature):
        feature["feature_status"]["value"] = "low_confidence"
        feature["usable_for_feature_vector"] = False
        feature["failure_reason"] = "all_features_not_assessable"
        return feature

    if min_mean_confidence > 0 and mean_confidence(feature) < min_mean_confidence:
        feature["feature_status"]["value"] = "low_confidence"
        feature["usable_for_feature_vector"] = False
        feature["failure_reason"] = "low_confidence"
        return feature

    partial_reason = first_partial_failure_reason(feature)
    if partial_reason != "none":
        feature["feature_status"]["value"] = "partial_usable"
        feature["usable_for_feature_vector"] = True
        feature["failure_reason"] = partial_reason
        return feature

    feature["feature_status"]["value"] = "usable"
    feature["usable_for_feature_vector"] = True
    feature["failure_reason"] = "none"
    return feature


def validate_and_complete_response(raw: Dict[str, Any], min_mean_confidence: float) -> Dict[str, Any]:
    out = default_response(api_status="ok", failure_reason="none")
    out["api_status"] = "ok"

    raw_core = raw.get("core_attributes", {})
    if not isinstance(raw_core, dict):
        raw_core = {}
    for name in CORE_ATTRIBUTES:
        item = raw_core.get(name, {})
        value = item.get("value") if isinstance(item, dict) else item
        out["core_attributes"][name] = {"value": normalize_core_value(name, value)}

    raw_features = raw.get("supporting_features", raw.get("feature_detection", {}))
    if not isinstance(raw_features, dict):
        raw_features = {}
    for name in FEATURE_NAMES:
        item = raw_features.get(name, {})
        if not isinstance(item, dict):
            item = {}
        out["supporting_features"][name] = {
            "state": normalize_state(item.get("state")),
            "confidence": clamp_confidence(item.get("confidence", 0.0)),
        }

    raw_feature_status = raw.get("feature_status", {})
    feature_status = raw_feature_status.get("value") if isinstance(raw_feature_status, dict) else raw_feature_status
    feature_status = str(feature_status or "").strip()
    if feature_status in FEATURE_STATUS_VALUES:
        out["feature_status"]["value"] = feature_status

    out["usable_for_feature_vector"] = normalize_bool(raw.get("usable_for_feature_vector", True))
    failure_reason = str(raw.get("failure_reason", "none")).strip().lower()
    out["failure_reason"] = failure_reason if failure_reason in FAILURE_REASONS else "none"
    return apply_decision_rules(out, min_mean_confidence)


def feature_vector(feature: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    values: List[float] = []
    names: List[str] = []
    features = feature["supporting_features"]
    for feature_name in FEATURE_NAMES:
        item = features[feature_name]
        state = item["state"]
        assessable = 0.0 if state == "not_assessable" else 1.0
        present_value = 1.0 if state == "present" else 0.0
        values.extend([present_value, assessable, float(item["confidence"])])
        names.extend([
            f"{feature_name}.value",
            f"{feature_name}.assessable_mask",
            f"{feature_name}.confidence",
        ])
    return values, names


def create_azure_client(args: argparse.Namespace) -> Any:
    from openai import AzureOpenAI

    endpoint = (args.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
    api_key = args.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
    api_version = args.azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION")
    if not endpoint:
        raise RuntimeError("Missing Azure endpoint. Set AZURE_OPENAI_ENDPOINT or pass --azure-endpoint.")
    if not api_key:
        raise RuntimeError("Missing Azure API key. Set AZURE_OPENAI_API_KEY or pass --azure-api-key.")

    if not api_version:
        raise RuntimeError("Missing Azure API version. Set AZURE_OPENAI_API_VERSION or pass --azure-api-version.")
    client = AzureOpenAI(azure_endpoint=endpoint.rstrip("/"), api_key=api_key, api_version=api_version)
    client._implant_endpoint_mode = "azure_deployments"
    return client


def azure_deployment_name(args: argparse.Namespace) -> str:
    deployment = (
        args.azure_deployment
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    )
    if not deployment:
        raise RuntimeError(
            "Missing Azure deployment name. Set AZURE_OPENAI_DEPLOYMENT, "
            "AZURE_OPENAI_CHAT_DEPLOYMENT, or pass --azure-deployment."
        )
    return deployment


def call_vlm(client: Any, args: argparse.Namespace, image_path: Path) -> Dict[str, Any]:
    data_url = image_to_data_url(image_path, args.max_image_side, args.jpeg_quality)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful dental implant radiograph feature extraction assistant. "
                "Use only visible image evidence. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": VLM_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url, "detail": args.image_detail}},
            ],
        },
    ]
    response = client.chat.completions.create(
        model=azure_deployment_name(args),
        messages=messages,
        max_completion_tokens=args.max_tokens,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    raw = extract_json_object(content)
    return validate_and_complete_response(raw, args.min_mean_confidence)


def read_manifest(path: Path, path_column: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or path_column not in reader.fieldnames:
            raise RuntimeError(f"Manifest must contain a '{path_column}' column.")
        for row in reader:
            image_path = (row.get(path_column) or "").strip()
            if not image_path:
                continue
            rows.append({"path": image_path, "label": row.get("label", ""), "split": row.get("split", "")})
    return rows


def discover_images(input_path: Path, path_column: str) -> List[Dict[str, str]]:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return read_manifest(input_path, path_column)
    if input_path.is_file():
        return [{"path": str(input_path), "label": "", "split": ""}]
    if input_path.is_dir():
        rows: List[Dict[str, str]] = []
        split_names = {"train", "valid", "val", "test"}
        for path in sorted(input_path.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = path.parent.name
            split = ""
            try:
                rel_parts = path.relative_to(input_path).parts
            except ValueError:
                rel_parts = path.parts
            if len(rel_parts) >= 3 and rel_parts[0] in split_names:
                split = "valid" if rel_parts[0] == "val" else rel_parts[0]
                label = rel_parts[1]
            rows.append({"path": str(path), "label": label, "split": split})
        return rows
    raise FileNotFoundError(input_path)


def validate_expected_label_count(rows: List[Dict[str, str]], expected_label_count: int) -> None:
    if expected_label_count <= 0:
        return
    labels = sorted({row.get("label", "") for row in rows if row.get("label", "")})
    if labels and len(labels) != expected_label_count:
        raise RuntimeError(f"Expected {expected_label_count} labels, found {len(labels)}: {labels}")


def completed_paths(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") == "ok" and rec.get("path"):
                done.add(str(rec["path"]))
    return done


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json_array(jsonl_path: Path, json_output: Path) -> None:
    rows = load_jsonl(jsonl_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with json_output.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_summary(output_path: Path, summary_csv: Path) -> None:
    rows: List[Dict[str, Any]] = []
    if not output_path.exists():
        return
    for rec in load_jsonl(output_path):
        feature = rec.get("feature", {})
        core = feature.get("core_attributes", {})
        supporting = feature.get("supporting_features", {})
        row: Dict[str, Any] = {
            "path": rec.get("path", ""),
            "label": rec.get("label", ""),
            "split": rec.get("split", ""),
            "status": rec.get("status", ""),
            "api_status": feature.get("api_status", ""),
            "feature_status": feature.get("feature_status", {}).get("value", ""),
            "usable_for_feature_vector": feature.get("usable_for_feature_vector", ""),
            "failure_reason": feature.get("failure_reason", rec.get("error", "")),
            "mean_confidence": rec.get("mean_confidence", ""),
            "error": rec.get("error", ""),
        }
        for name in CORE_ATTRIBUTES:
            row[name] = core.get(name, {}).get("value", "")
        for name in FEATURE_NAMES:
            item = supporting.get(name, {})
            row[name] = item.get("state", "")
            row[f"{name}_confidence"] = item.get("confidence", "")
        rows.append(row)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    load_dotenv_if_available(args.env_file)
    output_path = Path(args.output)
    summary_csv = Path(args.summary_csv)
    json_output = Path(args.json_output) if args.json_output else None

    if args.write_prompt_only:
        prompt_path = Path(args.prompt_output)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(VLM_PROMPT, encoding="utf-8")
        print(f"Wrote prompt: {prompt_path}")
        return output_path

    input_path = Path(args.input)
    rows = discover_images(input_path, args.path_column)
    validate_expected_label_count(rows, args.expected_label_count)
    if args.start_row > 1:
        rows = rows[args.start_row - 1 :]
    if args.max_images > 0:
        rows = rows[: args.max_images]
    if not rows:
        raise RuntimeError("No images found.")

    done = completed_paths(output_path) if args.resume else set()
    pending = [row for row in rows if str(row["path"]) not in done]

    labels = sorted({row.get("label", "") for row in rows if row.get("label", "")})
    splits = sorted({row.get("split", "") for row in rows if row.get("split", "")})
    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    if json_output:
        print(f"JSON:    {json_output}")
    print(f"Rows: total={len(rows)} done={len(done)} pending={len(pending)}")
    print(f"Labels: {labels}")
    print(f"Splits: {splits}")

    if args.dry_run:
        print("Dry run only. No API calls were made.")
        return output_path

    thread_local = threading.local()
    write_lock = threading.Lock()

    def get_client() -> Any:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = create_azure_client(args)
            thread_local.client = client
        return client

    def process_row(index: int, row: Dict[str, str]) -> Dict[str, Any]:
        image_path = Path(row["path"])
        record_base = {
            "path": str(image_path),
            "label": row.get("label", ""),
            "split": row.get("split", ""),
            "original_idx": index,
            "provider": "azure",
            "azure_deployment": azure_deployment_name(args),
        }
        try:
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise RuntimeError(f"Unsupported image extension: {image_path}")

            feature: Optional[Dict[str, Any]] = None
            last_error: Optional[Exception] = None
            for attempt in range(1, args.retries + 2):
                try:
                    feature = call_vlm(get_client(), args, image_path)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt <= args.retries:
                        sleep_s = args.retry_sleep * attempt
                        print(
                            f"[retry] {index}/{len(pending)} attempt={attempt} "
                            f"sleep={sleep_s:.1f}s error={repr(exc)}"
                        )
                        time.sleep(sleep_s)
                    else:
                        raise last_error

            assert feature is not None
            vector, vector_names = feature_vector(feature)
            return {
                **record_base,
                "status": "ok",
                "feature": feature,
                "feature_vector_names": vector_names,
                "feature_vector": vector,
                "mean_confidence": mean_confidence(feature),
            }
        except Exception as exc:
            return {
                **record_base,
                "status": "error",
                "feature": default_response(api_status="error", failure_reason="low_confidence"),
                "error": repr(exc),
            }

    worker_count = max(1, int(args.workers))
    print(f"Workers: {worker_count}")

    def write_record(done_count: int, original_idx: int, record: Dict[str, Any]) -> None:
        with write_lock:
            append_jsonl(output_path, record)
        image_path = Path(record["path"])
        if record["status"] == "ok":
            feature = record["feature"]
            print(
                f"[ok] {done_count}/{len(pending)} row={original_idx} "
                f"feature_status={feature['feature_status']['value']} "
                f"usable={feature['usable_for_feature_vector']} "
                f"failure={feature['failure_reason']} path={image_path.name}"
            )
        else:
            print(
                f"[error] {done_count}/{len(pending)} row={original_idx} "
                f"path={image_path} error={record.get('error', '')}"
            )

    if worker_count == 1:
        for done_count, row in enumerate(pending, start=1):
            record = process_row(done_count, row)
            write_record(done_count, done_count, record)
            if args.sleep > 0:
                time.sleep(args.sleep)
            if args.summary_every > 0 and done_count % args.summary_every == 0:
                write_summary(output_path, summary_csv)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(process_row, idx, row): idx
                for idx, row in enumerate(pending, start=1)
            }
            for done_count, future in enumerate(as_completed(futures), start=1):
                original_idx = futures[future]
                record = future.result()
                write_record(done_count, original_idx, record)
                if args.summary_every > 0 and done_count % args.summary_every == 0:
                    write_summary(output_path, summary_csv)

    write_summary(output_path, summary_csv)
    print(f"Summary CSV: {summary_csv}")
    if json_output:
        write_json_array(output_path, json_output)
        print(f"JSON output: {json_output}")
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 13feature_21reason JSON for 7-label dental implant ROI radiographs."
    )
    parser.add_argument("--input", default="data/7label", help="Image path, image directory, or CSV manifest.")
    parser.add_argument("--path-column", default="path", help="Image path column when --input is a CSV.")
    parser.add_argument("--output", default="data/manifests/13feature_21reason_7label_all.jsonl")
    parser.add_argument("--json-output", default="data/manifests/13feature_21reason_7label_all.json")
    parser.add_argument("--summary-csv", default="data/manifests/13feature_21reason_7label_all.summary.csv")
    parser.add_argument("--prompt-output", default="data/manifests/13feature_21reason_7label.prompt.md")
    parser.add_argument("--expected-label-count", type=int, default=7)

    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-api-key", default=None)
    parser.add_argument("--azure-api-version", default=None)
    parser.add_argument("--azure-deployment", default=None)
    parser.add_argument("--env-file", default=None)

    parser.add_argument("--image-detail", choices=["low", "high", "auto"], default="high")
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--max-tokens", type=int, default=2600)
    parser.add_argument("--min-mean-confidence", type=float, default=0.0)

    parser.add_argument("--start-row", type=int, default=1, help="1-based row offset after input loading.")
    parser.add_argument("--max-images", type=int, default=0, help="Debug limit. 0 means all images.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel VLM API calls. 1 means sequential.")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--summary-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-prompt-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
