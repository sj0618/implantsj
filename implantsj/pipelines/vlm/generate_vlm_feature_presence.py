#!/usr/bin/env python3
"""
Generate dental implant ROI feature-presence JSONL with Azure OpenAI.

This is a standalone experiment script for the 7-label ROI dataset. It scans a
single image, an image directory, or a CSV manifest and writes one JSONL record
per image. Directory input such as data/7label is scanned recursively, so train,
valid, and test can be processed in one run.

Examples:
  # Write only the prompt/schema to inspect it.
  python generate_vlm_feature_presence.py --write-prompt-only

  # Run the full 7-label dataset with Azure OpenAI.
  python generate_vlm_feature_presence.py \
    --env-file .env \
    --input data/7label \
    --output data/manifests/vlm_feature_presence_7label_all.jsonl \
    --summary-csv data/manifests/vlm_feature_presence_7label_all.summary.csv \
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
import time
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
STATE_VALUES = {"present", "absent", "not_assessable"}
FAILURE_REASONS = {
    "none",
    "implant_not_visible",
    "connection_not_visible",
    "thread_not_visible",
    "low_confidence",
    "invalid_json",
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

THREAD_DEPENDENT_FEATURES = [
    "micro_thread_zone_present",
    "macro_thread_zone_present",
    "regular_thread_pitch_visible",
]

CONNECTION_DEPENDENT_FEATURES = [
    "positive_connection_region_present",
    "straight_screw_entry_space_present",
    "negative_internal_connection_space_present",
    "platform_step_present",
    "internal_small_void_present",
]

STRUCTURAL_FEATURE_NAMES = [
    name
    for name in FEATURE_NAMES
    if name
    not in {
        "implant_visible",
        "connection_region_visible",
        "thread_region_visible",
        "abutment_or_prosthesis_present",
    }
]


VLM_PROMPT = """You are analyzing a dental implant ROI radiograph.

Do not identify or guess implant brand or system.
Do not classify the implant type.
Only check whether each requested visual feature is present, absent, or not_assessable.

Definitions:
- present: the feature is directly visible.
- absent: the relevant region is clearly visible and the feature is not present.
- not_assessable: the region is hidden, cropped, blurred, overexposed, underexposed, or obscured by abutment/prosthesis.

Important:
- Do not infer absent when the region is not visible.
- If unsure between absent and not_assessable, choose not_assessable.
- Return only valid JSON.
- Do not use null.
- Every feature must have state and confidence.
- Confidence must be 0.0 to 1.0.

Assess features in this practical order: connection, platform, outline, thread.
Use thread spacing/form, platform switching, and connection location only as visual
evidence for feature presence. Do not convert them into brand/system/type labels.

Feature definitions:
- implant_visible: implant fixture is visible inside the ROI.
- connection_region_visible: platform/connection region is visible.
- thread_region_visible: thread region is visible.
- abutment_or_prosthesis_present: an upper structure or prosthesis is connected.
- positive_connection_region_present: connection region protrudes as a positive/external-looking structure.
- straight_screw_entry_space_present: screw entry space appears straight.
- negative_internal_connection_space_present: connection region appears as a negative/internal recessed space.
- platform_step_present: a step is visible at the platform region.
- internal_small_void_present: a small internal void is visible below the platform or around the screw channel.
- smooth_no_thread_collar_present: a smooth coronal collar without threads is present.
- machined_surface_present: a long smooth/machined-surface zone is present.
- platform_boundary_visible: platform boundary is visible.
- platform_switching_step_present: width difference or step between abutment and fixture is visible.
- thread_lines_visible: thread lines are clearly visible.
- regular_thread_pitch_visible: regular thread spacing is observable.
- deep_thread_relief_present: deep thread relief/cut is visible.
- micro_thread_zone_present: narrow/fine coronal micro-thread zone is visible.
- macro_thread_zone_present: coarser macro-thread zone is visible.
- body_taper_present: implant body narrows toward the apex.
- parallel_body_wall_present: implant body side walls are nearly parallel.
- apex_boundary_visible: apical lower boundary is visible.

Allowed JSON schema:
{
  "api_status": "ok / error",
  "feature_detection": {
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
  "usable_for_feature_vector": true,
  "failure_reason": "none / implant_not_visible / connection_not_visible / thread_not_visible / low_confidence / invalid_json"
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


def feature_default(state: str = "not_assessable", confidence: float = 0.0) -> Dict[str, Any]:
    return {"state": state, "confidence": float(confidence)}


def default_response(api_status: str = "error", failure_reason: str = "invalid_json") -> Dict[str, Any]:
    return {
        "api_status": api_status,
        "feature_detection": {name: feature_default() for name in FEATURE_NAMES},
        "usable_for_feature_vector": False,
        "failure_reason": failure_reason,
    }


def normalize_state(value: Any) -> str:
    if value is None:
        return "not_assessable"
    value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if value in STATE_VALUES:
        return value
    if value in {"unknown", "uncertain", "not_visible", "hidden", "blurred", "na", "n/a"}:
        return "not_assessable"
    if value in {"yes", "true", "1", "visible"}:
        return "present"
    if value in {"no", "false", "0", "not_present"}:
        return "absent"
    return "not_assessable"


def clamp_confidence(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    return value in {"true", "1", "yes", "y"}


def validate_and_complete_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = default_response(api_status="ok", failure_reason="none")
    out["api_status"] = "ok" if str(raw.get("api_status", "ok")).lower() == "ok" else "error"

    raw_features = raw.get("feature_detection", {})
    if not isinstance(raw_features, dict):
        raw_features = {}

    for name in FEATURE_NAMES:
        item = raw_features.get(name, {})
        if not isinstance(item, dict):
            item = {}
        out["feature_detection"][name] = {
            "state": normalize_state(item.get("state")),
            "confidence": clamp_confidence(item.get("confidence", 0.0)),
        }

    out["usable_for_feature_vector"] = normalize_bool(raw.get("usable_for_feature_vector", True))
    failure_reason = str(raw.get("failure_reason", "none")).strip().lower()
    if failure_reason not in FAILURE_REASONS:
        failure_reason = "invalid_json" if out["api_status"] == "error" else "none"
    out["failure_reason"] = failure_reason
    return out


def infer_failure_reason(feature: Dict[str, Any], min_mean_confidence: float) -> Dict[str, Any]:
    features = feature["feature_detection"]

    if features["thread_region_visible"]["state"] != "present":
        for name in THREAD_DEPENDENT_FEATURES:
            features[name]["state"] = "not_assessable"

    if features["connection_region_visible"]["state"] != "present":
        for name in CONNECTION_DEPENDENT_FEATURES:
            features[name]["state"] = "not_assessable"

    if (
        features["abutment_or_prosthesis_present"]["state"] == "present"
        and features["connection_region_visible"]["state"] != "present"
    ):
        for name in CONNECTION_DEPENDENT_FEATURES:
            features[name]["state"] = "not_assessable"

    if features["implant_visible"]["state"] != "present":
        feature["usable_for_feature_vector"] = False
        feature["failure_reason"] = "implant_not_visible"
        return feature

    if all(features[name]["state"] == "not_assessable" for name in STRUCTURAL_FEATURE_NAMES):
        feature["usable_for_feature_vector"] = False
        feature["failure_reason"] = "low_confidence"
        return feature

    mean_conf = mean_confidence(feature)
    if min_mean_confidence > 0 and mean_conf < min_mean_confidence:
        feature["usable_for_feature_vector"] = False
        feature["failure_reason"] = "low_confidence"
        return feature

    if feature["failure_reason"] == "none":
        feature["usable_for_feature_vector"] = True
    return feature


def mean_confidence(feature: Dict[str, Any]) -> float:
    vals = [
        float(feature["feature_detection"][name].get("confidence", 0.0))
        for name in FEATURE_NAMES
    ]
    return sum(vals) / max(1, len(vals))


def feature_vector(feature: Dict[str, Any], include_confidence: bool = True) -> Tuple[List[float], List[str]]:
    values: List[float] = []
    names: List[str] = []
    for feature_name in FEATURE_NAMES:
        item = feature["feature_detection"][feature_name]
        state = item["state"]
        for state_name in ("present", "absent", "not_assessable"):
            values.append(1.0 if state == state_name else 0.0)
            names.append(f"{feature_name}={state_name}")
        if include_confidence:
            values.append(float(item["confidence"]))
            names.append(f"{feature_name}.confidence")
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

    return AzureOpenAI(
        azure_endpoint=endpoint.rstrip("/"),
        api_key=api_key,
        api_version=api_version,
    )


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
                "You are a careful dental radiograph feature detection assistant. "
                "Never identify implant brand, system, manufacturer, or implant type. "
                "Return only valid JSON."
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
    feature = validate_and_complete_response(raw)
    return infer_failure_reason(feature, args.min_mean_confidence)


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
            rows.append(
                {
                    "path": image_path,
                    "label": row.get("label", ""),
                    "split": row.get("split", ""),
                }
            )
    return rows


def discover_images(input_path: Path, path_column: str) -> List[Dict[str, str]]:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return read_manifest(input_path, path_column)
    if input_path.is_file():
        return [{"path": str(input_path), "label": "", "split": ""}]
    if input_path.is_dir():
        rows = []
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


def write_summary(output_path: Path, summary_csv: Path) -> None:
    rows: List[Dict[str, Any]] = []
    if not output_path.exists():
        return
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            feature = rec.get("feature", {})
            detections = feature.get("feature_detection", {})
            row = {
                "path": rec.get("path", ""),
                "label": rec.get("label", ""),
                "split": rec.get("split", ""),
                "status": rec.get("status", ""),
                "api_status": feature.get("api_status", ""),
                "usable_for_feature_vector": feature.get("usable_for_feature_vector", ""),
                "failure_reason": feature.get("failure_reason", rec.get("error", "")),
                "mean_confidence": rec.get("mean_confidence", ""),
                "error": rec.get("error", ""),
            }
            for name in FEATURE_NAMES:
                item = detections.get(name, {})
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

    if args.write_prompt_only:
        prompt_path = Path(args.prompt_output)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(VLM_PROMPT, encoding="utf-8")
        print(f"Wrote prompt: {prompt_path}")
        return output_path

    input_path = Path(args.input)
    rows = discover_images(input_path, args.path_column)
    if args.start_row > 1:
        rows = rows[args.start_row - 1 :]
    if args.max_images > 0:
        rows = rows[: args.max_images]
    if not rows:
        raise RuntimeError("No images found.")

    done = completed_paths(output_path) if args.resume else set()
    pending = [row for row in rows if str(row["path"]) not in done]

    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Rows: total={len(rows)} done={len(done)} pending={len(pending)}")

    if args.dry_run:
        print("Dry run only. No API calls were made.")
        return output_path

    thread_local = threading.local()

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
            vector, vector_names = feature_vector(feature, include_confidence=True)
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
                "feature": default_response(api_status="error", failure_reason="invalid_json"),
                "error": repr(exc),
            }

    worker_count = max(1, int(args.workers))
    print(f"Workers: {worker_count}")

    if worker_count == 1:
        completed_iter = (process_row(idx, row) for idx, row in enumerate(pending, start=1))
        for idx, record in enumerate(completed_iter, start=1):
            append_jsonl(output_path, record)
            image_path = Path(record["path"])
            if record["status"] == "ok":
                feature = record["feature"]
                print(
                    f"[ok] {idx}/{len(pending)} usable={feature['usable_for_feature_vector']} "
                    f"failure={feature['failure_reason']} path={image_path.name}"
                )
            else:
                print(f"[error] {idx}/{len(pending)} path={image_path} error={record.get('error', '')}")
            if args.sleep > 0:
                time.sleep(args.sleep)
            if args.summary_every > 0 and idx % args.summary_every == 0:
                write_summary(output_path, summary_csv)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(process_row, idx, row): idx
                for idx, row in enumerate(pending, start=1)
            }
            for done_count, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                append_jsonl(output_path, record)
                original_idx = futures[future]
                image_path = Path(record["path"])
                if record["status"] == "ok":
                    feature = record["feature"]
                    print(
                        f"[ok] {done_count}/{len(pending)} row={original_idx} "
                        f"usable={feature['usable_for_feature_vector']} "
                        f"failure={feature['failure_reason']} path={image_path.name}"
                    )
                else:
                    print(
                        f"[error] {done_count}/{len(pending)} row={original_idx} "
                        f"path={image_path} error={record.get('error', '')}"
                    )
                if args.summary_every > 0 and done_count % args.summary_every == 0:
                    write_summary(output_path, summary_csv)

    write_summary(output_path, summary_csv)
    print(f"Summary CSV: {summary_csv}")
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VLM feature-presence JSONL for dental implant ROI radiographs."
    )
    parser.add_argument("--input", default="data/7label", help="Image path, image directory, or CSV manifest.")
    parser.add_argument("--path-column", default="path", help="Image path column when --input is a CSV.")
    parser.add_argument("--output", default="data/manifests/vlm_feature_presence_7label_all.jsonl")
    parser.add_argument("--summary-csv", default="data/manifests/vlm_feature_presence_7label_all.summary.csv")
    parser.add_argument("--prompt-output", default="data/manifests/vlm_feature_presence.prompt.md")

    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-api-key", default=None)
    parser.add_argument("--azure-api-version", default=None)
    parser.add_argument("--azure-deployment", default=None)
    parser.add_argument("--env-file", default=None)

    parser.add_argument("--image-detail", choices=["low", "high", "auto"], default="high")
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--max-tokens", type=int, default=1800)
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
