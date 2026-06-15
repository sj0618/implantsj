#!/usr/bin/env python3
"""
Generate VLM 5-feature JSONL for every image in a 6-label implant dataset
using Azure OpenAI vision-capable chat deployments.

Output JSONL is directly compatible with train_vlm_fusion_5feature.py.
Each line contains:
  path,label,split,feature,vlm_feature_vector,vlm_feature_names,status
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import IMAGE_EXTENSIONS, build_manifest, detect_layout, load_manifest, make_stratified_splits
from train_vlm_fusion_5feature import (
    DEFAULT_FEATURE_GROUPS,
    VLM_PROMPT,
    feature_names,
    merge_default_features,
    vectorize_vlm_features,
)


DEFAULT_OUTPUT = ROOT / "data" / "manifests" / "vlm_5feature_6label.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "plain_vit_6label.csv"


def load_dotenv_if_available(path: Optional[str]) -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    if path:
        load_dotenv(path)
    else:
        load_dotenv(ROOT / ".env")
        load_dotenv(Path.cwd() / ".env")


def normalize_labels(labels: Optional[List[str]]) -> Optional[List[str]]:
    if labels is None:
        return None
    return [str(x) for x in labels]


def filter_labels(df: pd.DataFrame, labels: Optional[List[str]], exclude_labels: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].astype(str)
    if labels:
        allowed = set(labels)
        df = df[df["label"].isin(allowed)].copy()
        missing = sorted(allowed - set(df["label"].unique().tolist()))
        if missing:
            raise RuntimeError(f"Requested labels not found: {missing}")
    if exclude_labels:
        blocked = set(str(x) for x in exclude_labels)
        df = df[~df["label"].isin(blocked)].copy()
    if df.empty:
        raise RuntimeError("No rows left after label filtering.")
    return df.reset_index(drop=True)


def validate_label_manifest(df: pd.DataFrame, expected_label_count: int) -> None:
    labels = sorted(df["label"].astype(str).unique().tolist())
    if len(labels) != expected_label_count:
        raise RuntimeError(
            f"Expected exactly {expected_label_count} labels, got {len(labels)}: {labels}. "
            "Use --labels or --exclude-labels to select the target classes."
        )
    if "split" not in df.columns:
        raise RuntimeError("Manifest must include split column.")
    split_values = set(df["split"].astype(str).unique().tolist())
    if "train" not in split_values or "valid" not in split_values:
        raise RuntimeError(f"Manifest must include at least train and valid splits. Found: {sorted(split_values)}")


def build_or_load_6label_manifest(args: argparse.Namespace) -> Path:
    manifest = Path(args.manifest)
    labels = normalize_labels(args.labels)
    exclude_labels = normalize_labels(args.exclude_labels) or []

    if manifest.exists() and not args.rebuild_manifest and not labels and not exclude_labels:
        df = load_manifest(manifest, split=None)
        validate_label_manifest(df, args.expected_label_count)
        return manifest

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Missing data root: {data_root}. Pass --data-root or pass an existing --manifest."
        )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    layout = detect_layout(data_root)
    if layout == "split_class_folder":
        df = build_manifest(data_root, manifest, layout="split_class_folder")
        df = filter_labels(df, labels, exclude_labels)
        validate_label_manifest(df, args.expected_label_count)
        df.to_csv(manifest, index=False)
        return manifest

    unsplit = manifest.with_name(manifest.stem + "_unsplit.csv")
    filtered = manifest.with_name(manifest.stem + "_filtered_unsplit.csv")
    df = build_manifest(data_root, unsplit, layout="class_folder")
    df = filter_labels(df, labels, exclude_labels)
    df.to_csv(filtered, index=False)
    out_df = make_stratified_splits(
        filtered,
        manifest,
        train=args.train_ratio,
        valid=args.valid_ratio,
        test=args.test_ratio,
        seed=args.seed,
    )
    validate_label_manifest(out_df, args.expected_label_count)
    return manifest


def image_to_data_url(path: str | Path, max_side: int, jpeg_quality: int) -> str:
    path = Path(path)
    with Image.open(path) as img:
        img = img.convert("RGB")
        if max_side > 0:
            w, h = img.size
            scale = min(1.0, float(max_side) / float(max(w, h)))
            if scale < 1.0:
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def create_azure_client(args: argparse.Namespace):
    from urllib.parse import urlparse

    from openai import AzureOpenAI, OpenAI

    endpoint = (args.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
    api_key = args.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
    api_version = args.azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION")
    if not endpoint:
        raise RuntimeError("Missing Azure endpoint. Set AZURE_OPENAI_ENDPOINT or pass --azure-endpoint.")
    if not api_key:
        raise RuntimeError("Missing Azure API key. Set AZURE_OPENAI_API_KEY or pass --azure-api-key.")

    parsed = urlparse(endpoint)
    if parsed.path.rstrip("/").endswith("/openai/v1"):
        client = OpenAI(api_key=api_key, base_url=endpoint.rstrip("/") + "/", default_headers={"api-key": api_key})
        client._implant_endpoint_mode = "openai_v1"
        return client

    if not api_version:
        raise RuntimeError("Missing API version. Set AZURE_OPENAI_API_VERSION or pass --azure-api-version.")
    client = AzureOpenAI(azure_endpoint=endpoint.rstrip("/"), api_key=api_key, api_version=api_version)
    client._implant_endpoint_mode = "azure_deployments"
    return client


def call_vlm(client: Any, args: argparse.Namespace, image_path: str) -> Dict[str, Any]:
    deployment = args.azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv(
        "AZURE_OPENAI_CHAT_DEPLOYMENT"
    )
    if not deployment:
        raise RuntimeError(
            "Missing Azure deployment name. Set AZURE_OPENAI_DEPLOYMENT or pass --azure-deployment."
        )

    data_url = image_to_data_url(image_path, max_side=args.max_image_side, jpeg_quality=args.jpeg_quality)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful dental radiograph feature extraction assistant. "
                "Do not infer implant brand or system class. Return JSON only."
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
        model=deployment,
        messages=messages,
        max_completion_tokens=args.max_tokens,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    feature = extract_json_object(content)
    return merge_default_features(feature)


def completed_paths(output_path: Path) -> Set[str]:
    done: Set[str] = set()
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


def manifest_rows(args: argparse.Namespace, manifest: Path) -> pd.DataFrame:
    df = load_manifest(manifest, split=None)
    if args.splits:
        allowed = set(args.splits)
        df = df[df["split"].astype(str).isin(allowed)].copy()
    if args.start_row > 1:
        df = df.iloc[args.start_row - 1 :].copy()
    if args.max_images > 0:
        df = df.head(args.max_images).copy()
    return df.reset_index(drop=True)


def summarize_output(output_path: Path, summary_csv: Path) -> None:
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
            rows.append(
                {
                    "path": rec.get("path", ""),
                    "label": rec.get("label", ""),
                    "split": rec.get("split", ""),
                    "status": rec.get("status", ""),
                    "connection_type": rec.get("feature", {}).get("connection_type", {}).get("value", ""),
                    "connection_confidence": rec.get("feature", {}).get("connection_type", {}).get("confidence", ""),
                    "platform_switching": rec.get("feature", {}).get("platform", {}).get("platform_switching", ""),
                    "thread_pitch": rec.get("feature", {}).get("thread", {}).get("pitch", ""),
                    "thread_visibility": rec.get("feature", {}).get("image_quality", {}).get("thread_visibility", ""),
                    "mean_vlm_confidence": rec.get("mean_vlm_confidence", ""),
                    "error": rec.get("error", ""),
                }
            )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)


def mean_confidence(feature: Dict[str, Any]) -> float:
    vals = [
        feature.get("connection_type", {}).get("confidence", 0.0),
        feature.get("platform", {}).get("confidence", 0.0),
        feature.get("thread", {}).get("confidence", 0.0),
        feature.get("outline", {}).get("confidence", 0.0),
        feature.get("image_quality", {}).get("confidence", 0.0),
    ]
    nums = []
    for value in vals:
        try:
            nums.append(float(value))
        except Exception:
            nums.append(0.0)
    return float(sum(nums) / max(1, len(nums)))


def generate_features(args: argparse.Namespace) -> Path:
    load_dotenv_if_available(args.env_file)
    output_path = Path(args.output)
    summary_csv = Path(args.summary_csv)

    if args.write_prompt_only:
        prompt_path = output_path.with_suffix(".prompt.md")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(VLM_PROMPT, encoding="utf-8")
        print(f"Wrote prompt: {prompt_path}")
        return output_path

    manifest = build_or_load_6label_manifest(args)
    df = manifest_rows(args, manifest)
    if df.empty:
        raise RuntimeError("No rows to process.")

    done = completed_paths(output_path) if args.resume else set()
    pending = df[~df["path"].astype(str).isin(done)].reset_index(drop=True)

    print(f"Manifest: {manifest}")
    print(f"Output:   {output_path}")
    print(f"Rows: total={len(df)} done={len(done)} pending={len(pending)}")
    print("Labels:", sorted(df["label"].astype(str).unique().tolist()))
    print("Splits:", df["split"].astype(str).value_counts().sort_index().to_dict())

    if args.dry_run:
        print("Dry run only. No API calls were made.")
        return output_path

    client = create_azure_client(args)
    names = feature_names(DEFAULT_FEATURE_GROUPS)

    for idx, row in pending.iterrows():
        image_path = str(row["path"])
        label = str(row["label"])
        split = str(row["split"])
        record_base = {
            "path": image_path,
            "label": label,
            "split": split,
            "source_manifest": str(manifest),
            "azure_deployment": args.azure_deployment
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        }

        try:
            if not Path(image_path).exists():
                raise FileNotFoundError(image_path)
            if Path(image_path).suffix.lower() not in IMAGE_EXTENSIONS:
                raise RuntimeError(f"Unsupported image extension: {image_path}")

            feature: Optional[Dict[str, Any]] = None
            last_error: Optional[Exception] = None
            for attempt in range(1, args.retries + 2):
                try:
                    feature = call_vlm(client, args, image_path)
                    break
                except Exception as e:
                    last_error = e
                    if attempt <= args.retries:
                        sleep_s = args.retry_sleep * attempt
                        print(f"[retry] {idx + 1}/{len(pending)} attempt={attempt} sleep={sleep_s:.1f}s error={repr(e)}")
                        time.sleep(sleep_s)
                    else:
                        raise last_error

            assert feature is not None
            vector, flat = vectorize_vlm_features(feature, DEFAULT_FEATURE_GROUPS)
            record = {
                **record_base,
                "status": "ok",
                "feature": feature,
                "flat_feature": flat,
                "vlm_feature_names": names,
                "vlm_feature_vector": [float(x) for x in vector.tolist()],
                "mean_vlm_confidence": mean_confidence(feature),
            }
            append_jsonl(output_path, record)
            print(
                f"[ok] {idx + 1}/{len(pending)} split={split} label={label} "
                f"path={Path(image_path).name}"
            )
        except Exception as e:
            record = {**record_base, "status": "error", "error": repr(e)}
            append_jsonl(output_path, record)
            print(f"[error] {idx + 1}/{len(pending)} path={image_path} error={repr(e)}")

        if args.sleep > 0:
            time.sleep(args.sleep)
        if args.summary_every > 0 and (idx + 1) % args.summary_every == 0:
            summarize_output(output_path, summary_csv)

    summarize_output(output_path, summary_csv)
    print(f"Summary CSV: {summary_csv}")
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Azure OpenAI VLM 5-feature JSONL for implant label data.")
    p.add_argument("--data-root", default="/workspace/data/large_multiclass")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--summary-csv", default=str(DEFAULT_OUTPUT.with_suffix(".summary.csv")))
    p.add_argument("--labels", nargs="+", default=None, help="Exact labels to keep.")
    p.add_argument("--expected-label-count", type=int, default=6)
    p.add_argument("--exclude-labels", nargs="+", default=[], help="Labels to drop before validating 6 labels.")
    p.add_argument("--rebuild-manifest", action="store_true")
    p.add_argument("--train-ratio", type=float, default=8)
    p.add_argument("--valid-ratio", type=float, default=1)
    p.add_argument("--test-ratio", type=float, default=1)
    p.add_argument("--splits", nargs="+", default=None, help="Optional subset: train valid test")
    p.add_argument("--start-row", type=int, default=1, help="1-based row offset after split filtering.")
    p.add_argument("--max-images", type=int, default=0, help="Debug limit. 0 means all rows.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--env-file", default=None)
    p.add_argument("--azure-endpoint", default=None)
    p.add_argument("--azure-api-key", default=None)
    p.add_argument("--azure-api-version", default=None)
    p.add_argument("--azure-deployment", default=None)

    p.add_argument("--image-detail", choices=["low", "high", "auto"], default="high")
    p.add_argument("--max-image-side", type=int, default=1024)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--max-tokens", type=int, default=900)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=2.0)
    p.add_argument("--summary-every", type=int, default=25)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--write-prompt-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_features(args)


if __name__ == "__main__":
    main()
