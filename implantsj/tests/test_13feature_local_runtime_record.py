#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.vlm import generate_13feature_21reason_7label as mod


def make_args(model_path: Path) -> Namespace:
    return Namespace(
        local_model_path=str(model_path),
        local_torch_dtype="auto",
        local_device_map="auto",
        azure_endpoint=None,
        azure_api_key=None,
        azure_api_version=None,
        azure_deployment=None,
        min_mean_confidence=0.0,
    )


def test_local_qwen_record_construction() -> None:
    with tempfile.TemporaryDirectory(prefix="local_qwen_record_") as tmp:
        tmp_dir = Path(tmp)
        model_path = tmp_dir / "qwen_3B_VLM"
        image_path = tmp_dir / "sample.png"
        image_path.write_bytes(b"fake")

        args = make_args(model_path)
        row = {"path": str(image_path), "label": "L1", "split": "train"}

        raw = {
            "core_attributes": {},
            "supporting_features": {
                "implant_visible": {"state": "present", "confidence": 0.95},
                "connection_region_visible": {"state": "present", "confidence": 0.90},
                "thread_region_visible": {"state": "present", "confidence": 0.90},
                "negative_internal_connection_space_present": {"state": "present", "confidence": 0.80},
                "platform_step_present": {"state": "present", "confidence": 0.80},
                "internal_small_void_present": {"state": "present", "confidence": 0.80},
                "micro_thread_zone_present": {"state": "present", "confidence": 0.80},
                "thread_lines_visible": {"state": "present", "confidence": 0.85},
                "apex_boundary_visible": {"state": "present", "confidence": 0.90},
                "body_taper_present": {"state": "present", "confidence": 0.80},
                "parallel_body_wall_present": {"state": "absent", "confidence": 0.80},
                "smooth_no_thread_collar_present": {"state": "absent", "confidence": 0.80},
                "machined_surface_present": {"state": "absent", "confidence": 0.80},
            },
        }

        feature = mod.normalize_generated_json_text(json.dumps(raw), 0.0)
        record_base = mod.build_row_record_base(args, 1, row, image_path)
        success_record = mod.build_success_record(record_base, feature)

        assert success_record["provider"] == "qwen_3B_VLM"
        assert success_record["local_model_path"] == str(model_path)
        assert "azure_deployment" not in success_record
        assert success_record["status"] == "ok"
        assert success_record["path"] == str(image_path)
        assert success_record["label"] == "L1"
        assert success_record["split"] == "train"
        assert success_record["feature"]["feature_status"]["value"] == "usable"
        assert success_record["feature"]["usable_for_feature_vector"] is True
        assert success_record["feature"]["core_attributes"]["Connection_1"]["value"] == "Internal"
        assert success_record["feature"]["core_attributes"]["Microthread"]["value"] == "Yes"
        assert success_record["feature"]["core_attributes"]["Body_Type"]["value"] == "Threaded"
        assert len(success_record["feature_vector"]) == 63
        assert len(success_record["feature_vector_names"]) == 63

        err = RuntimeError("boom")
        error_record = mod.build_error_record(record_base, err)
        assert error_record["provider"] == "qwen_3B_VLM"
        assert error_record["local_model_path"] == str(model_path)
        assert "azure_deployment" not in error_record
        assert error_record["status"] == "error"
        assert "boom" in error_record["error"]
        assert error_record["feature"]["api_status"] == "error"

    print("OK: local qwen 13feature record construction test passed")


if __name__ == "__main__":
    test_local_qwen_record_construction()
