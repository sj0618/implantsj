You are analyzing a dental implant ROI radiograph.
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
