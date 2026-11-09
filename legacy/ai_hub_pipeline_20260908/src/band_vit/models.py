"""ViT classifier construction, checkpoints and band-attention capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _imports():
    try:
        import timm
        import torch
        from timm.data import create_transform, resolve_model_data_config
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("model execution requires torch and timm") from exc
    return torch, timm, create_transform, resolve_model_data_config


def choose_device(requested: str = "auto") -> str:
    torch, _, _, _ = _imports()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state.items():
        cleaned = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model."):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :]
                    changed = True
        result[cleaned] = value
    return result


def checkpoint_payload(
    path: str | Path,
    device: str = "cpu",
    checkpoint_format: str = "auto",
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_format == "auto":
        checkpoint_format = (
            "safetensors" if checkpoint_path.suffix == ".safetensors" else "torch"
        )
    if checkpoint_format == "ckpt":
        checkpoint_format = "torch"
    if checkpoint_format == "safetensors":
        from safetensors.torch import load_file

        metadata_path = checkpoint_path.with_suffix(".json")
        payload = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        payload["model_state_dict"] = load_file(str(checkpoint_path), device=device)
        return payload
    if checkpoint_format != "torch":
        raise ValueError("checkpoint_format must be auto, ckpt, torch or safetensors")

    torch, _, _, _ = _imports()
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded, dict):
        return loaded
    return {"model_state_dict": loaded}


def state_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    if payload and all(hasattr(value, "shape") for value in payload.values()):
        return payload
    raise ValueError("checkpoint has no model state mapping")


def _load_compatible_weights(model, payload: Mapping[str, Any]) -> tuple[int, int]:
    source = clean_state_dict(state_from_payload(payload))
    candidates = [
        source,
        {key.removeprefix("backbone."): value for key, value in source.items()},
    ]
    target = model.state_dict()
    compatible = max(
        (
            {
                key: value
                for key, value in candidate.items()
                if key in target and tuple(value.shape) == tuple(target[key].shape)
            }
            for candidate in candidates
        ),
        key=len,
    )
    if not compatible:
        raise ValueError("checkpoint has no weights compatible with the model")
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(target) - len(compatible)


def build_classifier(
    architecture: str,
    num_classes: int,
    device: str,
    *,
    pretrained: bool = True,
    initial_checkpoint: str | Path | None = None,
    checkpoint_format: str = "auto",
):
    _, timm, _, _ = _imports()
    model = timm.create_model(
        architecture,
        pretrained=pretrained and not initial_checkpoint,
        num_classes=num_classes,
    )
    if initial_checkpoint:
        payload = checkpoint_payload(
            initial_checkpoint, "cpu", checkpoint_format=checkpoint_format
        )
        _load_compatible_weights(model, payload)
    return model.to(device)


def load_classifier_checkpoint(
    path: str | Path,
    device: str,
    *,
    architecture: str | None = None,
    checkpoint_format: str = "auto",
):
    _, timm, _, _ = _imports()
    payload = checkpoint_payload(path, "cpu", checkpoint_format)
    labels = [str(value) for value in payload.get("labels", [])]
    if not labels and payload.get("label_to_index"):
        labels = [
            label
            for label, _ in sorted(
                payload["label_to_index"].items(), key=lambda item: int(item[1])
            )
        ]
    resolved_architecture = architecture or payload.get("architecture")
    num_classes = int(payload.get("num_classes") or len(labels))
    if not resolved_architecture:
        raise ValueError("checkpoint has no architecture; pass --model")
    if num_classes <= 0:
        raise ValueError("checkpoint has no labels or num_classes")

    model = timm.create_model(
        resolved_architecture,
        pretrained=False,
        num_classes=num_classes,
    )
    state = clean_state_dict(state_from_payload(payload))
    missing, unexpected = model.load_state_dict(state, strict=False)
    meaningful_missing = [
        key for key in missing if not key.endswith("num_batches_tracked")
    ]
    if meaningful_missing or unexpected:
        raise ValueError(
            "checkpoint does not match classifier; "
            f"missing={meaningful_missing[:8]}, unexpected={list(unexpected)[:8]}"
        )
    metadata = {
        **{
            key: value
            for key, value in payload.items()
            if not key.endswith("state_dict")
        },
        "architecture": resolved_architecture,
        "num_classes": num_classes,
        "labels": labels,
        "checkpoint": str(Path(path).expanduser().resolve()),
    }
    return model.to(device), metadata


def create_transforms(model, input_size: int):
    _, _, _, resolve_model_data_config = _imports()
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    from .data import SquarePad

    config = resolve_model_data_config(model)
    mean = tuple(config["mean"])
    std = tuple(config["std"])
    resize = transforms.Resize(
        (input_size, input_size), interpolation=InterpolationMode.BILINEAR
    )
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            SquarePad(fill=0),
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=10, interpolation=InterpolationMode.BILINEAR, fill=0
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            SquarePad(fill=0),
            resize,
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform, mean, std


def patch_grid_size(model) -> tuple[int, int]:
    patch_embed = getattr(model, "patch_embed", None)
    grid = getattr(patch_embed, "grid_size", None)
    if grid is None:
        raise TypeError(
            "band attention requires a ViT-style model with patch_embed.grid_size"
        )
    if isinstance(grid, int):
        return grid, grid
    if len(grid) != 2:
        raise ValueError(f"expected a 2D patch grid, got {grid!r}")
    return int(grid[0]), int(grid[1])


def attach_attention_capture(model, num_blocks: int = 6):
    blocks = getattr(model, "blocks", None)
    if blocks is None or not len(blocks):
        raise TypeError(
            "band attention requires a ViT-style model with transformer blocks"
        )
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")

    for handle in getattr(model, "_band_attention_handles", []):
        handle.remove()
    stores: list[dict[str, Any]] = []
    handles = []
    for block in blocks[-min(num_blocks, len(blocks)) :]:
        attention = getattr(block, "attn", None)
        attention_drop = getattr(attention, "attn_drop", None)
        if attention_drop is None:
            raise TypeError(
                "band attention requires transformer blocks with attn.attn_drop"
            )
        if hasattr(attention, "fused_attn"):
            attention.fused_attn = False
        store: dict[str, Any] = {}

        def capture(_module, inputs, _output, *, target=store):
            target["attention"] = inputs[0]

        handles.append(attention_drop.register_forward_hook(capture))
        stores.append(store)
    model._band_attention_stores = stores
    model._band_attention_handles = handles
    return model


def get_cls_patch_attention(model):
    torch, _, _, _ = _imports()
    stores = getattr(model, "_band_attention_stores", None)
    if not stores:
        raise RuntimeError("attention capture is not attached to this model")
    prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    values = []
    for store in stores:
        attention = store.get("attention")
        if attention is None:
            raise RuntimeError("run a model forward pass before reading attention")
        if attention.ndim != 4:
            raise ValueError(
                f"expected attention [batch, heads, tokens, tokens], got {attention.shape}"
            )
        values.append(attention.mean(dim=1)[:, 0, prefix_tokens:])
    result = torch.stack(values).mean(dim=0)
    return result / result.sum(dim=1, keepdim=True).clamp_min(1e-8)


def save_initial_checkpoint(
    architecture: str,
    output_path: str | Path,
    *,
    pretrained: bool = True,
    overwrite: bool = False,
) -> Path:
    torch, timm, _, _ = _imports()
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"initial checkpoint exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = timm.create_model(architecture, pretrained=pretrained)
    torch.save(model.state_dict(), output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default="vit_base_patch16_224")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = save_initial_checkpoint(
        args.architecture,
        args.output,
        pretrained=args.pretrained,
        overwrite=args.overwrite,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
