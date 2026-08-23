"""Swappable model specifications and embedding adapters."""

from __future__ import annotations

import abc
import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ModelSpec:
    name: str
    adapter: str = "timm_pretrained"
    architecture: str = "vit_base_patch16_224"
    checkpoint: str | None = None
    checkpoint_format: str = "auto"
    initial_checkpoint: str | None = None
    pretrained: bool = True
    training_kind: str = "none"
    train_scope: str = "all"
    embedding_source: str = "backbone"
    input_size: int = 224
    projection_dim: int = 128
    projection_hidden_dim: int = 512
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, default_name: str | None = None
    ) -> "ModelSpec":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        payload = dict(value)
        if default_name and "name" not in payload:
            payload["name"] = default_name
        extra = dict(payload.pop("extra", {}))
        for key in list(payload):
            if key not in known:
                extra[key] = payload.pop(key)
        payload["extra"] = extra
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_model_spec(path: str | Path, model_name: str | None = None) -> ModelSpec:
    spec_path = Path(path).expanduser().resolve()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    selected: Mapping[str, Any]
    if "models" in payload:
        models = payload["models"]
        if isinstance(models, list):
            if model_name is None:
                if len(models) != 1:
                    raise ValueError(
                        "--model-name is required when the file contains multiple models"
                    )
                selected = models[0]
            else:
                matches = [item for item in models if item.get("name") == model_name]
                if len(matches) != 1:
                    raise KeyError(
                        f"model {model_name!r} not found exactly once in {spec_path}"
                    )
                selected = matches[0]
        elif isinstance(models, dict):
            if model_name is None:
                if len(models) != 1:
                    raise ValueError(
                        "--model-name is required when the file contains multiple models"
                    )
                model_name, selected = next(iter(models.items()))
            else:
                selected = models[model_name]
        else:
            raise TypeError("models must be a list or object")
    else:
        selected = payload
    spec = ModelSpec.from_mapping(selected, default_name=model_name)
    resolved = spec.to_dict()
    for field_name in ("checkpoint", "initial_checkpoint"):
        value = resolved.get(field_name)
        if value and not Path(value).is_absolute():
            resolved[field_name] = str((spec_path.parent / value).resolve())
    spec = ModelSpec.from_mapping(resolved)
    return spec


def _imports():
    try:
        import torch
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("model execution requires torch and timm") from exc
    return torch, timm, create_transform, resolve_model_data_config


def clean_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    prefixes = ("module.", "model.")
    result: dict[str, Any] = {}
    for key, value in state.items():
        cleaned = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :]
                    changed = True
        result[cleaned] = value
    return result


def checkpoint_payload(
    path: str | Path, device: str = "cpu", checkpoint_format: str = "auto"
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
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
        if not isinstance(payload, dict):
            raise TypeError(f"checkpoint metadata must be an object: {metadata_path}")
        payload["state_dict"] = load_file(str(checkpoint_path), device=device)
        return payload
    if checkpoint_format != "torch":
        raise ValueError("checkpoint_format must be auto, ckpt, torch or safetensors")
    torch, _, _, _ = _imports()
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # older torch
        payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        return {"state_dict": payload}
    return payload


def state_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    if payload and all(hasattr(value, "shape") for value in payload.values()):
        return payload
    raise ValueError("checkpoint has no model_state_dict/state_dict/model mapping")


def load_initial_weights(model, spec: ModelSpec, loss_name: str) -> None:
    """Load only shape-compatible model weights from ckpt or safetensors."""

    payload = checkpoint_payload(spec.initial_checkpoint, "cpu", spec.checkpoint_format)
    source = clean_state_dict(state_from_payload(payload))
    candidates = [source]
    if loss_name == "ce":
        candidates.append(
            {
                key.removeprefix("backbone."): value
                for key, value in source.items()
                if not key.startswith("projector.")
            }
        )
    else:
        candidates.append(
            {
                key
                if key.startswith(("backbone.", "projector."))
                else f"backbone.{key}": value
                for key, value in source.items()
            }
        )
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
        raise ValueError(
            f"checkpoint has no weights compatible with {spec.architecture}"
        )
    model.load_state_dict(compatible, strict=False)


def build_training_model(
    spec: ModelSpec, loss_name: str, num_classes: int, device: str
):
    """Build a classifier for CE or an embedding network for metric losses."""

    _, timm, _, _ = _imports()
    if loss_name == "ce":
        model = timm.create_model(
            spec.architecture,
            pretrained=spec.pretrained and not spec.initial_checkpoint,
            num_classes=num_classes,
        )
    elif loss_name in {"supcon", "triplet"}:
        model = SupConNetwork.build(
            spec.architecture,
            spec.pretrained and not spec.initial_checkpoint,
            spec.projection_hidden_dim,
            spec.projection_dim,
        )
    else:
        raise ValueError("loss must be ce, supcon or triplet")
    if spec.initial_checkpoint:
        load_initial_weights(model, spec, loss_name)
    return model.to(device)


class ProjectionHead:
    """Namespace factory kept import-light until torch is actually needed."""

    @staticmethod
    def build(input_dim: int, hidden_dim: int, output_dim: int):
        torch, _, _, _ = _imports()
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )


class SupConNetwork:
    """Factory for a torch module exposing backbone and projection embeddings."""

    @staticmethod
    def build(
        architecture: str,
        pretrained: bool,
        hidden_dim: int,
        output_dim: int,
        checkpoint_path: str | None = None,
    ):
        torch, timm, _, _ = _imports()

        class Network(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if checkpoint_path:
                    self.backbone = timm.create_model(
                        architecture, pretrained=False, checkpoint_path=checkpoint_path
                    )
                    self.backbone.reset_classifier(0)
                else:
                    self.backbone = timm.create_model(
                        architecture, pretrained=pretrained, num_classes=0
                    )
                feature_dim = int(getattr(self.backbone, "num_features"))
                self.projector = ProjectionHead.build(
                    feature_dim, hidden_dim, output_dim
                )

            def forward(self, images, return_projection: bool = True):
                features = self.backbone(images)
                if not return_projection:
                    return features
                return torch.nn.functional.normalize(self.projector(features), dim=-1)

        return Network()


class ModelAdapter(abc.ABC):
    """One object controls loading, preprocessing and embedding extraction."""

    def __init__(self, spec: ModelSpec, device: str = "cpu") -> None:
        self.spec = spec
        self.device = device
        self.model = None
        self.checkpoint_metadata: dict[str, Any] = {}

    @abc.abstractmethod
    def load(self) -> "ModelAdapter":
        raise NotImplementedError

    def transform(self):
        if self.model is None:
            self.load()
        _, _, create_transform, resolve_model_data_config = _imports()
        config = resolve_model_data_config(getattr(self.model, "backbone", self.model))
        config["input_size"] = (3, self.spec.input_size, self.spec.input_size)
        return create_transform(**config, is_training=False)

    @abc.abstractmethod
    def encode_batch(self, images):
        raise NotImplementedError

    def encode(self, images):
        torch, _, _, _ = _imports()
        if self.model is None:
            self.load()
        self.model.eval()
        with torch.inference_mode():
            return self.encode_batch(images.to(self.device)).detach().cpu()

    def metadata(self) -> dict[str, Any]:
        return {"spec": self.spec.to_dict(), "checkpoint": self.checkpoint_metadata}


class TimmPretrainedAdapter(ModelAdapter):
    def load(self) -> "TimmPretrainedAdapter":
        _, timm, _, _ = _imports()
        if self.spec.checkpoint:
            self.model = timm.create_model(
                self.spec.architecture,
                pretrained=False,
                checkpoint_path=self.spec.checkpoint,
            )
            self.model.reset_classifier(0)
        else:
            self.model = timm.create_model(
                self.spec.architecture, pretrained=self.spec.pretrained, num_classes=0
            )
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.spec.checkpoint:
            self.checkpoint_metadata = {
                "path": str(Path(self.spec.checkpoint).expanduser().resolve()),
                "kind": "local_base_weights",
            }
        return self

    def encode_batch(self, images):
        return self.model(images)


class ClassifierCheckpointAdapter(ModelAdapter):
    def load(self) -> "ClassifierCheckpointAdapter":
        if not self.spec.checkpoint:
            raise ValueError(f"model {self.spec.name!r} requires checkpoint")
        _, timm, _, _ = _imports()
        payload = checkpoint_payload(
            self.spec.checkpoint, self.device, self.spec.checkpoint_format
        )
        labels = (
            payload.get("labels")
            or list((payload.get("label_to_index") or {}).keys())
            or list(self.spec.extra.get("labels", []))
        )
        state = clean_state_dict(state_from_payload(payload))
        classifier_weights = [
            value
            for key, value in state.items()
            if key.endswith(".weight")
            and any(
                part in {"head", "fc", "classifier"}
                for part in key.removesuffix(".weight").split(".")
            )
            and getattr(value, "ndim", 0) == 2
        ]
        num_classes = int(
            payload.get("num_classes")
            or self.spec.extra.get("num_classes", 0)
            or len(labels)
            or (classifier_weights[-1].shape[0] if classifier_weights else 0)
        )
        if num_classes <= 0:
            raise ValueError("classifier checkpoint must contain labels or num_classes")
        self.model = timm.create_model(
            payload.get("architecture", self.spec.architecture),
            pretrained=False,
            num_classes=num_classes,
        )
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        meaningful_missing = [
            key for key in missing if not key.endswith("num_batches_tracked")
        ]
        if meaningful_missing or unexpected:
            raise ValueError(
                f"checkpoint does not match classifier model; missing={meaningful_missing[:8]}, "
                f"unexpected={list(unexpected)[:8]}"
            )
        self.model = self.model.to(self.device).eval()
        self.checkpoint_metadata = {
            "path": str(Path(self.spec.checkpoint).expanduser().resolve()),
            "epoch": payload.get("epoch"),
            "labels": labels,
        }
        return self

    def encode_batch(self, images):
        features = self.model.forward_features(images)
        if hasattr(self.model, "forward_head"):
            return self.model.forward_head(features, pre_logits=True)
        return features


class SupConCheckpointAdapter(ModelAdapter):
    def load(self) -> "SupConCheckpointAdapter":
        if not self.spec.checkpoint:
            raise ValueError(f"model {self.spec.name!r} requires checkpoint")
        payload = checkpoint_payload(
            self.spec.checkpoint, self.device, self.spec.checkpoint_format
        )
        architecture = payload.get("architecture", self.spec.architecture)
        hidden = int(
            payload.get("projection_hidden_dim", self.spec.projection_hidden_dim)
        )
        output = int(payload.get("projection_dim", self.spec.projection_dim))
        self.model = SupConNetwork.build(architecture, False, hidden, output)
        missing, unexpected = self.model.load_state_dict(
            clean_state_dict(state_from_payload(payload)), strict=False
        )
        if missing or unexpected:
            raise ValueError(
                f"checkpoint does not match SupCon model; missing={list(missing)[:8]}, "
                f"unexpected={list(unexpected)[:8]}"
            )
        self.model = self.model.to(self.device).eval()
        self.checkpoint_metadata = {
            "path": str(Path(self.spec.checkpoint).expanduser().resolve()),
            "epoch": payload.get("epoch"),
            "labels": payload.get("labels", []),
        }
        return self

    def encode_batch(self, images):
        use_projection = self.spec.embedding_source == "projection"
        return self.model(images, return_projection=use_projection)


_ADAPTERS: dict[str, type[ModelAdapter]] = {}


def register_adapter(
    name: str, adapter_type: type[ModelAdapter], *, replace: bool = False
) -> None:
    if name in _ADAPTERS and not replace:
        raise KeyError(f"adapter {name!r} is already registered")
    _ADAPTERS[name] = adapter_type


def create_adapter(spec: ModelSpec, device: str = "cpu") -> ModelAdapter:
    try:
        adapter_type = _ADAPTERS[spec.adapter]
    except KeyError as exc:
        raise KeyError(
            f"unknown adapter {spec.adapter!r}; available={sorted(_ADAPTERS)}"
        ) from exc
    return adapter_type(spec, device)


register_adapter("timm_pretrained", TimmPretrainedAdapter)
register_adapter("classifier_checkpoint", ClassifierCheckpointAdapter)
register_adapter("supcon_checkpoint", SupConCheckpointAdapter)


def choose_device(requested: str) -> str:
    torch, _, _, _ = _imports()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_initial_checkpoint(
    architecture: str,
    output_path: str | Path,
    *,
    pretrained: bool = True,
    overwrite: bool = False,
) -> Path:
    """Save one reusable timm state dict for offline experiments."""
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
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("save-initial")
    save.add_argument("--architecture", default="vit_base_patch16_224")
    save.add_argument("--output", required=True)
    save.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    save.add_argument("--overwrite", action="store_true")
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
