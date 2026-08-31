"""ONNX export entry point for a validated GeoShield checkpoint.

Exports the requested model at the fixed serving resolution, then verifies
the exported graph against the PyTorch checkpoint it came from: ONNX Runtime
must be able to load and run it, and its argmax predictions must agree with
PyTorch's on at least ``--parity-threshold`` of pixels across random inputs.
Nothing is written to ``--output`` unless that parity check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import DAMAGE_CLASS_NAMES, NUM_CLASSES
from .models import PostOnlyUNet, SiameseUNet, require_torch, torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a GeoShield model to ONNX and verify PyTorch/ONNX parity.")
    parser.add_argument("--model", choices=("post_only", "siamese"), default="siamese")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("public/models/geoshield-siamese.onnx"))
    parser.add_argument("--metadata-output", type=Path, help="Defaults to <output>.json")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--parity-samples", type=int, default=8)
    parser.add_argument("--parity-threshold", type=float, default=0.999)
    return parser


def _load_model(model_name: str, checkpoint_path: Path):
    require_torch()
    model_class = PostOnlyUNet if model_name == "post_only" else SiameseUNet
    model = model_class(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _input_spec(model_name: str, size: int) -> tuple[list[str], tuple]:
    require_torch()
    dummy_after = torch.zeros(1, 3, size, size)
    if model_name == "post_only":
        return ["after"], (dummy_after,)
    dummy_before = torch.zeros(1, 3, size, size)
    return ["before", "after"], (dummy_before, dummy_after)


def _check_parity(model, session, model_name: str, size: int, samples: int, seed: int = 42) -> float:
    require_torch()
    rng = np.random.default_rng(seed)
    agreements: list[float] = []
    with torch.no_grad():
        for _ in range(samples):
            after_np = rng.random((1, 3, size, size), dtype=np.float32)
            if model_name == "post_only":
                torch_logits = model(torch.from_numpy(after_np))
                onnx_logits = session.run(None, {"after": after_np})[0]
            else:
                before_np = rng.random((1, 3, size, size), dtype=np.float32)
                torch_logits = model(torch.from_numpy(before_np), torch.from_numpy(after_np))
                onnx_logits = session.run(None, {"before": before_np, "after": after_np})[0]
            torch_mask = torch_logits.argmax(dim=1).numpy()
            onnx_mask = onnx_logits.argmax(axis=1)
            agreements.append(float((torch_mask == onnx_mask).mean()))
    return float(np.mean(agreements))


def export(args: argparse.Namespace) -> dict[str, object]:
    require_torch()
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required to export a model")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, checkpoint = _load_model(args.model, args.checkpoint)
    input_names, dummy_inputs = _input_spec(args.model, args.image_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_inputs,
        str(args.output),
        input_names=input_names,
        output_names=["logits"],
        opset_version=args.opset,
        # Keep weights embedded in one file: the browser Web Worker fetches
        # this by URL, so a second externalData sidecar (the exporter's
        # default for larger models) would add a relative-path fetch this
        # deployment target has no simple way to resolve.
        external_data=False,
    )

    import onnx as onnx_module
    import onnxruntime as ort

    onnx_module.checker.check_model(str(args.output))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])

    pixel_agreement = _check_parity(model, session, args.model, args.image_size, args.parity_samples)
    if pixel_agreement < args.parity_threshold:
        args.output.unlink(missing_ok=True)
        raise ValueError(
            f"PyTorch/ONNX pixel agreement {pixel_agreement:.6f} is below the required {args.parity_threshold}; "
            "the exported ONNX file was removed"
        )

    sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    is_trained = bool(checkpoint_config.get("data")) and checkpoint_config.get("data") not in (".", None)
    metadata = {
        "model": args.model,
        "onnx_path": str(args.output),
        "sha256": sha256,
        "size_bytes": args.output.stat().st_size,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "opset": args.opset,
        "input_shapes": {name: [1, 3, args.image_size, args.image_size] for name in input_names},
        "output_shape": [1, NUM_CLASSES, args.image_size, args.image_size],
        "class_mapping": list(DAMAGE_CLASS_NAMES),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_config": checkpoint_config,
        "pixel_agreement": pixel_agreement,
        "parity_samples": args.parity_samples,
        # False whenever the source checkpoint came from --smoke-test (synthetic
        # eight-tile fixture) rather than real xBD tiles; never claim otherwise.
        "trained_on_real_data": is_trained,
    }
    metadata_path = args.metadata_output or args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = export(args)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
