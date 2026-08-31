import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from geoshield.export_onnx import build_parser, export  # noqa: E402
from geoshield.train import build_parser as train_build_parser, train  # noqa: E402


@pytest.fixture(scope="module")
def smoke_checkpoint(tmp_path_factory):
    output = tmp_path_factory.mktemp("checkpoint")
    # Two epochs is enough to get a real (if untrained) state_dict; export
    # parity depends on the graph translation being correct, not on accuracy.
    train_args = train_build_parser().parse_args(
        ["--model", "siamese", "--smoke-test", "--output", str(output), "--seed", "42", "--epochs", "2"]
    )
    train(train_args)
    return output / "siamese_best.pt"


def test_export_produces_valid_onnx_with_parity(smoke_checkpoint, tmp_path):
    onnx_path = tmp_path / "geoshield-siamese.onnx"
    args = build_parser().parse_args(
        [
            "--model", "siamese",
            "--checkpoint", str(smoke_checkpoint),
            "--output", str(onnx_path),
            "--image-size", "64",
            "--parity-samples", "4",
        ]
    )
    metadata = export(args)

    assert onnx_path.exists()
    assert metadata["fp32_pixel_agreement"] >= 0.999
    assert 0.0 <= metadata["quantization_pixel_agreement"] <= 1.0
    assert metadata["size_bytes"] < metadata["fp32_size_bytes"]
    assert metadata["trained_on_real_data"] is False
    assert metadata["input_shapes"] == {"before": [1, 3, 64, 64], "after": [1, 3, 64, 64]}
    assert metadata["output_shape"] == [1, 5, 64, 64]
    assert len(metadata["sha256"]) == 64
    assert metadata["class_mapping"] == ["background", "undamaged", "minor", "major", "destroyed"]

    metadata_path = onnx_path.with_suffix(".json")
    assert metadata_path.exists()
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["sha256"] == metadata["sha256"]

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {entry.name for entry in session.get_inputs()}
    assert input_names == {"before", "after"}


def test_export_rejects_missing_checkpoint(tmp_path):
    args = build_parser().parse_args(
        ["--model", "siamese", "--checkpoint", str(tmp_path / "missing.pt"), "--output", str(tmp_path / "out.onnx")]
    )
    with pytest.raises(FileNotFoundError):
        export(args)
