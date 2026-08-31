import pytest

from geoshield.models import SiameseUNet, torch


@pytest.mark.skipif(torch is None, reason="PyTorch is installed in the training environment only")
def test_siamese_model_output_shape():
    import torch as torch_module

    model = SiameseUNet(pretrained=False)
    output = model(torch_module.zeros(1, 3, 512, 512), torch_module.zeros(1, 3, 512, 512))
    assert tuple(output.shape) == (1, 5, 512, 512)
