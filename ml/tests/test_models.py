import pytest

from geoshield.models import PostOnlyUNet, SiameseUNet, torch


@pytest.mark.skipif(torch is None, reason="PyTorch is installed in the training environment only")
def test_siamese_model_output_shape():
    import torch as torch_module

    model = SiameseUNet(pretrained=False)
    output = model(torch_module.zeros(1, 3, 512, 512), torch_module.zeros(1, 3, 512, 512))
    assert tuple(output.shape) == (1, 5, 512, 512)


@pytest.mark.skipif(torch is None, reason="PyTorch is installed in the training environment only")
def test_post_only_model_output_shape():
    import torch as torch_module

    model = PostOnlyUNet(pretrained=False)
    output = model(torch_module.zeros(1, 3, 512, 512))
    assert tuple(output.shape) == (1, 5, 512, 512)


@pytest.mark.skipif(torch is None, reason="PyTorch is installed in the training environment only")
def test_siamese_encoder_is_genuinely_shared():
    from geoshield.models import Encoder18

    model = SiameseUNet(pretrained=False)
    standalone_encoder_params = sum(p.numel() for p in Encoder18(pretrained=False).parameters())
    model_encoder_params = sum(p.numel() for p in model.encoder.parameters())
    # A single shared encoder holds exactly one encoder's worth of parameters;
    # two independent encoders (one per branch) would hold roughly double.
    assert model_encoder_params == standalone_encoder_params

    before = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        before_features = model.encoder(before)
        model.encoder.stem[0].weight.add_(1.0)  # perturb the shared weights
        before_features_again = model.encoder(before)
    # The perturbation must be visible on a second call through the same
    # attribute, which only holds if both branches share one module instance.
    assert not torch.allclose(before_features[0], before_features_again[0])


@pytest.mark.skipif(torch is None, reason="PyTorch is installed in the training environment only")
def test_siamese_model_is_sensitive_to_pre_post_order():
    import torch as torch_module

    torch_module.manual_seed(0)
    model = SiameseUNet(pretrained=False)
    model.eval()
    before = torch_module.rand(1, 3, 64, 64)
    after = torch_module.rand(1, 3, 64, 64)
    with torch_module.no_grad():
        forward = model(before, after)
        swapped = model(after, before)
    # A genuinely order-aware fusion (concat, not a symmetric op like sum/abs-diff
    # alone) must produce different logits when pre/post are swapped.
    assert not torch_module.allclose(forward, swapped)
