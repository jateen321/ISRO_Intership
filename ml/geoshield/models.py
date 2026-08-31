"""Lightweight paired-image U-Net models used by GeoShield."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:  # Keep dataset tooling and CLI help usable without ML extras.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    ResNet18_Weights = None  # type: ignore[assignment]
    resnet18 = None  # type: ignore[assignment]


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("Install ml project dependencies before creating a model")


if nn is not None:

    class Encoder18(nn.Module):
        def __init__(self, pretrained: bool = True):
            super().__init__()
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
            self.pool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4

        def forward(self, image):
            x0 = self.stem(image)
            x1 = self.layer1(self.pool(x0))
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            return (x0, x1, x2, x3, x4)


    class ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)


    class Decoder(nn.Module):
        def __init__(self, feature_channels: tuple[int, int, int, int, int], num_classes: int):
            super().__init__()
            c0, c1, c2, c3, c4 = feature_channels
            self.reduce4 = nn.Conv2d(c4, 256, 1)
            self.block3 = ConvBlock(256 + c3, 128)
            self.block2 = ConvBlock(128 + c2, 64)
            self.block1 = ConvBlock(64 + c1, 32)
            self.block0 = ConvBlock(32 + c0, 32)
            self.head = nn.Conv2d(32, num_classes, 1)

        def up(self, x, skip, block):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return block(torch.cat((x, skip), dim=1))

        def forward(self, features):
            x0, x1, x2, x3, x4 = features
            x = self.reduce4(x4)
            x = self.up(x, x3, self.block3)
            x = self.up(x, x2, self.block2)
            x = self.up(x, x1, self.block1)
            x = self.up(x, x0, self.block0)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            return self.head(x)


    class PostOnlyUNet(nn.Module):
        """Post-disaster-only baseline with a five-class output map."""

        def __init__(self, num_classes: int = 5, pretrained: bool = True):
            super().__init__()
            self.encoder = Encoder18(pretrained=pretrained)
            self.decoder = Decoder((64, 64, 128, 256, 512), num_classes)

        def forward(self, post_image):
            return self.decoder(self.encoder(post_image))


    class SiameseUNet(nn.Module):
        """Shared-encoder pre/post U-Net with multi-scale feature fusion."""

        def __init__(self, num_classes: int = 5, pretrained: bool = True):
            super().__init__()
            self.encoder = Encoder18(pretrained=pretrained)
            self.fuse = nn.ModuleList([
                nn.Conv2d(channels * 3, channels, 1) for channels in (64, 64, 128, 256, 512)
            ])
            self.decoder = Decoder((64, 64, 128, 256, 512), num_classes)

        def forward(self, before_image, after_image):
            batch_size = before_image.shape[0]
            # Run the shared encoder once on the two views concatenated along the
            # batch dimension, rather than as two separate calls. With a shared
            # encoder and small per-branch batch sizes, two separate calls make
            # BatchNorm accumulate running statistics from two different image
            # distributions (pre- vs post-disaster) in the same buffers, which
            # never stabilizes; a single joint-batch pass keeps BN statistics
            # consistent across both branches without changing the weights.
            combined_features = self.encoder(torch.cat((before_image, after_image), dim=0))
            fused = tuple(
                fuse(torch.cat((feature[:batch_size], feature[batch_size:], (feature[batch_size:] - feature[:batch_size]).abs()), dim=1))
                for fuse, feature in zip(self.fuse, combined_features)
            )
            return self.decoder(fused)

else:

    class PostOnlyUNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_torch()

    class SiameseUNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_torch()
