import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


FEATURE_DIMENSION = 512


class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1

        self.network = resnet18(
            weights=weights,
        )

        self.network.fc = nn.Identity()

        self.freeze_batchnorm_statistics()

    def freeze_batchnorm_statistics(self):
        for module in self.network.modules():
            if isinstance(
                module,
                nn.modules.batchnorm._BatchNorm,
            ):
                module.eval()

    def train(self, mode=True):
        super().train(mode)

        if mode:
            self.freeze_batchnorm_statistics()

        return self

    def forward(self, images):
        features = self.network(images)

        return features


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    backbone = ResNet18Backbone().to(device)
    backbone.train()

    example_images = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    features = backbone(example_images)

    batchnorm_layers = [
        module
        for module in backbone.modules()
        if isinstance(
            module,
            nn.modules.batchnorm._BatchNorm,
        )
    ]

    batchnorm_frozen = all(
        not layer.training
        for layer in batchnorm_layers
    )

    batchnorm_affine_trainable = all(
        layer.weight.requires_grad
        and layer.bias.requires_grad
        for layer in batchnorm_layers
    )

    print(f"Device: {device}")
    print(f"Feature shape: {features.shape}")
    print(
        "Backbone in training mode: "
        f"{backbone.training}"
    )
    print(
        "BatchNorm statistics frozen: "
        f"{batchnorm_frozen}"
    )
    print(
        "BatchNorm gamma/beta trainable: "
        f"{batchnorm_affine_trainable}"
    )


if __name__ == "__main__":
    main()