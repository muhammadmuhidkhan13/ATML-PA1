import torch
from torch import nn


FEATURE_DIMENSION = 512
NUMBER_OF_CLASSES = 7


class ClassifierHead(nn.Module):
    def __init__(
        self,
        feature_dimension=FEATURE_DIMENSION,
        number_of_classes=NUMBER_OF_CLASSES,
    ):
        super().__init__()

        self.linear = nn.Linear(
            feature_dimension,
            number_of_classes,
        )

    def forward(self, features):
        logits = self.linear(features)

        return logits


def main():
    classifier = ClassifierHead()

    example_features = torch.randn(
        4,
        FEATURE_DIMENSION,
    )

    logits = classifier(example_features)

    print(
        f"Input feature shape: "
        f"{example_features.shape}"
    )

    print(
        f"Output logits shape: "
        f"{logits.shape}"
    )


if __name__ == "__main__":
    main()