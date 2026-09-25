import torch
from torch import nn


HIDDEN_DIMENSION = 256
DROPOUT_PROBABILITY = 0.5
NUMBER_OF_DOMAINS = 2


class DomainDiscriminator(nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_dimension=HIDDEN_DIMENSION,
        dropout_probability=DROPOUT_PROBABILITY,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dimension,
                hidden_dimension,
            ),
            nn.ReLU(),
            nn.Dropout(
                p=dropout_probability,
            ),
            nn.Linear(
                hidden_dimension,
                NUMBER_OF_DOMAINS,
            ),
        )

    def forward(self, inputs):
        domain_logits = self.network(inputs)

        return domain_logits


def main():
    dann_discriminator = DomainDiscriminator(
        input_dimension=512,
    )

    cdan_discriminator = DomainDiscriminator(
        input_dimension=512 * 7,
    )

    dann_features = torch.randn(
        4,
        512,
    )

    cdan_features = torch.randn(
        4,
        512 * 7,
    )

    dann_logits = dann_discriminator(
        dann_features
    )

    cdan_logits = cdan_discriminator(
        cdan_features
    )

    print(
        f"DANN output shape: "
        f"{dann_logits.shape}"
    )

    print(
        f"CDAN output shape: "
        f"{cdan_logits.shape}"
    )


if __name__ == "__main__":
    main()