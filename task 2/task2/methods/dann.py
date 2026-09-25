import math

import torch
from torch.nn import functional as F

from task2.models.domain_discriminator import (
    DomainDiscriminator,
)


class GradientReversalFunction(
    torch.autograd.Function
):
    @staticmethod
    def forward(
        context,
        inputs,
        strength,
    ):
        context.strength = strength

        return inputs.view_as(inputs)

    @staticmethod
    def backward(
        context,
        output_gradient,
    ):
        reversed_gradient = (
            -context.strength
            * output_gradient
        )

        return reversed_gradient, None


def gradient_reverse(
    features,
    strength,
):
    return GradientReversalFunction.apply(
        features,
        strength,
    )


def gradient_reversal_strength(
    progress,
    maximum_strength=1.0,
):
    schedule_value = (
        2.0
        / (
            1.0
            + math.exp(-10.0 * progress)
        )
        - 1.0
    )

    return maximum_strength * schedule_value


def create_domain_labels(
    number_of_source_examples,
    number_of_target_examples,
    device,
):
    source_domain_labels = torch.zeros(
        number_of_source_examples,
        dtype=torch.long,
        device=device,
    )

    target_domain_labels = torch.ones(
        number_of_target_examples,
        dtype=torch.long,
        device=device,
    )

    domain_labels = torch.cat(
        [
            source_domain_labels,
            target_domain_labels,
        ],
        dim=0,
    )

    return domain_labels


def compute_dann_loss(
    source_logits,
    source_labels,
    source_features,
    target_features,
    domain_discriminator,
    progress,
    maximum_grl_strength=1.0,
    domain_loss_weight=1.0,
):
    classification_loss = F.cross_entropy(
        source_logits,
        source_labels,
    )

    combined_features = torch.cat(
        [
            source_features,
            target_features,
        ],
        dim=0,
    )

    grl_strength = gradient_reversal_strength(
        progress=progress,
        maximum_strength=maximum_grl_strength,
    )

    reversed_features = gradient_reverse(
        combined_features,
        grl_strength,
    )

    domain_logits = domain_discriminator(
        reversed_features
    )

    domain_labels = create_domain_labels(
        number_of_source_examples=(
            source_features.shape[0]
        ),
        number_of_target_examples=(
            target_features.shape[0]
        ),
        device=combined_features.device,
    )

    domain_loss = F.cross_entropy(
        domain_logits,
        domain_labels,
    )

    total_loss = (
        classification_loss
        + domain_loss_weight * domain_loss
    )

    domain_accuracy = (
        domain_logits.argmax(dim=1)
        == domain_labels
    ).float().mean()

    statistics = {
        "classification_loss": (
            classification_loss.detach().item()
        ),
        "domain_loss": (
            domain_loss.detach().item()
        ),
        "domain_accuracy": (
            domain_accuracy.detach().item()
        ),
        "grl_strength": grl_strength,
        "total_loss": (
            total_loss.detach().item()
        ),
    }

    return total_loss, statistics


def main():
    source_features = torch.randn(
        24,
        512,
        requires_grad=True,
    )

    target_features = torch.randn(
        24,
        512,
        requires_grad=True,
    )

    source_logits = torch.randn(
        24,
        7,
        requires_grad=True,
    )

    source_labels = torch.randint(
        low=0,
        high=7,
        size=(24,),
    )

    discriminator = DomainDiscriminator(
        input_dimension=512,
    )

    total_loss, statistics = compute_dann_loss(
        source_logits=source_logits,
        source_labels=source_labels,
        source_features=source_features,
        target_features=target_features,
        domain_discriminator=discriminator,
        progress=0.5,
    )

    total_loss.backward()

    print(f"Statistics: {statistics}")
    print(
        "Source-feature gradients created: "
        f"{source_features.grad is not None}"
    )


if __name__ == "__main__":
    main()