import torch
from torch.nn import functional as F

from task2.methods.dann import (
    create_domain_labels,
    gradient_reverse,
    gradient_reversal_strength,
)
from task2.models.domain_discriminator import (
    DomainDiscriminator,
)


FEATURE_DIMENSION = 512
NUMBER_OF_CLASSES = 7

CDAN_INPUT_DIMENSION = (
    FEATURE_DIMENSION
    * NUMBER_OF_CLASSES
)


def class_conditioned_features(
    features,
    logits,
):
    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    outer_products = torch.bmm(
        features.unsqueeze(2),
        probabilities.unsqueeze(1),
    )

    conditioned_features = outer_products.flatten(
        start_dim=1
    )

    return conditioned_features


def compute_cdan_loss(
    source_logits,
    target_logits,
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

    source_conditioned = class_conditioned_features(
        source_features,
        source_logits,
    )

    target_conditioned = class_conditioned_features(
        target_features,
        target_logits,
    )

    combined_conditioned = torch.cat(
        [
            source_conditioned,
            target_conditioned,
        ],
        dim=0,
    )

    grl_strength = gradient_reversal_strength(
        progress=progress,
        maximum_strength=maximum_grl_strength,
    )

    reversed_conditioned = gradient_reverse(
        combined_conditioned,
        grl_strength,
    )

    domain_logits = domain_discriminator(
        reversed_conditioned
    )

    domain_labels = create_domain_labels(
        number_of_source_examples=(
            source_features.shape[0]
        ),
        number_of_target_examples=(
            target_features.shape[0]
        ),
        device=combined_conditioned.device,
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
        FEATURE_DIMENSION,
        requires_grad=True,
    )

    target_features = torch.randn(
        24,
        FEATURE_DIMENSION,
        requires_grad=True,
    )

    source_logits = torch.randn(
        24,
        NUMBER_OF_CLASSES,
        requires_grad=True,
    )

    target_logits = torch.randn(
        24,
        NUMBER_OF_CLASSES,
        requires_grad=True,
    )

    source_labels = torch.randint(
        low=0,
        high=NUMBER_OF_CLASSES,
        size=(24,),
    )

    discriminator = DomainDiscriminator(
        input_dimension=CDAN_INPUT_DIMENSION,
    )

    conditioned = class_conditioned_features(
        source_features,
        source_logits,
    )

    total_loss, statistics = compute_cdan_loss(
        source_logits=source_logits,
        target_logits=target_logits,
        source_labels=source_labels,
        source_features=source_features,
        target_features=target_features,
        domain_discriminator=discriminator,
        progress=0.5,
    )

    total_loss.backward()

    print(
        "Conditioned feature shape: "
        f"{conditioned.shape}"
    )

    print(f"Statistics: {statistics}")

    print(
        "Source-logit gradients created: "
        f"{source_logits.grad is not None}"
    )


if __name__ == "__main__":
    main()