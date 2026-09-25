import torch
import torch.nn.functional as F


def classifier_placeholder_losses(
    logits,
    labels,
    known_classes=10
):
    known_logits = logits[:, :known_classes]
    dummy_logits = logits[:, known_classes:]

    classification_loss = F.cross_entropy(
        logits,
        labels
    )

    strongest_dummy = (
        dummy_logits
        .max(dim=1)
        .values
        .unsqueeze(1)
    )

    competitor_logits = torch.cat([
        known_logits,
        strongest_dummy
    ], dim=1)

    row_indices = torch.arange(
        labels.size(0),
        device=labels.device
    )

    competitor_logits = competitor_logits.clone()

    competitor_logits[
        row_indices,
        labels
    ] = torch.finfo(
        competitor_logits.dtype
    ).min

    dummy_target = torch.full(
        size=(labels.size(0),),
        fill_value=known_classes,
        dtype=torch.long,
        device=labels.device
    )

    classifier_placeholder_loss = (
        F.cross_entropy(
            competitor_logits,
            dummy_target
        )
    )

    return {
        "classification_loss":
            classification_loss,
        "classifier_placeholder_loss":
            classifier_placeholder_loss
    }


def data_placeholder_loss(
    mixed_logits,
    known_classes=10
):
    known_logits = mixed_logits[:, :known_classes]

    strongest_dummy = (
        mixed_logits[:, known_classes:]
        .max(dim=1)
        .values
        .unsqueeze(1)
    )

    combined_logits = torch.cat([
        known_logits,
        strongest_dummy
    ], dim=1)

    dummy_target = torch.full(
        size=(mixed_logits.size(0),),
        fill_value=known_classes,
        dtype=torch.long,
        device=mixed_logits.device
    )

    return F.cross_entropy(
        combined_logits,
        dummy_target
    )


def combined_proser_loss(
    classification_loss,
    classifier_placeholder_loss,
    data_placeholder_loss_value,
    beta=1.0,
    gamma=0.1
):
    return (
        classification_loss
        + beta * classifier_placeholder_loss
        + gamma * data_placeholder_loss_value
    )


def placeholder_unknownness(
    logits,
    known_classes=10,
    temperature=1024.0
):
    known_logits = logits[:, :known_classes]
    dummy_logits = logits[:, known_classes:]

    strongest_dummy = (
        dummy_logits
        .max(dim=1)
        .values
        .unsqueeze(1)
    )

    combined_logits = torch.cat([
        known_logits,
        strongest_dummy
    ], dim=1)

    probabilities = torch.softmax(
        combined_logits / temperature,
        dim=1
    )

    maximum_known_probability = (
        probabilities[:, :known_classes]
        .max(dim=1)
        .values
    )

    dummy_probability = probabilities[
        :, known_classes
    ]

    return (
        dummy_probability
        - maximum_known_probability
    )