import torch


def validation_threshold(
    validation_unknownness,
    percentile=0.95
):
    scores = torch.as_tensor(
        validation_unknownness,
        dtype=torch.float32
    )

    return torch.quantile(
        scores,
        percentile
    ).item()


def acceptance_mask(unknownness, threshold):
    scores = torch.as_tensor(unknownness)

    return scores <= threshold


def rejection_mask(unknownness, threshold):
    scores = torch.as_tensor(unknownness)

    return scores > threshold


def acceptance_rate(unknownness, threshold):
    accepted = acceptance_mask(
        unknownness,
        threshold
    )

    return accepted.float().mean().item()


def rejection_rate(unknownness, threshold):
    rejected = rejection_mask(
        unknownness,
        threshold
    )

    return rejected.float().mean().item()