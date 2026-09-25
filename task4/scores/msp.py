import torch


def msp_unknownness(logits):
    probabilities = torch.softmax(
        logits,
        dim=1
    )

    maximum_probability = (
        probabilities.max(dim=1).values
    )

    return 1.0 - maximum_probability