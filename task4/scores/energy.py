import torch


def energy_unknownness(logits):
    return -torch.logsumexp(
        logits,
        dim=1
    )