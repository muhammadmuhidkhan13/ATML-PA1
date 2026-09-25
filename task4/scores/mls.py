def mls_unknownness(logits):
    maximum_logit = (
        logits.max(dim=1).values
    )

    return -maximum_logit