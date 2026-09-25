import torch


def different_class_partners(labels):
    partner_indices = []

    for index in range(len(labels)):
        candidates = torch.nonzero(
            labels != labels[index]
        ).flatten()

        selected = candidates[
            torch.randint(
                low=0,
                high=len(candidates),
                size=(1,),
                device=labels.device
            )
        ]

        partner_indices.append(selected)

    return torch.cat(partner_indices)


def manifold_mixup(
    representations,
    labels,
    alpha=2.0
):
    partner_indices = different_class_partners(
        labels
    )

    distribution = torch.distributions.Beta(
        alpha,
        alpha
    )

    mixing_weight = distribution.sample().to(
        representations.device
    )

    partner_representations = representations[
        partner_indices
    ]

    mixed_representations = (
        mixing_weight * representations
        + (1.0 - mixing_weight)
        * partner_representations
    )

    return {
        "mixed_representations":
            mixed_representations,
        "partner_indices":
            partner_indices,
        "mixing_weight":
            mixing_weight
    }