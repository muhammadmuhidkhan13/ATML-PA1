from itertools import combinations

import torch

from task2.methods.dan import (
    maximum_mean_discrepancy,
)
from task3.methods.erm import compute_erm_loss


def compute_pairwise_mmd(features_by_domain):
    domain_names = list(features_by_domain.keys())

    if len(domain_names) != 3:
        raise ValueError(
            "DAN-DG requires exactly three source domains."
        )

    pairwise_losses = {}

    for first_domain, second_domain in combinations(
        domain_names,
        2,
    ):
        pair_name = (
            f"{first_domain}__{second_domain}"
        )

        pairwise_losses[pair_name] = (
            maximum_mean_discrepancy(
                features_by_domain[first_domain],
                features_by_domain[second_domain],
            )
        )

    average_mmd = torch.stack(
        list(pairwise_losses.values())
    ).mean()

    return average_mmd, pairwise_losses


def compute_dan_dg_loss(
    logits_by_domain,
    labels_by_domain,
    features_by_domain,
    lambda_dg=1.0,
):
    classification_loss, _ = compute_erm_loss(
        logits_by_domain,
        labels_by_domain,
    )

    average_mmd, pairwise_losses = (
        compute_pairwise_mmd(
            features_by_domain
        )
    )

    total_loss = (
        classification_loss
        + lambda_dg * average_mmd
    )

    statistics = {
        "classification_loss": (
            classification_loss.detach().item()
        ),
        "alignment_loss": (
            average_mmd.detach().item()
        ),
        "weighted_alignment_loss": (
            lambda_dg
            * average_mmd.detach().item()
        ),
        "total_loss": (
            total_loss.detach().item()
        ),
    }

    for pair_name, pair_loss in (
        pairwise_losses.items()
    ):
        statistics[
            f"mmd_{pair_name}"
        ] = pair_loss.detach().item()

    return total_loss, statistics


def main():
    domain_names = [
        "art_painting",
        "cartoon",
        "photo",
    ]

    features_by_domain = {
        domain: torch.randn(
            8,
            512,
            requires_grad=True,
        )
        for domain in domain_names
    }

    logits_by_domain = {
        domain: torch.randn(
            8,
            7,
            requires_grad=True,
        )
        for domain in domain_names
    }

    labels_by_domain = {
        domain: torch.randint(
            low=0,
            high=7,
            size=(8,),
        )
        for domain in domain_names
    }

    total_loss, statistics = (
        compute_dan_dg_loss(
            logits_by_domain=logits_by_domain,
            labels_by_domain=labels_by_domain,
            features_by_domain=features_by_domain,
            lambda_dg=1.0,
        )
    )

    total_loss.backward()

    print(f"Statistics: {statistics}")

    for domain in domain_names:
        print(
            f"{domain} gradients created: "
            f"{features_by_domain[domain].grad is not None}"
        )


if __name__ == "__main__":
    main()