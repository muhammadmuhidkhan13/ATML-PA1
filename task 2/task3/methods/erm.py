import torch
from torch.nn import functional as F


def compute_erm_loss(
    logits_by_domain,
    labels_by_domain,
):
    domain_losses = []

    for domain_name in logits_by_domain:
        domain_loss = F.cross_entropy(
            logits_by_domain[domain_name],
            labels_by_domain[domain_name],
        )

        domain_losses.append(domain_loss)

    erm_loss = torch.stack(domain_losses).mean()

    statistics = {
        "classification_loss": (
            erm_loss.detach().item()
        ),
        "total_loss": (
            erm_loss.detach().item()
        ),
    }

    return erm_loss, statistics


def main():
    domain_names = [
        "art_painting",
        "cartoon",
        "photo",
    ]

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

    loss, statistics = compute_erm_loss(
        logits_by_domain,
        labels_by_domain,
    )

    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"Statistics: {statistics}")


if __name__ == "__main__":
    main()