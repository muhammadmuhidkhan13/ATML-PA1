import torch
from torch.nn import functional as F


def compute_source_only_loss(
    source_logits,
    source_labels,
):
    classification_loss = F.cross_entropy(
        source_logits,
        source_labels,
    )

    statistics = {
        "classification_loss": (
            classification_loss.detach().item()
        ),
        "total_loss": (
            classification_loss.detach().item()
        ),
    }

    return classification_loss, statistics


def main():
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

    total_loss, statistics = (
        compute_source_only_loss(
            source_logits,
            source_labels,
        )
    )

    total_loss.backward()

    print(f"Loss: {total_loss.item():.4f}")
    print(f"Statistics: {statistics}")
    print(
        "Logit gradients created: "
        f"{source_logits.grad is not None}"
    )


if __name__ == "__main__":
    main()