import torch
from torch.nn import functional as F


KERNEL_MULTIPLIERS = (
    0.5,
    1.0,
    2.0,
)


def squared_distance_matrix(
    first_features,
    second_features,
):
    distances = torch.cdist(
        first_features,
        second_features,
        p=2,
    )

    return distances.pow(2)


def median_squared_distance(
    source_features,
    target_features,
):
    combined_features = torch.cat(
        [
            source_features,
            target_features,
        ],
        dim=0,
    )

    with torch.no_grad():
        distances = squared_distance_matrix(
            combined_features,
            combined_features,
        )

        number_of_examples = distances.shape[0]

        off_diagonal_mask = ~torch.eye(
            number_of_examples,
            dtype=torch.bool,
            device=distances.device,
        )

        nonzero_pairwise_distances = distances[
            off_diagonal_mask
        ]

        median_distance = (
            nonzero_pairwise_distances.median()
        )

        median_distance = median_distance.clamp_min(
            1e-8
        )

    return median_distance


def multi_kernel_rbf(
    first_features,
    second_features,
    base_bandwidth,
):
    squared_distances = squared_distance_matrix(
        first_features,
        second_features,
    )

    kernel_values = torch.zeros_like(
        squared_distances
    )

    for multiplier in KERNEL_MULTIPLIERS:
        bandwidth = base_bandwidth * multiplier

        kernel_values = (
            kernel_values
            + torch.exp(
                -squared_distances / bandwidth
            )
        )

    return kernel_values


def maximum_mean_discrepancy(
    source_features,
    target_features,
):
    base_bandwidth = median_squared_distance(
        source_features,
        target_features,
    )

    source_source_kernel = multi_kernel_rbf(
        source_features,
        source_features,
        base_bandwidth,
    )

    target_target_kernel = multi_kernel_rbf(
        target_features,
        target_features,
        base_bandwidth,
    )

    source_target_kernel = multi_kernel_rbf(
        source_features,
        target_features,
        base_bandwidth,
    )

    mmd_loss = (
        source_source_kernel.mean()
        + target_target_kernel.mean()
        - 2 * source_target_kernel.mean()
    )

    return mmd_loss


def compute_dan_loss(
    source_logits,
    source_labels,
    source_features,
    target_features,
    mmd_weight=1.0,
):
    classification_loss = F.cross_entropy(
        source_logits,
        source_labels,
    )

    mmd_loss = maximum_mean_discrepancy(
        source_features,
        target_features,
    )

    total_loss = (
        classification_loss
        + mmd_weight * mmd_loss
    )

    statistics = {
        "classification_loss": (
            classification_loss.detach().item()
        ),
        "alignment_loss": (
            mmd_loss.detach().item()
        ),
        "total_loss": (
            total_loss.detach().item()
        ),
    }

    return total_loss, statistics


def main():
    source_features = torch.randn(
        24,
        512,
        requires_grad=True,
    )

    target_features = torch.randn(
        24,
        512,
        requires_grad=True,
    )

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

    total_loss, statistics = compute_dan_loss(
        source_logits=source_logits,
        source_labels=source_labels,
        source_features=source_features,
        target_features=target_features,
        mmd_weight=1.0,
    )

    total_loss.backward()

    print(f"Statistics: {statistics}")
    print(
        "Source-feature gradients created: "
        f"{source_features.grad is not None}"
    )
    print(
        "Target-feature gradients created: "
        f"{target_features.grad is not None}"
    )


if __name__ == "__main__":
    main()