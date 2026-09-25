import torch


def fit_mahalanobis(
    features,
    labels,
    num_classes=10,
    epsilon=1e-6
):
    class_means = []

    for class_index in range(num_classes):
        class_features = features[
            labels == class_index
        ]

        if class_features.size(0) == 0:
            raise ValueError(
                "No training examples found for "
                f"class {class_index}."
            )

        class_means.append(
            class_features.mean(dim=0)
        )

    class_means = torch.stack(
        class_means
    )

    centered_features = (
        features
        - class_means[labels]
    )

    covariance_diagonal = (
        centered_features.pow(2).mean(dim=0)
        + epsilon
    )

    return (
        class_means,
        covariance_diagonal
    )


def mahalanobis_unknownness(
    features,
    class_means,
    covariance_diagonal
):
    class_distances = []

    for class_mean in class_means:
        differences = (
            features - class_mean
        )

        distances = (
            differences.pow(2)
            / covariance_diagonal
        ).sum(dim=1)

        class_distances.append(
            distances
        )

    class_distances = torch.stack(
        class_distances,
        dim=1
    )

    return class_distances.min(
        dim=1
    ).values