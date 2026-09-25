import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import CIFAR100


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def load_outputs(cache_directory, split_name):
    return torch.load(
        cache_directory / f"{split_name}_outputs.pt",
        map_location="cpu",
        weights_only=False
    )


def select_failures(
    outputs,
    unknownness,
    threshold,
    group,
    number=3
):
    known_logits = outputs["logits"][:, :10]
    predictions = known_logits.argmax(dim=1)

    accepted_positions = torch.nonzero(
        unknownness <= threshold
    ).flatten()

    accepted_scores = unknownness[
        accepted_positions
    ]

    confidence_order = torch.argsort(
        accepted_scores
    )

    selected_positions = accepted_positions[
        confidence_order[:number]
    ]

    failures = []

    for position in selected_positions:
        position = position.item()

        failures.append({
            "group": group,
            "dataset_index":
                outputs["indices"][position].item(),
            "unknown_class_id":
                outputs["labels"][position].item(),
            "predicted_known_class_id":
                predictions[position].item(),
            "unknownness_score":
                unknownness[position].item(),
            "threshold": threshold
        })

    return failures


def save_failure_csv(failures, class_names, path):
    rows = []

    for failure in failures:
        row = dict(failure)

        row["unknown_class"] = class_names[
            failure["unknown_class_id"]
        ]

        row["predicted_known_class"] = (
            CIFAR10_CLASSES[
                failure["predicted_known_class_id"]
            ]
        )

        rows.append(row)

    fieldnames = [
        "group",
        "dataset_index",
        "unknown_class_id",
        "unknown_class",
        "predicted_known_class_id",
        "predicted_known_class",
        "unknownness_score",
        "threshold"
    ]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


def save_failure_figure(
    rows,
    cifar100_dataset,
    path
):
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10, 7)
    )

    for axis, row in zip(axes.flatten(), rows):
        image, _ = cifar100_dataset[
            row["dataset_index"]
        ]

        axis.imshow(image)
        axis.axis("off")

        axis.set_title(
            f"{row['group'].title()}: "
            f"{row['unknown_class']}\n"
            f"Predicted: "
            f"{row['predicted_known_class']}\n"
            f"MLS: "
            f"{row['unknownness_score']:.3f}"
        )

    figure.suptitle(
        "Unknown Images Incorrectly Accepted by Vanilla MLS"
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        default="vanilla"
    )

    arguments = parser.parse_args()

    cache_directory = (
        Path("task4/cache")
        / arguments.model_name
    )

    results_directory = (
        Path("task4/results")
        / arguments.model_name
    )

    score_data = torch.load(
        results_directory / "posthoc_scores.pt",
        map_location="cpu",
        weights_only=False
    )

    threshold = score_data["mls"]["threshold"]

    near_outputs = load_outputs(
        cache_directory,
        "near_unknown"
    )

    far_outputs = load_outputs(
        cache_directory,
        "far_unknown"
    )

    near_failures = select_failures(
        near_outputs,
        score_data["mls"]["near_unknown"],
        threshold,
        group="near",
        number=3
    )

    far_failures = select_failures(
        far_outputs,
        score_data["mls"]["far_unknown"],
        threshold,
        group="far",
        number=3
    )

    failures = near_failures + far_failures

    cifar100_dataset = CIFAR100(
        root="data",
        train=False,
        download=False
    )

    rows = save_failure_csv(
        failures,
        cifar100_dataset.classes,
        results_directory
        / "vanilla_mls_failures.csv"
    )

    save_failure_figure(
        rows,
        cifar100_dataset,
        results_directory
        / "vanilla_mls_failures.png"
    )

    for row in rows:
        print(
            f"{row['group']:4s} | "
            f"{row['unknown_class']:14s} -> "
            f"{row['predicted_known_class']:10s} | "
            f"score: {row['unknownness_score']:.4f} | "
            f"threshold: {row['threshold']:.4f}"
        )


if __name__ == "__main__":
    main()