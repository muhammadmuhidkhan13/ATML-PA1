import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import CIFAR100


RESULTS_ROOT = Path("task4/results")
CACHE_ROOT = Path("task4/cache")

OUTPUT_DIRECTORY = (
    RESULTS_ROOT
    / "presentation_figures"
)

TABLE_PATH = (
    RESULTS_ROOT
    / "final_tables"
    / "table2_model_comparison.csv"
)

NEAR_CLASSES = [
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel",
]

FAR_CLASSES = [
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe",
]


def save_figure(
    figure,
    filename,
):
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        OUTPUT_DIRECTORY
        / f"{filename}.png"
    )

    pdf_path = (
        OUTPUT_DIRECTORY
        / f"{filename}.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def load_score_data(model_name):
    path = (
        RESULTS_ROOT
        / model_name
        / "posthoc_scores.pt"
    )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def load_cached_outputs(
    model_name,
    split_name,
):
    path = (
        CACHE_ROOT
        / model_name
        / f"{split_name}_outputs.pt"
    )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def as_numpy(values):
    if isinstance(values, torch.Tensor):
        return (
            values
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(values)


def average_ranks(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    sorted_values = values[order]

    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )

    start = 0

    while start < len(values):
        end = start + 1

        while (
            end < len(values)
            and sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        average_rank = (
            start + end - 1
        ) / 2.0

        ranks[
            order[start:end]
        ] = average_rank

        start = end

    return ranks


def correlation_matrix_without_blas(
    rank_matrix,
):
    rank_matrix = np.asarray(
        rank_matrix,
        dtype=np.float64,
    )

    number_of_columns = (
        rank_matrix.shape[1]
    )

    result = np.eye(
        number_of_columns,
        dtype=np.float64,
    )

    for first_index in range(
        number_of_columns
    ):
        first_values = rank_matrix[
            :,
            first_index,
        ]

        first_centered = (
            first_values
            - first_values.mean()
        )

        first_squared_sum = float(
            np.sum(
                first_centered
                * first_centered
            )
        )

        for second_index in range(
            first_index + 1,
            number_of_columns,
        ):
            second_values = rank_matrix[
                :,
                second_index,
            ]

            second_centered = (
                second_values
                - second_values.mean()
            )

            second_squared_sum = float(
                np.sum(
                    second_centered
                    * second_centered
                )
            )

            numerator = float(
                np.sum(
                    first_centered
                    * second_centered
                )
            )

            denominator = (
                first_squared_sum
                * second_squared_sum
            ) ** 0.5

            if denominator == 0.0:
                correlation = 0.0
            else:
                correlation = (
                    numerator
                    / denominator
                )

            result[
                first_index,
                second_index,
            ] = correlation

            result[
                second_index,
                first_index,
            ] = correlation

    return result


def load_model_comparison_table():
    with open(
        TABLE_PATH,
        "r",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def display_label(row):
    model_name = (
        row["model"].upper()
    )

    score_name = row["score"]

    if (
        score_name
        == "proser_placeholder"
    ):
        return (
            "PROSER\nPlaceholder"
        )

    return (
        f"{model_name}\n"
        f"{score_name.upper()}"
    )


def plot_known_acceptance():
    rows = (
        load_model_comparison_table()
    )

    labels = [
        display_label(row)
        for row in rows
    ]

    acceptance_rates = [
        100.0
        * float(
            row[
                "known_test_acceptance_rate"
            ]
        )
        for row in rows
    ]

    colors = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#B279A2",
    ]

    figure, axis = plt.subplots(
        figsize=(9, 5.5),
    )

    bars = axis.bar(
        labels,
        acceptance_rates,
        color=colors[:len(rows)],
        width=0.65,
    )

    target_line = axis.axhline(
        95.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=(
            "Target: 95% known acceptance"
        ),
    )

    for bar, value in zip(
        bars,
        acceptance_rates,
    ):
        axis.text(
            (
                bar.get_x()
                + bar.get_width() / 2
            ),
            value + 0.08,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    minimum_value = min(
        acceptance_rates
    )

    axis.set_ylim(
        min(
            93.5,
            minimum_value - 0.5,
        ),
        96.5,
    )

    axis.set_ylabel(
        "CIFAR-10 test "
        "acceptance rate (%)"
    )

    axis.set_title(
        "Known-Class Acceptance at the "
        "Validation-Calibrated Threshold",
        fontsize=14,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend(
        handles=[target_line],
        loc="lower right",
    )

    figure.tight_layout()

    save_figure(
        figure,
        "figure6_known_class_acceptance",
    )


def plot_score_agreement():
    score_data = load_score_data(
        "vanilla"
    )

    score_names = [
        "msp",
        "mls",
        "energy",
        "mahalanobis",
    ]

    display_names = [
        "MSP",
        "MLS",
        "Energy",
        "Mahalanobis",
    ]

    combined_scores = []

    for score_name in score_names:
        values = np.concatenate([
            as_numpy(
                score_data[
                    score_name
                ]["test"]
            ),
            as_numpy(
                score_data[
                    score_name
                ]["near_unknown"]
            ),
            as_numpy(
                score_data[
                    score_name
                ]["far_unknown"]
            ),
        ])

        combined_scores.append(
            values
        )

    score_matrix = np.column_stack(
        combined_scores
    )

    rank_matrix = np.column_stack([
        average_ranks(
            score_matrix[
                :,
                column_index,
            ]
        )
        for column_index in range(
            score_matrix.shape[1]
        )
    ])

    correlation_matrix = (
        correlation_matrix_without_blas(
            rank_matrix
        )
    )

    figure, axis = plt.subplots(
        figsize=(7.5, 6.3),
    )

    image = axis.imshow(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    axis.set_xticks(
        range(len(display_names)),
        display_names,
    )

    axis.set_yticks(
        range(len(display_names)),
        display_names,
    )

    for row_index in range(
        len(display_names)
    ):
        for column_index in range(
            len(display_names)
        ):
            value = correlation_matrix[
                row_index,
                column_index,
            ]

            if abs(value) > 0.55:
                text_color = "white"
            else:
                text_color = "black"

            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        shrink=0.85,
    )

    colorbar.set_label(
        "Spearman rank correlation"
    )

    axis.set_title(
        "Agreement Between Vanilla "
        "Unknownness Scores",
        fontsize=14,
        pad=14,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "figure7_score_agreement",
    )


def class_acceptance_rates(
    model_name,
    score_name,
    split_name,
    class_names,
    class_to_index,
):
    outputs = load_cached_outputs(
        model_name,
        split_name,
    )

    score_data = load_score_data(
        model_name
    )

    scores = torch.as_tensor(
        score_data[
            score_name
        ][split_name],
        dtype=torch.float32,
    )

    threshold = float(
        score_data[
            score_name
        ]["threshold"]
    )

    labels = torch.as_tensor(
        outputs["labels"],
        dtype=torch.long,
    )

    rates = []

    for class_name in class_names:
        class_id = class_to_index[
            class_name
        ]

        class_mask = (
            labels == class_id
        )

        number_of_examples = (
            class_mask.sum().item()
        )

        if number_of_examples == 0:
            rates.append(np.nan)
            continue

        accepted = (
            scores[class_mask]
            <= threshold
        )

        acceptance_rate = (
            accepted
            .float()
            .mean()
            .item()
            * 100.0
        )

        rates.append(
            acceptance_rate
        )

    return rates


def plot_cross_model_classwise_acceptance():
    cifar100 = CIFAR100(
        root="data",
        train=False,
        download=False,
    )

    class_to_index = (
        cifar100.class_to_idx
    )

    methods = [
        (
            "vanilla",
            "mls",
            "Vanilla\nMLS",
        ),
        (
            "gcsc",
            "mls",
            "GCSC\nMLS",
        ),
        (
            "proser",
            "mls",
            "PROSER\nMLS",
        ),
        (
            "proser",
            "proser_placeholder",
            "PROSER\nPlaceholder",
        ),
    ]

    all_classes = (
        NEAR_CLASSES
        + FAR_CLASSES
    )

    result_matrix = np.zeros(
        (
            len(all_classes),
            len(methods),
        ),
        dtype=np.float64,
    )

    for method_index, (
        model_name,
        score_name,
        _,
    ) in enumerate(methods):
        near_rates = (
            class_acceptance_rates(
                model_name=model_name,
                score_name=score_name,
                split_name=(
                    "near_unknown"
                ),
                class_names=NEAR_CLASSES,
                class_to_index=(
                    class_to_index
                ),
            )
        )

        far_rates = (
            class_acceptance_rates(
                model_name=model_name,
                score_name=score_name,
                split_name=(
                    "far_unknown"
                ),
                class_names=FAR_CLASSES,
                class_to_index=(
                    class_to_index
                ),
            )
        )

        result_matrix[
            :,
            method_index,
        ] = near_rates + far_rates

    figure, axis = plt.subplots(
        figsize=(9.5, 10),
    )

    image = axis.imshow(
        result_matrix,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=100.0,
        aspect="auto",
    )

    axis.set_xticks(
        range(len(methods)),
        [
            method[2]
            for method in methods
        ],
    )

    axis.set_yticks(
        range(len(all_classes)),
        all_classes,
    )

    axis.axhline(
        len(NEAR_CLASSES) - 0.5,
        color="black",
        linewidth=2,
    )

    for row_index in range(
        len(all_classes)
    ):
        for column_index in range(
            len(methods)
        ):
            value = result_matrix[
                row_index,
                column_index,
            ]

            if value >= 55.0:
                text_color = "white"
            else:
                text_color = "black"

            axis.text(
                column_index,
                row_index,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    axis.text(
        -0.95,
        3.5,
        "Near",
        rotation=90,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    axis.text(
        -0.95,
        11.5,
        "Far",
        rotation=90,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        shrink=0.8,
    )

    colorbar.set_label(
        "Incorrect acceptance rate (%)"
    )

    axis.set_title(
        "Classwise Unknown Acceptance "
        "Across Models and Scores",
        fontsize=14,
        pad=14,
    )

    figure.tight_layout()

    save_figure(
        figure,
        (
            "figure8_cross_model_"
            "classwise_acceptance"
        ),
    )

    csv_path = (
        OUTPUT_DIRECTORY
        / (
            "figure8_cross_model_"
            "classwise_acceptance.csv"
        )
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "group",
            "unknown_class",
            *[
                method[2].replace(
                    "\n",
                    " ",
                )
                for method in methods
            ],
        ])

        for row_index, class_name in enumerate(
            all_classes
        ):
            if row_index < len(
                NEAR_CLASSES
            ):
                group = "near"
            else:
                group = "far"

            writer.writerow([
                group,
                class_name,
                *result_matrix[
                    row_index
                ].tolist(),
            ])

    print(f"Saved: {csv_path}")


def plot_fixed_failure_examples():
    csv_path = (
        RESULTS_ROOT
        / "vanilla"
        / "vanilla_mls_failures.csv"
    )

    with open(
        csv_path,
        "r",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    cifar100 = CIFAR100(
        root="data",
        train=False,
        download=False,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13, 8.5),
        constrained_layout=True,
    )

    flattened_axes = (
        axes.flatten()
    )

    for axis, row in zip(
        flattened_axes,
        rows,
    ):
        dataset_index = int(
            row["dataset_index"]
        )

        image, _ = cifar100[
            dataset_index
        ]

        score = float(
            row["unknownness_score"]
        )

        threshold = float(
            row["threshold"]
        )

        axis.imshow(image)
        axis.axis("off")

        axis.set_title(
            f"{row['group'].title()}: "
            f"{row['unknown_class']}\n"
            f"Predicted: "
            f"{row['predicted_known_class']}\n"
            f"MLS u(x) = {score:.3f}\n"
            f"Threshold = {threshold:.3f}",
            fontsize=10,
            pad=8,
        )

    for axis in flattened_axes[
        len(rows):
    ]:
        axis.axis("off")

    figure.suptitle(
        "Unknown Images Incorrectly "
        "Accepted by Vanilla MLS",
        fontsize=17,
    )

    save_figure(
        figure,
        "figure9_failure_examples_fixed",
    )


def main():
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_known_acceptance()
    plot_score_agreement()
    plot_cross_model_classwise_acceptance()
    plot_fixed_failure_examples()

    print(
        "\nAdditional visualization "
        "generation complete."
    )


if __name__ == "__main__":
    main()