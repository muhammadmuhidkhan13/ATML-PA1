import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from shared.pacs import PACS_CLASSES


def load_yaml(file_path):
    with Path(file_path).open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def load_json(file_path):
    with Path(file_path).open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def create_output_folder(project_root, output_folder):
    output_path = project_root / output_folder

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_path


def plot_training_curves(
    project_root,
    run_names,
    output_folder,
):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    for run_name in run_names:
        history_path = (
            project_root
            / "task2"
            / "results"
            / run_name
            / "training_history.csv"
        )

        if not history_path.exists():
            raise FileNotFoundError(
                f"Training history not found: "
                f"{history_path}"
            )

        history = pd.read_csv(history_path)

        axes[0].plot(
            history["epoch"],
            history["train_classification_loss"],
            marker="o",
            label=run_name,
        )

        if "train_alignment_loss" in history.columns:
            axes[1].plot(
                history["epoch"],
                history["train_alignment_loss"],
                marker="o",
                label=f"{run_name}: MMD",
            )

        if "train_domain_loss" in history.columns:
            axes[1].plot(
                history["epoch"],
                history["train_domain_loss"],
                marker="o",
                label=f"{run_name}: domain",
            )

        axes[2].plot(
            history["epoch"],
            history[
                "mean_source_validation_macro_f1"
            ],
            marker="o",
            label=run_name,
        )

    axes[0].set_title("Classification loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Alignment or domain loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_title(
        "Mean source-validation macro-F1"
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    figure.tight_layout()

    figure.savefig(
        output_folder / "training_curves.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def filter_summary(summary, run_names):
    filtered = summary[
        summary["run_name"].isin(run_names)
    ].copy()

    filtered["run_name"] = pd.Categorical(
        filtered["run_name"],
        categories=run_names,
        ordered=True,
    )

    return filtered.sort_values("run_name")


def plot_method_comparison(
    evaluation_folder,
    run_names,
    output_folder,
):
    summary_path = (
        evaluation_folder
        / "method_comparison.csv"
    )

    summary = pd.read_csv(summary_path)

    summary = filter_summary(
        summary=summary,
        run_names=run_names,
    )

    x_positions = np.arange(len(summary))

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    axes[0].bar(
        x_positions,
        summary["mean_source_accuracy"],
    )

    axes[0].set_title("Mean source accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")

    axes[1].bar(
        x_positions,
        summary["target_accuracy"],
    )

    axes[1].set_title("Sketch target accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Accuracy")

    axes[2].bar(
        x_positions,
        summary["domain_separability"],
    )

    axes[2].axhline(
        0.5,
        color="red",
        linestyle="--",
        label="Chance",
    )

    axes[2].set_title("Domain separability")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Held-out accuracy")
    axes[2].legend()

    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            summary["method"],
            rotation=20,
        )
        axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        output_folder / "method_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_per_class_accuracy(
    evaluation_folder,
    run_names,
    output_folder,
):
    per_class_path = (
        evaluation_folder
        / "per_class_changes.csv"
    )

    per_class = pd.read_csv(per_class_path)

    per_class = per_class[
        per_class["run_name"].isin(run_names)
    ]

    figure, axis = plt.subplots(
        figsize=(14, 6),
    )

    x_positions = np.arange(len(PACS_CLASSES))
    bar_width = 0.8 / len(run_names)

    for run_index, run_name in enumerate(
        run_names
    ):
        run_data = per_class[
            per_class["run_name"] == run_name
        ].set_index("class_name")

        run_data = run_data.reindex(PACS_CLASSES)

        positions = (
            x_positions
            - 0.4
            + bar_width / 2
            + run_index * bar_width
        )

        axis.bar(
            positions,
            run_data["method_accuracy"],
            width=bar_width,
            label=run_name,
        )

    axis.set_title(
        "Per-class Sketch target accuracy"
    )

    axis.set_xlabel("Class")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)

    axis.set_xticks(x_positions)
    axis.set_xticklabels(PACS_CLASSES)

    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)

    figure.tight_layout()

    figure.savefig(
        output_folder / "per_class_accuracy.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_confusion_matrices(
    evaluation_folder,
    run_names,
    output_folder,
):
    for run_name in run_names:
        evaluation_path = (
            evaluation_folder
            / run_name
            / "evaluation.json"
        )

        result = load_json(evaluation_path)

        matrix = np.asarray(
            result["target"]["confusion_matrix"]
        )

        figure, axis = plt.subplots(
            figsize=(8, 7),
        )

        image = axis.imshow(
            matrix,
            cmap="Blues",
        )

        figure.colorbar(image, ax=axis)

        axis.set_title(
            f"Sketch confusion matrix: "
            f"{result['method']}"
        )

        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")

        axis.set_xticks(
            np.arange(len(PACS_CLASSES))
        )

        axis.set_yticks(
            np.arange(len(PACS_CLASSES))
        )

        axis.set_xticklabels(
            PACS_CLASSES,
            rotation=45,
            ha="right",
        )

        axis.set_yticklabels(PACS_CLASSES)

        for row in range(len(PACS_CLASSES)):
            for column in range(
                len(PACS_CLASSES)
            ):
                value = matrix[row, column]

                text_color = (
                    "white"
                    if value > matrix.max() / 2
                    else "black"
                )

                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

        figure.tight_layout()

        figure.savefig(
            output_folder
            / f"confusion_{run_name}.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(figure)


def plot_controlled_grl_study(
    project_root,
    evaluation_folder,
    controlled_runs,
    output_folder,
):
    if not controlled_runs:
        return

    summary = pd.read_csv(
        evaluation_folder
        / "method_comparison.csv"
    )

    controlled_rows = []

    for run_name in controlled_runs:
        config_path = (
            project_root
            / "task2"
            / "results"
            / run_name
            / "config.yaml"
        )

        configuration = load_yaml(config_path)

        maximum_grl_strength = configuration[
            "method"
        ]["maximum_grl_strength"]

        run_summary = summary[
            summary["run_name"] == run_name
        ]

        if len(run_summary) != 1:
            raise ValueError(
                f"Missing evaluation result for: "
                f"{run_name}"
            )

        row = run_summary.iloc[0]

        controlled_rows.append(
            {
                "maximum_grl_strength": (
                    maximum_grl_strength
                ),
                "mean_source_accuracy": row[
                    "mean_source_accuracy"
                ],
                "target_accuracy": row[
                    "target_accuracy"
                ],
                "domain_separability": row[
                    "domain_separability"
                ],
            }
        )

    controlled_data = pd.DataFrame(
        controlled_rows
    ).sort_values("maximum_grl_strength")

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
    )

    columns = [
        (
            "mean_source_accuracy",
            "Mean source accuracy",
        ),
        (
            "target_accuracy",
            "Sketch target accuracy",
        ),
        (
            "domain_separability",
            "Domain separability",
        ),
    ]

    for axis, (column, title) in zip(
        axes,
        columns,
    ):
        axis.plot(
            controlled_data[
                "maximum_grl_strength"
            ],
            controlled_data[column],
            marker="o",
        )

        axis.set_title(title)
        axis.set_xlabel(
            "Maximum GRL strength"
        )
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        output_folder
        / "controlled_grl_study.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    controlled_data.to_csv(
        output_folder
        / "controlled_grl_study.csv",
        index=False,
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--controlled-runs",
        nargs="*",
        default=[],
    )

    parser.add_argument(
        "--evaluation-folder",
        default=(
            "task2/results/final_evaluation"
        ),
    )

    parser.add_argument(
        "--output-folder",
        default="task2/results/figures",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    evaluation_folder = (
        project_root
        / arguments.evaluation_folder
    )

    if not evaluation_folder.exists():
        raise FileNotFoundError(
            "Final evaluation has not been run: "
            f"{evaluation_folder}"
        )

    output_folder = create_output_folder(
        project_root=project_root,
        output_folder=arguments.output_folder,
    )

    plot_training_curves(
        project_root=project_root,
        run_names=arguments.runs,
        output_folder=output_folder,
    )

    plot_method_comparison(
        evaluation_folder=evaluation_folder,
        run_names=arguments.runs,
        output_folder=output_folder,
    )

    plot_per_class_accuracy(
        evaluation_folder=evaluation_folder,
        run_names=arguments.runs,
        output_folder=output_folder,
    )

    plot_confusion_matrices(
        evaluation_folder=evaluation_folder,
        run_names=arguments.runs,
        output_folder=output_folder,
    )

    plot_controlled_grl_study(
        project_root=project_root,
        evaluation_folder=evaluation_folder,
        controlled_runs=(
            arguments.controlled_runs
        ),
        output_folder=output_folder,
    )

    print(
        f"Figures saved to:\n{output_folder}"
    )


if __name__ == "__main__":
    main()