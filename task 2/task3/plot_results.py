import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_csv(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def load_manifest(results_root, run_name):
    manifest_path = (
        results_root / run_name / "run_manifest.json"
    )

    if not manifest_path.exists():
        return {}

    return load_json(manifest_path)


def create_final_rows(
    run_names,
    source_results,
    sketch_results
):
    rows = []

    for run_name in run_names:
        source_result = source_results["runs"][run_name]
        sketch_result = sketch_results["runs"][run_name]

        source_validation = source_result[
            "source_validation"
        ]

        row = {
            "run_name": run_name,
            "method": source_result["method"]
        }

        for domain, metrics in source_validation.items():
            if domain == "summary":
                continue

            row[f"{domain}_accuracy"] = (
                metrics["accuracy"]
            )

            row[f"{domain}_macro_f1"] = (
                metrics["macro_f1"]
            )

        source_summary = source_validation["summary"]
        target_summary = sketch_result["summary"]

        row.update(
            {
                "mean_source_accuracy": (
                    source_summary["mean_accuracy"]
                ),
                "worst_source_accuracy": (
                    source_summary["worst_accuracy"]
                ),
                "mean_source_macro_f1": (
                    source_summary["mean_macro_f1"]
                ),
                "worst_source_macro_f1": (
                    source_summary["worst_macro_f1"]
                ),
                "target_accuracy": (
                    target_summary["target_accuracy"]
                ),
                "target_macro_f1": (
                    target_summary["target_macro_f1"]
                ),
                "target_accuracy_change_from_erm": (
                    target_summary[
                        "target_accuracy_change_from_erm"
                    ]
                ),
                "source_domain_separability": (
                    source_result[
                        "source_domain_separability"
                    ][
                        "source_domain_separability"
                    ]
                ),
                "sharpness_proxy": (
                    source_result[
                        "sharpness"
                    ]["sharpness_proxy"]
                )
            }
        )

        rows.append(row)

    return rows


def save_rows(rows, output_path):
    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def plot_method_comparison(rows, output_path):
    labels = [
        row["method"].upper()
        for row in rows
    ]

    metrics = [
        (
            "Mean source accuracy",
            "mean_source_accuracy"
        ),
        (
            "Worst source accuracy",
            "worst_source_accuracy"
        ),
        (
            "Sketch accuracy",
            "target_accuracy"
        ),
        (
            "Sketch macro-F1",
            "target_macro_f1"
        ),
        (
            "Sketch accuracy change from ERM",
            "target_accuracy_change_from_erm"
        ),
        (
            "Source-domain separability",
            "source_domain_separability"
        )
    ]

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    for axis, (title, field) in zip(axes.flat, metrics):
        values = [
            float(row[field])
            for row in rows
        ]

        bars = axis.bar(labels, values)
        axis.set_title(title)
        if field == "target_accuracy_change_from_erm":
            lower_limit = min(values + [0]) - 0.08
            upper_limit = max(values + [0]) + 0.08
            axis.set_ylim(lower_limit, upper_limit)
            axis.axhline(0, color="black", linewidth=0.8)
        else:
            axis.set_ylim(0, 1)
        axis.tick_params(
            axis="x",
            rotation=25
        )
        axis.grid(
            axis="y",
            alpha=0.3
        )

        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.015 if value >= 0 else -0.04),
                f"{value:.3f}",
                ha="center",
                fontsize=8,
                va="bottom" if value >= 0 else "top"
            )

        if field == "source_domain_separability":
            axis.axhline(
                1 / 3,
                color="red",
                linestyle="--",
                label="Chance"
            )
            axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(figure)


def plot_separability_vs_sketch(rows, output_path):
    figure, axis = plt.subplots(figsize=(7, 5.5))

    for row in rows:
        separability = float(
            row["source_domain_separability"]
        )
        sketch_accuracy = float(row["target_accuracy"])
        label = row["method"].upper()

        axis.scatter(
            separability,
            sketch_accuracy,
            s=90
        )
        axis.annotate(
            label,
            (separability, sketch_accuracy),
            xytext=(7, 7),
            textcoords="offset points"
        )

    axis.axvline(
        1 / 3,
        color="red",
        linestyle="--",
        label="Chance domain prediction"
    )
    axis.set_title(
        "Source-domain separability vs Sketch accuracy"
    )
    axis.set_xlabel("Source-domain separability")
    axis.set_ylabel("Sketch accuracy")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(figure)


def plot_sharpness(rows, output_path):
    labels = [
        row["method"].upper()
        for row in rows
    ]

    values = [
        float(row["sharpness_proxy"])
        for row in rows
    ]

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    bars = axis.bar(labels, values)

    axis.set_title(
        "Fixed-radius local sharpness proxy"
    )
    axis.set_ylabel(
        "Perturbed loss - original loss"
    )
    axis.tick_params(
        axis="x",
        rotation=25
    )
    axis.grid(
        axis="y",
        alpha=0.3
    )

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(figure)


def plot_training_curves(
    run_names,
    results_root,
    output_path
):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 4.5)
    )

    plotted_anything = False

    for run_name in run_names:
        history_path = (
            results_root
            / run_name
            / "training_history.csv"
        )

        if not history_path.exists():
            continue

        history = load_csv(history_path)

        if len(history) < 2:
            continue

        epochs = [
            int(row["epoch"])
            for row in history
        ]

        label = run_name

        if "train_classification_loss" in history[0]:
            classification_loss = [
                float(row["train_classification_loss"])
                for row in history
            ]

            axes[0].plot(
                epochs,
                classification_loss,
                marker="o",
                label=label
            )
            plotted_anything = True

        if "train_alignment_loss" in history[0]:
            alignment_loss = [
                float(row["train_alignment_loss"])
                for row in history
            ]

            axes[1].plot(
                epochs,
                alignment_loss,
                marker="o",
                label=f"{label}: MMD"
            )

        if "train_sharpness_during_training" in history[0]:
            training_sharpness = [
                float(
                    row[
                        "train_sharpness_during_training"
                    ]
                )
                for row in history
            ]

            axes[1].plot(
                epochs,
                training_sharpness,
                marker="o",
                label=f"{label}: SAM increase"
            )

        source_macro_f1 = [
            float(
                row[
                    "mean_source_validation_macro_f1"
                ]
            )
            for row in history
        ]

        axes[2].plot(
            epochs,
            source_macro_f1,
            marker="o",
            label=label
        )

    titles = [
        "Classification loss",
        "Method-specific training diagnostic",
        "Mean source-validation macro-F1"
    ]

    y_labels = [
        "Loss",
        "Loss contribution",
        "Macro-F1"
    ]

    for axis, title, y_label in zip(
        axes,
        titles,
        y_labels
    ):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.3)

        if axis.lines:
            axis.legend(fontsize=7)

    if plotted_anything:
        axes[0].set_yscale("log")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(figure)


def plot_per_class_accuracy(
    run_names,
    sketch_results,
    output_path
):
    class_names = sketch_results["class_names"]
    positions = np.arange(len(class_names))
    width = 0.8 / len(run_names)

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    for run_number, run_name in enumerate(run_names):
        result = sketch_results[
            "runs"
        ][run_name]

        per_class = result[
            "target"
        ]["per_class_accuracy"]

        values = [
            per_class[class_name]
            for class_name in class_names
        ]

        offset = (
            run_number
            - (len(run_names) - 1) / 2
        ) * width

        axis.bar(
            positions + offset,
            values,
            width=width,
            label=result["method"].upper()
        )

    axis.set_title(
        "Per-class Sketch accuracy"
    )
    axis.set_xlabel("Class")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    axis.set_xticks(positions)
    axis.set_xticklabels(class_names)
    axis.grid(
        axis="y",
        alpha=0.3
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(figure)


def plot_confusion_matrices(
    run_names,
    sketch_results,
    output_directory
):
    class_names = sketch_results["class_names"]

    for run_name in run_names:
        result = sketch_results[
            "runs"
        ][run_name]

        confusion_matrix = np.array(
            result["target"]["confusion_matrix"]
        )

        figure, axis = plt.subplots(
            figsize=(8, 7)
        )

        image = axis.imshow(
            confusion_matrix,
            cmap="Blues"
        )

        axis.set_title(
            "Sketch confusion matrix: "
            f"{result['method'].upper()}"
        )
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(
            range(len(class_names))
        )
        axis.set_yticks(
            range(len(class_names))
        )
        axis.set_xticklabels(
            class_names,
            rotation=45,
            ha="right"
        )
        axis.set_yticklabels(class_names)

        threshold = confusion_matrix.max() / 2

        for row in range(confusion_matrix.shape[0]):
            for column in range(
                confusion_matrix.shape[1]
            ):
                value = confusion_matrix[row, column]

                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if value > threshold
                        else "black"
                    )
                )

        figure.colorbar(image, ax=axis)
        figure.tight_layout()

        figure.savefig(
            output_directory
            / f"confusion_{run_name}.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(figure)


def plot_controlled_study(
    controlled_runs,
    results_root,
    source_results,
    sketch_results,
    output_path
):
    records = []

    for run_name in controlled_runs:
        manifest = load_manifest(
            results_root,
            run_name
        )

        lambda_dg = float(
            manifest["lambda_dg"]
        )

        source_result = source_results[
            "runs"
        ][run_name]

        sketch_result = sketch_results[
            "runs"
        ][run_name]

        history_path = (
            results_root
            / run_name
            / "training_history.csv"
        )

        history = load_csv(history_path)

        best_history_row = max(
            history,
            key=lambda row: float(
                row[
                    "mean_source_validation_macro_f1"
                ]
            )
        )

        alignment_loss = float(
            best_history_row["train_alignment_loss"]
        )

        records.append(
            {
                "lambda_dg": lambda_dg,
                "mean_source_accuracy": (
                    source_result[
                        "source_validation"
                    ]["summary"]["mean_accuracy"]
                ),
                "worst_source_accuracy": (
                    source_result[
                        "source_validation"
                    ]["summary"]["worst_accuracy"]
                ),
                "target_accuracy": (
                    sketch_result[
                        "target"
                    ]["accuracy"]
                ),
                "separability": (
                    source_result[
                        "source_domain_separability"
                    ][
                        "source_domain_separability"
                    ]
                ),
                "alignment_loss": alignment_loss
            }
        )

    records.sort(
        key=lambda record: record["lambda_dg"]
    )

    lambdas = [
        record["lambda_dg"]
        for record in records
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 9)
    )

    axes = axes.flat

    axes[0].plot(
        lambdas,
        [
            record["mean_source_accuracy"]
            for record in records
        ],
        marker="o",
        label="Mean source"
    )

    axes[0].plot(
        lambdas,
        [
            record["worst_source_accuracy"]
            for record in records
        ],
        marker="o",
        label="Worst source"
    )

    axes[0].set_title("Source performance")
    axes[0].legend()

    axes[1].plot(
        lambdas,
        [
            record["target_accuracy"]
            for record in records
        ],
        marker="o"
    )

    axes[1].set_title("Sketch accuracy")

    axes[2].plot(
        lambdas,
        [
            record["separability"]
            for record in records
        ],
        marker="o"
    )

    axes[2].axhline(
        1 / 3,
        color="red",
        linestyle="--",
        label="Chance"
    )

    axes[2].set_title(
        "Source-domain separability"
    )
    axes[2].legend()

    axes[3].plot(
        lambdas,
        [
            record["alignment_loss"]
            for record in records
        ],
        marker="o"
    )

    axes[3].set_title(
        "MMD at selected source checkpoint"
    )
    axes[3].set_ylabel("Unweighted MMD")

    for axis in axes:
        axis.set_xlabel(
            r"$\lambda_{\mathrm{DG}}$"
        )
        axis.grid(alpha=0.3)
        axis.set_xscale("log")

    for axis in axes[:3]:
        axis.set_ylim(0, 1)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create Task 3 tables and plots."
        )
    )

    parser.add_argument(
        "--main-runs",
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--controlled-runs",
        nargs="*",
        default=[]
    )

    parser.add_argument(
        "--results-root",
        default="task3/results"
    )

    parser.add_argument(
        "--source-folder",
        default="source_diagnostics"
    )

    parser.add_argument(
        "--sketch-folder",
        default="final_sketch_evaluation"
    )

    parser.add_argument(
        "--output-folder",
        default="plots"
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    results_root = Path(arguments.results_root)

    source_results = load_json(
        results_root
        / arguments.source_folder
        / "full_results.json"
    )

    sketch_results = load_json(
        results_root
        / arguments.sketch_folder
        / "full_results.json"
    )

    output_directory = (
        results_root / arguments.output_folder
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = create_final_rows(
        run_names=arguments.main_runs,
        source_results=source_results,
        sketch_results=sketch_results
    )

    save_rows(
        rows,
        output_directory
        / "final_comparison_table.csv"
    )

    plot_method_comparison(
        rows,
        output_directory
        / "method_comparison.png"
    )

    plot_separability_vs_sketch(
        rows,
        output_directory
        / "separability_vs_sketch.png"
    )

    plot_sharpness(
        rows,
        output_directory
        / "sharpness_comparison.png"
    )

    plot_training_curves(
        run_names=arguments.main_runs,
        results_root=results_root,
        output_path=(
            output_directory
            / "training_curves.png"
        )
    )

    plot_per_class_accuracy(
        run_names=arguments.main_runs,
        sketch_results=sketch_results,
        output_path=(
            output_directory
            / "per_class_accuracy.png"
        )
    )

    plot_confusion_matrices(
        run_names=arguments.main_runs,
        sketch_results=sketch_results,
        output_directory=output_directory
    )

    if arguments.controlled_runs:
        plot_controlled_study(
            controlled_runs=(
                arguments.controlled_runs
            ),
            results_root=results_root,
            source_results=source_results,
            sketch_results=sketch_results,
            output_path=(
                output_directory
                / "controlled_lambda_study.png"
            )
        )

    print("Plots and final table saved to:")
    print(output_directory.resolve())


if __name__ == "__main__":
    main()
