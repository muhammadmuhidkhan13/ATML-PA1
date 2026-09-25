from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = Path("task2/results")

EVALUATION_PATH = (
    RESULTS_ROOT
    / "final_evaluation"
    / "all_results.json"
)

OUTPUT_FOLDER = RESULTS_ROOT / "plots"

RUNS = {
    "Source-only": "source_only_final",
    "DAN": "dan_final",
    "DANN": "dann_disc_lr_3e_4_full",
    "CDAN": "cdan_disc_lr_1e_3_full",
}

EXPECTED_TARGET_ACCURACY = {
    "Source-only": 0.6750,
    "DAN": 0.6055,
    "DANN": 0.7233,
    "CDAN": 0.7826,
}

CLASS_NAMES = [
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person",
]

COLORS = {
    "Source-only": "#4C78A8",
    "DAN": "#F58518",
    "DANN": "#54A24B",
    "CDAN": "#E45756",
}


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_training_history(run_name):
    history_path = (
        RESULTS_ROOT
        / run_name
        / "training_history.csv"
    )

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found:\n{history_path}"
        )

    return pd.read_csv(history_path)


def find_run_record(data, run_name):
    if isinstance(data, dict):
        if run_name in data:
            return data[run_name]

        identifying_keys = [
            "run_name",
            "run",
            "name",
            "experiment_name",
        ]

        for key in identifying_keys:
            if str(data.get(key, "")) == run_name:
                return data

        for value in data.values():
            result = find_run_record(
                value,
                run_name,
            )

            if result is not None:
                return result

    elif isinstance(data, list):
        for value in data:
            result = find_run_record(
                value,
                run_name,
            )

            if result is not None:
                return result

    return None


def is_confusion_matrix(value):
    if not isinstance(value, list):
        return False

    if len(value) != len(CLASS_NAMES):
        return False

    for row in value:
        if not isinstance(row, list):
            return False

        if len(row) != len(CLASS_NAMES):
            return False

        for item in row:
            if not isinstance(item, (int, float)):
                return False

    return True


def collect_confusion_matrices(
    data,
    current_path="root",
):
    matrices = []

    if isinstance(data, dict):
        for key, value in data.items():
            child_path = (
                f"{current_path}.{key}"
            )

            if is_confusion_matrix(value):
                matrices.append(
                    (
                        child_path,
                        np.asarray(
                            value,
                            dtype=float,
                        ),
                    )
                )

            matrices.extend(
                collect_confusion_matrices(
                    value,
                    child_path,
                )
            )

    elif isinstance(data, list):
        if not is_confusion_matrix(data):
            for index, value in enumerate(data):
                child_path = (
                    f"{current_path}[{index}]"
                )

                matrices.extend(
                    collect_confusion_matrices(
                        value,
                        child_path,
                    )
                )

    return matrices


def target_path_score(path):
    path = path.lower()
    score = 0

    if "sketch" in path:
        score += 100

    if "target" in path:
        score += 60

    if "evaluation" in path:
        score += 10

    if "confusion" in path:
        score += 10

    source_terms = [
        "source",
        "validation",
        "art_painting",
        "cartoon",
        "photo",
    ]

    for term in source_terms:
        if term in path:
            score -= 50

    return score


def confusion_accuracy(confusion_matrix):
    total = confusion_matrix.sum()

    if total == 0:
        return 0.0

    return (
        np.trace(confusion_matrix)
        / total
    )


def select_target_confusion_matrix(
    evaluation_results,
    run_name,
    method_name,
):
    run_record = find_run_record(
        evaluation_results,
        run_name,
    )

    if run_record is None:
        raise KeyError(
            f"Could not find run '{run_name}' "
            f"inside {EVALUATION_PATH}."
        )

    candidates = collect_confusion_matrices(
        run_record
    )

    if not candidates:
        raise KeyError(
            f"No 7 x 7 confusion matrices were "
            f"found for '{run_name}'."
        )

    expected_accuracy = (
        EXPECTED_TARGET_ACCURACY[
            method_name
        ]
    )

    scored_candidates = []

    for path, matrix in candidates:
        path_score = target_path_score(path)
        matrix_accuracy = (
            confusion_accuracy(matrix)
        )

        accuracy_error = abs(
            matrix_accuracy
            - expected_accuracy
        )

        scored_candidates.append(
            {
                "path": path,
                "matrix": matrix,
                "path_score": path_score,
                "accuracy": matrix_accuracy,
                "accuracy_error": accuracy_error,
            }
        )

    accurate_candidates = [
        candidate
        for candidate in scored_candidates
        if candidate["accuracy_error"] < 0.002
    ]

    if not accurate_candidates:
        candidate_summary = "\n".join(
            (
                f"  {candidate['path']} | "
                f"accuracy="
                f"{candidate['accuracy']:.4f} | "
                f"path score="
                f"{candidate['path_score']}"
            )
            for candidate
            in scored_candidates
        )

        raise ValueError(
            f"No confusion matrix for "
            f"'{run_name}' matched the expected "
            f"Sketch accuracy "
            f"{expected_accuracy:.4f}.\n"
            f"Candidates found:\n"
            f"{candidate_summary}"
        )

    selected = max(
        accurate_candidates,
        key=lambda candidate: (
            candidate["path_score"],
            -candidate["accuracy_error"],
        ),
    )

    print(
        f"{method_name:12s} | "
        f"target matrix: "
        f"{selected['path']} | "
        f"accuracy: "
        f"{selected['accuracy']:.4f}"
    )

    return selected["matrix"]


def calculate_per_class_accuracy(
    confusion_matrix,
):
    row_totals = confusion_matrix.sum(axis=1)

    return np.divide(
        np.diag(confusion_matrix),
        row_totals,
        out=np.zeros(
            len(CLASS_NAMES),
            dtype=float,
        ),
        where=row_totals != 0,
    )


def get_classification_loss(history):
    possible_columns = [
        "train_classification_loss",
        "classification_loss",
        "train_loss",
    ]

    for column in possible_columns:
        if column in history.columns:
            return history[column]

    raise KeyError(
        "Classification-loss column not found.\n"
        f"Available columns: "
        f"{list(history.columns)}"
    )


def get_alignment_loss(
    history,
    method_name,
):
    if method_name == "DAN":
        possible_columns = [
            "train_alignment_loss",
            "train_mmd_loss",
        ]
    else:
        possible_columns = [
            "train_domain_loss",
            "domain_loss",
        ]

    for column in possible_columns:
        if column in history.columns:
            return history[column]

    return None


def get_validation_macro_f1(history):
    possible_columns = [
        "mean_source_validation_macro_f1",
        "source_validation_macro_f1",
        "validation_macro_f1",
    ]

    for column in possible_columns:
        if column in history.columns:
            return history[column]

    raise KeyError(
        "Source-validation macro-F1 "
        "column not found.\n"
        f"Available columns: "
        f"{list(history.columns)}"
    )


def create_per_class_tables(
    per_class_accuracies,
):
    accuracy_table = pd.DataFrame(
        per_class_accuracies,
        index=CLASS_NAMES,
    )

    accuracy_table.index.name = "class"

    change_table = pd.DataFrame(
        index=CLASS_NAMES,
    )

    source_only = accuracy_table[
        "Source-only"
    ]

    for method_name in [
        "DAN",
        "DANN",
        "CDAN",
    ]:
        change_table[method_name] = (
            accuracy_table[method_name]
            - source_only
        )

    accuracy_table.to_csv(
        OUTPUT_FOLDER
        / "task2_per_class_accuracy.csv"
    )

    change_table.to_csv(
        OUTPUT_FOLDER
        / "task2_per_class_changes.csv"
    )

    print("\nPer-class Sketch accuracy:")
    print(accuracy_table.round(4))

    print(
        "\nChange relative to Source-only:"
    )
    print(change_table.round(4))

    return change_table


def create_combined_figure(
    histories,
    change_table,
):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
    )

    classification_axis = axes[0, 0]
    alignment_axis = axes[0, 1]
    validation_axis = axes[1, 0]
    class_axis = axes[1, 1]

    for method_name, history in histories.items():
        classification_axis.plot(
            history["epoch"],
            get_classification_loss(history),
            marker="o",
            markersize=4,
            linewidth=2,
            color=COLORS[method_name],
            label=method_name,
        )

    classification_axis.set_yscale("log")
    classification_axis.set_title(
        "(a) Classification loss"
    )
    classification_axis.set_xlabel("Epoch")
    classification_axis.set_ylabel("Loss")
    classification_axis.grid(
        alpha=0.3,
        which="both",
    )
    classification_axis.legend(fontsize=8)

    for method_name in [
        "DAN",
        "DANN",
        "CDAN",
    ]:
        history = histories[method_name]

        alignment_loss = get_alignment_loss(
            history,
            method_name,
        )

        if alignment_loss is None:
            continue

        loss_name = (
            "MMD loss"
            if method_name == "DAN"
            else "Domain loss"
        )

        alignment_axis.plot(
            history["epoch"],
            alignment_loss,
            marker="o",
            markersize=4,
            linewidth=2,
            color=COLORS[method_name],
            label=(
                f"{method_name}: "
                f"{loss_name}"
            ),
        )

    alignment_axis.set_yscale("log")
    alignment_axis.set_title(
        "(b) Alignment or domain loss"
    )
    alignment_axis.set_xlabel("Epoch")
    alignment_axis.set_ylabel("Loss")
    alignment_axis.grid(
        alpha=0.3,
        which="both",
    )
    alignment_axis.legend(fontsize=8)

    for method_name, history in histories.items():
        validation_axis.plot(
            history["epoch"],
            get_validation_macro_f1(
                history
            ),
            marker="o",
            markersize=4,
            linewidth=2,
            color=COLORS[method_name],
            label=method_name,
        )

    validation_axis.set_title(
        "(c) Mean source-validation macro-F1"
    )
    validation_axis.set_xlabel("Epoch")
    validation_axis.set_ylabel("Macro-F1")
    validation_axis.set_ylim(0, 1)
    validation_axis.grid(alpha=0.3)
    validation_axis.legend(fontsize=8)

    x_positions = np.arange(
        len(CLASS_NAMES)
    )

    bar_width = 0.24

    methods = [
        "DAN",
        "DANN",
        "CDAN",
    ]

    offsets = [
        -bar_width,
        0,
        bar_width,
    ]

    for method_name, offset in zip(
        methods,
        offsets,
    ):
        class_axis.bar(
            x_positions + offset,
            change_table[
                method_name
            ].to_numpy(),
            width=bar_width,
            color=COLORS[method_name],
            label=method_name,
        )

    class_axis.axhline(
        0,
        color="black",
        linewidth=1,
    )

    class_axis.set_title(
        "(d) Per-class Sketch accuracy "
        "change relative to Source-only"
    )
    class_axis.set_xlabel("Class")
    class_axis.set_ylabel(
        "Change in accuracy"
    )
    class_axis.set_xticks(x_positions)
    class_axis.set_xticklabels(
        CLASS_NAMES,
        rotation=30,
        ha="right",
    )
    class_axis.grid(
        axis="y",
        alpha=0.3,
    )
    class_axis.legend(fontsize=8)

    figure.suptitle(
        "Task 2 training stability and "
        "class-level target performance",
        fontsize=15,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    return figure


def validate_files():
    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation file not found:\n"
            f"{EVALUATION_PATH}"
        )

    for run_name in RUNS.values():
        history_path = (
            RESULTS_ROOT
            / run_name
            / "training_history.csv"
        )

        if not history_path.exists():
            raise FileNotFoundError(
                f"Training history not found:\n"
                f"{history_path}"
            )


def main():
    validate_files()

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_results = load_json(
        EVALUATION_PATH
    )

    histories = {}
    per_class_accuracies = {}

    for method_name, run_name in RUNS.items():
        histories[method_name] = (
            load_training_history(
                run_name
            )
        )

        confusion_matrix = (
            select_target_confusion_matrix(
                evaluation_results,
                run_name,
                method_name,
            )
        )

        per_class_accuracies[
            method_name
        ] = calculate_per_class_accuracy(
            confusion_matrix
        )

    change_table = create_per_class_tables(
        per_class_accuracies
    )

    figure = create_combined_figure(
        histories,
        change_table,
    )

    pdf_path = (
        OUTPUT_FOLDER
        / "task2_combined_diagnostics.pdf"
    )

    png_path = (
        OUTPUT_FOLDER
        / "task2_combined_diagnostics.png"
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("\nCorrected figure saved to:")
    print(pdf_path.resolve())
    print(png_path.resolve())


if __name__ == "__main__":
    main()