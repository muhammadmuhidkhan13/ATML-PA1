import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


CLASS_NAMES = [
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person",
]


def to_integer_array(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    return np.asarray(values, dtype=np.int64)


def compute_classification_analysis(
    true_labels,
    predicted_labels,
    class_names=CLASS_NAMES,
):
    true_labels = to_integer_array(true_labels)
    predicted_labels = to_integer_array(
        predicted_labels
    )

    if true_labels.shape != predicted_labels.shape:
        raise ValueError(
            "true_labels and predicted_labels must "
            "have the same shape."
        )

    class_indices = list(range(len(class_names)))

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_indices,
    )

    per_class_accuracy = {}
    per_class_support = {}

    for class_index, class_name in enumerate(
        class_names
    ):
        class_total = int(matrix[class_index].sum())
        class_correct = int(
            matrix[class_index, class_index]
        )

        class_accuracy = (
            class_correct / class_total
            if class_total > 0
            else 0.0
        )

        per_class_accuracy[class_name] = float(
            class_accuracy
        )

        per_class_support[class_name] = class_total

    return {
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_accuracy": per_class_accuracy,
        "per_class_support": per_class_support,
        "confusion_matrix": matrix.tolist(),
        "number_of_examples": int(len(true_labels)),
    }


def compare_with_source_only(
    source_only_analysis,
    method_analysis,
    class_names=CLASS_NAMES,
):
    class_changes = []

    source_only_per_class = source_only_analysis[
        "per_class_accuracy"
    ]

    method_per_class = method_analysis[
        "per_class_accuracy"
    ]

    for class_name in class_names:
        baseline_accuracy = source_only_per_class[
            class_name
        ]

        method_accuracy = method_per_class[class_name]

        class_changes.append(
            {
                "class_name": class_name,
                "source_only_accuracy": float(
                    baseline_accuracy
                ),
                "method_accuracy": float(
                    method_accuracy
                ),
                "accuracy_change": float(
                    method_accuracy
                    - baseline_accuracy
                ),
            }
        )

    improvements = sorted(
        class_changes,
        key=lambda item: item["accuracy_change"],
        reverse=True,
    )

    degradations = sorted(
        class_changes,
        key=lambda item: item["accuracy_change"],
    )

    return {
        "target_accuracy_change": float(
            method_analysis["accuracy"]
            - source_only_analysis["accuracy"]
        ),
        "target_macro_f1_change": float(
            method_analysis["macro_f1"]
            - source_only_analysis["macro_f1"]
        ),
        "per_class_changes": class_changes,
        "largest_improvements": improvements[:3],
        "largest_degradations": degradations[:3],
    }


def find_largest_confusions(
    analysis,
    class_names=CLASS_NAMES,
    number_to_return=10,
):
    matrix = np.asarray(
        analysis["confusion_matrix"],
        dtype=np.int64,
    )

    confusions = []

    for true_index, true_name in enumerate(
        class_names
    ):
        for predicted_index, predicted_name in (
            enumerate(class_names)
        ):
            if true_index == predicted_index:
                continue

            count = int(
                matrix[true_index, predicted_index]
            )

            if count == 0:
                continue

            confusions.append(
                {
                    "true_class": true_name,
                    "predicted_class": predicted_name,
                    "count": count,
                }
            )

    confusions.sort(
        key=lambda item: item["count"],
        reverse=True,
    )

    return confusions[:number_to_return]


def main():
    true_labels = [0, 0, 1, 1, 2, 2, 3]
    predicted_labels = [0, 1, 1, 1, 2, 0, 3]

    analysis = compute_classification_analysis(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
    )

    print(analysis)
    print(find_largest_confusions(analysis))


if __name__ == "__main__":
    main()
