import argparse
import csv
import json
from pathlib import Path

import torch

from task4.evaluation.metrics import closed_set_accuracy
from task4.evaluation.metrics import known_unknown_auroc
from task4.evaluation.metrics import operating_point_metrics
from task4.evaluation.thresholds import validation_threshold
from task4.methods.proser import placeholder_unknownness
from task4.scores.energy import energy_unknownness
from task4.scores.mahalanobis import fit_mahalanobis
from task4.scores.mahalanobis import mahalanobis_unknownness
from task4.scores.mls import mls_unknownness
from task4.scores.msp import msp_unknownness


BASE_SCORE_NAMES = [
    "msp",
    "mls",
    "energy",
    "mahalanobis"
]


def load_outputs(cache_directory, split_name):
    path = (
        cache_directory
        / f"{split_name}_outputs.pt"
    )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


def calculate_score(
    score_name,
    outputs,
    known_classes,
    class_means=None,
    covariance_diagonal=None,
    placeholder_temperature=1024.0
):
    known_logits = outputs["logits"][
        :, :known_classes
    ]

    if score_name == "msp":
        return msp_unknownness(known_logits)

    if score_name == "mls":
        return mls_unknownness(known_logits)

    if score_name == "energy":
        return energy_unknownness(
            known_logits
        )

    if score_name == "mahalanobis":
        return mahalanobis_unknownness(
            outputs["features"],
            class_means,
            covariance_diagonal
        )

    if score_name == "proser_placeholder":
        return placeholder_unknownness(
            outputs["logits"],
            known_classes=known_classes,
            temperature=placeholder_temperature
        )

    raise ValueError(
        f"Unknown score: {score_name}"
    )


def save_csv(rows, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--known-classes",
        type=int,
        default=10
    )

    parser.add_argument(
        "--placeholder-temperature",
        type=float,
        default=1024.0
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

    results_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    train_outputs = load_outputs(
        cache_directory,
        "train"
    )

    validation_outputs = load_outputs(
        cache_directory,
        "validation"
    )

    test_outputs = load_outputs(
        cache_directory,
        "test"
    )

    near_outputs = load_outputs(
        cache_directory,
        "near_unknown"
    )

    far_outputs = load_outputs(
        cache_directory,
        "far_unknown"
    )

    total_outputs = validation_outputs[
        "logits"
    ].size(1)

    score_names = list(BASE_SCORE_NAMES)

    if total_outputs > arguments.known_classes:
        score_names.append(
            "proser_placeholder"
        )

    class_means, covariance_diagonal = (
        fit_mahalanobis(
            train_outputs["features"],
            train_outputs["labels"],
            num_classes=arguments.known_classes
        )
    )

    torch.save({
        "class_means": class_means,
        "covariance_diagonal":
            covariance_diagonal
    }, results_directory
       / "mahalanobis_statistics.pt")

    known_test_logits = test_outputs[
        "logits"
    ][:, :arguments.known_classes]

    known_test_predictions = (
        known_test_logits.argmax(dim=1)
    )

    csa = closed_set_accuracy(
        known_test_predictions.numpy(),
        test_outputs["labels"].numpy()
    )

    rows = []
    saved_scores = {}

    for score_name in score_names:
        validation_scores = calculate_score(
            score_name,
            validation_outputs,
            arguments.known_classes,
            class_means,
            covariance_diagonal,
            arguments.placeholder_temperature
        )

        test_scores = calculate_score(
            score_name,
            test_outputs,
            arguments.known_classes,
            class_means,
            covariance_diagonal,
            arguments.placeholder_temperature
        )

        near_scores = calculate_score(
            score_name,
            near_outputs,
            arguments.known_classes,
            class_means,
            covariance_diagonal,
            arguments.placeholder_temperature
        )

        far_scores = calculate_score(
            score_name,
            far_outputs,
            arguments.known_classes,
            class_means,
            covariance_diagonal,
            arguments.placeholder_temperature
        )

        all_unknown_scores = torch.cat([
            near_scores,
            far_scores
        ])

        threshold = validation_threshold(
            validation_scores,
            percentile=0.95
        )

        near_operating = (
            operating_point_metrics(
                test_scores,
                near_scores,
                threshold
            )
        )

        far_operating = (
            operating_point_metrics(
                test_scores,
                far_scores,
                threshold
            )
        )

        all_operating = (
            operating_point_metrics(
                test_scores,
                all_unknown_scores,
                threshold
            )
        )

        row = {
            "model": arguments.model_name,
            "score": score_name,
            "closed_set_accuracy": csa,
            "threshold": threshold,
            "known_test_acceptance_rate":
                all_operating[
                    "known_test_acceptance_rate"
                ],
            "near_auroc":
                known_unknown_auroc(
                    test_scores,
                    near_scores
                ),
            "far_auroc":
                known_unknown_auroc(
                    test_scores,
                    far_scores
                ),
            "all_unknown_auroc":
                known_unknown_auroc(
                    test_scores,
                    all_unknown_scores
                ),
            "near_unknown_rejection_rate":
                near_operating[
                    "unknown_rejection_rate"
                ],
            "far_unknown_rejection_rate":
                far_operating[
                    "unknown_rejection_rate"
                ],
            "all_unknown_rejection_rate":
                all_operating[
                    "unknown_rejection_rate"
                ],
            "near_fpr_at_95_tpr":
                near_operating[
                    "fpr_at_95_tpr"
                ],
            "far_fpr_at_95_tpr":
                far_operating[
                    "fpr_at_95_tpr"
                ],
            "all_unknown_fpr_at_95_tpr":
                all_operating[
                    "fpr_at_95_tpr"
                ]
        }

        rows.append(row)

        saved_scores[score_name] = {
            "threshold": threshold,
            "validation": validation_scores,
            "test": test_scores,
            "near_unknown": near_scores,
            "far_unknown": far_scores
        }

    save_csv(
        rows,
        results_directory
        / "posthoc_osr_metrics.csv"
    )

    torch.save(
        saved_scores,
        results_directory
        / "posthoc_scores.pt"
    )

    summary = {
        "model": arguments.model_name,
        "closed_set_accuracy": csa,
        "known_classes":
            arguments.known_classes,
        "total_outputs": total_outputs,
        "scores": score_names
    }

    with open(
        results_directory / "osr_summary.json",
        "w"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2
        )

    print(
        f"Closed-set accuracy: {csa:.4f}"
    )

    for row in rows:
        print(
            f"{row['score']:20s} | "
            f"near AUROC: "
            f"{row['near_auroc']:.4f} | "
            f"far AUROC: "
            f"{row['far_auroc']:.4f} | "
            f"all AUROC: "
            f"{row['all_unknown_auroc']:.4f}"
        )

    print(
        "\nResults saved to: "
        f"{results_directory}"
    )


if __name__ == "__main__":
    main()