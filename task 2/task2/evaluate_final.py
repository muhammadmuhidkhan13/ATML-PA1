import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from shared.pacs import (
    PACS_CLASSES,
    PACSLabeledDataset,
    get_image_transforms,
    list_image_paths,
)
from shared.pacs_protocol import load_or_create_split
from task2.evaluation.class_analysis import (
    compare_with_source_only,
    compute_classification_analysis,
    find_largest_confusions,
)
from task2.evaluation.domain_separability import (
    measure_domain_separability,
)
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead


SOURCE_DOMAINS = [
    "art_painting",
    "cartoon",
    "photo",
]

TARGET_DOMAIN = "sketch"
SEED = 6304
EVALUATION_BATCH_SIZE = 64


def save_json(content, output_path):
    with Path(output_path).open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(content, file, indent=2)


def save_csv(rows, output_path):
    if not rows:
        return

    with Path(output_path).open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def create_loader(dataset, number_of_workers):
    return DataLoader(
        dataset=dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=number_of_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_target_samples(
    images_root,
    target_domain=TARGET_DOMAIN,
):
    images_root = Path(images_root)
    samples = []

    for label, class_name in enumerate(PACS_CLASSES):
        class_folder = (
            images_root
            / target_domain
            / class_name
        )

        for image_path in list_image_paths(class_folder):
            relative_path = image_path.relative_to(
                images_root
            )

            samples.append(
                {
                    "path": relative_path.as_posix(),
                    "label": label,
                }
            )

    if not samples:
        raise ValueError(
            f"No target images found for {target_domain}."
        )

    return samples


def build_evaluation_data(
    project_root,
    number_of_workers,
):
    images_root = (
        project_root
        / "data"
        / "pacs"
        / "images"
    )

    split_path = (
        project_root
        / "shared"
        / "splits"
        / "pacs_sketch_seed6304.json"
    )

    split_data = load_or_create_split(
        images_root=images_root,
        split_path=split_path,
    )

    _, evaluation_transform = get_image_transforms()

    source_validation_loaders = {}

    for domain in SOURCE_DOMAINS:
        samples = split_data["domains"][domain][
            "validation"
        ]

        dataset = PACSLabeledDataset(
            images_root=images_root,
            samples=samples,
            transform=evaluation_transform,
        )

        source_validation_loaders[domain] = (
            create_loader(
                dataset=dataset,
                number_of_workers=number_of_workers,
            )
        )

    target_samples = create_target_samples(
        images_root=images_root,
    )

    target_dataset = PACSLabeledDataset(
        images_root=images_root,
        samples=target_samples,
        transform=evaluation_transform,
    )

    target_loader = create_loader(
        dataset=target_dataset,
        number_of_workers=number_of_workers,
    )

    return (
        source_validation_loaders,
        target_loader,
        target_samples,
    )


def load_trained_model(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    backbone = ResNet18Backbone().to(device)
    classifier = ClassifierHead().to(device)

    backbone.load_state_dict(
        checkpoint["backbone_state_dict"]
    )

    classifier.load_state_dict(
        checkpoint["classifier_state_dict"]
    )

    backbone.eval()
    classifier.eval()

    return backbone, classifier, checkpoint


def collect_outputs(
    backbone,
    classifier,
    data_loader,
    device,
):
    all_features = []
    all_true_labels = []
    all_predicted_labels = []

    backbone.eval()
    classifier.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            features = backbone(images)
            logits = classifier(features)
            predictions = logits.argmax(dim=1)

            all_features.append(
                features.cpu().numpy()
            )

            all_true_labels.extend(labels.tolist())

            all_predicted_labels.extend(
                predictions.cpu().tolist()
            )

    return {
        "features": np.concatenate(
            all_features,
            axis=0,
        ),
        "true_labels": all_true_labels,
        "predicted_labels": all_predicted_labels,
    }


def evaluate_source_validation(
    backbone,
    classifier,
    source_validation_loaders,
    device,
):
    domain_results = {}
    source_feature_batches = []

    for domain, data_loader in (
        source_validation_loaders.items()
    ):
        outputs = collect_outputs(
            backbone=backbone,
            classifier=classifier,
            data_loader=data_loader,
            device=device,
        )

        analysis = compute_classification_analysis(
            true_labels=outputs["true_labels"],
            predicted_labels=(
                outputs["predicted_labels"]
            ),
        )

        domain_results[domain] = analysis
        source_feature_batches.append(
            outputs["features"]
        )

    mean_accuracy = float(
        np.mean(
            [
                result["accuracy"]
                for result in domain_results.values()
            ]
        )
    )

    mean_macro_f1 = float(
        np.mean(
            [
                result["macro_f1"]
                for result in domain_results.values()
            ]
        )
    )

    source_features = np.concatenate(
        source_feature_batches,
        axis=0,
    )

    source_validation = {
        "domains": domain_results,
        "mean_source_accuracy": mean_accuracy,
        "mean_source_macro_f1": mean_macro_f1,
    }

    return source_validation, source_features


def create_prediction_rows(
    run_name,
    method_name,
    target_samples,
    true_labels,
    predicted_labels,
):
    rows = []

    for sample, true_label, prediction in zip(
        target_samples,
        true_labels,
        predicted_labels,
    ):
        rows.append(
            {
                "run_name": run_name,
                "method": method_name,
                "image_path": sample["path"],
                "true_label": true_label,
                "true_class": PACS_CLASSES[
                    true_label
                ],
                "predicted_label": prediction,
                "predicted_class": PACS_CLASSES[
                    prediction
                ],
                "correct": int(
                    true_label == prediction
                ),
            }
        )

    return rows


def evaluate_run(
    project_root,
    run_name,
    source_validation_loaders,
    target_loader,
    target_samples,
    device,
    output_root,
):
    checkpoint_path = (
        project_root
        / "task2"
        / "results"
        / run_name
        / "best_checkpoint.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    backbone, classifier, checkpoint = (
        load_trained_model(
            checkpoint_path=checkpoint_path,
            device=device,
        )
    )

    method_name = checkpoint["configuration"][
        "method"
    ]["name"]

    source_validation, source_features = (
        evaluate_source_validation(
            backbone=backbone,
            classifier=classifier,
            source_validation_loaders=(
                source_validation_loaders
            ),
            device=device,
        )
    )

    target_outputs = collect_outputs(
        backbone=backbone,
        classifier=classifier,
        data_loader=target_loader,
        device=device,
    )

    target_analysis = compute_classification_analysis(
        true_labels=target_outputs["true_labels"],
        predicted_labels=(
            target_outputs["predicted_labels"]
        ),
    )

    target_analysis["largest_confusions"] = (
        find_largest_confusions(target_analysis)
    )

    separability = measure_domain_separability(
        source_features=source_features,
        target_features=target_outputs["features"],
        seed=SEED,
    )

    result = {
        "run_name": run_name,
        "method": method_name,
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "checkpoint_selected_using": (
            "mean_source_validation_macro_f1"
        ),
        "source_validation": source_validation,
        "target": target_analysis,
        "domain_separability": separability,
        "target_labels_used_for_training": False,
        "target_labels_used_for_checkpoint_selection": (
            False
        ),
        "target_labels_used_for_final_analysis_only": (
            True
        ),
    }

    run_output_folder = output_root / run_name

    run_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_rows = create_prediction_rows(
        run_name=run_name,
        method_name=method_name,
        target_samples=target_samples,
        true_labels=target_outputs["true_labels"],
        predicted_labels=(
            target_outputs["predicted_labels"]
        ),
    )

    save_csv(
        prediction_rows,
        run_output_folder
        / "target_predictions.csv",
    )

    return result


def add_source_only_comparisons(results):
    source_only_results = [
        result
        for result in results
        if result["method"] == "source_only"
    ]

    if len(source_only_results) != 1:
        raise ValueError(
            "Exactly one Source-only run must be "
            "included in --runs."
        )

    source_only_target = source_only_results[0][
        "target"
    ]

    for result in results:
        result["relative_to_source_only"] = (
            compare_with_source_only(
                source_only_analysis=(
                    source_only_target
                ),
                method_analysis=result["target"],
            )
        )


def build_summary_rows(results):
    rows = []

    for result in results:
        source = result["source_validation"]
        target = result["target"]
        comparison = result[
            "relative_to_source_only"
        ]

        row = {
            "run_name": result["run_name"],
            "method": result["method"],
            "checkpoint_epoch": result[
                "checkpoint_epoch"
            ],
        }

        for domain in SOURCE_DOMAINS:
            row[
                f"{domain}_validation_accuracy"
            ] = source["domains"][domain][
                "accuracy"
            ]

            row[
                f"{domain}_validation_macro_f1"
            ] = source["domains"][domain][
                "macro_f1"
            ]

        row["mean_source_accuracy"] = source[
            "mean_source_accuracy"
        ]

        row["mean_source_macro_f1"] = source[
            "mean_source_macro_f1"
        ]

        row["target_accuracy"] = target[
            "accuracy"
        ]

        row["target_macro_f1"] = target[
            "macro_f1"
        ]

        row["target_accuracy_change"] = (
            comparison["target_accuracy_change"]
        )

        row["domain_separability"] = result[
            "domain_separability"
        ]["domain_separability_accuracy"]

        rows.append(row)

    return rows


def build_per_class_rows(results):
    rows = []

    for result in results:
        changes = result[
            "relative_to_source_only"
        ]["per_class_changes"]

        for change in changes:
            rows.append(
                {
                    "run_name": result["run_name"],
                    "method": result["method"],
                    **change,
                }
            )

    return rows


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--output-name",
        default="final_evaluation",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--confirm-settings-locked",
        action="store_true",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if not arguments.confirm_settings_locked:
        raise RuntimeError(
            "Final target evaluation is blocked. "
            "Lock every method, hyperparameter, and "
            "checkpoint first, then rerun with "
            "--confirm-settings-locked."
        )

    project_root = Path(__file__).resolve().parents[1]

    output_root = (
        project_root
        / "task2"
        / "results"
        / arguments.output_name
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print("Target-label evaluation unlocked.")

    (
        source_validation_loaders,
        target_loader,
        target_samples,
    ) = build_evaluation_data(
        project_root=project_root,
        number_of_workers=arguments.num_workers,
    )

    results = []

    for run_name in arguments.runs:
        print(f"\nEvaluating: {run_name}")

        result = evaluate_run(
            project_root=project_root,
            run_name=run_name,
            source_validation_loaders=(
                source_validation_loaders
            ),
            target_loader=target_loader,
            target_samples=target_samples,
            device=device,
            output_root=output_root,
        )

        results.append(result)

        print(
            "  Source macro-F1: "
            f"{result['source_validation']['mean_source_macro_f1']:.4f}"
        )

        print(
            "  Target accuracy: "
            f"{result['target']['accuracy']:.4f}"
        )

        print(
            "  Target macro-F1: "
            f"{result['target']['macro_f1']:.4f}"
        )

        print(
            "  Domain separability: "
            f"{result['domain_separability']['domain_separability_accuracy']:.4f}"
        )

    add_source_only_comparisons(results)

    for result in results:
        run_output_folder = (
            output_root / result["run_name"]
        )

        save_json(
            result,
            run_output_folder / "evaluation.json",
        )

    save_csv(
        build_summary_rows(results),
        output_root / "method_comparison.csv",
    )

    save_csv(
        build_per_class_rows(results),
        output_root / "per_class_changes.csv",
    )

    save_json(
        {
            "seed": SEED,
            "runs": arguments.runs,
            "target_domain": TARGET_DOMAIN,
            "target_labels_used_for_final_analysis_only": (
                True
            ),
            "results": results,
        },
        output_root / "all_results.json",
    )

    print(
        f"\nFinal evaluation saved to:\n{output_root}"
    )


if __name__ == "__main__":
    main()