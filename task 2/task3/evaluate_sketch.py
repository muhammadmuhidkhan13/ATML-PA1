import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from shared.pacs import PACSLabeledDataset, get_image_transforms
from shared.pacs_protocol import load_split_data
from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead
from task3.selection.source_validation import (
    evaluate_one_domain
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


def collect_sketch_samples(
    images_root,
    class_names,
    target_domain="sketch"
):
    images_root = Path(images_root)
    target_root = images_root / target_domain
    samples = []

    if not target_root.exists():
        raise FileNotFoundError(
            f"Sketch directory not found: {target_root}"
        )

    for label, class_name in enumerate(class_names):
        class_directory = target_root / class_name

        if not class_directory.exists():
            raise FileNotFoundError(
                f"Missing Sketch class: {class_directory}"
            )

        image_paths = sorted(
            path
            for path in class_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        )

        for image_path in image_paths:
            samples.append(
                {
                    "path": image_path.relative_to(
                        images_root
                    ).as_posix(),
                    "label": label,
                    "class_name": class_name
                }
            )

    if not samples:
        raise RuntimeError(
            "No Sketch images were found."
        )

    return samples


def build_sketch_loader(
    images_root,
    split_path,
    batch_size=64,
    num_workers=0
):
    split_data = load_split_data(split_path)

    target_domain = split_data["target_domain"]
    class_names = split_data["classes"]

    if target_domain != "sketch":
        raise ValueError(
            "Task 3 final evaluation expects Sketch "
            f"but found {target_domain}."
        )

    samples = collect_sketch_samples(
        images_root=images_root,
        class_names=class_names,
        target_domain=target_domain
    )

    _, evaluation_transform = get_image_transforms()

    dataset = PACSLabeledDataset(
        images_root=images_root,
        samples=samples,
        transform=evaluation_transform
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    return loader, class_names


def load_models(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    backbone = ResNet18Backbone().to(device)

    classifier = ClassifierHead(
        feature_dimension=512,
        number_of_classes=7
    ).to(device)

    backbone.load_state_dict(
        checkpoint["backbone_state_dict"]
    )

    classifier.load_state_dict(
        checkpoint["classifier_state_dict"]
    )

    return backbone, classifier, checkpoint


def load_json(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_run_information(
    results_root,
    run_name
):
    run_directory = results_root / run_name
    checkpoint_path = (
        run_directory / "best_checkpoint.pt"
    )

    source_metrics_path = (
        run_directory
        / "best_source_validation_metrics.json"
    )

    manifest_path = (
        run_directory / "run_manifest.json"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint missing for {run_name}: "
            f"{checkpoint_path}"
        )

    if not source_metrics_path.exists():
        raise FileNotFoundError(
            f"Source metrics missing for {run_name}: "
            f"{source_metrics_path}"
        )

    source_metrics = load_json(
        source_metrics_path
    )

    manifest = (
        load_json(manifest_path)
        if manifest_path.exists()
        else {}
    )

    return {
        "run_directory": run_directory,
        "checkpoint_path": checkpoint_path,
        "source_metrics": source_metrics,
        "manifest": manifest
    }


def evaluate_run(
    run_name,
    run_information,
    sketch_loader,
    class_names,
    device
):
    backbone, classifier, checkpoint = load_models(
        checkpoint_path=(
            run_information["checkpoint_path"]
        ),
        device=device
    )

    target_metrics = evaluate_one_domain(
        backbone=backbone,
        classifier=classifier,
        data_loader=sketch_loader,
        device=device,
        number_of_classes=len(class_names)
    )

    source_summary = run_information[
        "source_metrics"
    ]["summary"]

    method = run_information[
        "manifest"
    ].get("method", run_name)

    return {
        "run_name": run_name,
        "method": method,
        "checkpoint_epoch": checkpoint["epoch"],
        "source_validation": run_information[
            "source_metrics"
        ],
        "target": {
            "domain": "sketch",
            "accuracy": target_metrics["accuracy"],
            "macro_f1": target_metrics["macro_f1"],
            "per_class_accuracy": {
                class_name: class_accuracy
                for class_name, class_accuracy in zip(
                    class_names,
                    target_metrics[
                        "per_class_accuracy"
                    ]
                )
            },
            "confusion_matrix": target_metrics[
                "confusion_matrix"
            ]
        },
        "summary": {
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
                target_metrics["accuracy"]
            ),
            "target_macro_f1": (
                target_metrics["macro_f1"]
            )
        }
    }


def create_summary_rows(
    all_results,
    erm_run
):
    erm_accuracy = all_results[
        erm_run
    ]["target"]["accuracy"]

    rows = []

    for run_name, result in all_results.items():
        summary = result["summary"]

        target_accuracy_change = (
            summary["target_accuracy"]
            - erm_accuracy
        )

        result["summary"][
            "target_accuracy_change_from_erm"
        ] = target_accuracy_change

        row = {
            "run_name": run_name,
            "method": result["method"],
            "checkpoint_epoch": (
                result["checkpoint_epoch"]
            ),
            "mean_source_accuracy": (
                summary["mean_source_accuracy"]
            ),
            "worst_source_accuracy": (
                summary["worst_source_accuracy"]
            ),
            "mean_source_macro_f1": (
                summary["mean_source_macro_f1"]
            ),
            "worst_source_macro_f1": (
                summary["worst_source_macro_f1"]
            ),
            "target_accuracy": (
                summary["target_accuracy"]
            ),
            "target_macro_f1": (
                summary["target_macro_f1"]
            ),
            "target_accuracy_change_from_erm": (
                target_accuracy_change
            )
        }

        rows.append(row)

    return rows


def save_summary_csv(rows, output_path):
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Final locked Sketch evaluation for Task 3."
        )
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--erm-run",
        default="erm_task2_checkpoint"
    )

    parser.add_argument(
        "--results-root",
        default="task3/results"
    )

    parser.add_argument(
        "--output-name",
        default="final_sketch_evaluation"
    )

    parser.add_argument(
        "--images-root",
        default="data/pacs/images"
    )

    parser.add_argument(
        "--split-path",
        default=(
            "shared/splits/"
            "pacs_sketch_seed6304.json"
        )
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--confirm-settings-locked",
        action="store_true"
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if not arguments.confirm_settings_locked:
        raise RuntimeError(
            "Sketch evaluation is locked. Complete all "
            "Task 3 training, checkpoint selection, "
            "diagnostics, and hyperparameter decisions "
            "before adding --confirm-settings-locked."
        )

    runs = list(arguments.runs)

    if arguments.erm_run not in runs:
        runs.insert(0, arguments.erm_run)

    runs = list(dict.fromkeys(runs))

    results_root = Path(arguments.results_root)

    output_directory = (
        results_root / arguments.output_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    sketch_loader, class_names = build_sketch_loader(
        images_root=arguments.images_root,
        split_path=arguments.split_path,
        batch_size=64,
        num_workers=arguments.num_workers
    )

    print("Target-label evaluation unlocked.")
    print(f"Device: {device}")
    print(
        f"Sketch examples: {len(sketch_loader.dataset)}"
    )

    all_results = {}

    for run_name in runs:
        print(f"\nEvaluating: {run_name}")

        run_information = load_run_information(
            results_root=results_root,
            run_name=run_name
        )

        result = evaluate_run(
            run_name=run_name,
            run_information=run_information,
            sketch_loader=sketch_loader,
            class_names=class_names,
            device=device
        )

        all_results[run_name] = result

        print(
            "  Source mean macro-F1: "
            f"{result['summary']['mean_source_macro_f1']:.4f}"
        )

        print(
            "  Source worst macro-F1: "
            f"{result['summary']['worst_source_macro_f1']:.4f}"
        )

        print(
            "  Sketch accuracy: "
            f"{result['target']['accuracy']:.4f}"
        )

        print(
            "  Sketch macro-F1: "
            f"{result['target']['macro_f1']:.4f}"
        )

    summary_rows = create_summary_rows(
        all_results=all_results,
        erm_run=arguments.erm_run
    )

    save_summary_csv(
        summary_rows,
        output_directory / "summary.csv"
    )

    final_results = {
        "settings_locked_before_sketch": True,
        "target_domain": "sketch",
        "class_names": class_names,
        "erm_run": arguments.erm_run,
        "runs": all_results
    }

    with open(
        output_directory / "full_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(final_results, file, indent=2)

    print("\nFinal Sketch evaluation saved to:")
    print(output_directory.resolve())


if __name__ == "__main__":
    main()