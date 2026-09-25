import argparse
import csv
import json
from pathlib import Path

import torch

from task3.evaluation.sharpness import (
    calculate_sharpness_proxy,
    load_models,
    select_fixed_validation_batch
)
from task3.evaluation.source_domain_separability import (
    calculate_source_domain_separability
)
from task3.selection.source_validation import (
    build_source_data_loaders
)


def load_json(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_json(data, file_path):
    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(data, file, indent=2)


def evaluate_source_diagnostics(
    run_name,
    results_root,
    validation_loaders,
    sharpness_images,
    sharpness_labels,
    selected_indices,
    device,
    seed=6304
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
            f"Missing checkpoint: {checkpoint_path}"
        )

    if not source_metrics_path.exists():
        raise FileNotFoundError(
            f"Missing source metrics: "
            f"{source_metrics_path}"
        )

    backbone, classifier = load_models(
        checkpoint_path=checkpoint_path,
        device=device
    )

    source_metrics = load_json(
        source_metrics_path
    )

    manifest = (
        load_json(manifest_path)
        if manifest_path.exists()
        else {}
    )

    separability_results = (
        calculate_source_domain_separability(
            backbone=backbone,
            validation_loaders=validation_loaders,
            device=device,
            seed=seed
        )
    )

    sharpness_results = calculate_sharpness_proxy(
        backbone=backbone,
        classifier=classifier,
        images=sharpness_images,
        labels=sharpness_labels,
        device=device,
        radius=0.05
    )

    sharpness_results[
        "selected_validation_indices"
    ] = selected_indices

    return {
        "run_name": run_name,
        "method": manifest.get(
            "method",
            run_name
        ),
        "source_validation": source_metrics,
        "source_domain_separability": (
            separability_results
        ),
        "sharpness": sharpness_results
    }


def build_summary_row(result):
    source_summary = result[
        "source_validation"
    ]["summary"]

    return {
        "run_name": result["run_name"],
        "method": result["method"],
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
        "source_domain_separability": (
            result["source_domain_separability"][
                "source_domain_separability"
            ]
        ),
        "domain_chance_accuracy": (
            result["source_domain_separability"][
                "chance_accuracy"
            ]
        ),
        "sharpness_original_loss": (
            result["sharpness"][
                "original_validation_loss"
            ]
        ),
        "sharpness_perturbed_loss": (
            result["sharpness"][
                "perturbed_validation_loss"
            ]
        ),
        "sharpness_proxy": (
            result["sharpness"]["sharpness_proxy"]
        )
    }


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
            "Run source-only Task 3 diagnostics "
            "without accessing Sketch."
        )
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--results-root",
        default="task3/results"
    )

    parser.add_argument(
        "--output-name",
        default="source_diagnostics"
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

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    seed = 6304
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, validation_loaders, split_data = (
        build_source_data_loaders(
            images_root=arguments.images_root,
            split_path=arguments.split_path,
            training_batch_size=8,
            validation_batch_size=64,
            num_workers=arguments.num_workers,
            seed=seed
        )
    )

    sharpness_images, sharpness_labels, indices = (
        select_fixed_validation_batch(
            validation_loaders=validation_loaders,
            examples_per_domain=32,
            seed=seed
        )
    )

    results_root = Path(arguments.results_root)

    output_directory = (
        results_root / arguments.output_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Device: {device}")
    print(
        "Source domains:",
        ", ".join(split_data["source_domains"])
    )
    print(
        "Excluded target:",
        split_data["target_domain"]
    )

    all_results = {}
    summary_rows = []

    for run_name in arguments.runs:
        print(f"\nDiagnosing: {run_name}")

        result = evaluate_source_diagnostics(
            run_name=run_name,
            results_root=results_root,
            validation_loaders=validation_loaders,
            sharpness_images=sharpness_images,
            sharpness_labels=sharpness_labels,
            selected_indices=indices,
            device=device,
            seed=seed
        )

        all_results[run_name] = result
        summary_rows.append(
            build_summary_row(result)
        )

        source_summary = result[
            "source_validation"
        ]["summary"]

        separability = result[
            "source_domain_separability"
        ]["source_domain_separability"]

        sharpness = result[
            "sharpness"
        ]["sharpness_proxy"]

        print(
            "  Mean source macro-F1: "
            f"{source_summary['mean_macro_f1']:.4f}"
        )
        print(
            "  Worst source macro-F1: "
            f"{source_summary['worst_macro_f1']:.4f}"
        )
        print(
            "  Domain separability: "
            f"{separability:.4f}"
        )
        print(
            "  Sharpness proxy: "
            f"{sharpness:.6f}"
        )

    save_summary_csv(
        summary_rows,
        output_directory / "summary.csv"
    )

    save_json(
        {
            "sketch_accessed": False,
            "source_domains": (
                split_data["source_domains"]
            ),
            "excluded_target": (
                split_data["target_domain"]
            ),
            "seed": seed,
            "runs": all_results
        },
        output_directory / "full_results.json"
    )

    print("\nSource diagnostics saved to:")
    print(output_directory.resolve())


if __name__ == "__main__":
    main()