import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from task3.methods.dan_dg import compute_dan_dg_loss
from task3.methods.erm import compute_erm_loss
from task3.methods.sam import (
    apply_sam_perturbation,
    restore_sam_parameters,
    trainable_parameters
)
from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead
from task3.selection.source_validation import (
    build_source_data_loaders,
    evaluate_source_domains
)


CONFIG_PATHS = {
    "erm": Path("task3/configs/erm.yaml"),
    "dan_dg": Path("task3/configs/dan_dg.yaml"),
    "sam": Path("task3/configs/sam.yaml")
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_configuration(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    return configuration


def save_configuration(configuration, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            configuration,
            file,
            sort_keys=False
        )


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_history(history, output_path):
    if not history:
        return

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys())
        )

        writer.writeheader()
        writer.writerows(history)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def freeze_batchnorm_statistics(module):
    for child_module in module.modules():
        if isinstance(
            child_module,
            nn.modules.batchnorm._BatchNorm
        ):
            child_module.eval()


def build_models(configuration, device):
    backbone = ResNet18Backbone().to(device)

    classifier = ClassifierHead(
        feature_dimension=(
            configuration["model"]["feature_dimension"]
        ),
        number_of_classes=(
            configuration["model"]["number_of_classes"]
        )
    ).to(device)

    return backbone, classifier


def build_optimizer(
    backbone,
    classifier,
    learning_rate,
    weight_decay
):
    parameters = [
        parameter
        for parameter in list(backbone.parameters())
        + list(classifier.parameters())
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay
    )

    return optimizer


def get_next_batch(domain, loader, iterators):
    try:
        batch = next(iterators[domain])
    except StopIteration:
        iterators[domain] = iter(loader)
        batch = next(iterators[domain])

    return batch


def collect_domain_batches(
    training_loaders,
    iterators,
    device
):
    images_by_domain = {}
    labels_by_domain = {}

    for domain, loader in training_loaders.items():
        images, labels = get_next_batch(
            domain=domain,
            loader=loader,
            iterators=iterators
        )

        images_by_domain[domain] = images.to(
            device,
            non_blocking=True
        )

        labels_by_domain[domain] = labels.to(
            device,
            non_blocking=True
        )

    return images_by_domain, labels_by_domain


def forward_domains(
    backbone,
    classifier,
    images_by_domain
):
    features_by_domain = {}
    logits_by_domain = {}

    for domain, images in images_by_domain.items():
        features = backbone(images)
        logits = classifier(features)

        features_by_domain[domain] = features
        logits_by_domain[domain] = logits

    return features_by_domain, logits_by_domain


def average_statistics(statistic_totals, number_of_steps):
    return {
        name: value / number_of_steps
        for name, value in statistic_totals.items()
    }


def add_statistics(statistic_totals, statistics):
    for name, value in statistics.items():
        statistic_totals[name] = (
            statistic_totals.get(name, 0.0) + float(value)
        )


def train_dan_dg_epoch(
    backbone,
    classifier,
    optimizer,
    training_loaders,
    device,
    lambda_dg
):
    backbone.train()
    classifier.train()
    freeze_batchnorm_statistics(backbone)

    iterators = {
        domain: iter(loader)
        for domain, loader in training_loaders.items()
    }

    number_of_steps = max(
        len(loader)
        for loader in training_loaders.values()
    )

    statistic_totals = {}

    for _ in range(number_of_steps):
        images_by_domain, labels_by_domain = (
            collect_domain_batches(
                training_loaders=training_loaders,
                iterators=iterators,
                device=device
            )
        )

        optimizer.zero_grad(set_to_none=True)

        features_by_domain, logits_by_domain = (
            forward_domains(
                backbone=backbone,
                classifier=classifier,
                images_by_domain=images_by_domain
            )
        )

        total_loss, statistics = compute_dan_dg_loss(
            logits_by_domain=logits_by_domain,
            labels_by_domain=labels_by_domain,
            features_by_domain=features_by_domain,
            lambda_dg=lambda_dg
        )

        total_loss.backward()
        optimizer.step()

        add_statistics(statistic_totals, statistics)

    return average_statistics(
        statistic_totals,
        number_of_steps
    )


def train_sam_epoch(
    backbone,
    classifier,
    optimizer,
    training_loaders,
    device,
    rho
):
    backbone.train()
    classifier.train()
    freeze_batchnorm_statistics(backbone)

    iterators = {
        domain: iter(loader)
        for domain, loader in training_loaders.items()
    }

    number_of_steps = max(
        len(loader)
        for loader in training_loaders.values()
    )

    parameters = trainable_parameters(
        [backbone, classifier]
    )

    statistic_totals = {}

    for _ in range(number_of_steps):
        images_by_domain, labels_by_domain = (
            collect_domain_batches(
                training_loaders=training_loaders,
                iterators=iterators,
                device=device
            )
        )

        optimizer.zero_grad(set_to_none=True)

        _, first_logits_by_domain = forward_domains(
            backbone=backbone,
            classifier=classifier,
            images_by_domain=images_by_domain
        )

        original_loss, _ = compute_erm_loss(
            logits_by_domain=first_logits_by_domain,
            labels_by_domain=labels_by_domain
        )

        original_loss.backward()

        perturbations, gradient_norm = (
            apply_sam_perturbation(
                parameters=parameters,
                rho=rho
            )
        )

        optimizer.zero_grad(set_to_none=True)

        try:
            _, second_logits_by_domain = forward_domains(
                backbone=backbone,
                classifier=classifier,
                images_by_domain=images_by_domain
            )

            perturbed_loss, _ = compute_erm_loss(
                logits_by_domain=second_logits_by_domain,
                labels_by_domain=labels_by_domain
            )

            perturbed_loss.backward()

        finally:
            restore_sam_parameters(perturbations)

        optimizer.step()

        statistics = {
            "classification_loss": original_loss.detach().item(),
            "perturbed_loss": perturbed_loss.detach().item(),
            "sharpness_during_training": (
                perturbed_loss.detach().item()
                - original_loss.detach().item()
            ),
            "gradient_norm": gradient_norm,
            "total_loss": perturbed_loss.detach().item()
        }

        add_statistics(statistic_totals, statistics)

    return average_statistics(
        statistic_totals,
        number_of_steps
    )


def flatten_source_metrics(source_results):
    flattened = {}

    for domain, domain_results in source_results.items():
        if domain == "summary":
            continue

        flattened[
            f"{domain}_validation_accuracy"
        ] = domain_results["accuracy"]

        flattened[
            f"{domain}_validation_macro_f1"
        ] = domain_results["macro_f1"]

    summary = source_results["summary"]

    flattened["mean_source_validation_accuracy"] = (
        summary["mean_accuracy"]
    )

    flattened["worst_source_validation_accuracy"] = (
        summary["worst_accuracy"]
    )

    flattened["mean_source_validation_macro_f1"] = (
        summary["mean_macro_f1"]
    )

    flattened["worst_source_validation_macro_f1"] = (
        summary["worst_macro_f1"]
    )

    return flattened


def build_history_row(
    epoch,
    training_statistics,
    source_results
):
    row = {"epoch": epoch}

    for name, value in training_statistics.items():
        row[f"train_{name}"] = value

    row.update(flatten_source_metrics(source_results))

    return row


def save_checkpoint(
    output_path,
    epoch,
    configuration,
    backbone,
    classifier,
    optimizer,
    source_results
):
    checkpoint = {
        "epoch": epoch,
        "configuration": configuration,
        "backbone_state_dict": backbone.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "source_validation": source_results
    }

    torch.save(checkpoint, output_path)


def load_model_checkpoint(
    checkpoint_path,
    backbone,
    classifier,
    device
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    backbone.load_state_dict(
        checkpoint["backbone_state_dict"]
    )

    classifier.load_state_dict(
        checkpoint["classifier_state_dict"]
    )

    return checkpoint


def prepare_output_directory(configuration):
    output_root = Path(
        configuration["output"]["results_root"]
    )

    run_name = configuration["output"]["run_name"]
    output_directory = output_root / run_name
    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory


def register_erm_baseline(
    configuration,
    backbone,
    classifier,
    validation_loaders,
    device,
    output_directory
):
    checkpoint_path = Path(
        configuration["baseline"]["checkpoint_path"]
    )

    checkpoint = load_model_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        classifier=classifier,
        device=device
    )

    source_results = evaluate_source_domains(
        backbone=backbone,
        classifier=classifier,
        validation_loaders=validation_loaders,
        device=device,
        number_of_classes=(
            configuration["model"]["number_of_classes"]
        )
    )

    copied_checkpoint_path = (
        output_directory / "best_checkpoint.pt"
    )

    shutil.copy2(
        checkpoint_path,
        copied_checkpoint_path
    )

    save_json(
        source_results,
        output_directory
        / "best_source_validation_metrics.json"
    )

    history_row = build_history_row(
        epoch=checkpoint["epoch"],
        training_statistics={},
        source_results=source_results
    )

    save_history(
        [history_row],
        output_directory / "training_history.csv"
    )

    manifest = {
        "method": "erm",
        "status": "complete",
        "retrained": False,
        "original_checkpoint": str(checkpoint_path),
        "copied_checkpoint": str(copied_checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_selection": (
            "Task 2 mean source-validation macro-F1"
        ),
        "sketch_used": False
    }

    save_json(
        manifest,
        output_directory / "run_manifest.json"
    )

    summary = source_results["summary"]

    print("\nERM checkpoint reused without retraining.")
    print(
        "Mean source-validation accuracy: "
        f"{summary['mean_accuracy']:.4f}"
    )
    print(
        "Worst source-validation accuracy: "
        f"{summary['worst_accuracy']:.4f}"
    )
    print(
        "Mean source-validation macro-F1: "
        f"{summary['mean_macro_f1']:.4f}"
    )
    print(
        "Worst source-validation macro-F1: "
        f"{summary['worst_macro_f1']:.4f}"
    )


def train_new_method(
    configuration,
    backbone,
    classifier,
    training_loaders,
    validation_loaders,
    device,
    output_directory
):
    method_name = configuration["method"]["name"]
    training_configuration = configuration["training"]

    optimizer = build_optimizer(
        backbone=backbone,
        classifier=classifier,
        learning_rate=(
            training_configuration["learning_rate"]
        ),
        weight_decay=(
            training_configuration["weight_decay"]
        )
    )

    maximum_epochs = int(
        training_configuration["maximum_epochs"]
    )

    patience = int(
        training_configuration[
            "early_stopping_patience"
        ]
    )

    history = []
    best_score = float("-inf")
    best_epoch = None
    epochs_without_improvement = 0

    print(
        "Steps per epoch:",
        max(
            len(loader)
            for loader in training_loaders.values()
        )
    )

    for epoch in range(1, maximum_epochs + 1):
        if method_name == "dan_dg":
            training_statistics = train_dan_dg_epoch(
                backbone=backbone,
                classifier=classifier,
                optimizer=optimizer,
                training_loaders=training_loaders,
                device=device,
                lambda_dg=float(
                    configuration["method"]["lambda_dg"]
                )
            )

        elif method_name == "sam":
            training_statistics = train_sam_epoch(
                backbone=backbone,
                classifier=classifier,
                optimizer=optimizer,
                training_loaders=training_loaders,
                device=device,
                rho=float(
                    configuration["method"]["rho"]
                )
            )

        else:
            raise ValueError(
                f"Unsupported trainable method: {method_name}"
            )

        source_results = evaluate_source_domains(
            backbone=backbone,
            classifier=classifier,
            validation_loaders=validation_loaders,
            device=device,
            number_of_classes=(
                configuration["model"][
                    "number_of_classes"
                ]
            )
        )

        history_row = build_history_row(
            epoch=epoch,
            training_statistics=training_statistics,
            source_results=source_results
        )

        history.append(history_row)

        save_history(
            history,
            output_directory / "training_history.csv"
        )

        current_score = source_results[
            "summary"
        ]["mean_macro_f1"]

        total_loss = training_statistics["total_loss"]

        print(
            f"Epoch {epoch:02d} | "
            f"loss: {total_loss:.4f} | "
            f"source mean macro-F1: {current_score:.4f} | "
            f"source worst macro-F1: "
            f"{source_results['summary']['worst_macro_f1']:.4f}"
        )

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                output_path=(
                    output_directory
                    / "best_checkpoint.pt"
                ),
                epoch=epoch,
                configuration=configuration,
                backbone=backbone,
                classifier=classifier,
                optimizer=optimizer,
                source_results=source_results
            )

            save_json(
                source_results,
                output_directory
                / "best_source_validation_metrics.json"
            )

            print("  Saved new best checkpoint.")

        else:
            epochs_without_improvement += 1

            print(
                "  No improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping: {patience} epochs "
                "without improvement."
            )
            break

    manifest = {
        "method": method_name,
        "status": "complete",
        "best_epoch": best_epoch,
        "best_mean_source_validation_macro_f1": (
            best_score
        ),
        "checkpoint_selection": (
            "mean source-validation macro-F1"
        ),
        "maximum_epochs": maximum_epochs,
        "epochs_completed": len(history),
        "early_stopping_patience": patience,
        "seed": configuration["seed"],
        "sketch_used": False
    }

    if method_name == "dan_dg":
        manifest["lambda_dg"] = (
            configuration["method"]["lambda_dg"]
        )

    if method_name == "sam":
        manifest["rho"] = configuration["method"]["rho"]
        manifest["adaptive"] = (
            configuration["method"]["adaptive"]
        )

    save_json(
        manifest,
        output_directory / "run_manifest.json"
    )

    print(f"\nBest epoch: {best_epoch}")
    print(
        "Best mean source-validation macro-F1: "
        f"{best_score:.4f}"
    )


def apply_overrides(configuration, arguments):
    configuration["method"]["name"] = arguments.method

    if arguments.run_name is not None:
        configuration["output"]["run_name"] = (
            arguments.run_name
        )

    if arguments.num_workers is not None:
        configuration["data"]["num_workers"] = (
            arguments.num_workers
        )

    if (
        arguments.maximum_epochs is not None
        and arguments.method != "erm"
    ):
        configuration["training"]["maximum_epochs"] = (
            arguments.maximum_epochs
        )

    if (
        arguments.lambda_dg is not None
        and arguments.method == "dan_dg"
    ):
        configuration["method"]["lambda_dg"] = (
            arguments.lambda_dg
        )

    if (
        arguments.rho is not None
        and arguments.method == "sam"
    ):
        configuration["method"]["rho"] = arguments.rho

    return configuration


def run_experiment(arguments):
    config_path = (
        Path(arguments.config)
        if arguments.config is not None
        else CONFIG_PATHS[arguments.method]
    )

    configuration = load_configuration(config_path)

    configuration = apply_overrides(
        configuration,
        arguments
    )

    set_seed(int(configuration["seed"]))
    device = choose_device()

    output_directory = prepare_output_directory(
        configuration
    )

    save_configuration(
        configuration,
        output_directory / "config.yaml"
    )

    training_loaders, validation_loaders, split_data = (
        build_source_data_loaders(
            images_root=(
                configuration["data"]["images_root"]
            ),
            split_path=(
                configuration["data"]["split_path"]
            ),
            training_batch_size=(
                configuration["data"][
                    "source_batch_size_per_domain"
                ]
            ),
            validation_batch_size=64,
            num_workers=(
                configuration["data"]["num_workers"]
            ),
            seed=configuration["seed"]
        )
    )

    backbone, classifier = build_models(
        configuration=configuration,
        device=device
    )

    print(f"Method: {arguments.method}")
    print(
        f"Run: {configuration['output']['run_name']}"
    )
    print(f"Device: {device}")
    print(
        "Source domains:",
        ", ".join(split_data["source_domains"])
    )
    print(
        "Unavailable target:",
        split_data["target_domain"]
    )

    if arguments.method == "erm":
        register_erm_baseline(
            configuration=configuration,
            backbone=backbone,
            classifier=classifier,
            validation_loaders=validation_loaders,
            device=device,
            output_directory=output_directory
        )

    else:
        train_new_method(
            configuration=configuration,
            backbone=backbone,
            classifier=classifier,
            training_loaders=training_loaders,
            validation_loaders=validation_loaders,
            device=device,
            output_directory=output_directory
        )

    print("\nResults saved to:")
    print(output_directory.resolve())


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train Task 3 domain-generalization methods."
        )
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=["erm", "dan_dg", "sam"]
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--maximum-epochs",
        type=int,
        default=None
    )

    parser.add_argument(
        "--lambda-dg",
        type=float,
        default=None
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=None
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()
    run_experiment(arguments)


if __name__ == "__main__":
    main()