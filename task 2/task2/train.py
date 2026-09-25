import argparse
import copy
import csv
import json
import platform
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from shared.pacs import (
    PACSLabeledDataset,
    PACSUnlabeledDataset,
    get_image_transforms,
)
from shared.pacs_protocol import (
    load_or_create_split,
)
from task2.evaluation.metrics import (
    evaluate_source_domains,
)
from task2.methods.cdan import (
    CDAN_INPUT_DIMENSION,
    compute_cdan_loss,
)
from task2.methods.dan import compute_dan_loss
from task2.methods.dann import compute_dann_loss
from task2.methods.source_only import (
    compute_source_only_loss,
)
from task2.models.backbone import (
    FEATURE_DIMENSION,
    ResNet18Backbone,
)
from task2.models.classifier_head import (
    ClassifierHead,
)
from task2.models.domain_discriminator import (
    DomainDiscriminator,
)


VALIDATION_BATCH_SIZE = 64


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge_dictionaries(
    base_dictionary,
    override_dictionary,
):
    merged = copy.deepcopy(base_dictionary)

    for key, value in override_dictionary.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_dictionaries(
                merged[key],
                value,
            )
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def load_yaml_file(file_path):
    with Path(file_path).open(
        mode="r",
        encoding="utf-8",
    ) as file:
        content = yaml.safe_load(file)

    if content is None:
        content = {}

    return content


def load_configuration(
    project_root,
    method_name,
):
    config_folder = (
        project_root
        / "task2"
        / "configs"
    )

    base_config = load_yaml_file(
        config_folder / "base.yaml"
    )

    method_config = load_yaml_file(
        config_folder
        / f"{method_name}.yaml"
    )

    return merge_dictionaries(
        base_config,
        method_config,
    )


def apply_command_line_overrides(
    configuration,
    arguments,
):
    configuration = copy.deepcopy(
        configuration
    )

    if arguments.maximum_epochs is not None:
        configuration["training"][
            "maximum_epochs"
        ] = arguments.maximum_epochs

    if arguments.mmd_weight is not None:
        if arguments.method != "dan":
            raise ValueError(
                "--mmd-weight can only be used "
                "with DAN."
            )

        configuration["method"][
            "mmd_weight"
        ] = arguments.mmd_weight

    if (
        arguments.maximum_grl_strength
        is not None
    ):
        if arguments.method not in {
            "dann",
            "cdan",
        }:
            raise ValueError(
                "--maximum-grl-strength can "
                "only be used with DANN or CDAN."
            )

        configuration["method"][
            "maximum_grl_strength"
        ] = arguments.maximum_grl_strength

    if arguments.method in {
        "dann",
        "cdan",
    }:
        configuration["training"].setdefault(
            "discriminator_learning_rate",
            configuration["training"][
                "learning_rate"
            ],
        )

    if (
        arguments.discriminator_learning_rate
        is not None
    ):
        if arguments.method not in {
            "dann",
            "cdan",
        }:
            raise ValueError(
                "--discriminator-learning-rate "
                "can only be used with DANN or CDAN."
            )

        if (
            arguments.discriminator_learning_rate
            <= 0
        ):
            raise ValueError(
                "--discriminator-learning-rate "
                "must be greater than zero."
            )

        configuration["training"][
            "discriminator_learning_rate"
        ] = (
            arguments.discriminator_learning_rate
        )

    run_name = (
        arguments.run_name
        if arguments.run_name is not None
        else arguments.method
    )

    configuration["run_name"] = run_name

    return configuration


def create_data_loader(
    dataset,
    batch_size,
    shuffle,
    drop_last,
    number_of_workers,
    seed,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=number_of_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def build_data_loaders(
    project_root,
    configuration,
):
    data_config = configuration["data"]
    seed = configuration["seed"]

    images_root = (
        project_root
        / data_config["images_root"]
    )

    split_path = (
        project_root
        / data_config["split_path"]
    )

    split_data = load_or_create_split(
        images_root=images_root,
        split_path=split_path,
    )

    train_transform, evaluation_transform = (
        get_image_transforms()
    )

    source_train_loaders = {}
    source_validation_loaders = {}

    for domain_index, domain in enumerate(
        data_config["source_domains"]
    ):
        domain_split = split_data["domains"][
            domain
        ]

        training_dataset = PACSLabeledDataset(
            images_root=images_root,
            samples=domain_split["train"],
            transform=train_transform,
        )

        validation_dataset = PACSLabeledDataset(
            images_root=images_root,
            samples=domain_split["validation"],
            transform=evaluation_transform,
        )

        source_train_loaders[domain] = (
            create_data_loader(
                dataset=training_dataset,
                batch_size=data_config[
                    "source_batch_size_per_domain"
                ],
                shuffle=True,
                drop_last=True,
                number_of_workers=data_config[
                    "num_workers"
                ],
                seed=seed + domain_index,
            )
        )

        source_validation_loaders[domain] = (
            create_data_loader(
                dataset=validation_dataset,
                batch_size=VALIDATION_BATCH_SIZE,
                shuffle=False,
                drop_last=False,
                number_of_workers=data_config[
                    "num_workers"
                ],
                seed=seed,
            )
        )

    target_train_loader = None

    if configuration["method"][
        "uses_unlabeled_target"
    ]:
        target_dataset = PACSUnlabeledDataset(
            images_root=images_root,
            domain=data_config["target_domain"],
            transform=train_transform,
        )

        target_train_loader = create_data_loader(
            dataset=target_dataset,
            batch_size=data_config[
                "target_batch_size"
            ],
            shuffle=True,
            drop_last=True,
            number_of_workers=data_config[
                "num_workers"
            ],
            seed=seed + 100,
        )

    return (
        source_train_loaders,
        source_validation_loaders,
        target_train_loader,
    )


def build_models(
    configuration,
    device,
):
    method_name = configuration["method"]["name"]

    backbone = ResNet18Backbone().to(device)
    classifier = ClassifierHead().to(device)

    domain_discriminator = None

    if method_name == "dann":
        discriminator_config = configuration[
            "domain_discriminator"
        ]

        domain_discriminator = DomainDiscriminator(
            input_dimension=FEATURE_DIMENSION,
            hidden_dimension=discriminator_config[
                "hidden_dimension"
            ],
            dropout_probability=discriminator_config[
                "dropout_probability"
            ],
        ).to(device)

    elif method_name == "cdan":
        discriminator_config = configuration[
            "domain_discriminator"
        ]

        domain_discriminator = DomainDiscriminator(
            input_dimension=CDAN_INPUT_DIMENSION,
            hidden_dimension=discriminator_config[
                "hidden_dimension"
            ],
            dropout_probability=discriminator_config[
                "dropout_probability"
            ],
        ).to(device)

    return (
        backbone,
        classifier,
        domain_discriminator,
    )


def build_optimizer(
    configuration,
    backbone,
    classifier,
    domain_discriminator,
):
    task_parameters = list(
        backbone.parameters()
    )

    task_parameters += list(
        classifier.parameters()
    )

    training_config = configuration["training"]

    learning_rate = training_config[
        "learning_rate"
    ]

    if domain_discriminator is None:
        parameter_groups = task_parameters

    else:
        discriminator_learning_rate = (
            training_config.get(
                "discriminator_learning_rate",
                learning_rate,
            )
        )

        parameter_groups = [
            {
                "params": task_parameters,
                "lr": learning_rate,
            },
            {
                "params": (
                    domain_discriminator.parameters()
                ),
                "lr": (
                    discriminator_learning_rate
                ),
            },
        ]

    optimizer = AdamW(
        parameter_groups,
        lr=learning_rate,
        weight_decay=training_config[
            "weight_decay"
        ],
    )

    return optimizer


def get_next_batch(
    data_loader,
    data_iterator,
):
    try:
        batch = next(data_iterator)

    except StopIteration:
        data_iterator = iter(data_loader)
        batch = next(data_iterator)

    return batch, data_iterator


def calculate_steps_per_epoch(
    source_train_loaders,
    configuration,
):
    rule = configuration["training"][
        "steps_per_epoch"
    ]

    if rule != "longest_source_loader":
        raise ValueError(
            f"Unsupported epoch rule: {rule}"
        )

    return max(
        len(loader)
        for loader in source_train_loaders.values()
    )


def train_one_epoch(
    configuration,
    backbone,
    classifier,
    domain_discriminator,
    optimizer,
    source_train_loaders,
    target_train_loader,
    device,
    global_step,
    total_planned_steps,
):
    backbone.train()
    classifier.train()

    if domain_discriminator is not None:
        domain_discriminator.train()

    source_iterators = {
        domain: iter(loader)
        for domain, loader in (
            source_train_loaders.items()
        )
    }

    target_iterator = (
        iter(target_train_loader)
        if target_train_loader is not None
        else None
    )

    steps_per_epoch = calculate_steps_per_epoch(
        source_train_loaders,
        configuration,
    )

    statistic_totals = defaultdict(float)

    for _ in range(steps_per_epoch):
        source_image_batches = []
        source_label_batches = []

        for domain, loader in (
            source_train_loaders.items()
        ):
            batch, updated_iterator = (
                get_next_batch(
                    data_loader=loader,
                    data_iterator=source_iterators[
                        domain
                    ],
                )
            )

            source_iterators[domain] = (
                updated_iterator
            )

            images, labels = batch

            source_image_batches.append(images)
            source_label_batches.append(labels)

        source_images = torch.cat(
            source_image_batches,
            dim=0,
        ).to(
            device,
            non_blocking=True,
        )

        source_labels = torch.cat(
            source_label_batches,
            dim=0,
        ).to(
            device,
            non_blocking=True,
        )

        target_images = None

        if target_train_loader is not None:
            target_images, target_iterator = (
                get_next_batch(
                    data_loader=target_train_loader,
                    data_iterator=target_iterator,
                )
            )

            target_images = target_images.to(
                device,
                non_blocking=True,
            )

        optimizer.zero_grad(set_to_none=True)

        if target_images is None:
            source_features = backbone(
                source_images
            )

            target_features = None

        else:
            combined_images = torch.cat(
                [
                    source_images,
                    target_images,
                ],
                dim=0,
            )

            combined_features = backbone(
                combined_images
            )

            number_of_source_examples = (
                source_images.shape[0]
            )

            source_features = combined_features[
                :number_of_source_examples
            ]

            target_features = combined_features[
                number_of_source_examples:
            ]

        source_logits = classifier(
            source_features
        )

        progress = global_step / max(
            total_planned_steps - 1,
            1,
        )

        method_name = configuration["method"][
            "name"
        ]

        if method_name == "source_only":
            total_loss, statistics = (
                compute_source_only_loss(
                    source_logits=source_logits,
                    source_labels=source_labels,
                )
            )

        elif method_name == "dan":
            total_loss, statistics = (
                compute_dan_loss(
                    source_logits=source_logits,
                    source_labels=source_labels,
                    source_features=source_features,
                    target_features=target_features,
                    mmd_weight=configuration[
                        "method"
                    ]["mmd_weight"],
                )
            )

        elif method_name == "dann":
            total_loss, statistics = (
                compute_dann_loss(
                    source_logits=source_logits,
                    source_labels=source_labels,
                    source_features=source_features,
                    target_features=target_features,
                    domain_discriminator=(
                        domain_discriminator
                    ),
                    progress=progress,
                    maximum_grl_strength=(
                        configuration["method"][
                            "maximum_grl_strength"
                        ]
                    ),
                    domain_loss_weight=(
                        configuration["method"][
                            "domain_loss_weight"
                        ]
                    ),
                )
            )

        elif method_name == "cdan":
            target_logits = classifier(
                target_features
            )

            total_loss, statistics = (
                compute_cdan_loss(
                    source_logits=source_logits,
                    target_logits=target_logits,
                    source_labels=source_labels,
                    source_features=source_features,
                    target_features=target_features,
                    domain_discriminator=(
                        domain_discriminator
                    ),
                    progress=progress,
                    maximum_grl_strength=(
                        configuration["method"][
                            "maximum_grl_strength"
                        ]
                    ),
                    domain_loss_weight=(
                        configuration["method"][
                            "domain_loss_weight"
                        ]
                    ),
                )
            )

        else:
            raise ValueError(
                f"Unknown method: {method_name}"
            )

        total_loss.backward()
        optimizer.step()

        for key, value in statistics.items():
            statistic_totals[key] += float(value)

        global_step += 1

    mean_statistics = {
        key: value / steps_per_epoch
        for key, value in statistic_totals.items()
    }

    return mean_statistics, global_step


def build_history_row(
    epoch,
    training_statistics,
    validation_result,
):
    row = {
        "epoch": epoch,
    }

    for key, value in (
        training_statistics.items()
    ):
        row[f"train_{key}"] = value

    for domain, metrics in (
        validation_result["domains"].items()
    ):
        row[
            f"{domain}_validation_accuracy"
        ] = metrics["accuracy"]

        row[
            f"{domain}_validation_macro_f1"
        ] = metrics["macro_f1"]

    row["mean_source_validation_accuracy"] = (
        validation_result[
            "mean_source_accuracy"
        ]
    )

    row["mean_source_validation_macro_f1"] = (
        validation_result[
            "mean_source_macro_f1"
        ]
    )

    return row


def save_history(
    history,
    output_path,
):
    fieldnames = []

    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with Path(output_path).open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


def save_json(
    content,
    output_path,
):
    with Path(output_path).open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            content,
            file,
            indent=2,
        )


def save_configuration(
    configuration,
    output_path,
):
    with Path(output_path).open(
        mode="w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            configuration,
            file,
            sort_keys=False,
        )


def save_checkpoint(
    checkpoint_path,
    epoch,
    configuration,
    backbone,
    classifier,
    domain_discriminator,
    optimizer,
    validation_result,
):
    checkpoint = {
        "epoch": epoch,
        "configuration": configuration,
        "backbone_state_dict": (
            backbone.state_dict()
        ),
        "classifier_state_dict": (
            classifier.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "source_validation": (
            validation_result
        ),
    }

    if domain_discriminator is not None:
        checkpoint[
            "domain_discriminator_state_dict"
        ] = domain_discriminator.state_dict()

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def train_experiment(
    project_root,
    configuration,
    device,
):
    (
        source_train_loaders,
        source_validation_loaders,
        target_train_loader,
    ) = build_data_loaders(
        project_root=project_root,
        configuration=configuration,
    )

    (
        backbone,
        classifier,
        domain_discriminator,
    ) = build_models(
        configuration=configuration,
        device=device,
    )

    optimizer = build_optimizer(
        configuration=configuration,
        backbone=backbone,
        classifier=classifier,
        domain_discriminator=(
            domain_discriminator
        ),
    )

    print(
        "Backbone/classifier learning rate: "
        f"{configuration['training']['learning_rate']}"
    )

    if domain_discriminator is not None:
        print(
            "Discriminator learning rate: "
            f"{configuration['training']['discriminator_learning_rate']}"
        )

    output_root = (
        project_root
        / configuration["output"]["results_root"]
        / configuration["run_name"]
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_configuration(
        configuration,
        output_root / "config.yaml",
    )

    steps_per_epoch = calculate_steps_per_epoch(
        source_train_loaders,
        configuration,
    )

    maximum_epochs = configuration["training"][
        "maximum_epochs"
    ]

    grl_schedule_epochs = configuration["training"][
        "grl_schedule_epochs"
    ]

    total_planned_steps = (
        steps_per_epoch
        * grl_schedule_epochs
    )

    patience_limit = configuration["training"][
        "early_stopping_patience"
    ]

    best_score = float("-inf")
    best_epoch = None
    epochs_without_improvement = 0
    global_step = 0
    history = []

    print(
        f"Steps per epoch: {steps_per_epoch}"
    )

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        training_statistics, global_step = (
            train_one_epoch(
                configuration=configuration,
                backbone=backbone,
                classifier=classifier,
                domain_discriminator=(
                    domain_discriminator
                ),
                optimizer=optimizer,
                source_train_loaders=(
                    source_train_loaders
                ),
                target_train_loader=(
                    target_train_loader
                ),
                device=device,
                global_step=global_step,
                total_planned_steps=(
                    total_planned_steps
                ),
            )
        )

        validation_result = (
            evaluate_source_domains(
                backbone=backbone,
                classifier=classifier,
                validation_loaders=(
                    source_validation_loaders
                ),
                device=device,
            )
        )

        history_row = build_history_row(
            epoch=epoch,
            training_statistics=(
                training_statistics
            ),
            validation_result=(
                validation_result
            ),
        )

        history.append(history_row)

        save_history(
            history,
            output_root
            / "training_history.csv",
        )

        current_score = validation_result[
            "mean_source_macro_f1"
        ]

        print(
            f"Epoch {epoch:02d} | "
            f"loss: "
            f"{training_statistics['total_loss']:.4f} | "
            f"source accuracy: "
            f"{validation_result['mean_source_accuracy']:.4f} | "
            f"source macro-F1: "
            f"{current_score:.4f}"
        )

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_path=(
                    output_root
                    / "best_checkpoint.pt"
                ),
                epoch=epoch,
                configuration=configuration,
                backbone=backbone,
                classifier=classifier,
                domain_discriminator=(
                    domain_discriminator
                ),
                optimizer=optimizer,
                validation_result=(
                    validation_result
                ),
            )

            save_json(
                validation_result,
                output_root
                / "best_source_validation_metrics.json",
            )

            print("  Saved new best checkpoint.")

        else:
            epochs_without_improvement += 1

            print(
                "  No improvement for "
                f"{epochs_without_improvement} "
                "epoch(s)."
            )

        if (
            epochs_without_improvement
            >= patience_limit
        ):
            print(
                "\nEarly stopping: "
                f"{patience_limit} epochs "
                "without improvement."
            )

            break

    manifest = {
        "method": configuration["method"]["name"],
        "run_name": configuration["run_name"],
        "seed": configuration["seed"],
        "command": " ".join(
            [sys.executable] + sys.argv
        ),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "best_epoch": best_epoch,
        "best_mean_source_validation_macro_f1": (
            best_score
        ),
        "checkpoint_selection_metric": (
            "mean_source_validation_macro_f1"
        ),
        "target_labels_used_during_training": False,
        "target_labels_used_for_checkpoint_selection": (
            False
        ),
    }

    save_json(
        manifest,
        output_root / "run_manifest.json",
    )

    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        "Best mean source-validation macro-F1: "
        f"{best_score:.4f}"
    )

    print(
        f"Results saved to:\n{output_root}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "source_only",
            "dan",
            "dann",
            "cdan",
        ],
    )

    parser.add_argument(
        "--run-name",
        default=None,
    )

    parser.add_argument(
        "--maximum-epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--mmd-weight",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--maximum-grl-strength",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--discriminator-learning-rate",
        type=float,
        default=None,
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    configuration = load_configuration(
        project_root=project_root,
        method_name=arguments.method,
    )

    configuration = (
        apply_command_line_overrides(
            configuration=configuration,
            arguments=arguments,
        )
    )

    set_seed(configuration["seed"])

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Method: "
        f"{configuration['method']['name']}"
    )

    print(f"Run: {configuration['run_name']}")
    print(f"Device: {device}")

    train_experiment(
        project_root=project_root,
        configuration=configuration,
        device=device,
    )


if __name__ == "__main__":
    main()
