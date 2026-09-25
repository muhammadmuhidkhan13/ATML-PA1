import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from task4.data.cifar10 import get_cifar10_loaders
from task4.methods.manifold_mixup import manifold_mixup
from task4.methods.proser import classifier_placeholder_losses
from task4.methods.proser import combined_proser_loss
from task4.methods.proser import data_placeholder_loss
from task4.models.resnet_cifar import CIFARResNet18
from task4.train import get_output_paths
from task4.train import load_config
from task4.train import save_history
from task4.train import set_seed


def initialize_from_vanilla(
    checkpoint_path,
    known_classes,
    total_classes,
    device
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    vanilla_state = checkpoint["model_state_dict"]

    model = CIFARResNet18(
        num_classes=total_classes
    )

    shared_state = {
        name: value
        for name, value in vanilla_state.items()
        if not name.startswith("fc.")
    }

    model.load_state_dict(
        shared_state,
        strict=False
    )

    with torch.no_grad():
        model.fc.weight[:known_classes].copy_(
            vanilla_state["fc.weight"]
        )

        model.fc.bias[:known_classes].copy_(
            vanilla_state["fc.bias"]
        )

    return model.to(device)


def evaluate_known(
    model,
    loader,
    device,
    known_classes=10
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            logits = model(images)
            known_logits = logits[:, :known_classes]

            loss = F.cross_entropy(
                known_logits,
                labels
            )

            predictions = known_logits.argmax(dim=1)

            total_loss += (
                loss.item() * labels.size(0)
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += labels.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples
    }


def train_proser_epoch(
    model,
    loader,
    optimizer,
    device,
    known_classes,
    beta,
    gamma,
    mixup_alpha
):
    model.train()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_classifier_placeholder_loss = 0.0
    total_data_placeholder_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)

        usable_size = (
            images.size(0) // 2
        ) * 2

        if usable_size < 4:
            continue

        images = images[:usable_size]
        labels = labels[:usable_size]

        half_size = usable_size // 2

        ordinary_images = images[:half_size]
        ordinary_labels = labels[:half_size]

        placeholder_images = images[half_size:]
        placeholder_labels = labels[half_size:]

        if placeholder_labels.unique().numel() < 2:
            continue

        optimizer.zero_grad()

        ordinary_logits = model(
            ordinary_images
        )

        classifier_losses = (
            classifier_placeholder_losses(
                ordinary_logits,
                ordinary_labels,
                known_classes=known_classes
            )
        )

        placeholder_representations = (
            model.forward_until_layer2(
                placeholder_images
            )
        )

        mixup_output = manifold_mixup(
            placeholder_representations,
            placeholder_labels,
            alpha=mixup_alpha
        )

        mixed_logits, _ = (
            model.forward_from_layer2(
                mixup_output[
                    "mixed_representations"
                ]
            )
        )

        data_loss = data_placeholder_loss(
            mixed_logits,
            known_classes=known_classes
        )

        loss = combined_proser_loss(
            classification_loss=classifier_losses[
                "classification_loss"
            ],
            classifier_placeholder_loss=classifier_losses[
                "classifier_placeholder_loss"
            ],
            data_placeholder_loss_value=data_loss,
            beta=beta,
            gamma=gamma
        )

        loss.backward()
        optimizer.step()

        known_logits = ordinary_logits[
            :, :known_classes
        ]

        predictions = known_logits.argmax(dim=1)

        batch_size = ordinary_labels.size(0)

        total_loss += loss.item() * batch_size

        total_classification_loss += (
            classifier_losses[
                "classification_loss"
            ].item() * batch_size
        )

        total_classifier_placeholder_loss += (
            classifier_losses[
                "classifier_placeholder_loss"
            ].item() * batch_size
        )

        total_data_placeholder_loss += (
            data_loss.item() * batch_size
        )

        total_correct += (
            predictions == ordinary_labels
        ).sum().item()

        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError(
            "No usable PROSER training batches were produced."
        )

    return {
        "loss": total_loss / total_examples,
        "classification_loss":
            total_classification_loss
            / total_examples,
        "classifier_placeholder_loss":
            total_classifier_placeholder_loss
            / total_examples,
        "data_placeholder_loss":
            total_data_placeholder_loss
            / total_examples,
        "accuracy": total_correct / total_examples
    }


def train(
    config,
    maximum_epochs=None,
    run_name=None
):
    seed = config["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    known_classes = config["model"][
        "known_classes"
    ]

    total_classes = config["model"][
        "total_classes"
    ]

    initialization_checkpoint = Path(
        config["model"][
            "initialization_checkpoint"
        ]
    )

    if not initialization_checkpoint.exists():
        raise FileNotFoundError(
            "Vanilla checkpoint not found: "
            f"{initialization_checkpoint}"
        )

    loaders = get_cifar10_loaders(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_randaugment=False,
        seed=seed
    )

    model = initialize_from_vanilla(
        checkpoint_path=initialization_checkpoint,
        known_classes=known_classes,
        total_classes=total_classes,
        device=device
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"][
            "weight_decay"
        ]
    )

    configured_epochs = config["training"][
        "epochs"
    ]

    if maximum_epochs is None:
        epochs = configured_epochs
    else:
        epochs = maximum_epochs

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    )

    beta = config["proser"]["beta"]
    gamma = config["proser"]["gamma"]
    mixup_alpha = config["proser"][
        "mixup_alpha"
    ]

    output_paths = get_output_paths(
        config,
        run_name
    )

    output_paths["checkpoint"].parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history = []
    best_val_accuracy = -1.0
    best_epoch = None

    print("Method: proser")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(
        "Initialization checkpoint: "
        f"{initialization_checkpoint}"
    )
    print(
        f"Classes: {known_classes} known + "
        f"{total_classes - known_classes} dummy"
    )

    for epoch in range(1, epochs + 1):
        learning_rate = optimizer.param_groups[
            0
        ]["lr"]

        train_metrics = train_proser_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            known_classes=known_classes,
            beta=beta,
            gamma=gamma,
            mixup_alpha=mixup_alpha
        )

        val_metrics = evaluate_known(
            model=model,
            loader=loaders["val"],
            device=device,
            known_classes=known_classes
        )

        history.append({
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss":
                train_metrics["loss"],
            "classification_loss":
                train_metrics[
                    "classification_loss"
                ],
            "classifier_placeholder_loss":
                train_metrics[
                    "classifier_placeholder_loss"
                ],
            "data_placeholder_loss":
                train_metrics[
                    "data_placeholder_loss"
                ],
            "train_accuracy":
                train_metrics["accuracy"],
            "validation_loss":
                val_metrics["loss"],
            "validation_accuracy":
                val_metrics["accuracy"]
        })

        print(
            f"Epoch {epoch:03d} | "
            f"loss: {train_metrics['loss']:.4f} | "
            f"classification: "
            f"{train_metrics['classification_loss']:.4f} | "
            f"classifier placeholder: "
            f"{train_metrics['classifier_placeholder_loss']:.4f} | "
            f"data placeholder: "
            f"{train_metrics['data_placeholder_loss']:.4f} | "
            f"train accuracy: "
            f"{train_metrics['accuracy']:.4f} | "
            f"validation accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )

        if (
            val_metrics["accuracy"]
            > best_val_accuracy
        ):
            best_val_accuracy = (
                val_metrics["accuracy"]
            )

            best_epoch = epoch

            torch.save({
                "method": "proser",
                "epoch": epoch,
                "validation_accuracy":
                    best_val_accuracy,
                "model_state_dict":
                    model.state_dict(),
                "config": config
            }, output_paths["checkpoint"])

            print("  Saved new best checkpoint.")

        scheduler.step()

        save_history(
            history,
            output_paths["history"]
        )

    checkpoint = torch.load(
        output_paths["checkpoint"],
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_metrics = evaluate_known(
        model=model,
        loader=loaders["test"],
        device=device,
        known_classes=known_classes
    )

    summary = {
        "method": "proser",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_accuracy":
            best_val_accuracy,
        "cifar10_test_accuracy":
            test_metrics["accuracy"],
        "cifar10_test_loss":
            test_metrics["loss"],
        "known_classes": known_classes,
        "dummy_classes":
            total_classes - known_classes,
        "initialization_checkpoint":
            str(initialization_checkpoint),
        "checkpoint":
            str(output_paths["checkpoint"])
    }

    output_paths["summary"].parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_paths["summary"],
        "w"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2
        )

    print(f"\nBest epoch: {best_epoch}")

    print(
        "Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        "CIFAR-10 test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        "Checkpoint: "
        f"{output_paths['checkpoint']}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="task4/configs/proser.yaml"
    )

    parser.add_argument(
        "--maximum-epochs",
        type=int,
        default=None
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    config = load_config(
        arguments.config
    )

    train(
        config=config,
        maximum_epochs=arguments.maximum_epochs,
        run_name=arguments.run_name
    )


if __name__ == "__main__":
    main()